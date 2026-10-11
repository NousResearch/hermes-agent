"""Idle exit for an auto-started (unmanaged) gateway.

A client that finds no gateway for its home starts a detached ``gateway run --idle-exit``
(``hermes_cli.gateway_runtime_start``). Nothing else ever stops that process, so every throwaway home
(scripts, tests, one-shot ``chat -q``) used to leave a ~270 MB daemon behind. This watcher ends it once
it has served nobody and has nothing to do for ``gateway.unmanaged_idle_exit_seconds`` (default 600,
``0`` disables):

- no attached client (live ticketed WebSocket, outstanding ticket, or recent HTTP/WS activity);
- no messaging adapter configured, connected or reconnecting (any profile);
- no runnable cron job and no kanban task the dispatcher must act on;
- no queued/started admission, live managed worker, in-flight agent/cron/API run, background
  process or async delegation, active heartbeat/loop, pending handoff or hosted-room task.

Every probe fails CLOSED: an unreadable source counts as work, so the gateway stays up. A gateway with
a supervisor (systemd/launchd/Desktop/external) or started without ``--idle-exit`` (installed services,
``hermes gateway run`` by hand, ``hermes gateway ensure`` from Desktop) never arms this. The exit
publishes ``draining`` with ``drain_reason: idle_exit`` before anything else, so a client that races it
waits for the process to go and starts a fresh one instead of failing.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("gateway.run")

DEFAULT_IDLE_EXIT_SECONDS = 600.0
IDLE_EXIT_REASON = "idle_exit"
# Head of the ``exit_reason`` persisted with ``stopped``: a reconnecting client's ``--recover``
# (ui-tui/scripts/gateway_bootstrap.py) tells this self-ended exit from an operator stop by it.
IDLE_EXIT_STOP_PREFIX = "auto-started gateway idle"
# Kanban statuses the embedded dispatcher acts on without a human (promotion, spawn, reclaim).
_KANBAN_DUE_STATUSES = ("scheduled", "ready", "running", "review")


def idle_exit_seconds(runner) -> float:
    """``gateway.unmanaged_idle_exit_seconds`` from the launch home; 0 disables, junk keeps the default."""
    from gateway.run import _load_gateway_config
    section = (_load_gateway_config() or {}).get("gateway")
    value = section.get("unmanaged_idle_exit_seconds") if isinstance(section, dict) else None
    if value is None:
        return DEFAULT_IDLE_EXIT_SECONDS
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        logger.warning("gateway.unmanaged_idle_exit_seconds=%r is not a number; using %.0f", value,
                       DEFAULT_IDLE_EXIT_SECONDS)
        return DEFAULT_IDLE_EXIT_SECONDS
    return max(0.0, seconds)


def note_client_activity(runner) -> None:
    runner._idle_exit_last_activity = time.monotonic()


@contextlib.contextmanager
def attached_client(runner):
    """Count one live client connection for the idle check (and stamp activity on both ends)."""
    note_client_activity(runner)
    runner._idle_exit_clients = getattr(runner, "_idle_exit_clients", 0) + 1
    try:
        yield
    finally:
        runner._idle_exit_clients -= 1
        note_client_activity(runner)


def _messaging_busy(runner) -> str | None:
    from gateway.config import Platform
    local = {Platform.LOCAL}
    live = [p for p in (getattr(runner, "adapters", None) or {}) if p not in local]
    for adapters in (getattr(runner, "_profile_adapters", None) or {}).values():
        live.extend(adapters)
    for pending in (getattr(runner, "_profile_failed_platforms", None) or {}).values():
        live.extend(pending or {})
    live.extend(getattr(runner, "_failed_platforms", None) or {})
    config = getattr(runner, "config", None)
    if config is not None:
        live.extend(p for p, cfg in config.platforms.items() if p not in local and getattr(cfg, "enabled", False))
    return f"messaging {sorted({getattr(p, 'value', str(p)) for p in live})}" if live else None


def _served_homes(runner) -> list[Path]:
    from gateway.run import _cron_tick_profile_homes
    from gateway.session_authorities import all_authorities
    homes = {Path(a.profile_id) for a in all_authorities(runner)}
    homes.update(Path(h) for _name, h in _cron_tick_profile_homes(runner.config))
    return sorted(homes)


def _cron_busy(homes) -> str | None:
    from cron.jobs import is_job_runnable, is_terminal_job, load_jobs, use_cron_store
    from cron.scheduler import get_running_job_ids
    if get_running_job_ids():
        return "cron job running"
    for home in homes:
        if not (home / "cron" / "jobs.json").exists():
            continue
        with use_cron_store(home):
            jobs = load_jobs()
        if any(is_job_runnable(job) and not is_terminal_job(job) for job in jobs):
            return f"cron jobs scheduled in {home}"
    return None


def _kanban_busy() -> str | None:
    from hermes_cli import kanban_db
    marks = ",".join("?" * len(_KANBAN_DUE_STATUSES))
    for meta in kanban_db.list_boards(include_archived=False):
        path = kanban_db.kanban_db_path(board=meta.get("slug"))
        if not path.exists():
            continue
        with contextlib.closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)) as conn:
            if conn.execute(f"SELECT 1 FROM tasks WHERE status IN ({marks}) LIMIT 1",
                            _KANBAN_DUE_STATUSES).fetchone():
                return f"kanban work on board {meta.get('slug')}"
    return None


def _ledger_busy(runner) -> str | None:
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(runner):
        db = authority.db
        if db._read_all("SELECT 1 FROM session_admissions WHERE status IN ('queued','started') LIMIT 1"):
            return f"admissions pending in {authority.profile_id}"
        if db._read_all("SELECT 1 FROM worker_executions WHERE status != 'terminal' LIMIT 1"):
            return f"managed worker live in {authority.profile_id}"
        service = getattr(authority, "hosted_room_service", None)
        if service is not None:
            status = service.status()
            if status.get("current_tasks") or status.get("leased_rooms"):
                return f"hosted-room work in {authority.profile_id}"
    return None


def _store_gates_busy(homes) -> str | None:
    from gateway.run_idle_gates import (
        profile_has_active_heartbeat, profile_has_active_loop, profile_has_pending_handoff)
    for home in homes:
        for label, gate in (("heartbeat", profile_has_active_heartbeat), ("loop", profile_has_active_loop),
                            ("handoff", profile_has_pending_handoff)):
            if gate(home):
                return f"active {label} in {home}"
    return None


def busy_reason(runner, *, window: float) -> str | None:
    """Why the gateway must stay up, or None when it may exit. Fails closed on any error."""
    try:
        clients = getattr(runner, "_idle_exit_clients", 0)
        if clients:
            return f"{clients} client(s) attached"
        tickets = getattr(runner, "session_ticket_store", None)
        if tickets is not None and tickets.outstanding():
            return "client attaching"
        quiet = time.monotonic() - getattr(runner, "_idle_exit_last_activity", 0.0)
        if quiet < window:
            return f"client activity {quiet:.0f}s ago"
        if runner._draining or not runner._running:
            return "already stopping"
        if runner._active_work_count():
            return "agent work in flight"
        if runner._scale_to_zero_has_live_background_work():
            return "background work in flight"
        homes = _served_homes(runner)
        return (_messaging_busy(runner) or _ledger_busy(runner) or _cron_busy(homes) or _kanban_busy()
                or _store_gates_busy(homes))
    except Exception as exc:
        logger.debug("idle-exit probe failed; staying up", exc_info=True)
        return f"idle probe failed ({type(exc).__name__})"


def _begin_idle_exit(runner, window: float) -> None:
    """Withdraw admission synchronously (no await between the last idle check and here), then stop."""
    runner._draining = True
    runner.session_runtime_descriptor.update(state="draining", capabilities=[], drain_reason=IDLE_EXIT_REASON)
    runner._exit_reason = f"{IDLE_EXIT_STOP_PREFIX} for {window:.0f}s (gateway.unmanaged_idle_exit_seconds)"
    logger.info("Exiting: %s; the next client starts a fresh gateway", runner._exit_reason)
    runner._idle_exit_stop_task = asyncio.ensure_future(runner.stop())


async def unmanaged_idle_exit_watcher(runner) -> None:
    window = idle_exit_seconds(runner)
    if window <= 0:
        logger.info("Auto-started gateway idle exit disabled (gateway.unmanaged_idle_exit_seconds=0)")
        return
    logger.info("Auto-started gateway: exits after %.0fs with no clients, adapters, cron/kanban work or "
                "admissions (gateway.unmanaged_idle_exit_seconds)", window)
    interval = min(30.0, max(0.5, window / 4))
    idle_since = None
    while runner._running and not runner._draining:
        await asyncio.sleep(interval)
        window = idle_exit_seconds(runner)
        if window <= 0:
            idle_since = None
            continue
        # The store/cron/kanban reads touch disk; keep them off the loop, then re-check the
        # loop-only facts (clients, tickets, draining) synchronously right before acting.
        reason = await asyncio.to_thread(busy_reason, runner, window=window)
        if reason is not None:
            idle_since = None
            continue
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since < interval and window > interval:
            continue  # two consecutive idle readings, so one racing write cannot slip between them
        if busy_reason(runner, window=window) is None:
            _begin_idle_exit(runner, window)
            return


def arm_unmanaged_idle_exit(runner) -> bool:
    """Start the watcher for an ``--idle-exit`` launch that no supervisor owns. Returns whether it armed."""
    from gateway.control_socket import _detect_supervisor
    supervisor = _detect_supervisor()
    if supervisor != "manual":
        logger.info("Auto-started gateway idle exit not armed: supervised by %s", supervisor)
        return False
    note_client_activity(runner)
    runner._spawn_supervised(lambda: unmanaged_idle_exit_watcher(runner), "unmanaged_idle_exit_watcher")
    return True
