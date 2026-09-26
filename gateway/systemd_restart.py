"""Systemd restart state machine and recovery."""
from __future__ import annotations

import os
import subprocess
import time

from gateway import (
    process_liveness,
    restart,
    service_identity,
    signal_restart,
    systemd_identity,
    systemd_runtime,
    systemd_restart_state,
    systemd_unit_state,
)
from hermes_cli.config import read_raw_config


def _systemd_scope_preamble(action: str, system: bool, *, preflight_user: bool = False) -> bool:
    """Resolve scope and enforce action prerequisites."""
    system = systemd_runtime.select_scope(system)
    if system:
        systemd_identity.require_root(action)
    elif preflight_user:
        systemd_runtime.preflight_user()
    if not systemd_identity.unit_path(system).exists():
        print("✗ Gateway service is not installed")
        raise SystemExit(1)
    return system

def _agent_timeout_setting(env_var: str, key: str, parse) -> float:
    """Parse an agent timeout from environment first, then config."""
    env_raw = os.getenv(env_var)
    if env_raw is not None and str(env_raw).strip() != "":
        return parse(env_raw)
    cfg = read_raw_config()
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
    if isinstance(agent_cfg, dict) and key in agent_cfg:
        return parse(agent_cfg.get(key))
    return parse(None)


def _get_restart_exit_wait_budget() -> float:
    """CLI wait for gateway exit after SIGUSR1 / self-restart (#77184)."""
    return restart.resolve_restart_exit_wait_budget(
        restart.get_restart_drain_timeout(),
        _agent_timeout_setting(
            "HERMES_RESTART_AFTER_TURN_TIMEOUT",
            "restart_after_turn_timeout",
            restart.parse_restart_after_turn_timeout,
        ),
    )


def systemd_restart(system: bool = False):
    system = _systemd_scope_preamble("restart", system, preflight_user=True)
    # HERMES_HOME sync happens in refresh's systemd_unit_is_current gate; its os.environ mutation
    # persists for the get_running_pid / drain-timeout reads below.
    systemd_unit_state.refresh_if_needed(system=system)
    from gateway.status import get_running_pid
    pid = get_running_pid() or systemd_restart_state._systemd_main_pid(system=system)
    if pid is not None and process_liveness.probe_gateway_loop_liveness(pid) == process_liveness.GATEWAY_LOOP_WEDGED:
        # Event loop provably dead: SIGUSR1 can't drain it, so bounded SIGTERM → SIGKILL and let systemd relaunch.
        print(
            # Health probe says the event loop is provably dead (#81642): SIGUSR1 can never drain it, so the
            # graceful wait below would burn the full budget. A busy-but-alive gateway (fresh heartbeat)
            # never takes this path — its in-flight work, including the #86684 cron drain floor, keeps the
            # full graceful budget.
            # Health probe says the event loop is provably dead (#81642): the gateway cannot process a
            # graceful shutdown, so waiting the full drain budget only stalls the restart (and `hermes
            # update` behind it) for 180s. Bounded escalation instead: SIGTERM grace → SIGKILL → proceed,
            # ~10s worst case. Never taken for a busy-but-alive gateway — a fresh heartbeat keeps the drain
            # path (and the #86684 cron drain floor) fully intact.
            f"⚠ Gateway PID {pid} event loop is unresponsive — "
            "skipping graceful drain and forcing a bounded stop..."
        )
        process_liveness._escalate_wedged_gateway(pid)
        svc = service_identity.service_name()
        systemd_runtime.run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
        systemd_runtime.run_systemctl(["restart", svc], system=system, check=False, timeout=90)
        systemd_restart_state._wait_for_systemd_service_restart(system=system, previous_pid=pid)
        return
    if pid is not None:
        service_action = _systemd_graceful_restart_action(system, pid)
        if service_action:
            _systemd_reset_and_run(service_action, system=system, previous_pid=pid)
        return

    if systemd_restart_state._recover_pending_systemd_restart(system=system, previous_pid=pid):
        return
    _systemd_reset_and_run("restart", system=system, previous_pid=pid)

def _systemd_graceful_restart_action(system: bool, pid: int) -> str | None:
    """SIGUSR1-drain the live gateway ``pid``; return the follow-up ``systemctl`` verb (``"start"`` /
    ``"restart"``) the caller must still issue, or None when systemd already owns the relaunch."""
    scope_label = systemd_runtime.scope_label(system).capitalize()
    # Graceful in-band restart, mirroring the systemd branch. Previously this sent a bare SIGTERM and waited
    # ``restart.get_restart_drain_timeout()`` — which defaults to 0, so the wait could never succeed and every
    # restart fell through to ``kickstart -k``. A bare SIGTERM also leaves ``restart_requested`` False, so
    # the gateway exits 1 instead of 75 and reports itself to chat as "shutting down" rather than
    # "restarting", losing the resume_pending handoff. SIGUSR1 is the drain-aware path: refuse new turns,
    # wait for in-flight work (``agent.restart_after_turn_timeout``), then stop() within
    # ``agent.restart_drain_timeout``. The wait budget must cover BOTH phases plus headroom (#77184) — the
    # raw drain timeout covers only the second. Announce the wait BEFORE it runs: it can last the full
    # budget while the old gateway finishes in-flight agent runs, and it streams into surfaces with no other
    # feedback — the desktop updater's live output most of all, where a silent stop here reads as "update
    # stuck" (#44515).
    wait_budget = _get_restart_exit_wait_budget()
    print(
        f"⏳ {scope_label} service restarting gracefully (PID {pid}) — "
        f"waiting up to {wait_budget:.0f}s for in-flight turns + drain..."
    )
    from gateway.drain_report import drain_progress_reporter

    if not signal_restart._graceful_restart_via_sigusr1(
        pid,
        wait_budget,
        on_progress=drain_progress_reporter(budget_s=wait_budget),
    ):
        print(f"⚠ Graceful restart did not complete within {int(wait_budget)}s; forcing a service restart...")
        return "restart"

    # Exit 75 hands restart ownership to systemd; observe that replacement rather than restarting again.
    replacement_observed: list[bool] = []
    if systemd_restart_state._wait_for_systemd_service_restart(system=system, previous_pid=pid, replacement_observed=replacement_observed):
        return None
    if replacement_observed or systemd_restart_state._systemd_service_is_start_limited(system=system):
        return None

    # A replacement may have started but not reached runtime readiness in time; never stop that generation.
    props = systemd_restart_state._read_systemd_unit_properties(system=system)
    if not props:
        return None
    replacement_pid = systemd_restart_state._systemd_main_pid_from_props(props)
    if (
        props.get("ActiveState") in {"active", "activating", "reloading"}
        or props.get("SubState") == "auto-restart"
        or (replacement_pid is not None and replacement_pid != pid)
    ):
        return None

    print("⚠ Systemd did not relaunch the gateway after its graceful exit; starting the inactive service...")
    # ``start`` is intentionally idempotent: a replacement appearing after the snapshot must not be stopped.
    return "start"

def _systemd_reset_and_run(action: str, *, system: bool, previous_pid) -> None:
    """``reset-failed`` then ``systemctl <action>``, then wait for the relaunch. Start-limit
    rejection prints the wait hint instead of raising; a 90s timeout prints where to look."""
    svc = service_identity.service_name()
    systemd_runtime.run_systemctl(["reset-failed", svc], system=system, check=False, timeout=30)
    try:
        systemd_runtime.run_systemctl([action, svc], system=system, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        if systemd_restart_state._systemd_error_indicates_start_limit(exc) or systemd_restart_state._systemd_service_is_start_limited(system=system):
            systemd_restart_state._print_systemd_start_limit_wait(system=system)
            return
        raise
    except subprocess.TimeoutExpired:
        print(
            f"Gateway {systemd_runtime.scope_label(system)} service is still restarting after 90s; "
            "check `hermes gateway status` or logs for final state."
        )
        return
    systemd_restart_state._wait_for_systemd_service_restart(system=system, previous_pid=previous_pid)
