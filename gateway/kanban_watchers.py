"""Kanban board watcher methods for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3).
These are the background-loop methods that subscribe to kanban boards, deliver
notifications/artifacts, and drive the multi-agent dispatcher. They use only
``self`` state, so they live on a mixin that ``GatewayRunner`` inherits — the
``self._kanban_*`` call sites resolve identically via the MRO, making this a
behavior-neutral move that lifts ~1,000 LOC out of run.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
from contextvars import Context
from pathlib import Path
from typing import Any, Callable, Optional

from agent.i18n import t

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")


_LOCAL_PATH_RE = re.compile(
    r"(?<![\w:/])(?:/(?:Users|home|private|tmp|var|etc|workspace)/[^\s,;]+|"
    r"[A-Za-z]:\\[^\s,;]+)"
)


def _safe_review_reason(value: Any, limit: int = 160) -> str:
    """Return a mobile-friendly review reason safe for external delivery."""
    from agent.redact import redact_sensitive_text

    reason = redact_sensitive_text(
        "" if value is None else str(value),
        force=True,
        redact_url_credentials=True,
    )
    reason = _LOCAL_PATH_RE.sub("[local path]", reason)
    reason = " ".join(reason.split())
    if len(reason) > limit:
        reason = reason[: limit - 1].rstrip() + "…"
    return reason


def _resolve_auto_decompose_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int]":
    """Resolve the live (enabled, per_tick) auto-decompose settings.

    Read fresh from config on every dispatcher tick (#49638) so that flipping
    ``kanban.auto_decompose: false`` to STOP runaway fan-out takes effect on the
    next tick instead of requiring a gateway restart. Auto-decompose is a
    safety toggle — a user who sees it create and launch tasks they didn't
    intend reaches for this flag to halt it, and a stale boot-captured value
    silently ignoring that change is the bug reported in #49638.

    Fails **safe**: if the config read raises, return ``(False, 3)`` — a
    transient read error must never re-enable a feature the user turned off,
    nor fall back to the burst-prone default-on behaviour. ``per_tick`` is
    clamped to ``>= 1``.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("auto_decompose", True))
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    if per_tick < 1:
        per_tick = 1
    return enabled, per_tick


def _kanban_dispatch_allowed() -> bool:
    """Return False while the global emergency stop (`hermes pause`) is engaged.

    Checked every dispatcher tick BEFORE spawning new workers so a pause takes
    effect on the next tick without a gateway restart. In-flight workers are
    never touched — this only stops NEW spawns. Fails open: if the estop
    module is unimportable, dispatch proceeds (the sentinel gate must not
    become a new crash surface for the dispatcher).
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the calling task's context onto the worker
    thread. Supervised Kanban ticks are process-owned writers; if that copy
    still carries a ``delegate_task`` child marker, ``write_txn``
    false-trips. Since watchers spawn from a fresh ``Context``
    (``_spawn_supervised``), this offload-boundary scrub is defense in
    depth: it covers non-supervised spawn paths and any task context frozen
    before spawn isolation shipped. An empty Context keeps the DB guard
    intact for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work (dispatcher + notifier writers)
    without inheriting request-local ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take an exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway process machine-wide may run the embedded kanban
    dispatcher: concurrent dispatchers double the reclaim frequency (each
    runs its own ``release_stale_claims`` → promote → dispatch loop), double
    claim-attempt events in the event log, and — with ``wal_autocheckpoint=0`` —
    concurrent manual WAL checkpoints can corrupt index pages. The
    ``dispatch_in_gateway`` config flag is the primary control; this lock is the
    backstop that survives config drift and same-profile restart races.

    Delegates to :func:`gateway.status._try_acquire_file_lock` (``fcntl`` on
    POSIX, ``msvcrt`` on Windows) so the guard is cross-platform.

    Returns ``(handle, "held")`` on success — the caller keeps the file handle
    for the process lifetime and **must** release it via
    :func:`_release_singleton_lock` when done. ``(None, "contended")`` when
    another process holds the lock (caller must NOT dispatch). ``(None,
    "unavailable")`` when locking cannot be performed (non-POSIX filesystem
    without flock, or the status.py helpers are unimportable) — caller falls
    back to config-only control.
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a dispatcher singleton lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock
        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def _wake_scope_id(adapter: Any, sub: dict) -> Optional[str]:
    """Return the tenant scope (Slack workspace) a subscription's wake keys to.

    ``build_session_key()`` includes ``SessionSource.scope_id`` on platforms
    where one bot serves several isolated tenants, so a wake source must carry
    the same scope as inbound messages from that chat to resolve to the same
    session.

    The subscription's persisted ``delivery_metadata`` wins over the adapter's
    live chat → scope mapping, because it records the scope the subscription was
    created from; the mapping only covers rows that stored no metadata. ``None``
    means the chat has no scope, which is what an unscoped platform's key
    contains.
    """
    delivery_meta = sub.get("delivery_metadata")
    if isinstance(delivery_meta, dict):
        for key in ("scope_id", "slack_team_id", "team_id"):
            value = delivery_meta.get(key)
            if value:
                return str(value)
    resolver = getattr(adapter, "scope_id_for_chat", None)
    if callable(resolver):
        try:
            resolved = resolver(str(sub.get("chat_id") or ""))
        except Exception as exc:
            # An adapter-side lookup failure yields no scope, never an error.
            logger.debug(
                "kanban notifier: scope lookup failed for chat %s: %s",
                sub.get("chat_id"),
                exc,
                exc_info=True,
            )
            return None
        if resolved:
            return str(resolved)
    return None


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        """Return whether this gateway currently owns the singleton lock."""
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership before releasing the OS lock."""
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        _release_singleton_lock(handle)

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        For each subscription row, fetches ``task_events`` newer than the
        stored cursor with kind in the terminal set (``completed``,
        ``blocked``, ``gave_up``, ``crashed``, ``timed_out``,
        ``review_requested``, ``changes_requested``,
        ``block_loop_detected``). Sends one
        message per new event to ``(platform, chat_id, thread_id)``,
        then advances the cursor. The subscription is removed only when the
        task is ``archived``. A ``done`` task can be reopened for review or
        continuation, so its subscription and origin-session ownership must
        survive completion. Cursor advancement prevents old events replaying
        when that happens.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Each gateway polls only subscriptions owned by profiles whose
        adapters it hosts. The dispatch-owning gateway also handles legacy
        subscriptions without a profile stamp.
        """
        # Dispatch and delivery have separate ownership. A deployment may run
        # one dispatcher while each profile has its own gateway credentials;
        # those adapter-owning gateways must still poll and deliver their own
        # subscriptions. Legacy rows without a notifier_profile are visible
        # only while this process holds the actual singleton dispatcher lock.
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        # "status" covers dashboard drag-drop and `_set_status_direct()`
        # writes — surface those transitions to subscribers too.
        # ``review_requested`` wakes the origin subscriber like a block does,
        # but is not a block (see kanban_db.request_review); the task is not
        # archived, so the subscription stays alive and later review
        # cycles keep notifying.
        TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked", "block_loop_detected", "review_requested", "changes_requested")
        # Subscriptions are removed only when the task reaches the irreversible
        # archived status. ``done`` is reversible in review/controller flows,
        # so removing its subscription would silence a later reopen. We used
        # to also unsub on any terminal
        # event kind (gave_up / crashed / timed_out / blocked), but that
        # silently dropped the user out of the loop whenever the dispatcher
        # respawned the task: a worker that crashes, gets reclaimed, runs
        # again, and crashes a second time would only notify on the first
        # crash because the subscription was deleted after the first event.
        # Same shape as the reblock-after-unblock cycle that PR #22941
        # fixed for `blocked`. Keeping the subscription alive until the
        # task is archived lets the cursor (advanced atomically by
        # claim_unseen_events_for_sub) handle dedup, and any retry-loop
        # event reaches the user.
        # Per-subscription send-failure counter. Adapter.send raising
        # means the chat is dead (deleted, bot kicked, etc.) — after N
        # consecutive send failures the sub is dropped so we don't spin
        # against a dead chat every 5 seconds forever.
        # Raised from 3 to 12 (~60s at the 5s tick cadence): now that a
        # reported SendResult(success=False) also lands here (see the
        # delivery loop below), a transient Telegram/API outage of a few
        # ticks must NOT permanently unsubscribe a live review-gate channel.
        # A genuinely dead chat still drops, just ~60s later — a fine trade
        # for an unattended gate where a false drop means silent work pileup.
        MAX_SEND_FAILURES = 12
        sub_fail_counts: dict[tuple, int] = getattr(
            self, "_kanban_sub_fail_counts", {}
        )
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None)
        if not notifier_profile:
            notifier_profile = self._active_profile_name()
            self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        # Stale done-sub GC cadence. Subscriptions survive ``done`` (it is
        # reversible), so boards that never archive would otherwise
        # accumulate rows scanned on every 5s tick forever. The sweep is a
        # single DELETE per board, gated to once per watcher startup and at
        # most once per hour thereafter — cheap relative to the tick's own
        # per-sub claims. Retention is kanban.done_sub_retention_days in
        # config.yaml (default 30; 0 disables), re-read at each sweep so a
        # config change applies without a restart.
        _GC_INTERVAL_SECONDS = 3600.0
        _gc_next_at = 0.0  # 0 → sweep on the first tick after startup

        while self._running:
            try:
                _gc_due = time.monotonic() >= _gc_next_at
                _gc_retention_days = 30
                if _gc_due:
                    _gc_next_at = time.monotonic() + _GC_INTERVAL_SECONDS
                    try:
                        from hermes_cli.config import load_config as _load_cfg

                        _kanban_cfg = (_load_cfg() or {}).get("kanban") or {}
                        _gc_retention_days = int(
                            _kanban_cfg.get("done_sub_retention_days", 30)
                        )
                    except Exception:
                        # Fail safe on the shipped default; the sweep itself
                        # treats <= 0 as disabled.
                        _gc_retention_days = 30

                def _collect():
                    deliveries: list[dict] = []
                    include_unowned = self._owns_kanban_dispatcher_lock()
                    notifier_profiles = {notifier_profile}
                    notifier_profiles.update(
                        str(profile).strip()
                        for profile in getattr(self, "_profile_adapters", {})
                        if str(profile).strip()
                    )
                    active_platforms = {
                        getattr(platform, "value", str(platform)).lower()
                        for platform in self.adapters.keys()
                    }
                    # Widen to every platform any secondary profile has live,
                    # not just the default profile's. This is only a coarse
                    # pre-filter to skip claiming events for subs nobody can
                    # possibly deliver — the precise per-profile check (via
                    # gateway/authz_mixin.py::_authorization_adapter, which
                    # forbids default-profile fallback) still runs at delivery
                    # time below, rewinding the claim if it resolves to None.
                    # Without this, a subscription owned by a secondary
                    # profile on a platform the DEFAULT profile never
                    # connected (e.g. beta owns discord, default doesn't) was
                    # dropped here before ever being claimed — no rewind
                    # applies to an unclaimed event, so it silently never
                    # retries.
                    for _profile_adapter_map in getattr(self, "_profile_adapters", {}).values():
                        active_platforms.update(
                            getattr(platform, "value", str(platform)).lower()
                            for platform in _profile_adapter_map.keys()
                        )
                    if not active_platforms:
                        logger.debug("kanban notifier: no connected adapters; skipping tick")
                        return deliveries

                    # Enumerate every board on disk, but poll each resolved DB
                    # path once. Multiple slugs can point at the same DB when
                    # HERMES_KANBAN_DB pins the board path; without this guard
                    # one gateway could collect the same subscription/event
                    # more than once before advancing the cursor.
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: set[str] = set()
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            logger.debug(
                                "kanban notifier: skipping duplicate board slug %s for DB %s",
                                slug, resolved_db_path,
                            )
                            continue
                        seen_db_paths.add(resolved_db_path)
                        # Zero-subscription early exit: probe the board with a
                        # cheap read-only connection BEFORE the writable
                        # `connect()`. A board with no subscriptions has
                        # nothing to notify, and the writable open (schema
                        # init/migration on first open, WAL/-shm sidecars,
                        # checkpoint traffic) is exactly the per-tick cost
                        # this skip avoids.
                        try:
                            if _kb.count_notify_subs(
                                board=slug,
                                notifier_profiles=notifier_profiles,
                                include_unowned=include_unowned,
                            ) == 0:
                                logger.debug(
                                    "kanban notifier: board %s has no subscriptions owned by %s; skipping open",
                                    slug, sorted(notifier_profiles),
                                )
                                continue
                        except Exception as exc:
                            logger.debug(
                                "kanban notifier: read-only subscription probe failed "
                                "for board %s (%s); falling back to writable open",
                                slug, exc,
                            )
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                            continue
                        try:
                            if _gc_due:
                                # Hourly (plus once at startup) stale-sub GC:
                                # drop subscriptions for tasks that have been
                                # ``done`` untouched past the retention
                                # window. Best-effort — a failed sweep never
                                # blocks delivery; the next hourly gate
                                # retries it.
                                try:
                                    _purged = _kb.purge_stale_done_notify_subs(
                                        conn,
                                        max_age_days=_gc_retention_days,
                                    )
                                    if _purged:
                                        logger.info(
                                            "kanban notifier: purged %d stale done-task subscription(s) on board %s (retention %dd)",
                                            _purged, slug, _gc_retention_days,
                                        )
                                except Exception as _gc_exc:
                                    logger.debug(
                                        "kanban notifier: stale-sub GC failed for board %s: %s",
                                        slug, _gc_exc,
                                    )
                            # `connect()` runs the schema + idempotent migration
                            # on first open per process, so an explicit
                            # `init_db()` here would be redundant. Worse:
                            # `init_db()` deliberately busts the per-process
                            # cache and re-runs the migration on a *second*
                            # connection, which races the first and used to
                            # log a benign but noisy `duplicate column name`
                            # traceback (and intermittent "database is locked"
                            # — issue #21378) on every gateway start against
                            # a legacy DB. `_add_column_if_missing` now
                            # tolerates that race, but we still skip the
                            # redundant call to avoid the wasted work.
                            subs = _kb.list_notify_subs(
                                conn,
                                notifier_profiles=notifier_profiles,
                                include_unowned=include_unowned,
                            )
                            if not subs:
                                logger.debug("kanban notifier: board %s has no subscriptions", slug)
                            for sub in subs:
                                try:
                                    owner_profile = sub.get("notifier_profile") or None
                                    if owner_profile and owner_profile != notifier_profile:
                                        _owner_adapters = getattr(self, "_profile_adapters", {}).get(owner_profile)
                                        if not _owner_adapters:
                                            logger.debug(
                                                "kanban notifier: subscription for %s owned by profile %s; current profile %s has no adapter for it, skipping",
                                                sub.get("task_id"), owner_profile, notifier_profile,
                                            )
                                            continue
                                    platform = (sub.get("platform") or "").lower()
                                    if platform not in active_platforms:
                                        logger.debug(
                                            "kanban notifier: subscription for %s on %s skipped; adapter not connected",
                                            sub.get("task_id"), platform or "<missing>",
                                        )
                                        continue
                                    old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                        conn,
                                        task_id=sub["task_id"],
                                        platform=sub["platform"],
                                        chat_id=sub["chat_id"],
                                        thread_id=sub.get("thread_id") or "",
                                        kinds=TERMINAL_KINDS,
                                    )
                                    if not events:
                                        continue
                                    task = _kb.get_task(conn, sub["task_id"])
                                    logger.debug(
                                        "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                        len(events), sub["task_id"], slug, old_cursor, cursor,
                                    )
                                    deliveries.append({
                                        "sub": sub,
                                        "old_cursor": old_cursor,
                                        "cursor": cursor,
                                        "events": events,
                                        "task": task,
                                        "board": slug,
                                    })
                                except Exception as sub_exc:
                                    # Isolate per-subscription failures so one
                                    # bad subscription cannot block delivery for
                                    # all other subscriptions in this tick.
                                    logger.warning(
                                        "kanban notifier: subscription for %s on board %s failed: %s",
                                        sub.get("task_id"), slug, sub_exc,
                                    )
                        finally:
                            conn.close()
                    return deliveries

                deliveries = await asyncio.to_thread(_collect)
                for d in deliveries:
                    sub = d["sub"]
                    task = d["task"]
                    board_slug = d.get("board")
                    platform_str = (sub["platform"] or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string; skip and advance cursor so
                        # we don't replay forever.
                        await _to_thread_process_service(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        continue
                    sub_profile = sub.get("notifier_profile") or ""
                    # Route via the SAME chokepoint the authorization path uses
                    # (gateway/authz_mixin.py::_authorization_adapter): a stamped
                    # profile with its own adapter-registry entry must be served
                    # by THAT profile's same-platform adapter and must NOT silently
                    # fall back to the default profile's adapter — otherwise a
                    # secondary profile's task notification is delivered by the
                    # wrong bot (the cross-profile mis-delivery this whole change
                    # exists to fix). The helper returns None only when the profile
                    # (or default) genuinely has no adapter for the platform.
                    adapter = self._authorization_adapter(plat, sub_profile or None)
                    if adapter is None:
                        logger.debug(
                            "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                            platform_str, sub["task_id"],
                        )
                        await _to_thread_process_service(
                            self._kanban_rewind,
                            sub,
                            d["cursor"],
                            d.get("old_cursor", 0),
                            board_slug,
                        )
                        continue
                    title = (task.title if task else sub["task_id"])[:120]
                    board_tag = f"[{board_slug}] " if board_slug else ""
                    # Per-subscription failure-counter key. Hoisted out of the
                    # event loop: the wake self-post path (in the loop's
                    # ``else`` clause) needs it even when every event in the
                    # claim was skipped before reaching the send site.
                    sub_key = (
                        sub["task_id"], sub["platform"],
                        sub["chat_id"], sub.get("thread_id") or "",
                    )
                    mode = sub.get("delivery_mode") or "notify"
                    wake_agent = mode in ("notify+wake", "wake")
                    send_passive = mode != "wake"
                    # Worker handoff carried into the synthetic wake turn below
                    # (#70752): without it the woken creator only sees
                    # "Task X completed" and re-decomposes work that already
                    # exists on the board.
                    wake_handoff = ""
                    wake_review_detail = ""
                    for ev in d["events"]:
                        kind = ev.kind
                        # Identity prefix: attribute terminal pings to the
                        # worker that did the work. Makes fleets (where one
                        # chat subscribes to many tasks) legible at a glance.
                        who = (task.assignee if task and task.assignee else None)
                        tag = f"@{who} " if who else ""
                        if kind == "completed":
                            # Prefer the run's summary (the worker's
                            # intentional human-facing handoff, carried
                            # in the event payload), then fall back to
                            # task.result for legacy rows written before
                            # runs shipped.
                            handoff = ""
                            payload_summary = None
                            if ev.payload and ev.payload.get("summary"):
                                payload_summary = str(ev.payload["summary"])
                            if payload_summary:
                                lines = payload_summary.strip().splitlines()
                                h = lines[0][:200] if lines else payload_summary[:200]
                                handoff = f"\n{h}"
                                wake_handoff = h
                            elif task and task.result:
                                lines = task.result.strip().splitlines()
                                r = lines[0][:160] if lines else task.result[:160]
                                handoff = f"\n{r}"
                                wake_handoff = r
                            msg = (
                                f"✔ {board_tag}{tag}Kanban {sub['task_id']} done"
                                f" — {title}{handoff}"
                            )
                        elif kind == "blocked":
                            reason = ""
                            if ev.payload and ev.payload.get("reason"):
                                reason = f": {str(ev.payload['reason'])[:160]}"
                            msg = f"⏸ {board_tag}{tag}Kanban {sub['task_id']} blocked{reason}"
                        elif kind == "gave_up":
                            err = ""
                            if ev.payload and ev.payload.get("error"):
                                err = f"\n{str(ev.payload['error'])[:200]}"
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} gave up "
                                f"after repeated spawn failures{err}"
                            )
                        elif kind == "crashed":
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} worker crashed "
                                f"(pid gone); dispatcher will retry"
                            )
                        elif kind == "timed_out":
                            limit = 0
                            if ev.payload and ev.payload.get("limit_seconds"):
                                limit = int(ev.payload["limit_seconds"])
                            msg = (
                                f"⏱ {board_tag}{tag}Kanban {sub['task_id']} timed out "
                                f"(max_runtime={limit}s); will retry"
                            )
                        elif kind == "status":
                            new_status = ""
                            if ev.payload and ev.payload.get("status"):
                                new_status = str(ev.payload["status"])
                            msg = f"🔄 {board_tag}{tag}Kanban {sub['task_id']} → {new_status}"
                        elif kind == "review_requested":
                            # Implementation complete; task moved to the
                            # first-class review lane. Wake the origin thread.
                            handoff = ""
                            if ev.payload and ev.payload.get("summary"):
                                summary = str(ev.payload["summary"])
                                handoff = f"\n{summary[:200]}"
                                # Carry the worker's handoff into the wake turn
                                # like ``completed`` does: a reviewer woken with
                                # a bare "ready for review" has to re-read the
                                # board to learn what was implemented.
                                lines = summary.strip().splitlines()
                                wake_handoff = (
                                    lines[0][:200] if lines else summary[:200]
                                )
                            msg = (
                                f"👀 {board_tag}{tag}Kanban {sub['task_id']} ready for review"
                                f" — {title}{handoff}"
                            )
                        elif kind == "changes_requested":
                            payload = ev.payload or {}
                            reason = _safe_review_reason(payload.get("reason"))
                            reviewer = _safe_review_reason(payload.get("reviewer"), 48)
                            implementer = _safe_review_reason(payload.get("implementer"), 48)
                            reason_text = reason or "reviewer feedback requires changes"
                            provenance = ""
                            if reviewer:
                                provenance += f" — reviewer @{reviewer}"
                            if implementer:
                                provenance += f" → implementer @{implementer}"
                            msg = (
                                f"🛑 {board_tag}Kanban {sub['task_id']} review requested "
                                f"changes/BLOCK: {reason_text}{provenance}"
                            )
                            wake_review_detail = reason_text
                        elif kind == "block_loop_detected":
                            # A task re-blocked for the same cause past the
                            # recurrence limit and was routed to `triage` for a
                            # human decision. This is the ONE transition that
                            # exists to force human attention, yet it emits no
                            # `blocked`/`status` event — so before adding it to
                            # TERMINAL_KINDS it produced zero notification and
                            # the task stalled in triage silently. Ping loudly.
                            reason = ""
                            recurrences = None
                            if ev.payload:
                                if ev.payload.get("reason"):
                                    reason = f": {str(ev.payload['reason'])[:160]}"
                                recurrences = ev.payload.get("recurrences")
                            rc = f" (blocked {recurrences}x for the same cause)" if recurrences else ""
                            msg = (
                                f"🛑 {board_tag}{tag}Kanban {sub['task_id']} routed to TRIAGE"
                                f" — needs a human decision{rc}{reason}"
                            )
                        else:
                            # archived / unblocked are claimed by TERMINAL_KINDS
                            # (so the cursor advances past them and they can't
                            # wedge a later completed/blocked event behind an
                            # unclaimed row) but are intentionally SILENT: an
                            # archive needs no user ping, and unblocked is an
                            # internal transition. They are also excluded from
                            # _WAKE_KINDS below, so they never wake the creator.
                            continue
                        delivery_metadata = sub.get("delivery_metadata")
                        metadata: dict[str, Any] = (
                            dict(delivery_metadata)
                            if isinstance(delivery_metadata, dict)
                            else {}
                        )

                        if sub.get("thread_id") and not metadata.get("thread_id"):
                            metadata["thread_id"] = sub["thread_id"]
                        # Adapters with no push channel (the API server —
                        # ``supports_async_delivery = False``) can NEVER
                        # satisfy a text-send: ``send()`` always reports
                        # SendResult(success=False) by design (see
                        # ApiServerAdapter.send()). Treating that as a
                        # delivery failure would rewind/drop the subscription
                        # forever and — because the wake dispatch below lives
                        # in this loop's ``else`` clause — would also make the
                        # wake-on-completion path (the actual fix for the
                        # api_server wrong-session bug) unreachable. So for
                        # non-push adapters, skip the doomed send attempt
                        # entirely: there is nothing to text-notify, the
                        # creator is woken via the self-post below instead.
                        from gateway.wake import adapter_supports_push

                        if not adapter_supports_push(adapter) and wake_agent:
                            logger.debug(
                                "kanban notifier: adapter %s has no push "
                                "channel; skipping text ping for %s, relying "
                                "on wake self-post instead",
                                platform_str, sub["task_id"],
                            )
                            # Do NOT reset the failure counter here: on this
                            # path the wake self-post below IS the delivery,
                            # so the counter is resolved (reset or bumped) by
                            # the self-post outcome, not by skipping the send.
                            continue
                        if not send_passive:
                            # Wake-only subscriptions intentionally skip the
                            # visible platform message. The retained wake path
                            # below is the sole delivery — the failure counter
                            # is resolved (reset or bumped) by the wake
                            # outcome there, not by skipping the send here.
                            continue
                        try:
                            _send_res = await adapter.send(
                                sub["chat_id"], msg, metadata=metadata,
                            )
                            # A SendResult(success=False) without an exception
                            # (returned by push-capable adapters on a genuine
                            # transient failure) must count as a FAILED
                            # delivery — otherwise the cursor advances and the
                            # event is permanently lost. Adapters returning
                            # None (or anything non-SendResult shaped) keep
                            # the legacy "no exception == delivered" contract.
                            if getattr(_send_res, "success", True) is False:
                                raise RuntimeError(
                                    "adapter send() reported failure: "
                                    f"{getattr(_send_res, 'error', None) or 'unknown error'}"
                                )
                            logger.debug(
                                "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                                kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                            )
                            # After delivering the text notification, surface
                            # any artifact paths the worker referenced in
                            # ``kanban_complete(summary=..., artifacts=[...])``
                            # (or the legacy ``result`` field) as native
                            # uploads. ``extract_local_files`` finds bare
                            # absolute paths in the summary;
                            # ``send_document`` / ``send_image_file`` uploads
                            # them. Only fires on the ``completed`` event so
                            # we never spam attachments on retries.
                            if kind == "completed":
                                try:
                                    await self._deliver_kanban_artifacts(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        event_payload=getattr(ev, "payload", None),
                                        task=task,
                                    )
                                except Exception as art_exc:
                                    logger.debug(
                                        "kanban notifier: artifact delivery for %s failed: %s",
                                        sub["task_id"], art_exc,
                                    )
                            # Reset the failure counter on success.
                            sub_fail_counts.pop(sub_key, None)
                        except Exception as exc:
                            fails = sub_fail_counts.get(sub_key, 0) + 1
                            sub_fail_counts[sub_key] = fails
                            logger.warning(
                                "kanban notifier: send failed for %s on %s "
                                "(attempt %d/%d): %s",
                                sub["task_id"], platform_str, fails,
                                MAX_SEND_FAILURES, exc,
                            )
                            if fails >= MAX_SEND_FAILURES:
                                logger.warning(
                                    "kanban notifier: dropping subscription "
                                    "%s on %s after %d consecutive send failures",
                                    sub["task_id"], platform_str, fails,
                                )
                                await _to_thread_process_service(self._kanban_unsub, sub, board_slug)
                                sub_fail_counts.pop(sub_key, None)
                            else:
                                await _to_thread_process_service(
                                    self._kanban_rewind,
                                    sub,
                                    d["cursor"],
                                    d.get("old_cursor", 0),
                                    board_slug,
                                )
                            # Rewind the pre-send claim on transient failure so
                            # a later tick can retry. After too many failures,
                            # dropping the subscription is the terminal action.
                            break
                    else:
                        # All text pings delivered (or intentionally skipped
                        # for non-push adapters, whose delivery is the wake
                        # self-post below). Whether the cursor may advance now
                        # depends on the adapter class:
                        #
                        # * push-capable: the text send WAS the delivery, so
                        #   advance immediately (pre-existing behavior); the
                        #   wake injection below stays best-effort.
                        # * non-push (api_server): the wake self-post IS the
                        #   delivery. Advancing first would let a failed /
                        #   retry-exhausted self-post (swallowed by the
                        #   best-effort except) permanently lose the event.
                        #   So the self-post runs FIRST and the cursor only
                        #   advances after it succeeds — a failure rewinds the
                        #   claim exactly like a failed send() above, so the
                        #   next tick retries.
                        task_terminal = task and task.status == "archived"
                        # Kinds that hand a decision back to the origin, so the
                        # origin has to take a turn. ``review_requested`` (the
                        # implementation is done and waits for a reviewer),
                        # ``changes_requested`` (a reviewer BLOCKed and work
                        # returns to the implementer) and ``block_loop_detected``
                        # (routed to triage) belong here for the same reason
                        # ``blocked`` does. ``status`` / ``archived`` /
                        # ``unblocked`` stay out: bookkeeping.
                        _WAKE_KINDS = (
                            "completed", "gave_up", "crashed", "timed_out",
                            "blocked", "review_requested", "changes_requested",
                            "block_loop_detected",
                        )
                        _wake_kinds = (
                            {ev.kind for ev in d["events"] if ev.kind in _WAKE_KINDS}
                            if wake_agent
                            else set()
                        )
                        from gateway.wake import adapter_supports_push as _adapter_push_ok

                        _is_push_adapter = _adapter_push_ok(adapter)
                        _session_key = ""
                        _synth = ""
                        if _wake_kinds:
                            if _is_push_adapter:
                                _session_key = getattr(task, "session_id", None) or ""
                            else:
                                # Non-push (api_server) wakes go to the
                                # subscription's delivery destination —
                                # sub["chat_id"] IS the raw session id the
                                # subscriber registered with. task.session_id
                                # is worker/creator provenance and may point
                                # at a WORKER session for child tasks with
                                # inherited subscriptions; falling back to it
                                # only when chat_id is empty (legacy rows).
                                _session_key = (
                                    sub["chat_id"]
                                    or getattr(task, "session_id", None)
                                    or ""
                                )
                        if _wake_kinds:
                            _title = (task.title if task else sub["task_id"])[:120]
                            _assignee = task.assignee if task else ""
                            _parts = []
                            if "completed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.completed"))
                            if "gave_up" in _wake_kinds: _parts.append(t("gateway.kanban.wake.gave_up"))
                            if "crashed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.crashed"))
                            if "timed_out" in _wake_kinds: _parts.append(t("gateway.kanban.wake.timed_out"))
                            if "blocked" in _wake_kinds: _parts.append(t("gateway.kanban.wake.blocked"))
                            if "review_requested" in _wake_kinds: _parts.append(t("gateway.kanban.wake.review_requested"))
                            if "changes_requested" in _wake_kinds: _parts.append(t("gateway.kanban.wake.changes_requested"))
                            if "block_loop_detected" in _wake_kinds: _parts.append(t("gateway.kanban.wake.block_loop_detected"))
                            _status = t("gateway.kanban.wake.status_joiner").join(_parts) or t("gateway.kanban.wake.status_default")
                            _synth = t(
                                "gateway.kanban.wake.message",
                                task_id=sub["task_id"],
                                status=_status,
                                title=_title,
                                assignee=_assignee,
                                board=board_slug,
                            )
                            # Graph-safe wake turn (#70752): carry the worker's
                            # completion handoff into the synthetic turn and
                            # label it as an automatic notification so the woken
                            # creator inspects the board instead of
                            # re-decomposing work that already exists.
                            if wake_handoff:
                                _synth += "\n" + t(
                                    "gateway.kanban.wake.handoff",
                                    summary=wake_handoff,
                                )
                            if wake_review_detail:
                                _synth += "\n" + t(
                                    "gateway.kanban.wake.review_detail",
                                    reason=wake_review_detail,
                                )
                            _synth += "\n\n" + t(
                                "gateway.kanban.wake.guidance"
                            )

                        if not _is_push_adapter and _wake_kinds and _session_key:
                            # Wake self-post IS the delivery on this path —
                            # it must succeed BEFORE the cursor advances.
                            from gateway.wake import deliver_wake

                            try:
                                await deliver_wake(
                                    adapter,
                                    text=_synth,
                                    session_id=_session_key,
                                )
                                logger.info(
                                    "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                    sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                                )
                                sub_fail_counts.pop(sub_key, None)
                            except Exception as _wk_err:
                                fails = sub_fail_counts.get(sub_key, 0) + 1
                                sub_fail_counts[sub_key] = fails
                                logger.warning(
                                    "kanban notifier: wake self-post failed "
                                    "for %s (attempt %d/%d): %s",
                                    sub["task_id"], fails,
                                    MAX_SEND_FAILURES, _wk_err, exc_info=True,
                                )
                                if fails >= MAX_SEND_FAILURES:
                                    logger.warning(
                                        "kanban notifier: dropping subscription "
                                        "%s on %s after %d consecutive wake failures",
                                        sub["task_id"], platform_str, fails,
                                    )
                                    await _to_thread_process_service(self._kanban_unsub, sub, board_slug)
                                    sub_fail_counts.pop(sub_key, None)
                                else:
                                    # Rewind the pre-send claim so the next
                                    # tick retries the self-post — the event
                                    # is NOT lost.
                                    await _to_thread_process_service(
                                        self._kanban_rewind,
                                        sub,
                                        d["cursor"],
                                        d.get("old_cursor", 0),
                                        board_slug,
                                    )
                                continue

                        async def _push_wake() -> None:
                            """Wake the creator session behind a push adapter.

                            Shared by the wake-only (pre-advance, delivery)
                            and notify+wake (post-advance, best-effort)
                            branches below; raises on failure so the caller
                            decides whether to rewind or merely log.
                            """
                            from gateway.session import SessionSource
                            from gateway.wake import deliver_wake
                            # Rebuild the creator's real session scope from
                            # the chat_type persisted on the subscription
                            # row (#56580). build_session_key() keys DMs
                            # (":dm:<chat_id>") on a wholly different shape
                            # from group/thread, so the old hardcoded
                            # "group" mis-routed DM/thread creators into a
                            # fresh session. Legacy rows written before the
                            # column existed may still carry chat_type in
                            # delivery_metadata (#60600 rows) — fall back
                            # to that, then to "group" (the historical
                            # default that suits the dashboard/group flows).
                            # handle_message() get_or_create_session's the
                            # target, so a mismatch only ever degrades to a
                            # fresh session, never an exception.
                            _chat_type = str(sub.get("chat_type") or "").strip()
                            if not _chat_type:
                                _delivery_meta = sub.get("delivery_metadata")
                                if isinstance(_delivery_meta, dict):
                                    _chat_type = str(
                                        _delivery_meta.get("chat_type") or ""
                                    ).strip()
                            _chat_type = _chat_type or "group"
                            _source = SessionSource(
                                platform=plat,
                                chat_id=sub["chat_id"],
                                chat_type=_chat_type,
                                thread_id=sub.get("thread_id") or None,
                                user_id=sub.get("user_id"),
                                user_id_alt=sub.get("user_id_alt"),
                                profile=sub_profile or None,
                                scope_id=_wake_scope_id(adapter, sub),
                            )
                            # deliver_wake preserves the synthetic
                            # MessageEvent/handle_message path for
                            # push-capable adapters (the non-push /
                            # self-post branch is handled BEFORE the
                            # cursor advance above).
                            await deliver_wake(
                                adapter,
                                text=_synth,
                                session_id=_session_key,
                                source=_source,
                            )
                            logger.info(
                                "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                            )

                        if _is_push_adapter and not send_passive and _wake_kinds:
                            # Wake-only (delivery_mode='wake') push sub: the
                            # text ping was intentionally skipped above, so
                            # the wake IS the sole delivery. It must succeed
                            # BEFORE the cursor advances — advancing first
                            # would let a failed wake (previously swallowed
                            # by the best-effort except below) permanently
                            # lose the event. Mirrors the non-push
                            # (api_server) self-post ordering above.
                            try:
                                await _push_wake()
                                sub_fail_counts.pop(sub_key, None)
                            except Exception as _wk_err:
                                fails = sub_fail_counts.get(sub_key, 0) + 1
                                sub_fail_counts[sub_key] = fails
                                logger.warning(
                                    "kanban notifier: wake-only delivery failed "
                                    "for %s (attempt %d/%d): %s",
                                    sub["task_id"], fails,
                                    MAX_SEND_FAILURES, _wk_err, exc_info=True,
                                )
                                if fails >= MAX_SEND_FAILURES:
                                    logger.warning(
                                        "kanban notifier: dropping subscription "
                                        "%s on %s after %d consecutive wake failures",
                                        sub["task_id"], platform_str, fails,
                                    )
                                    await _to_thread_process_service(self._kanban_unsub, sub, board_slug)
                                    sub_fail_counts.pop(sub_key, None)
                                else:
                                    # Rewind the pre-send claim so the next
                                    # tick retries the wake — the event is
                                    # NOT lost.
                                    await _to_thread_process_service(
                                        self._kanban_rewind,
                                        sub,
                                        d["cursor"],
                                        d.get("old_cursor", 0),
                                        board_slug,
                                    )
                                continue

                        # Delivery complete (text ping for push adapters, wake
                        # self-post for non-push, wake injection for wake-only
                        # push subs): advance cursor. The cursor is the dedup
                        # mechanism — it prevents re-delivery of the same
                        # event on subsequent ticks.
                        await _to_thread_process_service(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        if not _is_push_adapter:
                            # Nothing left to deliver on this path (the wake,
                            # if any, already succeeded above).
                            sub_fail_counts.pop(sub_key, None)
                        # Unsubscribe only on archive. Completion (``done``)
                        # remains reversible: controllers reopen completed
                        # work for review corrections and continuation. The
                        # retained cursor prevents replay while preserving the
                        # original delivery and wake ownership for that cycle.
                        if _is_push_adapter and send_passive and _wake_kinds:
                            # notify+wake: the text ping above was the
                            # delivery and the cursor has advanced; the wake
                            # injection stays best-effort.
                            try:
                                await _push_wake()
                            except Exception as _wk_err:
                                # Best-effort: the notification itself already
                                # delivered and the cursor has advanced, so a
                                # broken wake path must not wedge the tick — but
                                # log at WARNING with a traceback rather than
                                # DEBUG so a persistently-failing wake is visible
                                # in normal logs instead of silently no-op'ing.
                                logger.warning(
                                    "kanban notifier: wakeup injection failed for %s: %s",
                                    sub["task_id"], _wk_err, exc_info=True,
                                )
                        if task_terminal:
                            await _to_thread_process_service(
                                self._kanban_unsub, sub, board_slug,
                            )
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            # Sleep with cancellation checks.
            for _ in range(int(max(1, interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _kanban_advance(
        self, sub: dict, cursor: int, board: Optional[str] = None,
    ) -> None:
        """Sync helper: advance a subscription's cursor. Runs in to_thread.

        ``board`` scopes the DB connection to the board that owns this
        subscription. Unsub cursors in one board can't touch another's.
        """
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.advance_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                new_cursor=cursor,
            )
        finally:
            conn.close()

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.remove_notify_sub(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
            )
        finally:
            conn.close()

    def _kanban_rewind(
        self,
        sub: dict,
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        """Sync helper: undo a claimed notification cursor after send failure."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.rewind_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
            )
        finally:
            conn.close()

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
        file paths through the completion event so downstream humans get
        the deliverable as a native upload instead of a path printed in
        chat.

        Sources scanned, in priority order:
          1. ``event_payload['artifacts']`` (explicit list — preferred)
          2. ``event_payload['summary']`` (truncated first line)
          3. ``task.result`` (legacy fallback)

        Files are deduplicated, missing files are silently skipped (the
        path may have been mentioned for reference only), and delivery
        errors are logged but do not break the notifier loop.
        """
        from pathlib import Path as _Path

        candidates: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            if not path:
                return
            expanded = os.path.expanduser(path)
            if expanded in seen:
                return
            if not os.path.isfile(expanded):
                return
            seen.add(expanded)
            candidates.append(expanded)

        # 1. Explicit artifacts list in payload.
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, str):
                        _add(item)

            # 2. Paths embedded in the payload summary.
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                paths, _ = adapter.extract_local_files(summary)
                for p in paths:
                    _add(p)

        # 3. Legacy: paths embedded in task.result.
        if task is not None and getattr(task, "result", None):
            result_text = str(task.result)
            paths, _ = adapter.extract_local_files(result_text)
            for p in paths:
                _add(p)

        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

        from urllib.parse import quote as _quote

        # Partition images so they ride a single send_multiple_images call
        # on platforms that support batch image uploads (Signal/Slack RPCs).
        image_paths = [p for p in candidates if _Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if _Path(p).suffix.lower() not in _IMAGE_EXTS]

        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(
                    chat_id=chat_id, images=batch, metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: image batch upload failed: %s", exc,
                )

        for path in other_paths:
            ext = _Path(path).suffix.lower()
            try:
                if ext in _VIDEO_EXTS:
                    await adapter.send_video(
                        chat_id=chat_id, video_path=path, metadata=metadata,
                    )
                else:
                    await adapter.send_document(
                        chat_id=chat_id, file_path=path, metadata=metadata,
                    )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: artifact upload (%s) failed: %s",
                    path, exc,
                )

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
        When true, the gateway hosts the single dispatcher for this profile:
        no separate `hermes kanban daemon` process needed. When false, the
        loop exits immediately and an external daemon is expected.

        Each tick calls :func:`kanban_db.dispatch_once` inside
        ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
        event loop. Failures in one tick don't stop subsequent ticks —
        same pattern as `_kanban_notifier_watcher`.

        Shutdown: the loop checks ``self._running`` between ticks; gateway
        stop() flips it to False and cancels pending tasks, and the
        in-flight ``to_thread`` returns on its own after the current
        ``dispatch_once`` call finishes (typically <1ms on an idle board).
        """
        # Read config once at boot. If the user flips the flag later, they
        # restart the gateway; same pattern as every other background
        # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
        # as an escape hatch (false-y value disables without editing YAML).
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return

        # Single-dispatcher backstop. dispatch_in_gateway defaults to true, so a
        # new profile gateway (or a same-profile restart race) can silently
        # start a second dispatcher; concurrent dispatchers double reclaim
        # frequency, double claim-attempt events, and — with
        # wal_autocheckpoint=0 — concurrent manual WAL checkpoints can corrupt
        # index pages. The lock lives at the machine-global kanban root
        # (shared across profiles by design), so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info(
                "kanban dispatcher: another gateway already holds the dispatcher "
                "lock (%s); this gateway will NOT dispatch.", _lock_path,
            )
            return
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning(
                "kanban dispatcher: advisory lock unavailable at %s; proceeding "
                "on config control alone.", _lock_path,
            )

        try:
            interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
        except (ValueError, TypeError):
            logger.warning(
                "kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                kanban_cfg.get("dispatch_interval_seconds"),
            )
            interval = 60.0
        interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

        # Read max_spawn config to limit concurrent kanban tasks
        max_spawn = kanban_cfg.get("max_spawn", None)
        if max_spawn is not None:
            logger.info("kanban dispatcher: max_spawn=%s", max_spawn)

        # Cap the number of simultaneously running tasks so slow workers
        # (local LLMs, resource-constrained hosts) don't pile up and time
        # out. When set, the dispatcher skips spawning when the board
        # already has this many tasks in 'running' status.
        raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
        max_in_progress = None
        if raw_max_in_progress is not None:
            try:
                max_in_progress = int(raw_max_in_progress)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                if max_in_progress < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                        raw_max_in_progress,
                    )
                    max_in_progress = None
                else:
                    logger.info("kanban dispatcher: max_in_progress=%s", max_in_progress)
        # When the operator never set kanban.max_in_progress, fall back to a
        # memory-derived default (OOF-30/OOF-77): unbounded fan-out on small
        # hosted VMs has repeatedly swap-thrashed the whole machine. Explicit
        # config always wins; None stays None on hosts where total memory
        # can't be read (macOS/Windows dev machines).
        effective_max_in_progress = _kb.resolve_max_in_progress(max_in_progress)
        if max_in_progress is None and effective_max_in_progress is not None:
            logger.info(
                "kanban dispatcher: kanban.max_in_progress unset; using "
                "memory-derived default max_in_progress=%d "
                "(set kanban.max_in_progress in config.yaml to override)",
                effective_max_in_progress,
            )
        max_in_progress = effective_max_in_progress

        raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
        try:
            failure_limit = int(raw_failure_limit)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT
        if failure_limit < 1:
            logger.warning(
                "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT

        # Read stale_timeout_seconds — 0 disables stale detection.
        raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
        try:
            stale_timeout_seconds = int(raw_stale or 0)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                "disabling stale detection",
                raw_stale,
            )
            stale_timeout_seconds = 0

        # kanban.reconcile_orphans (config.yaml, default true): each tick,
        # requeue 'running' cards whose claim bookkeeping is broken (no
        # valid claim, dead/gone worker) — the zombie-card reconciliation
        # pass. Set false to keep orphans frozen for manual forensics.
        reconcile_orphans = bool(kanban_cfg.get("reconcile_orphans", True))

        # Read kanban.default_assignee — fallback profile for tasks
        # created without an explicit assignee (e.g. via the dashboard).
        # When set, the dispatcher applies it to unassigned ready tasks
        # instead of skipping them indefinitely (#27145). Empty string
        # (the schema default) means "no fallback, keep skipping" —
        # backward-compatible with existing installs.
        default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
        if default_assignee:
            logger.info(
                "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                "will route to this profile)",
                default_assignee,
            )

        # Read kanban.max_in_progress_per_profile — per-profile concurrency
        # cap (#21582). When set, no single profile gets more than N
        # workers running at once, even if the global max_in_progress
        # would allow it. Prevents one profile's local model / API quota
        # / browser pool from being overwhelmed by a fan-out.
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
        max_in_progress_per_profile = None
        if raw_per_profile is not None:
            try:
                max_in_progress_per_profile = int(raw_per_profile)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                if max_in_progress_per_profile < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                        raw_per_profile,
                    )
                    max_in_progress_per_profile = None
                else:
                    logger.info(
                        "kanban dispatcher: max_in_progress_per_profile=%d",
                        max_in_progress_per_profile,
                    )

        # Initial delay so the gateway finishes wiring adapters before the
        # dispatcher spawns workers (those workers may hit gateway notify
        # subscriptions etc.). Matches the notifier watcher's delay.
        await asyncio.sleep(5)

        # Health telemetry mirrored from `_cmd_daemon`: warn when ready
        # queue is non-empty but spawns are 0 for N consecutive ticks —
        # usually means broken PATH, missing venv, or credential loss.
        HEALTH_WINDOW = 6
        bad_ticks = 0
        last_warn_at = 0
        # Avoid hot-looping corrupt-looking board DBs, but do not suppress
        # same-fingerprint retries forever: transient WAL/open races can
        # surface as "database disk image is malformed" for one tick.
        CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
        disabled_corrupt_boards: dict[
            str, tuple[tuple[str, int | None, int | None], float]
        ] = {}

        def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
            path = _kb.kanban_db_path(slug)
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            try:
                stat = path.stat()
            except OSError:
                return (resolved, None, None)
            return (resolved, stat.st_mtime_ns, stat.st_size)

        def _is_corrupt_board_db_error(exc: Exception) -> bool:
            corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
            if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
                return True
            if not isinstance(exc, sqlite3.DatabaseError):
                return False
            msg = str(exc).lower()
            return (
                "file is not a database" in msg
                or "database disk image is malformed" in msg
            )

        def _tick_once_for_board(slug: str) -> "Optional[object]":
            """Run one dispatch_once for a specific board.

            Runs in a worker thread via `asyncio.to_thread`. `board=slug`
            is passed through `dispatch_once` so `resolve_workspace` and
            `_default_spawn` see the right paths. The per-board DB is
            opened explicitly so concurrent boards never share a
            connection handle or accidentally claim across each other.
            """
            conn = None
            fingerprint = _board_db_fingerprint(slug)
            disabled_entry = disabled_corrupt_boards.get(slug)
            if disabled_entry is not None:
                disabled_fingerprint, disabled_at = disabled_entry
                age = time.monotonic() - disabled_at
                if (
                    disabled_fingerprint == fingerprint
                    and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
                ):
                    return None
                if disabled_fingerprint == fingerprint:
                    logger.info(
                        "kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch",
                        slug,
                        age,
                    )
                else:
                    logger.info(
                        "kanban dispatcher: board %s database changed; retrying dispatch",
                        slug,
                    )
                disabled_corrupt_boards.pop(slug, None)
            try:
                conn = _kb.connect(board=slug)
                # `connect()` runs the schema + idempotent migration on
                # first open per process; the previous explicit
                # `init_db()` call here busted the per-process cache and
                # re-ran the migration on a second connection, racing
                # the first. See the matching comment in
                # `_kanban_notifier_watcher` and issue #21378.
                return _kb.dispatch_once(
                    conn,
                    board=slug,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    stale_timeout_seconds=stale_timeout_seconds,
                    default_assignee=default_assignee,
                    max_in_progress_per_profile=max_in_progress_per_profile,
                    reconcile_orphans=reconcile_orphans,
                )
            except sqlite3.DatabaseError as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            except Exception as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _tick_once() -> "list[tuple[str, Optional[object]]]":
            """Run one dispatch_once per board. Returns (slug, result) pairs.

            Enumerating boards on every tick keeps the dispatcher honest
            when users create a new board mid-run: no restart required,
            the next tick picks it up automatically.
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            out: list[tuple[str, "Optional[object]"]] = []
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                out.append((slug, _tick_once_for_board(slug)))
            return out

        def _ready_nonempty() -> bool:
            """Cheap probe: is there at least one ready+assigned+unclaimed
            task on ANY board whose assignee maps to a real Hermes profile
            (i.e. one the dispatcher would actually spawn for)?

            Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
            ``orion-research``) are pulled by terminals via
            ``claim_task`` directly and never spawnable, so a queue full
            of those is "correctly idle", not "stuck". Filtering them out
            here keeps the stuck-warn fire only on real failures (broken
            PATH, missing venv, credential loss for a real Hermes profile).
            """
            # Only probe the review column when autonomous review dispatch is
            # actually on. With ``review_dispatch`` off (the default — no
            # sdlc-review agent), a task parked in 'review' is "correctly idle"
            # waiting for a human, not a stuck dispatcher; probing it here would
            # fire a false "dispatcher stuck" warning that never clears. Shares
            # the exact gate the dispatcher uses so the two can't drift.
            _review_probe = _kb.review_dispatch_enabled()
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                conn = None
                try:
                    conn = _kb.connect(board=slug)
                    if _kb.has_spawnable_ready(conn):
                        return True
                    if _review_probe and _kb.has_spawnable_review(conn):
                        return True
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
            return False

        # Auto-decompose: turn fresh triage tasks into ready workgraphs
        # before the dispatcher fans out workers. Gated by
        # ``kanban.auto_decompose`` (default True). Capped by
        # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
        # of triage tasks doesn't burst-spend the aux LLM in one tick;
        # remainder defers to subsequent ticks.
        #
        # The flag is re-read from config EVERY tick (#49638) rather than
        # captured once at boot. Auto-decompose is a safety toggle: a user who
        # sees it fan out and run tasks they didn't intend reaches for
        # ``kanban.auto_decompose: false`` to STOP it — and that must take
        # effect on the next tick, not require a gateway restart. (Reported:
        # auto-decompose created and launched destructive tasks while the user
        # was still typing the task description, and the flag "couldn't be
        # disabled" because the gateway had captured its boot-time value.)
        def _read_auto_decompose_settings() -> tuple[bool, int]:
            """Re-resolve (enabled, per_tick) from current config each tick."""
            return _resolve_auto_decompose_settings(_load_config)

        def _auto_decompose_tick(auto_decompose_per_tick: int) -> int:
            """Run the auto-decomposer for up to N triage tasks across all
            boards. Returns the number of triage tasks that were
            successfully decomposed or specified this tick.
            """
            try:
                from hermes_cli import kanban_decompose as _decomp
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "kanban auto-decompose: import failed (%s); skipping", exc,
                )
                return 0
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            attempted = 0
            successes = 0
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin this board for the duration of the call — same
                # pattern as the dashboard specify endpoint. The
                # decomposer module connects with no board kwarg and
                # relies on the env var.
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug(
                            "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                            slug, exc,
                        )
                        triage_ids = []
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        attempted += 1
                        try:
                            outcome = _decomp.decompose_task(
                                tid, author="auto-decomposer",
                            )
                        except Exception:
                            logger.exception(
                                "kanban auto-decompose: decompose_task crashed on %s",
                                tid,
                            )
                            continue
                        if outcome.ok:
                            successes += 1
                            if outcome.fanout and outcome.child_ids:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → %d children",
                                    slug, tid, len(outcome.child_ids),
                                )
                            else:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                    slug, tid,
                                )
                        else:
                            # Common no-op reasons (no aux client configured) shouldn't
                            # spam logs every tick. Log at debug.
                            logger.debug(
                                "kanban auto-decompose [%s]: %s skipped: %s",
                                slug, tid, outcome.reason,
                            )
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
            return successes

        logger.info(
            "kanban dispatcher: embedded in gateway (interval=%.1fs)", interval
        )
        while self._running:
            try:
                # Reap zombie children before per-board work so a board DB
                # failure cannot block cleanup of unrelated workers.
                pids = await _to_thread_process_service(_kb.reap_worker_zombies)
                if pids:
                    logger.info(
                        "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                        len(pids),
                        pids,
                    )
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Global emergency stop (`hermes pause`): skip auto-decompose
                # and dispatch entirely — no new workers while paused. Running
                # workers finish naturally; zombie reaping above still runs.
                if not _kanban_dispatch_allowed():
                    ready_pending = False
                    bad_ticks = 0
                else:
                    # Re-read the auto-decompose toggle live each tick so a user
                    # flipping kanban.auto_decompose=false to STOP runaway fan-out
                    # takes effect on the next tick, not on gateway restart (#49638).
                    _ad_enabled, _ad_per_tick = _read_auto_decompose_settings()
                    if _ad_enabled:
                        await _to_thread_process_service(_auto_decompose_tick, _ad_per_tick)
                    results = await _to_thread_process_service(_tick_once)
                    any_spawned = False
                    for slug, res in (results or []):
                        if res is not None and getattr(res, "spawned", None):
                            any_spawned = True
                            # Quiet by default — only log when something actually
                            # happened, so an idle gateway stays silent.
                            logger.info(
                                "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                                "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                                slug,
                                len(res.spawned),
                                res.reclaimed,
                                len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
                                len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
                                res.promoted,
                                len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
                            )
                    # Health telemetry (aggregate across boards)
                    ready_pending = await _to_thread_process_service(_ready_nonempty)
                    if ready_pending and not any_spawned:
                        bad_ticks += 1
                    else:
                        bad_ticks = 0
                if bad_ticks >= HEALTH_WINDOW:
                    now = int(time.time())
                    if now - last_warn_at >= 300:
                        logger.warning(
                            "kanban dispatcher stuck: ready queue non-empty for "
                            "%d consecutive ticks but 0 workers spawned. Check "
                            "profile health (venv, PATH, credentials) and "
                            "`hermes kanban list --status ready`.",
                            bad_ticks,
                        )
                        last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                self._release_kanban_dispatcher_lock()
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

        self._release_kanban_dispatcher_lock()
