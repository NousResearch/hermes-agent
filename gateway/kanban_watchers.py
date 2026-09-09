"""Kanban board watcher methods for GatewayRunner.

Background loops that subscribe to kanban boards, deliver notifications and
artifacts, and drive the multi-agent dispatcher. They use only ``self`` state,
so they live on a mixin ``GatewayRunner`` inherits. Per-tick work lives in
``kanban_watchers_notifier`` / ``kanban_watchers_dispatcher``; shared plumbing
in ``kanban_watchers_common``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _kanban_dispatch_allowed,
    _release_singleton_lock,
    _resolve_auto_decompose_settings,
    _gc_retention_days,
    _to_thread_process_service,
    logger,
)
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.kanban_watchers_dispatcher import (
    _KanbanDispatcher,
    _log_spawn_results,
    _resolve_dispatcher_settings,
)

# FLEET: dispatcher lease/takeover support (t_9e151ee8). Ownership is a
# LEASE held only for the duration of one tick, plus a heartbeat contenders
# probe, because boot-time acquisition made a losing gateway give up once
# and never re-probe while a winning-but-frozen gateway pinned the flock
# forever — together, a four-hour dispatch outage (00:28-04:30).
_DISPATCHER_HEARTBEAT_FILENAME = ".dispatcher.heartbeat"

_DISPATCHER_HEARTBEAT_STALE_SECONDS = 300  # >5 min

_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS = 120


def _dispatcher_heartbeat_path(lock_path) -> Path:
    """Heartbeat file lives beside the lock:  `<kanban>/.dispatcher.heartbeat`."""
    return Path(lock_path).with_name(_DISPATCHER_HEARTBEAT_FILENAME)


def _touch_dispatcher_heartbeat(heartbeat_path) -> None:
    """Record liveness: the file's mtime is the last healthy dispatcher tick."""
    try:
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        heartbeat_path.touch()
    except OSError:
        logger.debug(
            "kanban dispatcher: heartbeat touch failed at %s", heartbeat_path
        )


def _dispatcher_heartbeat_is_stale(heartbeat_path, *, now: Optional[float] = None) -> bool:
    """True when the heartbeat is absent or older than the stale window.

    An absent heartbeat (no owner has ever ticked under this scheme) counts
    as stale so the first claimant wins under the same takeover rules.
    """
    try:
        mtime = heartbeat_path.stat().st_mtime
    except (OSError, ValueError):
        return True
    ref = time.time() if now is None else now
    return ref - mtime > _DISPATCHER_HEARTBEAT_STALE_SECONDS


def _root_gateway_in_grace(kanban_root: Path) -> bool:
    """True when the ROOT gateway is alive AND freshly (re)started.

    Root preference is bounded to a short post-restart grace window: only a
    root that (re)started within ``_DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS``
    is deferred to, so the root keeps winning the initial claim race after the
    deploy/restart that brings its dispatcher loop online. Once the root has
    settled (up longer than the grace), a non-root no longer defers to it even
    if the root *process* is alive — a root process whose dispatcher loop
    never claims (the card's process-alive / dispatcher-dead failure, now on
    the root) must not keep the board frozen for hours.

    Anchored to the root process start time (recorded in gateway.pid), not the
    dispatch heartbeat, so a healthy non-root OWNER that refreshes its own
    heartbeat still honours the bounded root preference. The uptime is computed
    from a TRUE epoch creation timestamp (``psutil.create_time()``) — NOT the
    PID-reuse fingerprint from ``get_running_pid_identity_strict``, whose units
    are platform-dependent (centisecond-epoch on macOS, ticks-since-boot on
    Linux) and therefore not comparable to ``time.time()``. Fail-safe: any probe
    error returns False (do not defer).
    """
    try:
        from gateway import status as _st
    except Exception:
        return False
    try:
        identity = _st.get_running_pid_identity_strict(
            kanban_root / "gateway.pid"
        )
    except Exception:
        return False
    if identity is None:
        return False
    _pid, _start = identity
    if not _pid or _pid <= 0:
        return False
    create_time = _st.get_process_create_time_epoch(_pid)
    if create_time is None:
        return False
    try:
        uptime = time.time() - float(create_time)
    except (TypeError, ValueError):
        return False
    return 0 < uptime <= _DISPATCHER_ROOT_PREFERENCE_GRACE_SECONDS


def _should_seize_dispatcher(
    *, am_root: bool, owner_stale: bool, defer_to_root: bool,
) -> bool:
    """Pure takeover decision for a contender that just won the OS flock.

    A claimant that is NOT the marked owner takes over only when the previous
    owner's heartbeat is stale (>5 min). Root preference is a *bounded* yield:
    a non-root claimant defers only while ``defer_to_root`` is true — the root
    gateway (the one deploys restart) is alive AND freshly (re)started, i.e.
    still inside the post-deploy window where it can reasonably claim. Once the
    root settles (up longer than the grace period) or is unprobeable, the
    non-root seizes the stale owner even if the root *process* is alive —
    process-alive / dispatcher-dead on the root is exactly the original 4-hour
    freeze, and deferring to a live root whose loop never claims would
    re-create it.
    """
    if not owner_stale:
        return False
    if defer_to_root and not am_root:
        return False
    return True


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_GC_INTERVAL_SECONDS = 3600.0
_HEALTH_WINDOW = 6


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        """Whether this gateway is the designated dispatcher owner.

        FLEET: ownership is a stable marker that persists across inter-tick
        sleeps; the underlying OS lease handle is acquired and released per tick
        (see ``_release_kanban_dispatcher_lease``) so the notifier's judgement
        on ``unowned`` subscriptions does not flicker while the owner sleeps.
        """
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lease(self) -> None:
        """FLEET: release the per-tick lease handle WITHOUT clearing ownership.

        Called before each inter-tick sleep so a live-but-frozen owner loop
        never pins the flock forever (a frozen loop cannot be forcibly unlocked
        by another gateway). The owner marker persists so the notifier still
        treats this gateway as owner across the sleep.
        """
        handle = getattr(self, "_kanban_dispatcher_lease_handle", None)
        self._kanban_dispatcher_lease_handle = None
        _release_singleton_lock(handle)

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership and release the OS lease (shutdown)."""
        self._kanban_dispatcher_lock_handle = None
        self._release_kanban_dispatcher_lease()

    def _try_claim_dispatcher_lease(self, lock_path, heartbeat_path, kanban_root) -> bool:
        """FLEET: hold the dispatcher lease for THIS tick; True when active.

        A retired owner re-claims on the next tick; a contender only seizes
        ownership once the previous owner's heartbeat has gone stale, so a
        healthy owner is never fought over. The ROOT (default) gateway wins
        genuine takeovers during its bounded post-restart grace — it is the one
        deploys restart — but a settled root whose own loop never claims must
        not starve the board. Lock-unavailable falls back to config-only
        control (dispatch with no lock), the pre-lease fail-safe.
        """
        am_root = self._active_profile_name() == "default"
        try:
            handle, state = _acquire_singleton_lock(lock_path)
        except Exception:
            handle, state = None, "unavailable"
        if state == "unavailable":
            self._kanban_dispatcher_lease_handle = None
            return True
        if state != "held":
            # Contended: an owner holds the flock mid-tick. Never steal
            # un-gated — wait and re-probe on the next tick.
            self._kanban_dispatcher_lease_handle = None
            return False
        self._kanban_dispatcher_lease_handle = handle
        if self._owns_kanban_dispatcher_lock():
            # Standing owner re-hosting an inter-tick lease. Yield only to a
            # LIVE, freshly-restarted root, and only inside its grace window.
            if not am_root and _root_gateway_in_grace(kanban_root):
                logger.info(
                    "kanban dispatcher: root gateway freshly restarted; non-root "
                    "owner (%s) yields dispatcher to root at %s",
                    self._active_profile_name(), lock_path,
                )
                self._kanban_dispatcher_lock_handle = None
                self._release_kanban_dispatcher_lease()
                return False
            # Touch only while we remain owner, so contenders know we are alive.
            _touch_dispatcher_heartbeat(heartbeat_path)
            return True
        owner_stale = _dispatcher_heartbeat_is_stale(heartbeat_path)
        defer_to_root = _root_gateway_in_grace(kanban_root)
        if _should_seize_dispatcher(
            am_root=am_root, owner_stale=owner_stale, defer_to_root=defer_to_root,
        ):
            self._kanban_dispatcher_lock_handle = handle
            _touch_dispatcher_heartbeat(heartbeat_path)
            logger.info(
                "kanban dispatcher: took over dispatcher lease (previous owner "
                "heartbeat stale) at %s", lock_path,
            )
            return True
        logger.debug(
            "kanban dispatcher: yielding lease at %s (owner_stale=%s "
            "defer_to_root=%s am_root=%s)",
            lock_path, owner_stale, defer_to_root, am_root,
        )
        self._release_kanban_dispatcher_lease()
        return False

    async def _sleep_between_ticks(self, interval: float) -> None:
        """Sleep *interval* (floored to 1s) in 1s slices so stop() never waits a full interval."""
        interval = max(interval, 1.0)
        slept = 0.0
        while slept < interval and self._running:
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        Per subscription, claims ``task_events`` newer than the stored cursor
        (kinds in TERMINAL_KINDS), sends one message per event, then advances
        the cursor. The subscription is removed only when the task is
        ``archived``: ``done`` is reversible, so the cursor — not unsubscribing
        — is the dedup mechanism (unsub-on-terminal dropped users when the
        dispatcher respawned a crashed task). All SQLite work runs in a thread;
        one tick's failure never stops the next.
        """
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        sub_fail_counts: dict[tuple, int] = getattr(self, "_kanban_sub_fail_counts", {})
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name()
        self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        # Stale done-sub GC: subs survive ``done``, so boards that never
        # archive would accumulate rows scanned every tick. One DELETE per
        # board, at startup (0 → first tick) and at most hourly.
        _gc_next_at = 0.0

        while self._running:
            try:
                _gc_due = time.monotonic() >= _gc_next_at
                _retention = 30
                if _gc_due:
                    _gc_next_at = time.monotonic() + _GC_INTERVAL_SECONDS
                    _retention = _gc_retention_days()

                deliveries = await asyncio.to_thread(
                    _notifier_collect, self, _kb,
                    notifier_profile=notifier_profile, gc_due=_gc_due, gc_retention_days=_retention,
                )
                for d in deliveries:
                    await _KanbanNotification(
                        self, d, platform_cls=_Platform, sub_fail_counts=sub_fail_counts,
                    ).deliver()
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            await self._sleep_between_ticks(interval)

    def _kanban_sub_op(self, board: Optional[str], op: str, sub: dict, **extra: Any) -> None:
        """Sync helper (runs in to_thread): call ``kanban_db_notify.<op>`` for one subscription on its board."""
        from hermes_cli import kanban_db_connect as _kbc
        from hermes_cli import kanban_db_notify as _kbn
        conn = _kbc.connect(board=board)
        try:
            getattr(_kbn, op)(
                conn, task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "", **extra,
            )
        finally:
            conn.close()

    def _kanban_advance(self, sub: dict, cursor: int, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "advance_notify_cursor", sub, new_cursor=cursor)

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "remove_notify_sub", sub)

    def _kanban_rewind(self, sub: dict, claimed_cursor: int, old_cursor: int, board: Optional[str] = None) -> None:
        """Undo a claimed notification cursor after send failure."""
        self._kanban_sub_op(board, "rewind_notify_cursor", sub, claimed_cursor=claimed_cursor, old_cursor=old_cursor)

    async def _deliver_kanban_artifacts(self, *, adapter, chat_id: str, metadata: dict, event_payload: Optional[dict], task) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Sources, in priority order: ``event_payload['artifacts']``,
        ``event_payload['summary']``, then ``task.result`` (legacy). Paths are
        deduplicated, missing files are skipped (may be mentioned for
        reference only), and upload errors are logged, never raised.
        """
        raw_paths: list[str] = []
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                raw_paths += [item for item in raw if isinstance(item, str)]
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                raw_paths += adapter.extract_local_files(summary)[0]
        if task is not None and getattr(task, "result", None):
            raw_paths += adapter.extract_local_files(str(task.result))[0]
        candidates: list[str] = []
        for path in raw_paths:
            expanded = os.path.expanduser(path) if path else ""
            if expanded and expanded not in candidates and os.path.isfile(expanded):
                candidates.append(expanded)
        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        from urllib.parse import quote as _quote

        # Images ride one send_multiple_images call (batch uploads on Signal/Slack).
        image_paths = [p for p in candidates if Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if Path(p).suffix.lower() not in _IMAGE_EXTS]
        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(chat_id=chat_id, images=batch, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: image batch upload failed: %s", exc)
        for path in other_paths:
            try:
                if Path(path).suffix.lower() in _VIDEO_EXTS:
                    await adapter.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
                else:
                    await adapter.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: artifact upload (%s) failed: %s", path, exc)

    def _kanban_dispatcher_boot(self) -> Optional[tuple]:
        """Resolve config, kanban_db and the singleton lock; None when the dispatcher must not run.

        Config is read once at boot (restart to apply), except the auto-decompose
        toggle which is re-read every tick. The env var is an escape hatch to
        disable without editing YAML.
        """
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return None
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return None
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return None
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info("kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false")
            return None
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return None

        # Single-dispatcher backstop (see _acquire_singleton_lock). The lock
        # lives at the machine-global kanban root, so it serialises ALL gateways.
        #
        # FLEET (t_9e151ee8): the flock is NOT taken here for the process
        # lifetime — see _try_claim_dispatcher_lease. Boot-time acquisition made
        # a losing gateway give up once and never re-probe, and let a
        # winning-but-frozen gateway hold the flock forever.
        self._kanban_dispatcher_lock_handle = None      # stable owner marker
        self._kanban_dispatcher_lease_handle = None     # per-tick OS lease
        return _load_config, _kb, kanban_cfg

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` (default True); when false the
        loop exits and an external `hermes kanban daemon` is expected. Each
        tick runs :func:`kanban_db_dispatch.dispatch_once` in a thread; one tick's
        failure never stops the next. Shutdown: ``self._running`` is checked
        between ticks and the in-flight ``to_thread`` returns on its own.
        """
        boot = self._kanban_dispatcher_boot()
        if boot is None:
            return
        _load_config, _kb, kanban_cfg = boot
        settings = _resolve_dispatcher_settings(kanban_cfg, _kb)
        interval = settings.interval

        # Initial delay so adapters are wired before workers spawn (matches the notifier).
        await asyncio.sleep(5)

        # Health telemetry (mirrors `_cmd_daemon`): warn when the ready queue
        # is non-empty but spawns are 0 for N consecutive ticks — usually a
        # broken PATH, missing venv, or credential loss.
        bad_ticks = 0
        last_warn_at = 0
        dispatcher = _KanbanDispatcher(_kb, settings)

        logger.info("kanban dispatcher: embedded in gateway (interval=%.1fs)", interval)
        _kanban_root = _kb.kanban_home()
        _lock_path = _kanban_root / "kanban" / ".dispatcher.lock"
        _heartbeat_path = _dispatcher_heartbeat_path(_lock_path)
        while self._running:
            # FLEET lease acquisition / takeover (t_9e151ee8). The OS flock is
            # held only for this tick and released during the sleep below, so a
            # live-but-frozen owner loop cannot pin it: contenders re-probe every
            # tick and claim once the heartbeat goes stale.
            if not self._try_claim_dispatcher_lease(_lock_path, _heartbeat_path, _kanban_root):
                await self._sleep_between_ticks(interval)
                continue
            try:
                # Reap zombies before per-board work so a board DB failure
                # cannot block cleanup of unrelated workers.
                from hermes_cli import kanban_db_dispatch as _kbd
                pids = await _to_thread_process_service(_kbd.reap_worker_zombies)
                if pids:
                    logger.info("kanban dispatcher: reaped %d zombie worker(s), pids=%s", len(pids), pids)
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Emergency stop (`hermes pause`): no auto-decompose or
                # dispatch while paused; running workers finish naturally.
                if not _kanban_dispatch_allowed():
                    bad_ticks = 0
                else:
                    # Re-read the auto-decompose toggle live so disabling it
                    # takes effect on the next tick, not on restart.
                    _ad_enabled, _ad_per_tick = _resolve_auto_decompose_settings(_load_config)
                    # See #49638.
                    if _ad_enabled:
                        await _to_thread_process_service(dispatcher.auto_decompose_tick, _ad_per_tick)
                    results = await _to_thread_process_service(dispatcher.tick_once)
                    any_spawned = _log_spawn_results(results)
                    ready_pending = await _to_thread_process_service(dispatcher.ready_nonempty)
                    bad_ticks = bad_ticks + 1 if ready_pending and not any_spawned else 0
                now = int(time.time())
                if bad_ticks >= _HEALTH_WINDOW and now - last_warn_at >= 300:
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

            # FLEET: release the OS lease for the sleep window so a frozen owner
            # never pins the flock; a healthy owner re-claims on the next tick.
            self._release_kanban_dispatcher_lease()
            await self._sleep_between_ticks(interval)

        self._release_kanban_dispatcher_lock()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
from contextvars import Context  # noqa: F401,E402
import logging  # noqa: F401,E402
import re  # noqa: F401,E402
import sqlite3  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    't': ('agent.i18n', 't'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
