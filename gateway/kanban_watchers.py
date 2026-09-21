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
from gateway.kanban_watchers_notifier import (
    _KanbanNotification,
    _notifier_collect,
    _profile_gate_collect,
    _profile_gate_mark_notified,
)
from gateway.kanban_watchers_dispatcher import (
    _KanbanDispatcher,
    _log_spawn_results,
    _resolve_dispatcher_settings,
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_GC_INTERVAL_SECONDS = 3600.0
_HEALTH_WINDOW = 6


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership before releasing the OS lock."""
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        _release_singleton_lock(handle)

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
        try:
            from hermes_cli.config import load_config as _load_config

            cfg = _load_config()
            kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        except Exception as exc:
            logger.warning("kanban notifier: cannot load config (%s); continuing enabled", exc)
            kanban_cfg = {}
        if not kanban_cfg.get("notify_in_gateway", True):
            logger.info("kanban notifier: disabled via config kanban.notify_in_gateway=false")
            return

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
                    # KENSEI CUSTOM: event-driven lead shared-memory injection
                    # (locked decision 2 / R5). Best-effort — the helper never
                    # raises into this tick; see _kanban_shared_injection.
                    if getattr(d.get("task"), "status", "") == "archived":
                        await asyncio.to_thread(
                            self._kanban_shared_injection,
                            d["sub"], d["task"], d.get("board"),
                        )
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            await self._sleep_between_ticks(interval)

    async def _profile_gate_watcher(self, interval: float = 5.0) -> None:
        """Deliver pending PROFILE-GATE approvals to Discord as button prompts.

        KENSEI CUSTOM (restored from fork commit 863c2bbecc; the delivery
        loop was lost in the 20260904 watcher refactor and is now re-ported
        into the split watcher modules). Polls each board's
        ``profile_lifecycle_approvals`` for undelivered pending rows and
        posts an Approve / Reject prompt to the Discord home channel. Runs
        only on the dispatch-owning gateway (same gate as the dispatcher) so
        a single process owns kanban-DB access. Marking a row ``notified``
        is the idempotency guard against re-posting every tick.
        """
        boot = self._kanban_dispatcher_boot()
        if boot is None:
            return
        _load_config, _kb, _kanban_cfg = boot  # noqa: F841 — boot gate only
        from gateway.config import Platform as _Platform

        await asyncio.sleep(6)  # let adapters wire up

        while self._running:
            try:
                adapter = self.adapters.get(_Platform.DISCORD)
                home = self.config.get_home_channel(_Platform.DISCORD)
                if adapter is None or home is None:
                    await self._sleep_between_ticks(interval)
                    continue

                pending = await asyncio.to_thread(_profile_gate_collect, _kb)
                for slug, row in pending:
                    metadata = {"thread_id": home.thread_id} if home.thread_id else None
                    try:
                        res = await adapter.send_profile_gate(
                            home.chat_id, row, board=slug, metadata=metadata,
                        )
                    except Exception as exc:
                        logger.warning(
                            "profile-gate watcher: send failed for %s: %s",
                            row.get("id"), exc,
                        )
                        continue
                    if getattr(res, "success", False):
                        await asyncio.to_thread(
                            _profile_gate_mark_notified, _kb, slug, row["id"],
                        )
                    else:
                        logger.warning(
                            "profile-gate watcher: delivery unsuccessful for %s: %s",
                            row.get("id"), getattr(res, "error", "?"),
                        )
            except Exception:
                logger.exception("profile-gate watcher tick failed")
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
    # KENSEI CUSTOM: event-driven lead shared-memory injection (locked
    # decision 2 / R5). Distils the team's shared surface into the lead's
    # scope on terminal task transition. Best-effort and scope-safe: only a
    # REGISTERED team (task tenant key present in the TEAMS registry) with
    # SEVERIAN_STORAGE set triggers anything; all failures are logged,
    # never raised — the notifier tick must not wedge.
    def _kanban_shared_injection(
        self,
        sub: dict,
        task: Optional[Any],
        board_slug: Optional[str],
    ) -> None:
        """Event-driven lead shared-memory injection (locked decision 2 / R5)."""
        from gateway.shared_injection import (
            TEAMS as _SI_TEAMS,
            inject_on_task_completion,
        )
        from severian.composition import build_bundle as _si_bundle
        from severian.infrastructure.embedding_resolver import (
            embedding_from_env as _si_embedding_from_env,
        )

        _team_key = getattr(task, "tenant", None) or ""
        _team_key = _team_key if _team_key in _SI_TEAMS else ""
        _store = os.environ.get("SEVERIAN_STORAGE", "").strip()
        if not (_store and _team_key):
            return
        # embedding_from_env is the resolver's single SEVERIAN_EMBEDDING
        # entry point (F-03 contract): the seam honours the exact same
        # embedding space as gateways and cron scripts.
        _bundle = _si_bundle(
            backend="sqlite",
            database=Path(_store) / "severian.db",
            fts=Path(_store) / "severian.fts",
            vectors=Path(_store) / "severian.vec",
            embedding=_si_embedding_from_env(),
        )
        try:
            inject_on_task_completion(
                bundle=_bundle,
                task_id=sub["task_id"],
                title=(task.title if task else "")[:120],
                board=board_slug or "",
                team=_team_key,
            )
        finally:
            _bundle.close()


    async def _deliver_kanban_artifacts(self, *, adapter, chat_id: str, metadata: dict, event_payload: Optional[dict], task) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Sources, in priority order: ``event_payload['artifacts']``,
        ``event_payload['summary']``, then ``task.result`` (legacy). Paths are
        deduplicated, missing files are skipped (may be mentioned for
        reference only), and upload errors are logged, never raised.
        """
        raw_paths: list[str] = []
        prose_paths: list[str] = []
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                raw_paths += [item for item in raw if isinstance(item, str)]
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                prose_paths += adapter.extract_local_files(summary)[0]
        if task is not None and getattr(task, "result", None):
            prose_paths += adapter.extract_local_files(str(task.result))[0]
        # A staged copy and the scratch original it was copied from are the
        # same deliverable; on a review handoff the original still exists, so
        # prose mentions of it must not upload the file a second time.
        staged_names = {os.path.basename(p) for p in raw_paths}
        raw_paths += [p for p in prose_paths if os.path.basename(p) not in staged_names]
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
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info("kanban dispatcher: another gateway already holds the dispatcher "
                        "lock (%s); this gateway will NOT dispatch.", _lock_path)
            return None
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning("kanban dispatcher: advisory lock unavailable at %s; proceeding "
                           "on config control alone.", _lock_path)
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
        results: Optional[list] = None
        dispatcher = _KanbanDispatcher(_kb, settings)

        logger.info("kanban dispatcher: embedded in gateway (interval=%.1fs)", interval)
        while self._running:
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
                    held = _kbd.describe_suppression(res for _slug, res in (results or []))
                    logger.warning(
                        "kanban dispatcher stuck: ready queue non-empty for "
                        "%d consecutive ticks but 0 workers spawned.%s Check "
                        "profile health (venv, PATH, credentials) and "
                        "`hermes kanban list --status ready`.",
                        bad_ticks, f" Last tick held back: {held}." if held else "",
                    )
                    last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                self._release_kanban_dispatcher_lock()
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

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
