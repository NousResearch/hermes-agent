"""Kanban board watcher methods for GatewayRunner.

Background loops that subscribe to kanban boards, deliver notifications and
artifacts, and drive the multi-agent dispatcher. They use only ``self`` state,
so they live on a mixin ``GatewayRunner`` inherits. Per-tick work lives in
``kanban_watchers_notifier`` / ``kanban_watchers_dispatcher``; shared plumbing
in ``kanban_watchers_common``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import weakref
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

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_GC_INTERVAL_SECONDS = 3600.0
_HEALTH_WINDOW = 6


# Bounded wait for a CONFIRMED instruction wake. The in-process transport, the
# control-socket verb and the CLI/dashboard client all bound their wait here:
# ``delivered`` (and therefore ``routed_to``) may only ever reflect a wake that
# actually completed, never a task merely queued on the loop.
INSTRUCTION_DELIVERY_TIMEOUT = 60.0


def _instruction_owner_scope(runner, platform, chat_id: str, served: Optional[str]):
    """Runtime scope of the subscription's profile under multiplex, else a no-op.

    Mirrors ``_KanbanNotification._owner_scope``: a served profile's
    raw-session wake must run in THAT profile's home, in-process.
    """
    if not served:
        return contextlib.nullcontext()
    from gateway.run import _async_profile_runtime_scope
    from gateway.session import SessionSource
    source = SessionSource(platform=platform, chat_id=chat_id, profile=served)
    return _async_profile_runtime_scope(runner._resolve_profile_home_for_source(source))


async def deliver_kanban_instruction_wake(runner, *, profile: str, task_id: str,
                                          text: str, subs: list) -> tuple[bool, str]:
    """Deliver the decompose instruction to ``profile`` as ONE wake — never a fan-out.

    ``subs`` are every destination subscribed for that one profile (the routing
    gate has already collapsed them to a single eligible profile). They are
    delivery CANDIDATES, not a recipient list: delivery stops at the first
    destination whose wake this gateway actually accepts, so a profile with
    several chats or threads receives one instruction, never a broadcast
    (the no-subscriber-fan-out contract).

    Push-capable adapters get the internal ``SessionSource`` wake; stateless
    (``supports_async_delivery=False``) adapters the raw-session self-post. The
    subscription's ``delivery_mode`` is deliberately ignored: this is an
    instruction to the agent, not an event notification, so a ``notify``-mode
    row must wake too — a passive ``send()`` alone never reaches the agent.

    Read-only for the board: no task, link, event or comment row is touched, so
    delivering the instruction can never change the task graph by itself.
    Returns ``(delivered, detail)``; ``detail`` names the single destination
    woken, or exactly why none could be.
    """
    from gateway.config import Platform as _Platform
    from gateway.kanban_watchers_notifier import _adapter_for_subscription, _wake_scope_id
    from gateway.session import SessionSource
    from gateway.wake import adapter_supports_push, deliver_wake

    if not subs:
        return False, f"no live subscription destination for profile {profile!r} on task {task_id}"
    multiplex = bool(getattr(getattr(runner, "config", None), "multiplex_profiles", False))
    served = str(profile) if multiplex and profile else None
    last_problem = "no destination was attempted"
    for sub in subs:
        try:
            platform = _Platform(str(sub.get("platform") or ""))
        except Exception:
            last_problem = f"unknown platform {sub.get('platform')!r}"
            continue
        chat_id = str(sub.get("chat_id") or "")
        try:
            adapter = _adapter_for_subscription(runner, platform, sub, profile)
        except Exception as exc:
            last_problem = (f"adapter lookup failed for {platform.value}/{chat_id}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        if adapter is None:
            last_problem = f"no live adapter for {platform.value}/{chat_id}"
            continue
        if served:
            from hermes_cli.profiles import profile_exists
            if not profile_exists(served):
                last_problem = f"profile {served!r} no longer exists"
                continue
        try:
            if adapter_supports_push(adapter):
                meta = dict(sub.get("delivery_metadata") or {})
                source = SessionSource(
                    platform=platform, chat_id=chat_id,
                    chat_type=(str(sub.get("chat_type") or meta.get("chat_type") or "").strip()
                               or "group"),
                    thread_id=sub.get("thread_id") or None,
                    user_id=sub.get("user_id"), user_id_alt=sub.get("user_id_alt"),
                    profile=profile or None, scope_id=_wake_scope_id(adapter, sub),
                    parent_chat_id=meta.get("parent_chat_id"))
                source._transport_adapter_ref = weakref.ref(adapter)
                from gateway.run import _async_profile_runtime_scope
                async with _async_profile_runtime_scope(
                        runner._resolve_profile_home_for_source(source)):
                    await deliver_wake(adapter, text=text, session_id=chat_id,
                                       source=source, notification_category="diagnostic")
                kind = "push wake"
            else:
                async with _instruction_owner_scope(runner, platform, chat_id, served):
                    await deliver_wake(adapter, text=text, session_id=chat_id,
                                       profile=served, notification_category="diagnostic")
                kind = "session self-post wake"
            return True, (
                f"instruction delivered to {profile!r} as a single {kind} on "
                f"{platform.value}/{chat_id}"
                + (f" ({len(subs) - 1} other destination(s) of the same profile "
                   f"deliberately not fanned out to)" if len(subs) > 1 else ""))
        except Exception as exc:
            last_problem = (f"wake to {platform.value}/{chat_id} failed: "
                            f"{type(exc).__name__}: {exc}")
            logger.debug("kanban decompose instruction: %s", last_problem)
            continue
    return False, (
        f"the gateway could not wake {profile!r} on any of {len(subs)} subscription "
        f"destination(s): {last_problem}")


def install_kanban_instruction_transport(runner) -> None:
    """Register the in-process wake transport ``decompose_task`` uses when auto-decompose is off.

    With ``kanban.auto_decompose`` disabled the auxiliary model is never called;
    instead the existing prompt is delivered to the task's single eligible
    subscriber as a wake. Installed at gateway start (see
    ``GatewayStartupMixin._start_spawn_background_watchers``), NOT from inside
    the kanban dispatcher: a gateway running with ``kanban.dispatch_in_gateway``
    disabled must still serve its own in-process callers, and callers outside
    this process (CLI, out-of-process dashboard) reach the same delivery through
    the ``deliver-decompose-instruction`` control-socket verb. Only a running
    gateway holds adapters, so only it can confirm a wake; any process with
    neither path answers honestly via
    ``hermes_cli.kanban_decompose.deliver_decompose_instruction`` rather than
    claiming a handoff. Registration is idempotent (last writer wins) and binds
    THIS gateway's event loop at install time.
    """
    from hermes_cli import kanban_decompose as _decomp

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("kanban decompose instruction: no running loop; transport not installed")
        return

    def _transport(*, profile: str, task_id: str, text: str, subs: list) -> tuple[bool, str]:
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            # Confirming a wake means awaiting a coroutine ON this loop; blocking
            # here would deadlock the gateway and create_task() cannot confirm
            # anything. Report the truth (not delivered) so ``routed_to`` stays
            # unset instead of being claimed for a queued task.
            return False, (
                "delivery was requested from the gateway's own event loop, where a "
                "wake cannot be confirmed synchronously; retry from a worker or "
                "CLI/dashboard context (which reaches this gateway over the "
                "control socket) for a confirmed handoff")
        try:
            pending = asyncio.run_coroutine_threadsafe(
                deliver_kanban_instruction_wake(runner, profile=profile, task_id=task_id,
                                                text=text, subs=subs), loop)
        except Exception as exc:
            return False, f"gateway delivery unavailable: {type(exc).__name__}: {exc}"
        try:
            delivered, detail = pending.result(timeout=INSTRUCTION_DELIVERY_TIMEOUT)
            return bool(delivered), str(detail)
        except Exception as exc:
            return False, f"gateway delivery not confirmed: {type(exc).__name__}: {exc}"

    _decomp.set_instruction_transport(_transport)
    logger.info("kanban: decompose instruction transport installed")


def decompose_instruction_control_verb(runner, main_loop):
    """Build the ``deliver-decompose-instruction`` control-socket handler.

    This is the reachable delivery path for the shared CLI/dashboard backend: a
    process that owns no adapters asks the gateway that DOES to push the wake.
    The handler runs on the socket's executor thread, so the coroutine is
    marshalled onto ``main_loop`` and the answer waits (bounded by
    ``INSTRUCTION_DELIVERY_TIMEOUT``) for the CONFIRMED result — ``delivered``
    is true only after the wake completed, never for a merely queued task and
    never when delivery failed.
    """
    def _handler(params: dict) -> dict:
        params = params if isinstance(params, dict) else {}
        profile = str(params.get("profile") or "").strip()
        task_id = str(params.get("task_id") or "").strip()
        text = str(params.get("text") or "")
        subs = params.get("subs")
        if not profile or not task_id or not text or not isinstance(subs, list) or not subs:
            return {
                "delivered": False,
                "detail": "profile, task_id, text and a non-empty subs list are required",
            }
        try:
            pending = asyncio.run_coroutine_threadsafe(
                deliver_kanban_instruction_wake(runner, profile=profile, task_id=task_id,
                                                text=text, subs=list(subs)), main_loop)
        except Exception as exc:
            return {"delivered": False,
                    "detail": f"gateway delivery unavailable: {type(exc).__name__}: {exc}"}
        try:
            delivered, detail = pending.result(timeout=INSTRUCTION_DELIVERY_TIMEOUT)
        except Exception as exc:
            # Timed out or the loop is gone. The wake may still be in flight,
            # but it is NOT confirmed — and routed_to only ever reflects
            # confirmation, so say exactly that instead of guessing.
            return {
                "delivered": False,
                "detail": (f"wake not confirmed within {INSTRUCTION_DELIVERY_TIMEOUT:.0f}s "
                           f"(it may still complete in the gateway): "
                           f"{type(exc).__name__}: {exc}"),
            }
        return {"delivered": bool(delivered), "detail": str(detail)}

    return _handler


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
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            await self._sleep_between_ticks(interval)

    def _kanban_sub_op(self, board: Optional[str], op: str, sub: dict, **extra: Any) -> None:
        """Sync helper (runs in to_thread): call ``kanban_db_notify.<op>`` for one subscription on its board."""
        from hermes_cli import kanban_db as _kb
        from hermes_cli import kanban_db_connect as _kbc
        from hermes_cli import kanban_db_notify as _kbn
        # Cursor writes are machine flow: the sub's board slug must resolve
        # through the env pin on a dispatcher-pinned box (same as the notifier
        # tick), or advance/rewind land on a per-slug DB the notifier never
        # reads and cursors silently reset.
        with _kb.pin_first_board_resolution():
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
