"""Startup sequence, resume/restore and handoff methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import faulthandler
import logging
import os
import signal
import time
from contextlib import suppress
from datetime import datetime
from gateway.config import Platform
from gateway.delivery import looks_like_telegram_private_chat_id
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, GATEWAY_FATAL_CONFIG_EXIT_CODE, is_global_startup_conflict
)
from gateway.run_shutdown import _log_suppressed, _send_error
from gateway.shutdown_watchdog import (
    DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_LOOP_WATCHDOG_INTERVAL_S,
    DEFAULT_LOOP_WATCHDOG_MAX_STRIKES, DEFAULT_LOOP_WATCHDOG_TIMEOUT_S, loop_heartbeat_forever,
)
from typing import Any, Dict, Optional, Tuple

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayStartupMixin:
    """Startup sequence, resume/restore and handoff methods for GatewayRunner."""

    async def _run_startup_resume_event(
        self, adapter: BasePlatformAdapter, event: MessageEvent, session_key: str,
    ) -> None:
        """Dispatch one synthetic startup resume and wait for its agent turn (inbound stays queued
        until it finishes, else a user message can race it)."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        try:
            await adapter.handle_message(event)
            session_tasks = getattr(adapter, "_session_tasks", {})
            task = session_tasks.get(session_key) if isinstance(session_tasks, dict) else None
            if task is not None:
                await asyncio.shield(task)
        finally:
            # Release the pre-claimed slot if handle_message raised before _handle_message took ownership.
            _pre_state = self._peek_session_state(session_key)
            if (_pre_state.turn.agent if _pre_state else None) is _AGENT_PENDING_SENTINEL:
                self._release_running_agent_state(session_key)

    def _queue_startup_restore_event(self, event: MessageEvent) -> None:
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            queue = self._startup_restore_queue = []
        queue.append(event)
        with suppress(Exception):
            source = event.source
            logger.info(
                "Queued inbound message during gateway startup restore: platform=%s chat=%s",
                source.platform.value if source and source.platform else "unknown",
                source.chat_id if source else "unknown",
            )

    async def _drain_startup_restore_queue(self) -> int:
        """Replay inbound messages queued while startup auto-resume ran."""
        drained = 0
        queue = getattr(self, "_startup_restore_queue", None) or []
        while queue:
            event = queue.pop(0)
            source = getattr(event, "source", None)
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Dropping startup-restore queued message: adapter unavailable for %s",
                    getattr(getattr(source, "platform", None), "value", None),
                )
                continue
            # Mark the replay so _handle_message does not re-queue it while the restore gate is closed.
            with suppress(Exception):
                setattr(event, "_hermes_startup_restore_replay", True)
            await adapter.handle_message(event)
            drained += 1
        return drained

    def _start_startup_warmup(self) -> None:
        """Kick off the boot turn-machinery warm-up so it overlaps the network-bound platform
        connects; ``_finish_startup_restore`` awaits it (bounded)."""
        from gateway.run import _startup_warmup_timeout_secs
        if _startup_warmup_timeout_secs() <= 0:
            self._startup_warmup_task = None
            return
        self._startup_warmup_task = asyncio.ensure_future(self._warm_turn_prerequisites())

    async def _warm_turn_prerequisites(self) -> None:
        """Initialize turn machinery on an executor thread before the gate opens. Never raises: a
        failed warm-up degrades to lazy init and must not block startup."""
        from gateway.run import _warm_turn_machinery_sync
        with _log_suppressed(
            logging.WARNING, "Turn-machinery warm-up failed; first inbound turn will initialize lazily",
            exc_info=True,
        ):
            loop = asyncio.get_running_loop()
            t0 = time.monotonic()
            tool_count = await loop.run_in_executor(None, _warm_turn_machinery_sync)
            logger.info(
                "Turn machinery warmed in %.1fs (%d tool schema(s) materialized)",
                time.monotonic() - t0, tool_count,
            )

    async def _await_startup_warmup(self) -> None:
        """Bounded wait for the boot warm-up. On timeout the gate opens anyway (availability outranks
        prompt completeness for a WEDGED init); the warm-up continues and a late failure is logged."""
        from gateway.run import _startup_warmup_timeout_secs
        task = getattr(self, "_startup_warmup_task", None)
        if task is None or task.done():
            return
        timeout = _startup_warmup_timeout_secs()
        if timeout <= 0:
            return
        await self._wait_bounded_or_release(
            {task}, timeout,
            "Turn-machinery warm-up still running after %.0fs; opening inbound gate anyway — the "
            "first turn may see lazily initialized machinery (#99373). Warm-up continues in the background.",
            "boot turn-machinery warm-up failed after gate release", level=logging.DEBUG,
        )

    async def _wait_bounded_or_release(
        self, tasks: set, timeout: float, warn_fmt: str, late_msg: str, *,
        level: int = logging.WARNING, track: bool = False,
    ) -> set:
        """``asyncio.wait`` (which, unlike wait_for/gather+timeout, does NOT cancel on timeout) with
        the gate-release warning + late-failure callback every boot gate uses. Returns ``done``.
        ``warn_fmt`` takes ``%.0fs`` (timeout) and optionally ``%d`` (pending count). A non-positive
        ``timeout`` opts out of the bound ("wait forever")."""
        if timeout <= 0:
            await asyncio.gather(*tasks, return_exceptions=True)
            return set(tasks)
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            args = (timeout, len(pending)) if warn_fmt.count("%") > 1 else (timeout,)
            logger.warning(warn_fmt, *args)
            late = self._late_failure_callback(late_msg, level=level)
            for task in pending:
                task.add_done_callback(late)
                if track:
                    self._retain_background_task(task)
        return done

    async def _finish_startup_restore(self) -> None:
        """Wait (BOUNDED by ``_startup_restore_drain_timeout_secs``) for startup auto-resume, then
        release + drain inbound. On timeout the gate opens and resume turns finish in the background
        (NOT cancelled) — safe because ``_schedule_resume_pending_sessions`` claims each
        ``_running_agents`` slot SYNCHRONOUSLY first, so drained inbound queues behind."""
        from gateway.run import _startup_restore_drain_timeout_secs
        tasks = list(getattr(self, "_startup_restore_tasks", []) or [])
        if tasks:
            # Tasks outliving the gate get a late-failure callback (their done-callback only discards them).
            done = await self._wait_bounded_or_release(
                set(tasks), _startup_restore_drain_timeout_secs(),
                "Startup-restore gate released after %.0fs with %d boot auto-resume turn(s) "
                "still running; draining inbound queue now (resume slots already claimed, so no "
                "duplicate agents). Slow turn(s) continue in the background.",
                "background startup auto-resume task failed after gate release", level=logging.DEBUG,
            )
            report = self._late_failure_callback("startup auto-resume task failed", level=logging.DEBUG)
            for task in done:
                report(task)
        self._startup_restore_tasks = []
        # Warm the turn machinery BEFORE the queue drains: inbound turns must not build skeleton prompts.
        await self._await_startup_warmup()
        drained = await self._drain_startup_restore_queue()
        self._startup_restore_in_progress = False
        if drained:
            logger.info("Drained %d inbound message(s) queued during startup restore", drained)

    @staticmethod
    def _late_failure_callback(message: str, *, level: int = logging.WARNING):
        """Done-callback for boot-path tasks that outlive the startup-restore gate: surface a late
        failure otherwise swallowed once the task leaves ``_background_tasks``. Cancellation is
        expected (shutdown), not an error."""
        def _report(task: "asyncio.Task") -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.log(level, message, exc_info=(type(exc), exc, exc.__traceback__))
        return _report

    async def _await_startup_boot_sends(self, *, planned_restart_notification_pending: bool) -> None:
        """Run boot-path sends without letting them pin the inbound restore gate (one Telegram
        flood-control sleep must not freeze inbound on every platform): same bounded wait as the
        resume gate, sends finish in the background on timeout. The ledger claim + ``resume_pending``
        clear run INLINE before the send task exists — deferring it let a hung notification expire the
        gate with zero rows claimed, so answered turns were replayed AND redelivered.

        ``_send_restart_notification`` and ``_redeliver_pending_obligations`` used to be awaited inline
        *before* ``_finish_startup_restore`` released the gate. See #91969.
        """
        from gateway.run import _clear_planned_restart_notification, _startup_restore_drain_timeout_secs
        claimed = await self._claim_pending_obligations()

        async def _boot_sends() -> None:
            await self._send_restart_notification()
            if planned_restart_notification_pending:
                try:
                    await self._send_home_channel_startup_notifications(skip_targets=None)
                finally:
                    _clear_planned_restart_notification()
            await self._redeliver_claimed_obligations(claimed)

        boot_task = asyncio.create_task(_boot_sends())
        timeout = _startup_restore_drain_timeout_secs()
        if timeout <= 0:
            await boot_task  # unbounded: a failing send surfaces here (unlike the gate path)
            return
        await self._wait_bounded_or_release(
            {boot_task}, timeout,
            "Boot-path sends still running after %.0fs; releasing inbound gate so other platforms are not "
            "frozen. Restart notification / obligation redelivery continue in the background.",
            "background boot-path send failed after gate release: see traceback", track=True,
        )

    async def _clear_resume_pending_for_claimed_obligations(
        self, claimed: list, *, require_success: bool = False
    ) -> list:
        """Clear resume flags and return rows safe to redeliver. Startup recovery is best-effort;
        runtime reconnect recovery (``require_success``) is stricter: if the session-store write
        fails the response must not be sent, or the turn could be resumed too."""
        sendable = []
        for row in claimed:
            session_key = row.get("session_key") or ""
            if not session_key:
                sendable.append(row)
                continue
            try:
                await self.async_session_store.clear_resume_pending(session_key)
            except Exception:
                logger.debug("clear_resume_pending failed for %s", session_key, exc_info=True)
                if require_success:
                    continue
            sendable.append(row)
        return sendable

    async def _claim_pending_obligations(self) -> list:
        """Claim recoverable delivery-ledger rows and clear their ``resume_pending`` flags (pure DB
        work, no sends). Must run INLINE BEFORE ``_schedule_resume_pending_sessions`` and the
        abandonable boot-send task: these sessions already produced their answer, so the resume path
        must not re-run (re-pay for) the turn however long the sends take. Mid-send / rejected rows
        carry a visible recovered-reply marker (gateway/delivery_ledger.py). Returns the rows.

        A session with a recoverable obligation already produced its answer — the turn completed and only
        delivery is owed — so clearing ``resume_pending`` here prevents the resume path from re-running (and
        re-paying for) a turn whose output we hold, regardless of how long the sends ahead of redelivery
        take (#91969).
        """
        try:
            from gateway.delivery_ledger import ledger_enabled, sweep_recoverable
            if not await asyncio.to_thread(ledger_enabled):
                return []
            # Claim only rows whose exact transport owner is connected: platform-only filtering would spend
            # a disconnected bot's retry budget because another bot on that platform is online.
            _profile_adapters = getattr(self, "_profile_adapters", None) or {}
            _pval = lambda p: getattr(p, "value", str(p))  # noqa: E731
            _deliverable_targets = {(_pval(p), "default") for p in self.adapters}
            # Legacy rows (no adapter_profile) are unambiguous only without multiplexing; else fail closed.
            if not _profile_adapters:
                _deliverable_targets.update((_pval(p), None) for p in self.adapters)
            for _profile, _adapters in _profile_adapters.items():
                _deliverable_targets.update((_pval(p), _profile) for p in _adapters)
            claimed = await asyncio.to_thread(
                sweep_recoverable, None,
                deliverable_platforms={platform for platform, _ in _deliverable_targets},
                deliverable_targets=_deliverable_targets,
            )
        except Exception:
            logger.debug("delivery ledger sweep failed", exc_info=True)
            return []
        if not claimed:
            return []
        # Clear resume_pending for EVERY claimed row before any send: the answer is in the ledger.
        # Claiming already spent one of the row's redelivery attempts — the answer is in the ledger, so the
        # resume path must never re-run these turns (#91969).
        await self._clear_resume_pending_for_claimed_obligations(claimed)
        return claimed

    @staticmethod
    async def _release_runtime_claim_quiet(obligation_id, log_fmt: str) -> None:
        """Release a runtime delivery-ledger claim as ``send_path_degraded``; log-only on failure."""
        from gateway.delivery_ledger import release_runtime_claim
        try:
            await asyncio.to_thread(release_runtime_claim, obligation_id, "send_path_degraded")
        except Exception:
            logger.debug(log_fmt, obligation_id, exc_info=True)

    async def _redeliver_claimed_obligations(self, claimed: list) -> int:
        """Redeliver final responses for claimed rows (network half of the split): runs inside the
        bounded boot-send task, so a flood-limited send can be abandoned by the restore gate without
        reopening the turn-replay window. Returns the redelivered count."""
        if not claimed:
            return 0
        try:
            from gateway.delivery_ledger import (
                RECOVERED_MARKER, mark_delivered, mark_failed, mark_failed_with_content)
        except Exception:
            logger.debug("delivery ledger import failed", exc_info=True)
            return 0
        redelivered = 0
        for row in claimed:
            adapter = await self._obligation_adapter(row)
            if adapter is None:
                continue
            content = row["content"]
            if row.get("needs_marker"):
                content = row.get("marker", RECOVERED_MARKER) + content
            metadata = {"thread_id": row["thread_id"]} if row.get("thread_id") else None
            try:
                result = await adapter.send(chat_id=row["chat_id"], content=content, metadata=metadata)
            except Exception as send_err:
                logger.warning("obligation %s: redelivery send raised: %s", row["obligation_id"], send_err)
                result = None
            with _log_suppressed(logging.DEBUG, "delivery ledger update failed", exc_info=True):
                if result is not None and getattr(result, "success", False):
                    await asyncio.to_thread(mark_delivered, row["obligation_id"])
                    redelivered += 1
                    logger.info(
                        "Redelivered recovered final response to %s:%s (obligation %s, attempt %d)",
                        row["platform"], row["chat_id"], row["obligation_id"], row["attempts"],
                    )
                else:
                    error = str(getattr(result, "error", "") or "send failed")
                    raw = getattr(result, "raw_response", None)
                    undelivered = raw.get("undelivered_content") if isinstance(raw, dict) else None
                    if isinstance(undelivered, str) and undelivered:
                        await asyncio.to_thread(
                            mark_failed_with_content, row["obligation_id"], undelivered, error)
                    else:
                        await asyncio.to_thread(mark_failed, row["obligation_id"], error)
        return redelivered

    async def _obligation_adapter(self, row: dict):
        """Resolve the adapter for a claimed ledger row, or None when it cannot be delivered now."""
        try:
            platform = Platform(row["platform"])
        except Exception:
            logger.debug("obligation %s: unknown platform %r", row["obligation_id"], row.get("platform"))
            return None
        if "profile" in row:
            adapter = self._authorization_adapter(platform, row.get("profile"))
        else:
            # Startup rows preserve the historical default-adapter route.
            adapter = self.adapters.get(platform)
        # A runtime claim whose reconnect vanished before dispatch is released without spending an
        # attempt; startup claims keep their state (attempts cap + stale cutoff bound retries).
        if adapter is None and row.get("runtime_recovery"):
            await self._release_runtime_claim_quiet(
                row["obligation_id"], "failed to release undispatched runtime obligation %s"
            )
        return adapter

    async def _redeliver_pending_obligations(self) -> int:
        """Claim + redeliver in one call. Stable shape for tests/external callers; the startup path
        calls the halves separately so the DB half runs inline before the abandonable send task."""
        return await self._redeliver_claimed_obligations(await self._claim_pending_obligations())

    async def _redeliver_failed_obligations_for_platform(
        self, platform: Platform, *, profile: Optional[str] = None,
    ) -> int:
        """Replay one adapter identity's transient failures after reconnect: the startup sweep cannot
        claim live-owner rows, so ``send_path_degraded`` responses would otherwise stay failed until
        the next restart. Best-effort; reuses the startup redelivery contract."""
        try:
            from gateway.delivery_ledger import ledger_enabled, sweep_failed_for_runtime
            if not await asyncio.to_thread(ledger_enabled):
                return 0
            claimed = await asyncio.to_thread(sweep_failed_for_runtime, platform.value, profile=profile)
        except Exception:
            logger.debug(
                "runtime delivery ledger sweep failed after %s reconnect", platform.value, exc_info=True,
            )
            return 0
        if not claimed:
            return 0
        # Clear before any send so the reconnect path cannot both redeliver AND resume the same turn.
        sendable = await self._clear_resume_pending_for_claimed_obligations(claimed, require_success=True)
        sendable_ids = {row["obligation_id"] for row in sendable}
        for row in claimed:
            if row["obligation_id"] not in sendable_ids:
                await self._release_runtime_claim_quiet(
                    row["obligation_id"], "failed to release runtime delivery claim %s"
                )
        return await self._redeliver_claimed_obligations(sendable)

    def _resume_pending_candidates(self, platform=None) -> Optional[list]:
        """Snapshot resume-pending entries (optionally scoped to ``platform``); None when
        enumeration failed or the restart-loop breaker tripped for this boot."""
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                    and (platform is None or entry.origin.platform == platform)
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return None
        # Restart-loop breaker: only boots WITH restart-interrupted sessions count; when tripped, skip
        # auto-resume for THIS boot only (inbound still served; sessions stay resume_pending).
        if candidates:
            try:
                from gateway import restart_loop_guard as _rlg
                _max_restarts, _window, _max_gap = self._restart_loop_guard_config()
                if _rlg.check_and_record(_max_restarts, _window, max_gap_seconds=_max_gap):
                    return None
            except Exception as exc:  # noqa: BLE001 — breaker must fail OPEN
                logger.debug("Restart-loop guard check skipped: %s", exc)
        return candidates

    def _resume_owner_authorized(self, session_key: str, source) -> bool:
        """Validate the session owner against the CURRENT allowlist: a session created before the
        allowlist existed (or whose owner was since removed) must not silently receive a full agent
        response just because it carries a resume marker."""
        try:
            if self._is_user_authorized(source):
                return True
            logger.warning(
                "Skipping auto-resume for %s: session owner is no "
                "longer authorized under the current allowlist", session_key,
            )
        except Exception as exc:
            logger.warning("Skipping auto-resume for %s: authorization check failed: %s", session_key, exc)
        return False

    def _schedule_resume_pending_sessions(self, platform=None) -> int:
        """Auto-continue fresh restart-interrupted sessions: synthesize an empty-text turn (the
        ``_is_resume_pending`` injection path owns the wording). Sessions whose adapter is offline stay
        ``resume_pending`` for the reconnect watcher, which re-calls this scoped to that ``platform``;
        sessions with a running agent are skipped so none is resumed twice."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _auto_continue_freshness_window
        window = _auto_continue_freshness_window()
        candidates = self._resume_pending_candidates(platform)
        if candidates is None:
            return 0
        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue
            # Already being resumed (e.g. scheduled at startup, still in-flight) — no second turn.
            if self._is_session_running(entry.session_key):
                continue
            source = entry.origin
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s", entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue
            if not self._resume_owner_authorized(entry.session_key, source):
                continue
            # Claim the slot *before* spawning so an inbound message arriving before the task's first
            # await queues instead of building a duplicate AIAgent.
            _resume_state = self._session_state(entry.session_key)
            _resume_state.turn.agent = _AGENT_PENDING_SENTINEL
            _resume_state.turn.started_ts = time.time()
            self._persist_active_agents()
            # Empty-text internal event: the _is_resume_pending branch prepends the reason-aware note.
            event = MessageEvent(text="", message_type=MessageType.TEXT, source=source, internal=True)
            task = self._retain_background_task(
                asyncio.create_task(self._run_startup_resume_event(adapter, event, entry.session_key))
            )
            if getattr(self, "_startup_restore_in_progress", False):
                tasks = getattr(self, "_startup_restore_tasks", None)
                if tasks is None:
                    tasks = self._startup_restore_tasks = []
                tasks.append(task)
            scheduled += 1
        if scheduled:
            logger.info("Scheduled auto-resume for %d restart-interrupted session(s)", scheduled)
        return scheduled

    def _startup_should_abort(self) -> bool:
        return self._restart_requested or self._draining or self._shutdown_event.is_set()

    async def _startup_teardown_adapter(self, adapter, platform) -> None:
        """Cancel an adapter's background tasks (best-effort) then disconnect it."""
        with _log_suppressed(logging.DEBUG, "✗ %s background-task cancel error: %s", platform.value):
            await adapter.cancel_background_tasks()
        await self._safe_adapter_disconnect(adapter, platform)

    def _startup_retry_entry(self, platform, adapter, platform_config, *, queued: bool = True) -> dict:
        """``_failed_platforms`` entry for a platform that failed at startup (first retry in 30s)."""
        return self._reconnect_queue_entry(platform, adapter, platform_config, attempts=1, delay=30, queued=queued)

    async def _abort_startup_if_shutdown_requested(
        self, adapter: Optional[BasePlatformAdapter] = None, platform: Optional[Platform] = None
    ) -> bool:
        """Clean up and exit startup when restart/shutdown begins mid-startup."""
        if not self._startup_should_abort():
            return False
        if adapter is not None and platform is not None:
            await self._startup_teardown_adapter(adapter, platform)
        stop_task = self._stop_task
        current_task = asyncio.current_task()
        if stop_task is not None and stop_task is not current_task:
            await stop_task
        elif not self._shutdown_event.is_set():
            await self.stop(
                restart=self._restart_requested, detached_restart=self._restart_detached,
                service_restart=self._restart_via_service,
            )
        return True

    def _start_loop_liveness_guards(self, loop: asyncio.AbstractEventLoop) -> None:
        """Arm the selector floor and out-of-loop watchdog before adapters. Disabled entirely with
        ``gateway.loop_watchdog: false`` in config.yaml (config-only knob).

        See #69089.
        """
        from gateway.shutdown_watchdog import _arm_loop_floor_timer, start_loop_liveness_watchdog
        config = getattr(self, "config", None)
        if config is not None and not getattr(config, "loop_watchdog", True):
            return
        if getattr(self, "_loop_floor_timer_handle", None) is None:
            with _log_suppressed(logging.DEBUG, "Failed to arm gateway loop floor timer", exc_info=True):
                self._loop_floor_timer_handle = _arm_loop_floor_timer(loop)
        watchdog = getattr(self, "_loop_liveness_watchdog", None)
        if watchdog is None or not watchdog.is_alive():
            try:
                # getattr defaults cover config=None test paths; loaded values are already clamped.
                self._loop_liveness_watchdog = start_loop_liveness_watchdog(
                    loop,
                    probe_interval=float(getattr(
                        config, "loop_watchdog_probe_interval_s", DEFAULT_LOOP_WATCHDOG_INTERVAL_S
                    )),
                    probe_timeout=float(getattr(
                        config, "loop_watchdog_probe_timeout_s", DEFAULT_LOOP_WATCHDOG_TIMEOUT_S
                    )),
                    max_strikes=int(getattr(
                        config, "loop_watchdog_max_strikes", DEFAULT_LOOP_WATCHDOG_MAX_STRIKES
                    )),
                )
            except Exception:
                logger.debug("Failed to start gateway loop liveness watchdog", exc_info=True)

    def _stop_loop_liveness_guards(self) -> None:
        """Disarm lifetime liveness guards before shutdown can load the loop — including the
        heartbeat writer, which would otherwise make a draining gateway look healthy to probes."""
        for attr, method, what in (
            ("_loop_liveness_watchdog", "stop", "stop gateway loop liveness watchdog"),
            ("_loop_floor_timer_handle", "cancel", "cancel gateway loop floor timer"),
            ("_loop_heartbeat_task", "cancel", "cancel gateway loop heartbeat task"),
        ):
            guard = getattr(self, attr, None)
            setattr(self, attr, None)
            if guard is not None:
                with _log_suppressed(logging.DEBUG, "Failed to %s", what, exc_info=True):
                    getattr(guard, method)()

    async def _consume_clean_shutdown_marker(self, marker_path) -> int:
        """Discard orphan turn markers before consuming a clean-exit receipt. Raises (fail closed):
        continuing with the old receipt would let a later unclean exit masquerade as clean."""
        discarded = await self.async_session_store.discard_active_turn_markers()
        marker_path.unlink()
        return discarded

    async def _recover_unclean_sessions(self) -> tuple[int, int]:
        """Recover exact active turns, then run the legacy recency fallback."""
        from gateway.run import _float_env
        exact = 0
        fallback = 0
        with _log_suppressed(logging.WARNING, "Exact active-turn recovery on startup failed: %s"):
            agent_timeout = max(1.0, _float_env("HERMES_AGENT_TIMEOUT", 1800))
            exact = await self.async_session_store.recover_interrupted_turns(
                max_age_seconds=max(60 * 60, int(agent_timeout * 2))
            )
        with _log_suppressed(logging.WARNING, "Legacy session recovery on startup failed: %s"):
            fallback = await self.async_session_store.suspend_recently_active(max_age_seconds=120)
        return exact, fallback

    @staticmethod
    def _start_hosted_room_worker_sync():
        """Start the local Group Chat worker without importing the dashboard."""
        import tui_gateway.server  # noqa: F401
        from tui_gateway import methods_groups
        service = methods_groups.get_hosted_room_service()
        if service is None:
            service = methods_groups.start_hosted_room_service()
        if service is None:
            raise RuntimeError("Group Chat worker has no bound session backend")
        status = service.runtime.status()
        if not status.get("running") or status.get("stopping"):
            raise RuntimeError("Group Chat worker did not start")
        return service

    async def _ensure_hosted_room_worker(self):
        return await asyncio.to_thread(self._start_hosted_room_worker_sync)

    async def _hosted_room_worker_watcher(self, interval: float = 1.0) -> None:
        """Keep the room worker alive for the messaging gateway lifetime."""
        while self._running:
            await self._ensure_hosted_room_worker()
            await asyncio.sleep(interval)

    async def _stop_hosted_room_worker(self, timeout: float = 5.0) -> bool:
        """Pause room execution durably without interrupting accepted turns."""
        from tui_gateway import methods_groups
        return await asyncio.to_thread(methods_groups.stop_hosted_room_service, timeout=timeout)

    def _start_loop_heartbeat_task(self) -> None:
        """Start the loop-liveness heartbeat task (idempotent, best-effort). An asyncio task so a
        frozen loop stops refreshing ``state/gateway.heartbeat``; cancelled with the others in stop().

        See #66892.
        """
        with _log_suppressed(logging.DEBUG, "Failed to start gateway loop heartbeat", exc_info=True):
            _existing_hb = getattr(self, "_loop_heartbeat_task", None)
            if _existing_hb is not None and not _existing_hb.done():
                return
            task = self._loop_heartbeat_task = asyncio.create_task(
                loop_heartbeat_forever(
                    interval_s=DEFAULT_HEARTBEAT_INTERVAL_S,
                    start_time=getattr(self, "_gateway_started_at", 0.0),
                )
            )
            # PERMANENT watcher tag so the scale-to-zero idle check doesn't count it as busy forever.
            task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                self._track_task_in(_bg, task)

    def _open_faulthandler_log(self):
        """Open (append) ``<log_dir>/gateway_faulthandler.log``, creating the directory."""
        from gateway.run import get_hermes_home
        log_dir = getattr(self.config, "log_dir", None) or os.path.join(str(get_hermes_home()), "logs")
        os.makedirs(log_dir, exist_ok=True)
        return open(os.path.join(log_dir, "gateway_faulthandler.log"), "a", encoding="utf-8")

    def _start_install_faulthandler(self) -> None:
        """Enable faulthandler (stderr or a log file) plus the SIGUSR2 stack-dump hook."""
        # sys.stderr may be None (Windows VBS / pythonw / detached service): fall back to a log file.
        try:
            # Enable faulthandler for stack dumps on freezes/crashes (#70344). Falls back to a log file when
            # sys.stderr is None (Windows VBS / pythonw / detached service) — otherwise the gateway would
            # die here and take every adapter offline. See #71671.
            faulthandler.enable()
        except (RuntimeError, ValueError, OSError):
            with _log_suppressed(logging.DEBUG, "faulthandler.enable() unavailable", exc_info=True):
                faulthandler.enable(file=self._open_faulthandler_log(), all_threads=True)
        # SIGUSR2 stack dump to file for service managers that drop stderr; POSIX-only.
        _sigusr2 = getattr(signal, "SIGUSR2", None)
        if _sigusr2 is not None and hasattr(faulthandler, "register"):
            with _log_suppressed(logging.DEBUG, "Could not set up faulthandler file logging", exc_info=True):
                faulthandler.register(
                    _sigusr2, file=self._open_faulthandler_log(), all_threads=True, chain=True,
                )

    def _start_log_startup_environment(self) -> None:
        """Bind the gateway loop, disarm the startup watchdog, and log the startup environment."""
        from gateway.run import _write_runtime_status_quiet
        try:
            self._gateway_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._gateway_loop = None
        if self._gateway_loop is not None:
            self._start_loop_liveness_guards(self._gateway_loop)
            # Loop live: the loop-liveness watchdog takes over from the startup watchdog. Disarm even
            # when loop guards are config-disabled; only inside this branch (no live loop = stay armed).
            with _log_suppressed(logging.DEBUG, "Startup watchdog disarm failed", exc_info=True):
                from hermes_startup_watchdog import disarm_startup_watchdog
                disarm_startup_watchdog()
        logger.info("Session storage: %s", self.config.sessions_dir)
        self._start_log_systemd_timing_alignment()
        # Log the resolved max_iterations so operators can verify the config.yaml → env bridge.
        with suppress(Exception):
            logger.info(
                "Agent budget: max_iterations=%d (agent.max_turns from config.yaml, "
                "or HERMES_MAX_ITERATIONS from .env, or default 500)",
                int(os.getenv("HERMES_MAX_ITERATIONS", "500")),
            )
        # Warn prominently when redaction is opted out; the redactor snapshots its state at import time,
        # so this line is the source of truth for the process lifetime.
        with suppress(Exception):
            # Redaction status: ON by default (#17691).
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() in {"1", "true", "yes", "on"}:
                logger.info(
                    "Secret redaction: ENABLED (tool output, logs, and chat "
                    "responses are scrubbed before delivery)"
                )
            else:
                logger.warning(
                    "Secret redaction: DISABLED (HERMES_REDACT_SECRETS=%s). API keys and tokens may appear "
                    "verbatim in chat output, session JSONs, and logs. Set security.redact_secrets: true "
                    "in config.yaml to re-enable.", _redact_raw,
                )
        with suppress(Exception):
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        _write_runtime_status_quiet(gateway_state="starting", exit_reason=None, clear_profile_platforms=True)
        with _log_suppressed(logging.DEBUG, "gateway health OTLP export startup failed", exc_info=True):
            from hermes_cli.config import load_config
            from agent.monitoring.gateway_health_export import start_gateway_health_export
            self._gateway_health_export_runtime = start_gateway_health_export(load_config())
            if getattr(self._gateway_health_export_runtime, "enabled", False):
                logger.info("Gateway health OTLP export: enabled")
        # Supply-chain advisories: log only (never block startup or surface to users; only the operator can act).
        with _log_suppressed(logging.DEBUG, "security advisory check failed at gateway startup", exc_info=True):
            from hermes_cli.security_advisories import detect_compromised, gateway_log_message
            _adv_msg = gateway_log_message(detect_compromised())
            if _adv_msg:
                logger.warning("%s", _adv_msg)
                logger.warning("Run `hermes doctor` on the gateway host for full remediation steps.")

    def _start_log_systemd_timing_alignment(self) -> None:
        """Warn when systemd's TimeoutStopSec does not cover the drain window (a unit file from before
        an upgrade may encode the old default, so SIGKILL hits mid-drain). Never raises."""
        with _log_suppressed(logging.DEBUG, "check_systemd_timing_alignment failed: %s"):
            from gateway.shutdown_forensics import check_systemd_timing_alignment
            _alignment = check_systemd_timing_alignment(
                self._restart_drain_timeout,
                getattr(self, "_cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT),
            )
            if _alignment is not None and _alignment.get("mismatch"):
                logger.warning(
                    "Stale systemd unit detected: %s has TimeoutStopSec=%.0fs but drain_timeout=%.0fs "
                    "cron_drain_timeout=%.0fs (expected >=%.0fs). systemd may SIGKILL the gateway "
                    "mid-drain. Run `hermes gateway install --force` to regenerate the unit, or shorten "
                    "agent.restart_drain_timeout / agent.cron_drain_timeout.",
                    _alignment.get("unit", "(unknown)"), _alignment["timeout_stop_sec"],
                    _alignment["drain_timeout"],
                    _alignment.get("cron_drain_timeout", DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT),
                    _alignment["expected_min"],
                )

    # Builtin platforms whose ``<P>_ALLOWED_USERS`` / ``<P>_ALLOW_ALL_USERS`` env vars count as an
    # allowlist / open-access opt-in; plugin platforms are appended at check time.
    _ALLOWLIST_ENV_PLATFORMS = (
        "TELEGRAM", "DISCORD", "WHATSAPP", "WHATSAPP_CLOUD", "SLACK", "SIGNAL", "EMAIL", "SMS",
        "MATTERMOST", "MATRIX", "DINGTALK", "FEISHU", "WECOM", "WECOM_CALLBACK", "WEIXIN",
        "BLUEBUBBLES", "QQ", "YUANBAO",
    )
    _BUILTIN_ALLOWED_USERS_VARS = tuple(f"{p}_ALLOWED_USERS" for p in _ALLOWLIST_ENV_PLATFORMS) + (
        "SIGNAL_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS", "GATEWAY_ALLOWED_USERS",
    )
    _BUILTIN_ALLOW_ALL_VARS = tuple(f"{p}_ALLOW_ALL_USERS" for p in _ALLOWLIST_ENV_PLATFORMS)

    def _start_check_access_policy(self) -> bool:
        """Warn about missing allowlists; return True when startup must be refused."""
        from gateway.run import (
            _OWN_POLICY_OPEN_ENV, _own_policy_open_startup_violation, _write_runtime_status_quiet
        )
        # Plugin platforms declare their own allowed_users_env / allow_all_env.
        allowed_vars = list(self._BUILTIN_ALLOWED_USERS_VARS)
        allow_all_vars = ["GATEWAY_ALLOW_ALL_USERS", *self._BUILTIN_ALLOW_ALL_VARS]
        with suppress(Exception):
            from gateway.platform_registry import platform_registry
            entries = platform_registry.plugin_entries()
            allowed_vars += [e.allowed_users_env for e in entries if e.allowed_users_env]
            allow_all_vars += [e.allow_all_env for e in entries if e.allow_all_env]
        if not any(os.getenv(v) for v in allowed_vars) and not any(
            os.getenv(v, "").lower() in {"true", "1", "yes"} for v in allow_all_vars
        ):
            logger.warning(
                "No env user allowlists configured. Messaging platforms default to pairing/allowlist "
                "policies and will deny unknown senders unless you configure platform allowlists (e.g., "
                "TELEGRAM_ALLOWED_USERS=your_id) or explicitly opt in with GATEWAY_ALLOW_ALL_USERS=true "
                "plus dm_policy/group_policy: open on the platform."
            )
        reason = _own_policy_open_startup_violation(self.config)
        if reason:
            platform_value = reason.split(":", 1)[0]
            allow_all_env = next(
                (env[2] for p, env in _OWN_POLICY_OPEN_ENV.items() if p.value == platform_value), None
            )
            logger.error(
                "Refusing to start: %s has dm_policy/group_policy set to 'open' "
                "but neither GATEWAY_ALLOW_ALL_USERS nor %s is enabled.", platform_value,
                allow_all_env or "a platform allow-all flag",
            )
            _write_runtime_status_quiet(gateway_state="startup_failed", exit_reason=reason)
            self._request_clean_exit(reason)
            return True
        return False

    @staticmethod
    def _start_register_plugins_relay_hooks() -> None:
        """Plugin discovery, relay registration and shell-hook/webhook registration. Never raises."""
        # Discover plugins before shell hooks (plugin block decisions win ties). Explicit: the gateway
        # lazily imports run_agent, so model_tools' discover_plugins() side-effect may not have run.
        with _log_suppressed(logging.WARNING, "plugin discovery failed at gateway startup", exc_info=True):
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
        # Generic relay adapter only if GATEWAY_RELAY_URL / gateway.relay_url is set; no URL -> no-op.
        try:
            from gateway.relay import (
                register_relay_adapter, relay_url, self_provision_relay, send_relay_policy
            )
            # Relay self-provision sets GATEWAY_RELAY_* in os.environ BEFORE registration reads them.
            self_provision_relay()
            if register_relay_adapter():
                logger.info("relay adapter registered (connector at %s)", relay_url())
                # Declare the relevance policy to the connector so the SAME behavior governs relay delivery.
                send_relay_policy()
        except Exception:
            logger.warning("relay adapter registration failed at gateway startup", exc_info=True)
        GatewayStartupMixin._register_config_hooks("shell-hook registration failed at gateway startup")

    @staticmethod
    def _register_config_hooks(fail_fmt: str, *fail_args, level: int = logging.DEBUG) -> None:
        """Register declarative shell hooks + outbound webhooks from the CURRENT scope's config.

        Gateway has no TTY, so consent must come from --accept-hooks, HERMES_ACCEPT_HOOKS, or
        hooks_auto_accept: true; ``accept_hooks=False`` lets register_from_config resolve env + config.
        Never raises (logged at ``level``).
        """
        try:
            from hermes_cli.config import load_config
            from agent.shell_hooks import register_from_config
            from agent.outbound_webhooks import register_from_config as register_outbound_webhooks
            _hooks_cfg = load_config()
            register_from_config(_hooks_cfg, accept_hooks=False)
            register_outbound_webhooks(_hooks_cfg)
        except Exception:
            logger.log(level, fail_fmt, *fail_args, exc_info=True)

    async def _start_recover_previous_run(self) -> None:
        """Plugins, relay, hooks, then crash/clean-exit recovery of processes and sessions."""
        from gateway.run import _hermes_home
        self._start_register_plugins_relay_hooks()
        self.hooks.discover_and_load()
        # Recover background processes from checkpoint (crash recovery)
        with _log_suppressed(logging.WARNING, "Process checkpoint recovery: %s"):
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        # Recover sessions active at last exit (exact turn markers + 120s recency fallback for
        # marker-less older turns). SKIP after a clean exit — the previous process already drained.
        _clean_marker = _hermes_home / ".clean_shutdown"
        if _clean_marker.exists():
            logger.info("Previous gateway exited cleanly — skipping session suspension")
            try:
                discarded = await self._consume_clean_shutdown_marker(_clean_marker)
            except Exception as exc:
                logger.error(
                    "Clean-start marker cleanup failed; refusing startup so the "
                    "clean-exit receipt cannot mask a later unclean exit: %s", exc,
                )
                raise RuntimeError("clean-start recovery cleanup failed") from exc
            if discarded:
                logger.info("Discarded %d orphan active-turn marker(s) after clean shutdown", discarded)
        else:
            exact, fallback = await self._recover_unclean_sessions()
            if exact + fallback:
                logger.info(
                    "Marked %d in-flight session(s) as resumable from previous run "
                    "(%d exact, %d legacy)", exact + fallback, exact, fallback,
                )
        # Stuck-loop detection: a session active across 3+ consecutive restarts is auto-suspended.
        with _log_suppressed(logging.DEBUG, "Stuck-loop detection failed: %s"):
            # Auto-suspend it so the user gets a clean slate on the next message. See #7536.
            stuck = self._suspend_stuck_loop_sessions()
            if stuck:
                logger.warning("Auto-suspended %d stuck-loop session(s)", stuck)

    async def _start_prefilter_platforms(self) -> Tuple[bool, int, list, list]:
        """Create + wire an adapter per enabled platform (no connects). Returns
        (aborted, enabled_platform_count, multiplex_skipped_platforms, pending_connects)."""
        from gateway.run import _platform_has_bot_credential
        enabled_platform_count = 0
        _multiplex_on = self._multiplex_on()
        _multiplex_skipped_platforms: list[Platform] = []
        _pending_connects = []  # (platform, platform_config, adapter); connected concurrently later
        for platform, platform_config in self.config.platforms.items():
            if await self._abort_startup_if_shutdown_requested():
                return True, enabled_platform_count, _multiplex_skipped_platforms, _pending_connects
            if not platform_config.enabled:
                continue
            # Multiplex: a platform enabled in the shared config.yaml may hold its token only in a
            # secondary profile's .env; an empty primary would queue a reconnect loop that never heals.
            # Starting that primary adapter with an empty token fails immediately and queues an infinite
            # reconnect loop that can never heal (#64674). Secondary profiles still start their own adapters
            # under _profile_runtime_scope with the real token -- skip the empty primary instead of failing
            # loudly.
            if _multiplex_on and not _platform_has_bot_credential(platform, platform_config):
                logger.info(
                    "Skipping %s on default profile: no bot credential in this "
                    "profile's secrets. Secondary multiplexed profiles that "
                    "provide the token will still connect.", platform.value,
                )
                _multiplex_skipped_platforms.append(platform)
                continue
            enabled_platform_count += 1
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                # Distinguish between missing builtin deps and missing plugin
                if platform.value in {m.value for m in Platform.__members__.values()}:
                    logger.warning("No adapter available for %s", platform.value)
                else:
                    logger.warning(
                        "No adapter for '%s' -- is the plugin installed? "
                        "(platform is enabled in config.yaml but no plugin registered it)", platform.value,
                    )
                continue
            # Under multiplexing the default profile needs the same whole-handler runtime scope as a
            # secondary (authorization and prompt rendering run before the agent-turn scope).
            self._wire_adapter_handlers(adapter)
            _pending_connects.append((platform, platform_config, adapter))
        return False, enabled_platform_count, _multiplex_skipped_platforms, _pending_connects

    async def _start_connect_pending(self, _pending_connects: list) -> Optional[list]:
        """Connect the pre-filtered adapters concurrently. Returns the raw per-platform results, or
        None when a restart/shutdown aborted startup mid-connect (adapters already torn down)."""
        async def _connect_one_startup(p, p_cfg, adp):
            """Connect a single platform; never let one block the others (#83791)."""
            if await self._abort_startup_if_shutdown_requested(adp, p):
                return (p, adp, p_cfg, "aborted", None)
            logger.info("Connecting to %s...", p.value)
            self._update_platform_runtime_status(
                p.value, platform_state="connecting", error_code=None, error_message=None,
            )
            try:
                ok = await self._connect_initial_adapter_with_timeout(adp, p)
            except Exception as _exc:  # noqa: BLE001 - surfaced below as a retryable error
                return (p, adp, p_cfg, "exception", _exc)
            return (p, adp, p_cfg, "ok" if ok else "failed", None)

        if not _pending_connects:
            return []
        # Abort-aware concurrent wait: a restart/shutdown mid-connect cancels pending connects, tears down
        # completed ones, and aborts startup.
        _task_map = {
            asyncio.ensure_future(_connect_one_startup(p, c, a)): (p, c, a) for (p, c, a) in _pending_connects
        }
        _pending_tasks = set(_task_map)
        while _pending_tasks:
            _done, _pending_tasks = await asyncio.wait(_pending_tasks, timeout=0.05)
            if _pending_tasks and self._startup_should_abort():
                break
        else:
            return [_t.exception() or _t.result() for _t in _task_map]
        # Settle in-flight connects FIRST so a completed adapter's disconnect cannot unblock a sibling.
        for _t in _pending_tasks:
            _t.cancel()
        await asyncio.gather(*_pending_tasks, return_exceptions=True)
        # Then tear down adapters whose connect succeeded — never registered, so stop() won't reach them.
        _connected_ok = [
            _t for _t in _task_map
            if _t not in _pending_tasks and not _t.cancelled() and _t.exception() is None and _t.result()[3] == "ok"
        ]
        for _t in [*_pending_tasks, *_connected_ok]:
            await self._startup_teardown_adapter(_task_map[_t][2], _task_map[_t][0])
        await self._abort_startup_if_shutdown_requested()
        return None

    def _startup_queue_transient_failure(
        self, platform, adapter, platform_config, message: str, startup_retryable_errors: list
    ) -> None:
        """Mark a platform ``retrying`` with ``message`` and queue it in ``_failed_platforms``."""
        self._update_platform_runtime_status(
            platform.value, platform_state="retrying", error_code=None, error_message=message,
        )
        startup_retryable_errors.append(f"{platform.value}: {message}")
        self._failed_platforms[platform] = self._startup_retry_entry(platform, adapter, platform_config)

    async def _start_aggregate_connect_results(
        self, _raw: list, startup_retryable_errors: list, startup_nonretryable_errors: list
    ) -> int:
        """Apply connect outcomes to shared state single-threaded (exactly as the original serial
        loop did); returns the connected adapter count."""
        connected_count = 0
        for _item in _raw:
            if isinstance(_item, Exception):
                # Unexpected escape from _connect_one_startup (shouldn't happen); log and skip.
                logger.error("Unexpected startup connect error: %s", _item)
                continue
            platform, adapter, platform_config, outcome, exc = _item
            if outcome == "aborted":
                continue
            if outcome == "exception":
                logger.error("\u2717 %s error: %s", platform.value, exc)
                # An adapter that raised mid-connect may still hold a ClientSession/subprocess; treat
                # unexpected exceptions as transient and queue for retry.
                await self._safe_adapter_disconnect(adapter, platform)
                self._startup_queue_transient_failure(
                    platform, adapter, platform_config, str(exc), startup_retryable_errors
                )
                continue
            if outcome == "ok":
                self._publish_primary_adapter(platform, adapter)
                connected_count += 1
                # connect() may return True on a degraded (unconfirmed) receive path; don't stamp "connected".
                _degraded = adapter.send_path_degraded
                self._update_platform_runtime_status(
                    platform.value, platform_state="retrying" if _degraded else "connected", error_code=None,
                    error_message=adapter.DEGRADED_STATUS_MESSAGE if _degraded else None,
                )
                logger.info("\u2713 %s connected%s", platform.value, " (degraded)" if _degraded else "")
                continue
            # outcome == "failed"
            logger.warning("\u2717 %s failed to connect", platform.value)
            # A failed connect() may have allocated ClientSessions / poll tasks / subprocesses.
            await self._safe_adapter_disconnect(adapter, platform)
            if not adapter.has_fatal_error:
                # No fatal error info means likely a transient issue -- queue for retry
                self._startup_queue_transient_failure(
                    platform, adapter, platform_config, "failed to connect", startup_retryable_errors
                )
                continue
            # A live foreign token holder is an ownership conflict, not a blip: retryable only for
            # MID-RUN reconnects; at startup route it non-retryable so the gateway exits 78, not deaf.
            _retryable = adapter.fatal_error_retryable and not is_global_startup_conflict(adapter.fatal_error_code)
            self._update_platform_runtime_status(
                platform.value, platform_state="retrying" if _retryable else "fatal",
                error_code=adapter.fatal_error_code, error_message=adapter.fatal_error_message,
            )
            target = startup_retryable_errors if _retryable else startup_nonretryable_errors
            target.append(f"{platform.value}: {adapter.fatal_error_message}")
            if _retryable:
                self._failed_platforms[platform] = self._startup_retry_entry(
                    platform, adapter, platform_config, queued=False
                )
        return connected_count

    def _startup_fail_fatal_config(self, reason: str) -> None:
        """Record a fatal-config startup failure (exit 78) and request a clean exit."""
        from gateway.run import _write_runtime_status_quiet
        _write_runtime_status_quiet(gateway_state="startup_failed", exit_reason=reason)
        self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
        self._request_clean_exit(reason)
        self._startup_restore_in_progress = False

    async def _start_secondary_profiles(
        self, connected_count: int, _multiplex_skipped_platforms: list
    ) -> Tuple[bool, int]:
        """Bring up multiplexed secondary-profile adapters. Returns (aborted, connected_count)."""
        from gateway.run import MultiplexConfigError
        # Secondary-profile adapters connect under their own home + credential scope.
        try:
            connected_count += await self._start_secondary_profile_adapters()
        except MultiplexConfigError as e:
            # Invalid multiplexer config — abort cleanly rather than run a half-wired gateway.
            logger.error("Gateway multiplexer config error: %s", str(e))
            self._startup_fail_fatal_config(str(e))
            return True, connected_count
        except Exception as e:
            logger.error("Secondary-profile adapter startup failed: %s", e, exc_info=True)
        finally:
            # Startup authority is one phase: from here on every adapter retry is non-evicting.
            self._platform_lock_takeover_on_start = False
        # A platform skipped on the primary should have been picked up by a secondary owning the token;
        # if none did it is enabled yet silently unserved — say so loudly.
        # If none did, the platform is enabled in config.yaml yet silently unserved — surface it loudly so
        # the operator sees a config problem instead of a quiet dead channel (#64674 follow-up).
        for _skipped in _multiplex_skipped_platforms:
            if not any(_skipped in _profile_map for _profile_map in self._profile_adapters.values()):
                logger.warning(
                    "%s is enabled but no profile (default or secondary) provided a bot credential for it — "
                    "the platform is not being served. Add its token to the profile that should "
                    "own it, or disable the platform.", _skipped.value,
                )
        return False, connected_count

    def _start_handle_no_connections(
        self, connected_count: int, enabled_platform_count: int, startup_retryable_errors: list,
        startup_nonretryable_errors: list,
    ) -> bool:
        """Log/degrade when nothing connected; return True when startup must exit."""
        from gateway.run import _write_runtime_status_quiet
        if connected_count != 0:
            return False
        if startup_nonretryable_errors and not startup_retryable_errors:
            reason = "; ".join(startup_nonretryable_errors)
            logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
            self._startup_fail_fatal_config(reason)
            return True
        if startup_nonretryable_errors:
            # Mixed (some fatal, some transient): exiting 78 would take the gateway PERMANENTLY down
            # over a blip. Log the fatal side loudly and fall through to the degraded/retry path.
            logger.error(
                # WhatsApp enabled but never paired) while others hit merely transient errors (e.g. Telegram
                # TimedOut during polling startup). Exiting with GATEWAY_FATAL_CONFIG_EXIT_CODE here is
                # wrong in both supervision worlds: under supervisors that honor the exit-78 contract
                # (systemd RestartPreventExitStatus, s6 finish→125 since #51228) the gateway goes
                # PERMANENTLY down over a network blip; under anything else it crash-loops. Either way the
                # retryable platforms never get their retry. Log the fatal side loudly, then fall through to
                # the degraded/retry path below: the reconnect watcher recovers the retryable platforms; the
                # non-retryable ones remain fatal-parked and visible in runtime status.
                "%d platform(s) fatally misconfigured and parked: %s. "
                "Staying alive so retryable platforms can recover.",
                len(startup_nonretryable_errors), "; ".join(startup_nonretryable_errors),
            )
        if enabled_platform_count <= 0:
            logger.warning("No messaging platforms enabled.")
            logger.info("Gateway will continue running for cron job execution.")
            return False
        if startup_retryable_errors:
            # All retryable: stay alive (cron runs, watcher recovers) rather than systemd restart-loop.
            logger.warning(
                "Gateway started with no connected platforms — %d platform(s) queued for retry: %s",
                len(self._failed_platforms), "; ".join(startup_retryable_errors),
            )
            _write_runtime_status_quiet(gateway_state="degraded", exit_reason=None)
        # No adapter for any enabled platform: fleet nodes share one config.yaml but hold a subset of
        # credentials, so degrade gracefully.
        logger.warning(
            # Fall through to the normal "running" state — reconnect watcher takes it from here. In fleet
            # deployments the same config.yaml is shared across nodes that may only have credentials for a
            # subset of platforms. Rather than failing hard, degrade gracefully and allow cron jobs to run
            # (#5196).
            "No adapter could be created for any of the %d configured platform(s). "
            "Check that required dependencies are installed and credentials are set. "
            "Gateway will continue for cron job execution.", enabled_platform_count,
        )
        return False

    async def _start_post_connect_services(self, connected_count: int) -> None:
        """Room worker, heartbeat, gateway:startup hook, channel directory, /update notice."""
        from gateway.run import _hermes_home
        try:
            await self._ensure_hosted_room_worker()
        except Exception:
            logger.error(
                "Group Chat worker failed to start; mutating Group Chat commands "
                "will fail closed until supervision recovers it", exc_info=True,
            )
        self._spawn_supervised(self._hosted_room_worker_watcher, "hosted_room_worker")
        self._start_loop_heartbeat_task()
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {"platforms": [p.value for p in self.adapters]})
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        # Initial channel directory for send_message name resolution
        with _log_suppressed(logging.WARNING, "Channel directory build failed: %s"):
            from gateway.channel_directory import build_channel_directory
            directory = await build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        # Restarting after a /update still in progress: keep watching so we notify when it finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            (_hermes_home / name).exists()
            for name in (".update_pending.json", ".update_pending.claimed.json")
        ):
            self._schedule_update_notification_watch()

    async def _start_finish_wiring(self, connected_count: int) -> None:
        """Post-connect wiring: services, boot notifications, startup restore, recovered watchers."""
        from gateway.run import _planned_restart_notification_pending, _restart_notification_pending
        await self._start_post_connect_services(connected_count)
        # Let fresh adapters settle before lifecycle sends (helps Discord thread deliveries).
        if connected_count > 0:
            await asyncio.sleep(1.0)
        # Before _send_restart_notification() unlinks the marker: did we boot from a chat /restart?
        # One-shot signal for _is_stale_restart_redelivery.
        if _restart_notification_pending():
            self._booted_from_restart = True
        # Boot-path adapter.send() calls must not pin the inbound restore gate (a Telegram flood-
        # control sleep here once froze every platform).
        # Restart notification, home-channel startup notice, and obligation redelivery all call
        # adapter.send(). Bound them the same way _finish_startup_restore bounds resume turns. See #91969.
        await self._await_startup_boot_sends(
            planned_restart_notification_pending=_planned_restart_notification_pending(),
        )
        # Auto-resume restart-interrupted sessions (ledger-answered ones were cleared above); a failed
        # auto-resume stays visible on the next user message.
        self._schedule_resume_pending_sessions()
        await self._finish_startup_restore()
        # Surface state.db init failures to messaging platforms before the user loses data.
        # See #88235.
        await self._send_session_db_warning_notifications()
        # Resume recovered process watchers. Detach the batch atomically (fresh list, not clear(): a
        # concurrent append during the yield must not be lost); yield every 100 to keep the loop live.
        with _log_suppressed(logging.ERROR, "Recovered watcher setup error: %s"):
            from tools.process_registry import process_registry
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            for i, watcher in enumerate(watchers):
                self._spawn_supervised(
                    lambda w=watcher: self._run_process_watcher(w),
                    f"process_watcher:{watcher.get('session_id')}", restart=False,
                )
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
                if i % 100 == 99:
                    await asyncio.sleep(0)

    # Long-lived supervised watchers spawned at the end of start(), in order; supervised name = method
    # name minus the leading underscore.
    _PRE_RECONNECT_WATCHERS = (
        "_session_expiry_watcher", "_model_catalog_refresh_watcher", "_session_stall_watcher",
        "_kanban_notifier_watcher", "_kanban_dispatcher_watcher",
    )
    _POST_RECONNECT_WATCHERS = ("_handoff_watcher", "_async_delegation_watcher", "_loop_wakeup_watcher")

    def _start_spawn_background_watchers(self) -> None:
        """Spawn the long-lived supervised background watchers."""
        for method in self._PRE_RECONNECT_WATCHERS:
            self._spawn_supervised(getattr(self, method), method[1:])
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms), ", ".join(p.value for p in self._failed_platforms),
            )
        # Supervised: an escaping exception is restarted with backoff instead of stranding queued
        # platforms (the ensure hook only runs on a NEW fatal). ``on_spawn`` keeps the handle current.
        self._spawn_reconnect_watcher()
        for method in self._POST_RECONNECT_WATCHERS:
            self._spawn_supervised(getattr(self, method), method[1:])
        # Scale-to-zero watcher ONLY when opted in, messaging is relay-only/absent, and a wakeUrl exists.
        try:
            if self._scale_to_zero_should_arm():
                logger.info(
                    "scale-to-zero: armed (idle timeout %.0fs) — watching for idle",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                self._spawn_supervised(self._scale_to_zero_watcher, "scale_to_zero_watcher")
            else:
                # Say WHY an OPTED-IN instance didn't arm (non-opted stays silent).
                self._log_scale_to_zero_not_armed_reason()
        except Exception:  # noqa: BLE001 - arming must never block startup
            logger.debug("scale-to-zero: arm check failed at startup", exc_info=True)
        # Drain-control watcher: reconciles new-turn acceptance with the dashboard's ``.drain_request.json``
        # marker (prior-instantiation markers are ignored via epoch).
        self._spawn_supervised(self._drain_control_watcher, "drain_control_watcher")

    async def start(self) -> bool:
        """Start the gateway and all configured platform adapters."""
        logger.info("Starting Hermes Gateway...")
        self._start_install_faulthandler()
        self._start_log_startup_environment()
        if await self._abort_startup_if_shutdown_requested():
            return True
        if self._start_check_access_policy():
            return True
        await self._start_recover_previous_run()
        # Serialize startup restore against inbound: adapters receive as soon as they connect, so inbound
        # queues until every synthetic resume turn has finished.
        self._startup_restore_in_progress = True
        self._startup_restore_queue = []
        self._startup_restore_tasks = []
        # Fresh boot: the gate opens while the turn machinery is still cold (skeleton prompts). Warm NOW
        # to overlap the connects; _finish_startup_restore awaits it (bounded).
        self._start_startup_warmup()
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        (
            _aborted, enabled_platform_count, _multiplex_skipped_platforms, _pending_connects
        ) = await self._start_prefilter_platforms()
        if _aborted:
            return True
        if await self._abort_startup_if_shutdown_requested():
            return True
        _raw = await self._start_connect_pending(_pending_connects)
        if _raw is None:
            return True
        connected_count = await self._start_aggregate_connect_results(
            _raw, startup_retryable_errors, startup_nonretryable_errors
        )
        if await self._abort_startup_if_shutdown_requested():
            return True
        _aborted, connected_count = await self._start_secondary_profiles(
            connected_count, _multiplex_skipped_platforms
        )
        if _aborted:
            return True
        if self._start_handle_no_connections(
            connected_count, enabled_platform_count, startup_retryable_errors, startup_nonretryable_errors
        ):
            return True
        if await self._abort_startup_if_shutdown_requested():
            return True
        self.delivery_router.adapters = self.adapters
        self._wire_teams_pipeline_runtime()
        self._running = True
        self._install_plugin_message_injector()
        self._update_runtime_status("running")
        await self._start_finish_wiring(connected_count)
        self._start_spawn_background_watchers()
        logger.info("Press Ctrl+C to stop")
        return True

    @dataclasses.dataclass
    class _HandoffDestination:
        """Resolved destination for one handoff row."""
        platform: Platform
        platform_name: str
        transport: Any
        home: Any
        home_chat_id: str
        effective_thread_id: Optional[str]
        source: SessionSource
        handoff_config: Any

    def _handoff_resolve_scope(self, profile_name: Optional[str]):
        """Return (config, adapters) for the profile that queued the handoff. For a secondary
        profile the watcher already entered _profile_runtime_scope, so a fresh load resolves THAT
        profile's config; fail closed — self.config would deliver to the WRONG chat."""
        from gateway.run import load_gateway_config
        if not profile_name or profile_name == "default":
            return self.config, self.adapters
        secondary = (self._profile_adapters or {}).get(profile_name)
        if not secondary:
            raise RuntimeError(f"profile '{profile_name}' has no live adapters in this gateway")
        try:
            return load_gateway_config(), secondary
        except Exception as exc:
            logger.error(
                "Handoff: could not load config for profile %s; failing the handoff instead of "
                "delivering via the primary's config", profile_name, exc_info=True,
            )
            raise RuntimeError(f"could not load config for profile '{profile_name}': {exc}") from exc

    async def _handoff_resolve_destination(
        self, row: Dict[str, Any], profile_name: Optional[str]
    ) -> "GatewayStartupMixin._HandoffDestination":
        """Resolve platform, transport, home channel, thread and destination source for a row."""
        from gateway.delivery import resolve_delivery_transport
        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")
        handoff_config, handoff_adapters = self._handoff_resolve_scope(profile_name)
        # Alias-aware transport: a relay-fronted gateway registers ONE Platform.RELAY adapter fronting
        # N logical platforms, so a literal adapters.get() would miss a deliverable one.
        transport = resolve_delivery_transport(platform, handoff_config, handoff_adapters)
        if not transport:
            raise RuntimeError(f"platform '{platform_name}' is not active in this gateway")
        home = handoff_config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; run /sethome on the desired chat first"
            )
        home_chat_id = str(home.chat_id)
        # Fresh thread for the handoff's own scrollback; None when unsupported or creation failed.
        cli_title = row.get("title") or cli_session_id[:8]
        try:
            new_thread_id = await transport.adapter.create_handoff_thread(
                home_chat_id, f"Hermes — {cli_title}",
            )
        except Exception as exc:
            logger.debug("Handoff: create_handoff_thread raised on %s: %s", platform_name, exc, exc_info=True)
            new_thread_id = None
        effective_thread_id = new_thread_id or (str(home.thread_id) if home.thread_id else None)
        # Telegram private-chat DM topics use the DM-topic source shape (user_id == chat_id) so the
        # synthetic turn binds the same key later inbound turns arrive on (`dm`, not `thread`).
        is_telegram_private_chat = (
            platform == Platform.TELEGRAM and looks_like_telegram_private_chat_id(home_chat_id)
        )
        is_thread = bool(new_thread_id) and not is_telegram_private_chat
        # Discord builds in-thread messages with ``chat_id == thread id``: key on the thread's OWN id.
        dest_source = SessionSource(
            platform=platform,
            chat_id=str(effective_thread_id) if (
                is_thread and platform == Platform.DISCORD and effective_thread_id
            ) else home_chat_id,
            chat_name=home.name,
            chat_type="thread" if is_thread else "dm",
            user_id=home_chat_id if is_telegram_private_chat else "system:handoff",
            user_name="Handoff", thread_id=effective_thread_id, profile=profile_name,
        )
        return self._HandoffDestination(
            platform=platform, platform_name=platform_name, transport=transport, home=home,
            home_chat_id=home_chat_id, effective_thread_id=effective_thread_id, source=dest_source,
            handoff_config=handoff_config,
        )

    def _handoff_session_key(self, dest, profile_name: Optional[str]) -> str:
        """Destination session_key by the adapters' own rules. Thread keys omit user_id so the next
        message shares it. Namespaced to the queuing profile (else a multiplexed gateway builds
        ``agent:main:...`` while the profile's adapter routes on ``agent:<profile>:...``); the store
        resolver is only the root fallback. The isinstance check is load-bearing: a Mock store returns
        a truthy MagicMock."""
        platform_cfg = dest.handoff_config.platforms.get(dest.platform)
        extra = platform_cfg.extra if platform_cfg else {}
        handoff_profile = profile_name if (profile_name and profile_name != "default") else None
        if handoff_profile is None:
            try:
                store = getattr(self.async_session_store, "_store", self.async_session_store)
                resolver = getattr(store, "_resolve_profile_for_key", None)
                # Resolve the bound text channel's channel_prompt so voice input gets the same per-channel
                # context as typed messages (#50149).
                if callable(resolver):
                    resolved = resolver(dest.source)
                    if isinstance(resolved, str) and resolved.strip():
                        handoff_profile = resolved
            except Exception:
                logger.debug("Handoff: could not resolve profile namespace", exc_info=True)
        return build_session_key(
            dest.source, group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False), profile=handoff_profile,
        )

    async def _process_handoff(self, row: Dict[str, Any], profile_name: Optional[str] = None) -> None:
        """Execute one handoff row; raises on failure (caller marks failed). ``profile_name`` (None =
        root) is the profile whose store queued it — load-bearing under multiplex: secondaries live in
        ``_profile_adapters`` and the key must be namespaced ``agent:<profile>:...`` or nobody reads it."""
        cli_session_id = row["id"]
        dest = await self._handoff_resolve_destination(row, profile_name)
        session_key = self._handoff_session_key(dest, profile_name)
        # Ensure a session_store entry exists for this key; switch_session then re-points it.
        await self.async_session_store.get_or_create_session(dest.source)
        # switch_session ends the prior session and reopens the CLI session under the new key.
        switched = await self.async_session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(f"could not switch session key {session_key} → {cli_session_id}")
        # Evict the cached AIAgent (rebuild against the CLI session_id, like /resume) and clear stale
        # running-agent state so the synthetic turn isn't queued behind it.
        self._evict_cached_agent(session_key)
        self._release_running_agent_state(session_key)
        cli_title = row.get("title") or cli_session_id[:8]
        synthetic_event = MessageEvent(
            text=(
                f"[Session was just handed off from CLI (\"{cli_title}\") to this "
                f"channel. The full prior conversation history is loaded above. "
                f"Briefly confirm you're working here and summarize what we were "
                f"working on, so the user can continue from this device.]"
            ),
            source=dest.source,
            internal=True,
        )
        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, dest.platform_name, dest.home.chat_id, dest.effective_thread_id, session_key,
        )
        # Inline _handle_message keeps success/failure observable (handle_message would detach it).
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have delivered inline; the agent ran without raising — success.
            return
        # Reply into the new thread (else the home channel) via the resolved transport, so a relay-fronted
        # logical platform is stamped on the outbound frame.
        send_metadata = {"thread_id": dest.effective_thread_id} if dest.effective_thread_id else None
        try:
            result = await dest.transport.send(
                dest.platform, str(dest.home.chat_id), response_text, send_metadata,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc
        if not getattr(result, "success", True):
            raise RuntimeError(f"adapter.send failed: {_send_error(result)}")
