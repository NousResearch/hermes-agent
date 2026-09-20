"""Requester and planned-home notifications after a gateway restart."""

import asyncio
import json
import logging
import math
from typing import TYPE_CHECKING, Optional, cast

from agent.async_utils import consume_detached_task_result
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.restart import restart_notification_marker_lock
from gateway.run_shutdown import _send_error, _send_failed

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")
_RESTART_NOTICE_TIMEOUT = 180.0
_RESTART_NOTICE_MAX_DELAY = 15.0


def _known_unsent(result) -> bool:
    """Only an explicit adapter refusal permits replay; never infer safety from error text."""
    get = result.get if isinstance(result, dict) else lambda key, default=None: getattr(result, key, default)
    return (get("success") is False and get("known_unsent") is True and get("retryable") is True
            and not get("message_id") and not get("continuation_message_ids"))


class GatewayRestartNotificationsMixin:
    _restart_notice_lock: Optional[asyncio.Lock] = None

    def _capture_planned_restart_notification(self) -> None:
        from gateway.run import _planned_restart_notification_path
        self._planned_restart_notification_payload = None
        self._planned_restart_notice_progress = None
        try:
            self._planned_restart_notification_payload = _planned_restart_notification_path().read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            logger.warning("Could not read planned restart notification; preserving it", exc_info=True)

    async def _replay_pending_planned_restart_notification(self) -> None:
        """Replay homes independently while a reconnected adapter proves readiness (#66589).

        Boot and reconnect already run this behind the bounded startup gate/in a background task.
        A missing transport is left for reconnect; a present but unready transport gets a bounded
        retry window. Only an explicit whole-call known-unsent result permits a second send.
        Progress survives cancellation/reboot, and old workers never adopt a newer restart marker.
        """
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _planned_restart_notification_path
        from gateway.run_notifications import _notice_target_key
        from utils import atomic_json_write

        runner = cast("GatewayRunner", self)
        if not hasattr(self, "_planned_restart_notification_payload"):
            self._capture_planned_restart_notification()
        owner = asyncio.current_task()
        path = _planned_restart_notification_path()

        def owns_marker():
            try:
                return path.read_text(encoding="utf-8") == self._planned_restart_notification_payload
            except FileNotFoundError:
                return False

        def active():
            return (getattr(runner, "_running", True) and not getattr(runner, "_restart_requested", False)
                    and owns_marker())

        payload = self._planned_restart_notification_payload
        if payload is None or not active():
            return
        try:
            data = json.loads(payload)
            if self._planned_restart_notice_progress is None:
                # Shared by concurrent boot/reconnect passes. All progress/claim mutations
                # are synchronous on the gateway loop; no lock spans transport waits.
                self._planned_restart_notice_progress = (
                    {tuple(target) for target in data.get("delivered_targets", [])},
                    {tuple(target) for target in data.get("attempted_targets", [])},
                    set(),
                )
            delivered, attempted, inflight = self._planned_restart_notice_progress
            homes = {
                _notice_target_key(platform.value, cfg.home_channel.chat_id, cfg.home_channel.thread_id):
                (platform, cfg.home_channel)
                for platform, cfg in runner.config.platforms.items()
                if cfg.home_channel and cfg.home_channel.chat_id and cfg.gateway_restart_notification
            }

            def persist():
                # The planned marker's publisher runs synchronously on this same event loop.
                # No await between compare/write: a newer shutdown generation cannot be clobbered.
                if owns_marker():
                    data["delivered_targets"] = [list(target) for target in delivered]
                    data["attempted_targets"] = [list(target) for target in attempted]
                    atomic_json_write(path, data, indent=0)
                    self._planned_restart_notification_payload = path.read_text(encoding="utf-8")

            loop = asyncio.get_running_loop()
            deadline = loop.time() + _RESTART_NOTICE_TIMEOUT
            message = "♻️ Gateway online — Hermes is back and ready."
            free_tier_line = runner._free_tier_startup_line()
            if free_tier_line:
                message = f"{message}\n{free_tier_line}"

            async def notify(target, platform, home):
                if target in inflight:
                    return
                inflight.add(target)
                delay, send_task = 1.0, None
                try:
                    while active() and target not in delivered | attempted:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            logger.warning("Planned-restart notification retry budget exhausted for %s; marker remains pending", target)
                            return
                        transport = resolve_delivery_transport(platform, runner.config, runner.adapters)
                        if transport is None:
                            return  # _install_reconnected_adapter will replay once it exists.
                        retry_delay = delay
                        if (getattr(transport.adapter, "is_connected", True) is not False
                                and getattr(transport.adapter, "send_path_degraded", False) is not True):
                            async def dispatch():
                                # Owner cancellation can precede gather cancelling this child.
                                if ((owner is not None and owner.cancelling())
                                        or not active() or loop.time() >= deadline):
                                    return SendResult(success=False, retryable=True, known_unsent=True)
                                attempted.add(target)
                                persist()  # Cancellation/crash after dispatch is ambiguous, never replay it.
                                return await runner._dispatch_home_channel_message(platform, home, transport, message)

                            send_task = asyncio.create_task(dispatch())
                            done, _ = await asyncio.wait({send_task}, timeout=remaining)
                            if not done:
                                logger.warning("Planned-restart notification send timed out for %s; not replaying", target)
                                return
                            result = send_task.result()
                            if not _known_unsent(result):
                                failed = result.get("success") is False if isinstance(result, dict) else _send_failed(result)
                                if not failed:
                                    attempted.discard(target)
                                    delivered.add(target)
                                else:
                                    logger.warning("Planned-restart notification failed for %s; not replaying an ambiguous send", target)
                                return
                            attempted.discard(target)
                            persist()
                            raw_delay = result.get("retry_after") if isinstance(result, dict) else getattr(result, "retry_after", None)
                            if raw_delay is not None:
                                retry_after = float(raw_delay)
                                if math.isfinite(retry_after) and retry_after > 0:
                                    retry_delay = retry_after
                        await asyncio.sleep(min(retry_delay, max(0.0, deadline - loop.time())))
                        delay = min(delay * 2, _RESTART_NOTICE_MAX_DELAY)
                except Exception:
                    logger.warning("Planned-restart notification failed for %s", target, exc_info=True)
                finally:
                    if send_task is not None:
                        if send_task.done() and not send_task.cancelled():
                            try:
                                if _known_unsent(send_task.result()):
                                    attempted.discard(target)
                            except Exception:
                                pass
                        elif not send_task.done():
                            send_task.cancel()
                            send_task.add_done_callback(consume_detached_task_result)
                    inflight.discard(target)
                    persist()

            results = await asyncio.gather(*(notify(target, *home) for target, home in homes.items()
                                             if target not in delivered | attempted), return_exceptions=True)
            if any(isinstance(result, BaseException) for result in results):
                logger.warning("Could not persist planned-restart notification progress; marker remains pending")
                return
            if owns_marker() and not inflight and homes.keys() <= delivered | attempted:
                path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Planned-restart notification remains pending", exc_info=True)

    def _capture_restart_notification(self) -> None:
        """Pin the boot generation BEFORE connecting adapters (which can accept /restart)."""
        from gateway.run import _hermes_home
        self._restart_notification_payload = None
        try:
            self._restart_notification_payload = (_hermes_home / ".restart_notify.json").read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            logger.warning("Could not read restart notification; preserving it for a later boot", exc_info=True)
        self._booted_from_restart = self._restart_notification_payload is not None

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Retry only known-unsent notices within a bounded window; preserve those still owed.

        An ambiguous send is consumed to avoid duplicates. Cleanup compares the exact boot payload
        under the publisher's file lock, so an old worker cannot remove a newer /restart marker.
        """
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _hermes_home, _non_conversational_metadata

        runner = cast("GatewayRunner", self)

        if self._restart_notice_lock is None:
            self._restart_notice_lock = asyncio.Lock()
        async with self._restart_notice_lock:
            # Also supports callers outside start(); never recapture a newer generation on replay.
            if not hasattr(self, "_restart_notification_payload"):
                self._capture_restart_notification()
            payload = self._restart_notification_payload
            if payload is None:
                return None
            path = _hermes_home / ".restart_notify.json"

            def owns_marker():
                try:
                    return path.read_text(encoding="utf-8") == payload
                except FileNotFoundError:
                    return False

            consume = False
            send_task = None
            try:
                data = json.loads(payload)
                platform = Platform(data["platform"])
                chat_id, thread_id = data["chat_id"], data.get("thread_id")
                if not chat_id:
                    return None
                cfg = runner.config.platforms.get(platform)
                if cfg is not None and not cfg.gateway_restart_notification:
                    consume = True
                    return None
                loop = asyncio.get_running_loop()
                deadline, delay = loop.time() + _RESTART_NOTICE_TIMEOUT, 1.0
                while runner._running and not runner._restart_requested and owns_marker():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        logger.warning("Restart notification retry budget exhausted; marker remains pending")
                        return None
                    # Keep current main's requester-profile/relay routing, not the default bot map.
                    transport = resolve_delivery_transport(
                        platform, runner.config, runner._adapters_for_profile(runner._marker_profile(data)))
                    retry_delay = delay
                    if (transport is not None
                            and getattr(transport.adapter, "is_connected", True) is not False
                            and getattr(transport.adapter, "send_path_degraded", False) is not True):
                        metadata = runner._pending_marker_metadata(platform, chat_id, data, transport.adapter)
                        if data.get("delivered_via_upstream_relay") is True:
                            metadata = dict(metadata or {})
                            for field in ("user_id", "scope_id"):
                                if data.get(field):
                                    metadata[field] = str(data[field])

                        async def dispatch():
                            nonlocal consume
                            # create_task yields: a /restart may replace the marker before we run.
                            if (not runner._running or runner._restart_requested
                                    or loop.time() >= deadline or not owns_marker()):
                                return SendResult(success=False, retryable=True, known_unsent=True)
                            consume = True  # From this point a cancellation/exception is ambiguous.
                            return await transport.send(
                                platform, str(chat_id), "♻ Gateway restarted successfully. Your session continues.",
                                metadata=_non_conversational_metadata(metadata, platform=platform))

                        send_task = asyncio.create_task(dispatch())
                        done, _ = await asyncio.wait({send_task}, timeout=remaining)
                        if not done:
                            logger.warning("Restart notification send timed out; not replaying an ambiguous send")
                            return None
                        result = send_task.result()
                        failed = result.get("success") is False if isinstance(result, dict) else _send_failed(result)
                        if not failed:
                            logger.info("Sent restart notification to %s:%s", platform.value, chat_id)
                            return platform.value, str(chat_id), str(thread_id) if thread_id else None
                        if not _known_unsent(result):
                            logger.warning("Restart notification to %s:%s was not delivered: %s",
                                           platform.value, chat_id, result.get("error") if isinstance(result, dict) else _send_error(result))
                            return None
                        consume = False
                        raw_delay = result.get("retry_after") if isinstance(result, dict) else getattr(result, "retry_after", None)
                        if raw_delay is not None:
                            retry_after = float(raw_delay)
                            if math.isfinite(retry_after) and retry_after > 0:
                                retry_delay = retry_after
                    remaining = deadline - loop.time()
                    if remaining > 0:
                        await asyncio.sleep(min(max(retry_delay, 0.0), remaining))
                    delay = min(delay * 2, _RESTART_NOTICE_MAX_DELAY)
            except Exception:
                logger.warning("Restart notification failed", exc_info=True)
            finally:
                if send_task is not None:
                    # Cancellation can race a completed, explicitly known-unsent refusal.
                    if send_task.done() and not send_task.cancelled():
                        try:
                            if _known_unsent(send_task.result()):
                                consume = False
                        except Exception:
                            pass
                    elif not send_task.done():
                        send_task.cancel()
                        send_task.add_done_callback(consume_detached_task_result)
                if consume:
                    def clear_owned_marker():
                        try:
                            with restart_notification_marker_lock(_hermes_home):
                                if owns_marker():
                                    path.unlink(missing_ok=True)
                        except Exception:
                            logger.warning("Could not clear owned restart notification", exc_info=True)
                    # A publisher may hold the process-shared lock: never block the event loop on it.
                    await asyncio.to_thread(clear_owned_marker)
            return None
