"""Long-running turn heartbeat and typing indicator delivery."""
from __future__ import annotations
import asyncio
import logging
import time
from contextlib import suppress
from typing import Optional, TYPE_CHECKING
from agent.i18n import t
from agent.session_activity import format_iteration_progress
from gateway.turn_context import TurnContext

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


class GatewayTurnNotificationsMixin:
    async def _hmwa_stop_typing_for_turn(self, event, source):
        """Stop the typing indicator (never raises). Slack AI status is scoped to a thread/
        workspace, so preserve the routing metadata used by the response delivery path."""
        with suppress(Exception):
            _typing_adapter = self._delivery_adapter_for(source)
            _kind = type(_typing_adapter)
            if _typing_adapter and callable(getattr(_kind, "_stop_typing_with_metadata", None)):
                await _typing_adapter._stop_typing_with_metadata(source.chat_id, self._event_thread_metadata(event, source))
            elif _typing_adapter and callable(getattr(_kind, "stop_typing", None)):
                await _typing_adapter.stop_typing(source.chat_id)




    async def _run_agent_notify_long_running(
        self, disp: GatewayRunner._RunAgentDisplay, turn_ctx: TurnContext, _executor_task_holder: list,
    ) -> None:
        """Periodic "still working" heartbeat, edited in place where supported. Stops once this run
        no longer owns the session slot or the executor finished. ``_executor_task_holder[0]`` is
        bound just after this task is scheduled (reads as None until then).

        Interval: agent.gateway_notify_interval / HERMES_AGENT_NOTIFY_INTERVAL (default 180s; 0 or
        long_running_notifications=off disables)."""
        from gateway.run import _float_env, _interim_metadata, _non_conversational_metadata
        _notify_start = time.time()
        _NOTIFY_INTERVAL = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _long_running_mode = disp._display_surface_mode("long_running_notifications", default=True, allow_generic=True)
        if _NOTIFY_INTERVAL <= 0 or _long_running_mode == "off":
            return
        source, session_key, agent_holder = turn_ctx.source, turn_ctx.session_key, turn_ctx.agent_holder
        _status_thread_metadata = turn_ctx._status_thread_metadata
        _notify_adapter = self._delivery_adapter_for(source)
        if not _notify_adapter:
            return
        _heartbeat_msg_id: Optional[str] = None
        while True:
            await asyncio.sleep(_NOTIFY_INTERVAL)
            if not self._should_emit_long_running_notification(
                session_key, agent_holder[0], _executor_task_holder[0]
            ):
                break
            _elapsed_mins = int((time.time() - _notify_start) // 60)
            # Terse heartbeat by default; the iteration counter is gated on busy_ack_detail.
            _status_detail = ""
            _want_iteration_detail = bool(
                disp.resolve_display_setting(disp.user_config, disp.platform_key, "busy_ack_detail", True)
            )
            _a = self._agent_activity_summary(agent_holder[0])
            with suppress(Exception):
                if _a:
                    _parts = []
                    if _want_iteration_detail:
                        _parts.append(format_iteration_progress(_a["api_call_count"], _a["max_iterations"]))
                    _action = _a.get("current_tool") or _a.get("last_activity_desc")
                    if _action:
                        _parts.append(str(_action))
                    if _parts:
                        _status_detail = " — " + ", ".join(_parts)
            _heartbeat_text = (
                disp._generic_status_phrase("status")
                if _long_running_mode == "generic"
                else t("gateway.progress.working_heartbeat", minutes=_elapsed_mins, detail=_status_detail)
            )
            try:
                _notify_res = None
                if _heartbeat_msg_id:
                    try:
                        _notify_res = await _notify_adapter.edit_message(source.chat_id, _heartbeat_msg_id, _heartbeat_text)
                    except Exception as _ee:
                        logger.debug("Heartbeat edit failed: %s", _ee)
                        _notify_res = None
                if not (_notify_res and getattr(_notify_res, "success", False)):
                    # The edit above awaited; a drain/restart notice may have gone out meanwhile, and
                    # a fresh "Working" bubble after it reads as a contradiction (#10990).
                    if not self._should_emit_long_running_notification(
                        session_key, agent_holder[0], _executor_task_holder[0]
                    ):
                        break
                    _notify_res = await _notify_adapter.send(
                        source.chat_id, _heartbeat_text,
                        metadata=_interim_metadata(_non_conversational_metadata(_status_thread_metadata, platform=source.platform)),
                    )
                    if getattr(_notify_res, "success", False) and getattr(_notify_res, "message_id", None):
                        _heartbeat_msg_id = str(_notify_res.message_id)
                        if turn_ctx._cleanup_progress:
                            turn_ctx._cleanup_msg_ids.append(_heartbeat_msg_id)
            except Exception as _ne:
                logger.debug("Long-running notification error: %s", _ne)
