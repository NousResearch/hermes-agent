"""Per-turn wiring for GatewayRunner: progress threading, the sync→async callback bridges and the
end-of-turn task cleanup. Inherited by GatewayTurnMixin.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.turn_context import TurnContext

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run_turn_runner import TurnRunner

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayTurnWiringMixin:
    """Turn wiring and cleanup phases of GatewayTurnMixin."""

    async def _run_agent_cleanup_turn_tasks(
        self, turn_ctx: TurnContext, *, progress_task: Any, log_task: Any, interrupt_monitor: asyncio.Task,
        _notify_task: asyncio.Task, tracking_task: asyncio.Task, stream_task: Any,
    ) -> None:
        """``finally`` half of a turn: cancel background tasks, flush stream, release the session slot."""
        stream_consumer_holder, session_key = turn_ctx.stream_consumer_holder, turn_ctx.session_key
        for task in (progress_task, log_task, interrupt_monitor, _notify_task):
            if task:
                task.cancel()

        if stream_task:
            # No stream consumer was created: nothing to flush, cancel instead of waiting out 5s.
            if not (stream_consumer_holder and stream_consumer_holder[0] is not None):
                stream_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stream_task
            else:
                await self._await_stream_task(stream_task)

        # Abort + bounded wait for streaming TTS: covers paths where normal finalisation was skipped.
        _stts_finally = turn_ctx.streaming_tts_consumer_holder[0]
        # See #60671.
        if _stts_finally is not None and not _stts_finally.done:
            _stts_finally.abort("cleanup")
            with suppress(Exception):
                await _stts_finally.wait_complete(timeout=2.0)

        tracking_task.cancel()
        if session_key:
            # Release the slot only if this run's generation still owns it (/stop or /new may have
            # installed its own state).
            self._release_running_agent_state(session_key, run_generation=turn_ctx.run_generation)
        if self._draining:
            self._update_runtime_status("draining")

        for task in (progress_task, log_task, interrupt_monitor, tracking_task, _notify_task):
            if task:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # A background task that died of a real error must not abort the cleanup path.
                    logger.debug("background turn task failed during cleanup", exc_info=True)

        pane = getattr(turn_ctx, "matrix_commentary_pane", None)
        if pane is not None:
            await pane.close()

    def _run_agent_bind_turn_wiring(
        self, turn_ctx: TurnContext, turn_runner: TurnRunner, source: SessionSource,
        event_message_id: Optional[str], _native_slack_task_cards: bool,
    ) -> Optional[dict[str, Any]]:
        """Resolve progress threading, then publish progress metadata and the sync→async bridges onto
        ``turn_ctx`` (the one-slot holders shared with run_sync's executor thread are TurnContext
        defaults). Returns ``_status_thread_metadata``."""
        turn_ctx._progress_metadata, turn_ctx._progress_reply_to, _status_thread_metadata = (
            self._run_agent_progress_threading(source, event_message_id, _native_slack_task_cards)
        )
        adapter = self._delivery_adapter_for(source)
        from gateway.session_detached import detached_route_of
        if (
            turn_ctx.interim_assistant_messages_enabled
            and adapter is not None
            # A detached conversation answers app viewers through the local adapter.
            and detached_route_of(source) is None
            and (getattr(adapter, "name", "") == "matrix" or source.platform == Platform.MATRIX)
        ):
            from gateway.matrix_commentary_pane import MatrixCommentaryPane
            turn_ctx.matrix_commentary_pane = MatrixCommentaryPane(
                adapter=adapter,
                chat_id=source.chat_id,
                reply_to=turn_ctx._progress_reply_to,
                metadata=turn_ctx._progress_metadata,
            )
        # Bridges: sync step/event/status callbacks → async hooks.emit and adapter.send.
        turn_ctx._loop_for_step = asyncio.get_running_loop()
        turn_ctx._hooks_ref = self.hooks
        turn_ctx._step_callback_sync = turn_runner._step_callback_sync
        turn_ctx._event_callback_sync = turn_runner._event_callback_sync
        turn_ctx._status_callback_sync = turn_runner._status_callback_sync
        turn_ctx._status_adapter = self._delivery_adapter_for(source)
        turn_ctx._status_chat_id = source.chat_id
        turn_ctx._status_thread_metadata = _status_thread_metadata
        return _status_thread_metadata
