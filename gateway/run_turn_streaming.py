"""Settle per-turn background tasks without swallowing parent cancellation."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from gateway.turn_context import TurnContext

logger = logging.getLogger("gateway.run")


async def cleanup_turn_tasks(
    self, turn_ctx: TurnContext, *, progress_task: Any, log_task: Any, interrupt_monitor: asyncio.Task,
    _notify_task: asyncio.Task, tracking_task: asyncio.Task, stream_task: Any,
) -> None:
    """``finally`` half of a turn: cancel background tasks, flush stream, release the session slot."""
    stream_consumer_holder, session_key = turn_ctx.stream_consumer_holder, turn_ctx.session_key
    for task in (progress_task, log_task, interrupt_monitor, _notify_task):
        if task:
            task.cancel()

    try:
        try:
            if stream_task:
                # No stream consumer was created: nothing to flush, cancel instead of waiting out 5s.
                if not (stream_consumer_holder and stream_consumer_holder[0] is not None):
                    stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stream_task
                else:
                    await self._await_stream_task(stream_task)
        finally:
            # Abort + bounded wait for streaming TTS: covers paths where normal finalisation was skipped.
            _stts_finally = turn_ctx.streaming_tts_consumer_holder[0]
            # See #60671. Text-flush cancellation must not skip TTS cleanup.
            if _stts_finally is not None and not _stts_finally.done:
                _stts_finally.abort("cleanup")
                with suppress(Exception):
                    await _stts_finally.wait_complete(timeout=2.0)
    finally:
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
        # Child-task cancellation is expected above, but must not consume a
        # new cancellation of this turn while one of those children settles.
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError

