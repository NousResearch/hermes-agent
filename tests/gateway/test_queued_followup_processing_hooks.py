"""Regression coverage for issue #103429 (part 1).

``busy_input_mode: queue`` follow-ups are drained IN-BAND by
``GatewayRunner._run_agent_queued_followup`` (via ``_run_agent_drain_pending``),
which pops the pending event straight out of ``adapter._pending_messages``
before ``BasePlatformAdapter._process_message_background``'s own end-of-turn
drain ever sees it. That out-of-band drain is what normally fires
``on_processing_start`` / ``on_processing_complete`` (the hooks that drive a
Telegram reaction ack) — so a queued follow-up consumed in-band got neither
hook call, and so no reaction, ever.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType, Platform, ProcessingOutcome
from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


class _StubAdapter:
    def __init__(self):
        self.hook_calls = []
        self._active_sessions = {}
        self._expected_cancelled_tasks = set()

    async def _run_processing_hook(self, hook_name, *args, **kwargs):
        self.hook_calls.append((hook_name, args, kwargs))


class _FakeGatewayRunner(GatewayRunner):
    """Overrides every collaborator of ``_run_agent_queued_followup`` except the
    method under test, so the real recursion/hook-firing logic runs unmodified."""

    def __init__(self, adapter, followup_result=None, followup_exception=None):
        self._adapter = adapter
        self._followup_result = followup_result
        self._followup_exception = followup_exception
        self.run_agent_calls = []

    def _adapter_for_source(self, source):
        return self._adapter

    def _session_key_for_source(self, source):
        return "telegram:dm:1"

    async def _prepare_profile_scoped_inbound_message_text(self, *, event, source, history, session_key=None):
        return event.text

    async def _run_agent_deliver_first_response(self, *args, **kwargs):
        return None

    async def _refresh_agent_cache_message_count(self, *args, **kwargs):
        return None

    async def _run_agent(self, **kwargs):
        self.run_agent_calls.append(kwargs)
        if self._followup_exception is not None:
            raise self._followup_exception
        return self._followup_result


def _plain_dm_pending_event() -> MessageEvent:
    # A plain Telegram private chat: no forum/DM topic, so thread_id is None.
    source = MagicMock(platform=Platform.TELEGRAM, thread_id=None, chat_type="dm", profile=None)
    return MessageEvent(
        text="second message", message_type=MessageType.TEXT, source=source, message_id="msg2",
    )


def _make_turn_ctx(pending_event: MessageEvent) -> TurnContext:
    return TurnContext(
        source=MagicMock(platform=Platform.TELEGRAM, thread_id=None, chat_type="dm", profile=None),
        session_id="sess-1", session_key="telegram:dm:1", history=[],
        _interrupt_depth=0, _status_thread_metadata=None,
    )


class TestQueuedFollowupFiresProcessingHooks:
    def test_on_processing_start_and_complete_fire_for_pending_event(self):
        pending_event = _plain_dm_pending_event()
        adapter = _StubAdapter()
        turn_ctx = _make_turn_ctx(pending_event)
        runner = _FakeGatewayRunner(adapter, followup_result={"final_response": "the answer"})

        result = asyncio.run(runner._run_agent_queued_followup(
            turn_ctx, adapter, pending="second message", pending_event=pending_event,
            response={}, result={"messages": []}, stream_task=None,
        ))

        assert result == {"final_response": "the answer"}
        # The queued message must get its OWN on_processing_start/complete calls —
        # not zero, and not the original message's.
        hook_names = [name for name, _args, _kwargs in adapter.hook_calls]
        assert hook_names == ["on_processing_start", "on_processing_complete"]
        start_call = adapter.hook_calls[0]
        assert start_call[1] == (pending_event,)
        complete_call = adapter.hook_calls[1]
        assert complete_call[1] == (pending_event, ProcessingOutcome.SUCCESS)

    def test_failed_followup_reports_failure_outcome(self):
        pending_event = _plain_dm_pending_event()
        adapter = _StubAdapter()
        turn_ctx = _make_turn_ctx(pending_event)
        runner = _FakeGatewayRunner(adapter, followup_result={"final_response": "", "failed": True})

        asyncio.run(runner._run_agent_queued_followup(
            turn_ctx, adapter, pending="second message", pending_event=pending_event,
            response={}, result={"messages": []}, stream_task=None,
        ))

        complete_call = adapter.hook_calls[1]
        assert complete_call[1] == (pending_event, ProcessingOutcome.FAILURE)

    def test_unhandled_exception_still_fires_processing_complete(self):
        """A raised exception from the recursive turn must not leave on_processing_start unpaired
        (else the platform reaction it drives, e.g. Telegram's 👀, is stuck forever)."""
        pending_event = _plain_dm_pending_event()
        adapter = _StubAdapter()
        turn_ctx = _make_turn_ctx(pending_event)
        runner = _FakeGatewayRunner(adapter, followup_exception=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(runner._run_agent_queued_followup(
                turn_ctx, adapter, pending="second message", pending_event=pending_event,
                response={}, result={"messages": []}, stream_task=None,
            ))

        hook_names = [name for name, _args, _kwargs in adapter.hook_calls]
        assert hook_names == ["on_processing_start", "on_processing_complete"]
        complete_call = adapter.hook_calls[1]
        assert complete_call[1] == (pending_event, ProcessingOutcome.FAILURE)

    def test_cancelled_error_still_fires_processing_complete(self):
        """A CancelledError (e.g. gateway shutdown/restart) from the recursive turn must still pair
        with on_processing_complete, classified as CANCELLED when the cancellation was expected."""
        pending_event = _plain_dm_pending_event()
        adapter = _StubAdapter()
        turn_ctx = _make_turn_ctx(pending_event)
        runner = _FakeGatewayRunner(adapter, followup_exception=asyncio.CancelledError())

        async def _run():
            task = asyncio.current_task()
            adapter._expected_cancelled_tasks.add(task)
            with pytest.raises(asyncio.CancelledError):
                await runner._run_agent_queued_followup(
                    turn_ctx, adapter, pending="second message", pending_event=pending_event,
                    response={}, result={"messages": []}, stream_task=None,
                )

        asyncio.run(_run())

        hook_names = [name for name, _args, _kwargs in adapter.hook_calls]
        assert hook_names == ["on_processing_start", "on_processing_complete"]
        complete_call = adapter.hook_calls[1]
        assert complete_call[1] == (pending_event, ProcessingOutcome.CANCELLED)
