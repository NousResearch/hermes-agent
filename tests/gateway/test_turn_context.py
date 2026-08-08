"""Unit tests for the TurnContext/TurnRunner seam extracted from
``GatewayRunner._run_agent_inner`` (gateway/turn_context.py + gateway/run.py).

The extraction contract: the closure bodies moved onto ``TurnRunner`` methods
byte-identically (modulo local -> ctx.field rewrites), with every closed-over
local carried as a ``TurnContext`` field. These tests pin the seam's wiring —
shared mutable containers, no-queue early returns — not the progress behavior
itself (that's covered by test_run_progress_topics.py et al.).
"""

import asyncio
import queue as queue_mod

import pytest

from agent.turn_context import bind_turn_source_identity, current_turn_source_identity
from gateway.platforms.base import MessageEvent, trusted_source_message_id
from gateway.turn_context import TurnContext


def _make_runner(ctx):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

    return TurnRunner(_StubGatewayRunner(), ctx)


class TestTurnContext:
    def test_defaults_are_independent_containers(self):
        a, b = TurnContext(), TurnContext()
        a.last_progress_msg[0] = "x"
        a.repeat_count[0] = 3
        a._cleanup_msg_ids.append("1")
        assert b.last_progress_msg == [None]
        assert b.repeat_count == [0]
        assert b._cleanup_msg_ids == []

    def test_source_identity_is_separate_from_reply_routing_anchor(self):
        ctx = TurnContext(
            event_message_id="reply-or-thread-anchor",
            source_message_id="adapter-event-id",
        )

        assert ctx.event_message_id == "reply-or-thread-anchor"
        assert ctx.source_message_id == "adapter-event-id"

    def test_internal_event_never_has_trusted_source_identity(self):
        event = MessageEvent(
            text="[SYSTEM: background process completed]",
            message_id="stale-reply-anchor",
            internal=True,
        )

        assert trusted_source_message_id(event) is None

    def test_turn_source_identity_is_normalized_nested_and_restored(self):
        assert current_turn_source_identity() == ("", False)

        with bind_turn_source_identity("  wamid-current  "):
            assert current_turn_source_identity() == ("wamid-current", True)
            with bind_turn_source_identity(None):
                assert current_turn_source_identity() == ("", False)
            assert current_turn_source_identity() == ("wamid-current", True)

        assert current_turn_source_identity() == ("", False)

    def test_shared_containers_visible_to_outer_scope(self):
        # The outer body and the runner share the SAME list objects, so
        # mutation through the ctx is visible to locals captured elsewhere.
        last_progress_msg = [None]
        ctx = TurnContext(last_progress_msg=last_progress_msg)
        ctx.last_progress_msg[0] = "🔍 web_search"
        assert last_progress_msg[0] == "🔍 web_search"


class TestTurnRunner:
    def test_methods_exist_and_bind(self):
        from gateway.run import TurnRunner

        ctx = TurnContext()
        runner = _make_runner(ctx)
        assert callable(runner.progress_callback)
        assert asyncio.iscoroutinefunction(TurnRunner.send_progress_messages)
        assert runner._ctx is ctx

    def test_send_progress_messages_no_queue_returns(self):
        ctx = TurnContext(progress_queue=None)
        runner = _make_runner(ctx)
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_send_progress_messages_no_adapter_returns(self):
        ctx = TurnContext(progress_queue=queue_mod.Queue())
        runner = _make_runner(ctx)  # stub adapter resolver returns None
        assert asyncio.run(runner.send_progress_messages()) is None
