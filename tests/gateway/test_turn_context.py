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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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

    @pytest.mark.asyncio
    async def test_interactive_card_sender_binds_current_turn_destination(self):
        from gateway.config import Platform
        from gateway.run import TurnRunner
        from gateway.session import SessionSource

        manager = SimpleNamespace(deliver_card=AsyncMock(return_value="delivery-1"))
        adapter = SimpleNamespace()

        class _Runner:
            _interactive_action_manager = manager

            def _adapter_for_source(self, _source):
                return adapter

        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_current",
            user_id="ou_current",
            user_name="Alice",
            thread_id="om_root",
            message_id="om_source",
            profile="work",
        )
        ctx = TurnContext(
            source=source,
            event_message_id="om_trigger",
            _run_still_current=lambda: True,
            _status_thread_metadata={"thread_id": "om_root"},
            _interactive_card_sender_active=True,
        )
        ctx._loop_for_step = asyncio.get_running_loop()
        turn = TurnRunner(_Runner(), ctx)

        delivery = await asyncio.to_thread(
            turn._send_interactive_card_sync,
            plugin_id="proposal-plugin",
            envelope=SimpleNamespace(),
        )

        assert delivery == "delivery-1"
        kwargs = manager.deliver_card.await_args.kwargs
        assert kwargs["plugin_id"] == "proposal-plugin"
        assert kwargs["adapter"] is adapter
        assert kwargs["origin"].platform == "feishu"
        assert kwargs["origin"].profile_id == "work"
        assert kwargs["origin"].chat_id == "oc_current"
        assert kwargs["origin"].thread_id == "om_root"
        assert kwargs["origin"].initiator_id == "ou_current"
        assert kwargs["origin"].message_id == "om_trigger"

    def test_interactive_card_sender_rejects_a_copied_context_after_turn_end(self):
        from gateway.interactive_actions import InteractiveCardUnavailableError
        from gateway.run import TurnRunner

        ctx = TurnContext(
            _run_still_current=lambda: True,
            _interactive_card_sender_active=False,
        )
        turn = TurnRunner(object(), ctx)

        with pytest.raises(InteractiveCardUnavailableError, match="no longer active"):
            turn._send_interactive_card_sync(
                plugin_id="proposal-plugin",
                envelope=SimpleNamespace(),
            )

    @pytest.mark.asyncio
    async def test_interactive_card_sender_never_deadlocks_gateway_loop(self):
        from gateway.interactive_actions import InteractiveCardUnavailableError
        from gateway.run import TurnRunner

        adapter = SimpleNamespace()

        class _Runner:
            def _adapter_for_source(self, _source):
                return adapter

        ctx = TurnContext(
            source=SimpleNamespace(),
            _run_still_current=lambda: True,
            _interactive_card_sender_active=True,
            _loop_for_step=asyncio.get_running_loop(),
        )
        turn = TurnRunner(_Runner(), ctx)

        with pytest.raises(InteractiveCardUnavailableError, match="gateway loop"):
            turn._send_interactive_card_sync(
                plugin_id="proposal-plugin",
                envelope=SimpleNamespace(),
            )
