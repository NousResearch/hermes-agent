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
from unittest.mock import MagicMock

import pytest


from gateway.config import Platform
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


def _make_runner(ctx):
    from gateway.run_turn_runner import TurnRunner

    class _StubGatewayRunner:
        def _delivery_adapter_for(self, source):
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



class TestTurnRunner:

    def test_send_progress_messages_no_queue_returns(self):
        ctx = TurnContext(progress_queue=None)
        runner = _make_runner(ctx)
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_send_progress_messages_no_adapter_returns(self):
        ctx = TurnContext(progress_queue=queue_mod.Queue())
        runner = _make_runner(ctx)  # stub adapter resolver returns None
        assert asyncio.run(runner.send_progress_messages()) is None

    def test_normal_response_preserves_compression_exhausted(self):
        """A non-empty exhaustion response must still reach auto-reset consumers."""

        class _ExhaustedAgent:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                self.session_id = kwargs["session_id"]
                self.tools = []
                self.context_compressor = SimpleNamespace(
                    last_prompt_tokens=0,
                    context_length=200_000,
                )
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0

            def run_conversation(self, _message, **_kwargs):
                return {
                    "final_response": "Context length exceeded. Cannot compress further.",
                    "failed": True,
                    "compression_exhausted": True,
                    "messages": [],
                }

        gateway_runner = MagicMock()
        gateway_runner.config = SimpleNamespace(streaming=None)
        gateway_runner._provider_routing = {}
        gateway_runner._agent_cache_lock = None
        gateway_runner._agent_cache = {}
        gateway_runner._session_db = None
        gateway_runner._prefill_messages = None
        gateway_runner._pending_model_notes = {}
        gateway_runner._pending_skills_reload_notes = {}
        gateway_runner.session_store._entries = {}
        gateway_runner._get_system_prompt_for_channel.return_value = None
        gateway_runner._resolve_session_agent_runtime.return_value = ("test-model", {})
        gateway_runner._resolve_session_reasoning_config.return_value = None
        gateway_runner._resolve_session_service_tier.return_value = None
        gateway_runner._resolve_turn_agent_config.return_value = {
            "model": "test-model",
            "runtime": {},
        }
        gateway_runner._agent_config_signature.return_value = ("test-signature",)
        gateway_runner._extract_cache_busting_config.return_value = {}
        gateway_runner._refresh_fallback_model.return_value = None
        gateway_runner._consume_pending_native_image_paths.return_value = []
        gateway_runner._consume_pending_turn_sidecar_notes.return_value = []
        gateway_runner._is_telegram_topic_lane.return_value = False
        gateway_runner._is_discord_auto_thread_lane.return_value = False
        gateway_runner._is_relay_discord_channel_lane.return_value = False

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="test-chat",
            user_id="test-user",
        )
        ctx = TurnContext(
            source=source,
            message="continue",
            history=[],
            session_id="test-session",
            session_key="test-session-key",
            user_config={},
            AIAgent=_ExhaustedAgent,
            resolve_display_setting=lambda *_args: False,
            _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False),
        )

        from gateway.run_turn_runner import TurnRunner

        result = TurnRunner(gateway_runner, ctx).run_sync()

        assert result["final_response"] == (
            "Context length exceeded. Cannot compress further."
        )
        assert result["compression_exhausted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["route", "agent", "callbacks", "history", "message", "conversation", "finalization"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_exception_after_stream_creation_finishes_consumer(monkeypatch, boundary, error_type):
    from gateway.config import StreamingConfig
    from gateway.run_turn_runner import TurnRunner
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = SimpleNamespace(
        SUPPORTS_MESSAGE_EDITING=True, SUPPORTS_NATIVE_STREAMING=False,
        max_message_length=4000, message_len_fn=len,
    )
    gateway_runner = MagicMock()
    gateway_runner.config = SimpleNamespace(streaming=StreamingConfig(enabled=True))
    gateway_runner._provider_routing = {}
    gateway_runner._pre_agent_fallback_notice = None
    gateway_runner._resolve_session_agent_runtime.return_value = ("test-model", {})
    gateway_runner._delivery_adapter_for.return_value = adapter
    gateway_runner._build_stream_consumer_config.return_value = (StreamConsumerConfig(), None)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="test-chat"),
        message="continue", session_id="test-session", session_key="test-key",
        user_config={}, resolve_display_setting=lambda *_args: True,
        _run_still_current=lambda: True,
    )
    turn = TurnRunner(gateway_runner, ctx)
    monkeypatch.setattr(turn, "_combined_ephemeral_prompt", lambda: "")
    monkeypatch.setattr(turn, "_resolve_turn_agent", lambda *_args: (object(), False))
    monkeypatch.setattr(turn, "_wire_turn_agent_callbacks", lambda *_args: None)
    monkeypatch.setattr(turn, "_load_turn_history", lambda *_args: ([], None, []))
    monkeypatch.setattr(turn, "_prepare_turn_message", lambda *_args: ("continue", None))
    monkeypatch.setattr(turn, "_run_conversation_with_approval", lambda *_args: {"final_response": "answer"})
    error = error_type("failed after stream creation")

    def fail(*_args):
        raise error

    target, method = {
        "route": (gateway_runner, "_resolve_turn_agent_config"),
        "agent": (turn, "_resolve_turn_agent"),
        "callbacks": (turn, "_wire_turn_agent_callbacks"),
        "history": (turn, "_load_turn_history"),
        "message": (turn, "_prepare_turn_message"),
        "conversation": (turn, "_run_conversation_with_approval"),
        "finalization": (turn, "_finish_stream_consumer"),
    }[boundary]
    monkeypatch.setattr(target, method, fail)
    finishes = []
    real_finish = GatewayStreamConsumer.finish

    def finish(consumer, final_text=None):
        finishes.append(final_text)
        real_finish(consumer, final_text)

    monkeypatch.setattr(GatewayStreamConsumer, "finish", finish)
    with pytest.raises(error_type) as caught:
        turn.run_sync()
    consumer = ctx.stream_consumer_holder[0]
    assert isinstance(consumer, GatewayStreamConsumer)
    assert (caught.value is error, finishes, ctx.result_holder) == (True, [None], [None])
    await consumer.run()
    assert consumer.final_response_sent is False
