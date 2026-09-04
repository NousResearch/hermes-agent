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
from gateway.platforms.base import resolve_channel_prompt
from gateway.session import SessionSource
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

        from gateway.run import TurnRunner

        result = TurnRunner(gateway_runner, ctx).run_sync()

        assert result["final_response"] == (
            "Context length exceeded. Cannot compress further."
        )
        assert result["compression_exhausted"] is True

    def test_wildcard_channel_prompt_composes_with_channel_override(self):
        class _CapturingAgent:
            init_kwargs: dict = {}

            def __init__(self, **kwargs):
                type(self).init_kwargs = kwargs
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
                    "final_response": "ok",
                    "messages": [],
                    "api_calls": 1,
                    "completed": True,
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
        gateway_runner._get_system_prompt_for_channel.return_value = "Override persona"
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
            platform=Platform.DISCORD,
            chat_id="channel-a",
            user_id="test-user",
        )
        channel_prompt = resolve_channel_prompt(
            {
                "channel_prompts": {
                    "*": "Global policy",
                    "channel-a": "Exact context",
                }
            },
            source.chat_id,
        )
        ctx = TurnContext(
            source=source,
            message="continue",
            history=[],
            session_id="test-session",
            session_key="test-session-key",
            user_config={},
            context_prompt="Platform context",
            channel_prompt=channel_prompt,
            AIAgent=_CapturingAgent,
            resolve_display_setting=lambda *_args: False,
            _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False),
        )

        from gateway.run import TurnRunner

        result = TurnRunner(gateway_runner, ctx).run_sync()

        assert result["final_response"] == "ok"
        assert _CapturingAgent.init_kwargs["ephemeral_system_prompt"] == (
            "Platform context\n\nGlobal policy\n\nExact context\n\nOverride persona"
        )
