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
from types import SimpleNamespace
from unittest.mock import MagicMock


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

    def test_turn_routed_model_owns_reasoning_config(self):
        """Gateway must resolve per-model reasoning after turn_route selects the effective model."""

        captured = {}

        class _Agent:
            def __init__(self, **kwargs):
                captured["model"] = kwargs["model"]
                captured["reasoning_config"] = kwargs["reasoning_config"]
                self.model = kwargs["model"]
                self.session_id = kwargs["session_id"]
                self.tools = []
                self.context_compressor = SimpleNamespace(last_prompt_tokens=0, context_length=200_000)
                self.session_prompt_tokens = 0
                self.session_completion_tokens = 0

            def run_conversation(self, _message, **_kwargs):
                return {"final_response": "ok", "messages": []}

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
        gateway_runner._resolve_session_agent_runtime.return_value = ("alpha", {})
        gateway_runner._resolve_session_reasoning_config.side_effect = (
            lambda **kwargs: {"enabled": True, "effort": "high" if kwargs["model"] == "beta" else "low"}
        )
        gateway_runner._resolve_session_service_tier.return_value = None
        gateway_runner._resolve_turn_agent_config.return_value = {"model": "beta", "runtime": {}}
        gateway_runner._agent_config_signature.return_value = ("test-signature",)
        gateway_runner._extract_cache_busting_config.return_value = {}
        gateway_runner._refresh_fallback_model.return_value = None
        gateway_runner._consume_pending_native_image_paths.return_value = []
        gateway_runner._consume_pending_turn_sidecar_notes.return_value = []
        gateway_runner._is_telegram_topic_lane.return_value = False
        gateway_runner._is_discord_auto_thread_lane.return_value = False
        gateway_runner._is_relay_discord_channel_lane.return_value = False

        source = SessionSource(platform=Platform.LOCAL, chat_id="test-chat", user_id="test-user")
        ctx = TurnContext(
            source=source,
            message="route me",
            history=[],
            session_id="test-session",
            session_key="test-session-key",
            user_config={},
            AIAgent=_Agent,
            resolve_display_setting=lambda *_args: False,
            _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False),
        )

        from gateway.run_turn_runner import TurnRunner

        result = TurnRunner(gateway_runner, ctx).run_sync()

        assert result["final_response"] == "ok"
        assert captured == {
            "model": "beta",
            "reasoning_config": {"enabled": True, "effort": "high"},
        }
        assert gateway_runner._resolve_session_reasoning_config.call_args.kwargs["model"] == "beta"
        assert ctx.realized_route is gateway_runner._resolve_turn_agent_config.return_value



class TestTurnRoutingLifecycle:
    @pytest.mark.parametrize(
        ("configured_window", "selected_window", "injected_tokens", "admitted"),
        [
            (8_192, 131_072, 6_000, True),
            (131_072, 8_192, 16_000, False),
        ],
    )
    def test_context_admission_uses_selected_route_window(
        self, monkeypatch, configured_window, selected_window, injected_tokens, admitted
    ):
        """Both admission directions use the route selected for this turn, not config."""
        from agent.context_references import ContextReferenceResult
        from gateway.run_turn_runner import TurnRunner

        from gateway.run import GatewayRunner
        runner = object.__new__(GatewayRunner)
        source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
        ctx = TurnContext(source=source, session_key="session", message="inspect @diff")
        route = {
            "model": "selected-model",
            "runtime": {"provider": "custom", "context_length": selected_window},
        }
        seen = {}

        async def context_length(_source, _session_key, *, turn_route=None):
            seen["route"] = turn_route
            return turn_route["runtime"]["context_length"]

        async def preprocess(message, *, context_length, **_kwargs):
            seen["context_length"] = context_length
            return ContextReferenceResult(
                message=message + " expanded", original_message=message,
                expanded=admitted, blocked=not admitted,
                injected_tokens=injected_tokens, warnings=["blocked"] if not admitted else [],
            )

        runner._inbound_model_context_length = context_length
        runner._delivery_adapter_for = lambda _source: None
        assert configured_window != selected_window
        monkeypatch.setattr(
            "agent.context_references.preprocess_context_references_async", preprocess
        )

        result = TurnRunner(runner, ctx)._prepare_context_references_for_realized_route(route)

        assert result is admitted
        assert seen["route"] is route
        assert seen["context_length"] == selected_window
        if admitted:
            assert ctx.message.endswith(" expanded")
        else:
            assert ctx.context_reference_blocked is True

    def test_context_reference_expansion_receives_the_realized_route(self):
        """The worker must admit @-context using the route it is about to execute."""
        from gateway.run_turn_runner import TurnRunner

        seen = {}
        runner = SimpleNamespace()
        source = SessionSource(platform=Platform.LOCAL, chat_id="chat", user_id="user")
        ctx = TurnContext(
            source=source, session_key="session",
            message="[2026-09-30 12:00] inspect @diff",
            persist_user_message="inspect @diff",
        )
        selected_route = {"model": "large-window", "runtime": {"provider": "custom"}}

        async def expand(source_arg, session_key, message, *, turn_route, warning_sender):
            seen.update(source=source_arg, session_key=session_key, message=message, route=turn_route)
            assert warning_sender is not None
            return message + "\n\nattached context"

        runner._expand_inbound_context_references = expand
        runner._delivery_adapter_for = lambda _source: None

        assert TurnRunner(runner, ctx)._prepare_context_references_for_realized_route(selected_route)
        assert seen["route"] is selected_route
        assert ctx.message.endswith("attached context")
        assert ctx.persist_user_message == "inspect @diff\n\nattached context"

    def test_fallback_cleanup_uses_realized_route_and_preserves_true_fallback_eviction(self, monkeypatch):
        """A successful A→B route is reusable; a successful in-turn B→C fallback is not."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        evicted = []
        runner._evict_cached_agent = lambda key: evicted.append(key)
        runner._is_intentional_model_switch = lambda *_args: False
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "configured-alpha")

        def cleanup(expected_model, actual_model):
            ctx = TurnContext(
                session_key="durable",
                realized_route={"model": expected_model, "runtime": {"provider": "custom"}},
            )
            ctx.agent_holder[0] = SimpleNamespace(model=actual_model, provider="custom")
            ctx.result_holder[0] = {"final_response": "ok"}
            runner._run_agent_evict_on_fallback(ctx)

        cleanup("selected-beta", "selected-beta")
        assert evicted == []

        cleanup("selected-beta", "fallback-gamma")
        assert evicted == ["durable"]
