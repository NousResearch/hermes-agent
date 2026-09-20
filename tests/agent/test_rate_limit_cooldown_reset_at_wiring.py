"""Wiring tests for #117484: the reset_at must actually flow from the classified error
through ``_try_activate_fallback`` into the armed cooldown on the real routing paths."""

import time
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason, ClassifiedError
from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


class TestResetAtFlowsThroughRealRoutes:
    def test_codex_app_server_route_honors_reset(self):
        """``activate_codex_app_server_fallback`` passes ``classified.error_context`` through:
        a rate_limit verdict carrying reset_at=+90s must arm 90s (provider truth), not the
        60s exponential guess. (The route's real input is error text without a status code,
        so the context comes from the classifier's verdict — here pinned directly.)"""
        from agent.turn_recovery import activate_codex_app_server_fallback
        from agent.error_classifier import ClassifiedError
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        classified = ClassifiedError(
            reason=FailoverReason.rate_limit, status_code=429,
            error_context={"reset_at": now + 90},
        )
        with (
            patch("agent.turn_recovery.classify_api_error", return_value=classified),
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=((lambda: (m := MagicMock(), setattr(m, "base_url", "https://openrouter.ai/api/v1"),
                                          setattr(m, "api_key", "k"))[0])(), "resolved")),
        ):
            ok = activate_codex_app_server_fallback(agent, {"error": "429 rate limited"})
            cooldown = agent._rate_limited_until - frozen
        assert ok is True
        assert cooldown == 90.0, f"expected the provider-stated 90s window, got {cooldown}s"

    def test_route_classified_error_honors_reset_at_context(self):
        """``route_classified_error``'s eager-fallback route passes ``classified.error_context``
        through; a classified 429 with reset_at=+90s arms 90s on the real path."""
        from agent.turn_recovery import route_classified_error
        from agent.turn_retry_state import TurnRetryState

        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        classified = ClassifiedError(
            reason=FailoverReason.rate_limit, status_code=429, retryable=True,
            error_context={"reset_at": now + 90},
        )
        _retry = TurnRetryState()
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=((lambda: (m := MagicMock(), setattr(m, "base_url", "https://openrouter.ai/api/v1"),
                                          setattr(m, "api_key", "k"))[0])(), "resolved")),
        ):
            verdict = route_classified_error(
                agent, RuntimeError("429"), classified, _retry, error_msg="429",
                error_context=classified.error_context, recovered_with_pool=False,
                base_url="https://x", model="m", messages=[], api_messages=[], system_message=None,
                active_system_prompt=None, conversation_history=[], retry_count=99, max_retries=3,
                compression_attempts=0, max_compression_attempts=2, api_call_count=1,
                effective_task_id=None,
            )
            cooldown = agent._rate_limited_until - frozen
        assert verdict.action == "break"
        assert cooldown == 90.0, f"expected 90s from error_context reset_at, got {cooldown}s"

    def test_forwarder_accepts_error_context_kwarg(self):
        """``AIAgent._try_activate_fallback`` (the lazy forwarder) forwards the
        ``error_context`` keyword to the chokepoint untouched."""
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        frozen = 1_000.0
        now = 50_000.0
        with (
            patch("agent.chat_completion_helpers.time.monotonic", return_value=frozen),
            patch("agent.fallback_cooldown.time.time", return_value=now),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=((lambda: (m := MagicMock(), setattr(m, "base_url", "https://openrouter.ai/api/v1"),
                                          setattr(m, "api_key", "k"))[0])(), "resolved")),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit,
                                                error_context={"reset_at": now + 45}) is True
        assert agent._rate_limited_until - frozen == 45.0
