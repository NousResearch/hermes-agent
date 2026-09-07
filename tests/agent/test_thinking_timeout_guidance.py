"""Tests for the reasoning-model thinking-timeout detection + guidance.

Two layers:

1. **Classifier override (Part 1, ``agent/error_classifier.py:720-738``)**:
   A transport disconnect on a reasoning model is reclassified as
   ``FailoverReason.timeout`` even when the session is large — instead
   of routing to the compression branch via
   ``FailoverReason.context_overflow`` which would silently delete
   conversation history on a phantom context-length error.

2. **Detection + guidance (Part 2, ``agent/thinking_timeout_guidance.py``)**:
   When the classifier says timeout AND the model is in the reasoning
   allowlist AND the error message has a transport-kill signature,
   the user gets actionable guidance (raise stale_timeout, lower
   reasoning_budget, or switch models) instead of the misleading
   "use execute_code with Python's open() for large files" advice
   that fires for the unrelated large-file-write stream-drop case.

Both behaviors were previously broken: the existing
``test_disconnect_large_session_context_overflow`` test in
``tests/agent/test_error_classifier.py`` confirms that non-reasoning
models still route to context_overflow on a large session, so the
reasoning-model override is strictly targeted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


# ── helpers ──────────────────────────────────────────────────────────────


class _TimeoutReason:
    """Minimal FailoverReason stand-in for unit tests."""

    def __init__(self, value: str = "timeout") -> None:
        self.value = value


def _classified(reason: str = "timeout", **kwargs) -> SimpleNamespace:
    """Construct a ClassifiedError stand-in with the given reason."""
    defaults = dict(
        reason=_TimeoutReason(reason),
        status_code=None,
        retryable=True,
        should_compress=False,
        should_rotate_credential=False,
        should_fallback=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── Part 1: classifier override (agent/error_classifier.py:720-738) ──


def _make_session(disconnect_message: str, model: str, *, num_messages: int = 250):
    """Construct inputs to classify_api_error for a disconnect+large-session case."""
    e = Exception(disconnect_message)
    # 128k context_length; 130k approx_tokens puts us over 0.6 of context
    # AND > 120k absolute threshold; 250 messages is also > 200 threshold.
    # Without the reasoning-model override, this routes to context_overflow.
    return e, {
        "provider": "nvidia",
        "model": model,
        "approx_tokens": 130_000,
        "context_length": 200_000,
        "num_messages": num_messages,
    }


class TestClassifierOverride:
    """The reasoning-model override at error_classifier.py:720-738.

    Verifies the new behavior: a transport disconnect on a reasoning
    model on a LARGE session now routes to FailoverReason.timeout
    instead of context_overflow.  Without this fix, the compression
    branch would fire on a phantom overflow and silently delete
    conversation history.
    """


    @pytest.mark.parametrize("model", [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "openai/o3-mini",
        "anthropic/claude-opus-4-6",
        "deepseek/deepseek-r1",
        "qwen/qwq-32b-preview",
        "x-ai/grok-4-fast-reasoning",
    ])
    def test_all_known_reasoning_models_override(self, model):
        from agent.error_classifier import classify_api_error, FailoverReason
        e, kwargs = _make_session(
            "server disconnected without sending complete message",
            model=model,
        )
        result = classify_api_error(e, **kwargs)
        assert result.reason == FailoverReason.timeout
        assert result.should_compress is False

    def test_non_reasoning_model_large_session_still_routes_to_context_overflow(self):
        """Regression guard: existing test_disconnect_large_session_context_overflow
        behavior must be preserved for non-reasoning models.

        Without the override, this case routes to context_overflow +
        should_compress=True (the existing, intentional behavior for
        chat models that hit true context-length errors via proxy
        disconnect).  With the override, it stays that way.
        """
        from agent.error_classifier import classify_api_error, FailoverReason
        e, kwargs = _make_session(
            "server disconnected without sending complete message",
            model="gpt-4o",
        )
        result = classify_api_error(e, **kwargs)
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True





# ── Part 2: detection (agent/thinking_timeout_guidance.py:is_thinking_timeout) ──


class TestIsThinkingTimeout:

    @pytest.mark.parametrize("model,msg", [
        ("nvidia/nemotron-3-ultra-550b-a55b", "connection reset by peer"),
        ("openai/o3-mini", "remote protocol error"),
        ("anthropic/claude-opus-4-6", "peer closed connection"),
        ("deepseek/deepseek-r1", "connection lost"),
        ("x-ai/grok-4-fast-reasoning", "server disconnected"),
    ])
    def test_known_reasoning_models_match(self, model, msg):
        from agent.thinking_timeout_guidance import is_thinking_timeout
        classified = _classified(reason="timeout")
        assert is_thinking_timeout(classified, model, msg) is True




    def test_empty_error_msg_returns_false(self):
        from agent.thinking_timeout_guidance import is_thinking_timeout
        classified = _classified(reason="timeout")
        assert is_thinking_timeout(
            classified, "nvidia/nemotron-3-ultra-550b-a55b", "",
        ) is False

    def test_none_error_msg_returns_false(self):
        from agent.thinking_timeout_guidance import is_thinking_timeout
        classified = _classified(reason="timeout")
        assert is_thinking_timeout(
            classified, "nvidia/nemotron-3-ultra-550b-a55b", None,
        ) is False


# ── Part 2: guidance text (agent/thinking_timeout_guidance.py:build_thinking_timeout_guidance) ──


class TestBuildThinkingTimeoutGuidance:
    def test_guidance_mentions_config_path(self):
        from agent.thinking_timeout_guidance import build_thinking_timeout_guidance
        text = build_thinking_timeout_guidance(
            provider="nvidia", model="nvidia/nemotron-3-ultra-550b-a55b",
        )
        assert "providers.nvidia.models.nvidia/nemotron-3-ultra-550b-a55b.stale_timeout_seconds" in text


    def test_guidance_mentions_known_providers(self):
        from agent.thinking_timeout_guidance import build_thinking_timeout_guidance
        text = build_thinking_timeout_guidance(provider="nvidia", model="x")
        # At least one of the known cloud providers should be mentioned
        # to give the user context.
        assert any(p in text for p in (
            "NVIDIA NIM", "OpenAI", "Anthropic", "DeepSeek",
        ))




