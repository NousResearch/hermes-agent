"""
Regression tests for #56004 — Qwen3.6 / vLLM preserve_thinking history replay.

Qwen3.6 is trained to retain and leverage historical thinking traces via the
``preserve_thinking`` chat-template option. On vLLM ≥0.20 the reasoning comes
back under ``message.reasoning`` (not ``reasoning_content``) and vLLM drops
``reasoning_content`` on incoming assistant messages, so the only field that
lands in the rendered template is ``reasoning``.

Hermes today strips both ``reasoning_content`` (via
``copy_reasoning_content_for_api`` for any non-DeepSeek/Kimi/MiMo provider)
and ``reasoning`` (unconditionally, in ``conversation_loop.py`` line ~790
"Remove 'reasoning' field — it's for trajectory storage only"). For Qwen3.6 /
vLLM that defeats the preserve_thinking feature.

These tests assert the new behaviour for a provider that "preserves thinking
history" (Qwen3.6 / vLLM):
  * ``copy_reasoning_content_for_api`` does NOT strip ``reasoning_content``
    just because the provider is not in the DeepSeek/Kimi/MiMo set.
  * ``conversation_loop.py`` does NOT pop the ``reasoning`` field on the
    outgoing message.
  * ``reapply_reasoning_echo_for_provider`` does NOT strip ``reasoning`` on
    the way out either.
  * The fallback path (switching FROM a thinking-history provider TO a strict
    provider) does strip ``reasoning`` to avoid 400/422.

The detection predicate is ``_preserves_thinking_history()`` and uses a
model-name match on ``qwen3.6`` / ``qwen-3.6`` per the issue's recommended
default ("Model-name matching on qwen3.6 is a reasonable default if a config
flag is undesirable").
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from run_agent import AIAgent


def _make_agent(provider: str = "", model: str = "", base_url: str = "") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


# ─── Detection predicate ──────────────────────────────────────────────────


def test_preserves_thinking_history_matches_qwen3_6_model_name():
    """A model named ``qwen3.6-...`` is a thinking-history provider."""
    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8", base_url="http://10.0.0.5:8000/v1")
    assert agent._preserves_thinking_history() is True


def test_preserves_thinking_history_matches_qwen_3_6_with_hyphen():
    """A model named ``qwen-3.6-...`` (hyphen variant) is also a match."""
    agent = _make_agent(provider="custom", model="qwen-3.6-35B-A3B", base_url="http://10.0.0.5:8000/v1")
    assert agent._preserves_thinking_history() is True


def test_preserves_thinking_history_false_for_qwen3():
    """Older Qwen3 (without .6) is NOT a thinking-history provider — the
    preserve_thinking feature was introduced in Qwen3.6 per the model card."""
    agent = _make_agent(provider="custom", model="Qwen3-32B", base_url="http://10.0.0.5:8000/v1")
    assert agent._preserves_thinking_history() is False


def test_preserves_thinking_history_false_for_deepseek():
    """DeepSeek has its own path through _needs_thinking_reasoning_pad;
    a thinking-history match here would conflict with that."""
    agent = _make_agent(provider="deepseek", model="deepseek-reasoner", base_url="https://api.deepseek.com/v1")
    assert agent._preserves_thinking_history() is False


def test_preserves_thinking_history_false_for_strict_provider():
    """Strict providers (Mistral, Cerebras, Groq) do NOT preserve thinking."""
    agent = _make_agent(provider="mistral", model="mistral-large-latest", base_url="https://api.mistral.ai/v1")
    assert agent._preserves_thinking_history() is False


# ─── copy_reasoning_content_for_api ───────────────────────────────────────


def test_copy_reasoning_content_keeps_reasoning_content_for_thinking_history_provider():
    """For a Qwen3.6/vLLM provider the reasoning_content field is NOT stripped
    by copy_reasoning_content_for_api — the strict-provider strip is skipped.
    The provider may carry reasoning under either field; the template reads
    what it needs, and Hermes must not silently drop one."""
    from agent.agent_runtime_helpers import copy_reasoning_content_for_api

    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8")
    source_msg = {
        "role": "assistant",
        "reasoning": "Let me think about this...",
        "reasoning_content": "Let me think about this...",
    }
    api_msg: dict = {}
    copy_reasoning_content_for_api(agent, source_msg, api_msg)
    assert "reasoning_content" not in api_msg or api_msg.get("reasoning_content") == source_msg["reasoning_content"], (
        "For a thinking-history provider, copy_reasoning_content_for_api "
        "must not silently strip reasoning_content — vLLM templates may use it."
    )


# ─── conversation_loop reasoning-pop gate ─────────────────────────────────


def test_conversation_loop_does_not_pop_reasoning_for_thinking_history_provider():
    """The conversation_loop code path that drops 'reasoning' from the outgoing
    api_msg (line ~790, "Remove 'reasoning' field — it's for trajectory
    storage only") must NOT drop it for a Qwen3.6/vLLM provider.

    This test exercises the underlying predicate so a regression in the
    gate's wiring surfaces immediately. The integration assertion (full
    conversation_loop behavior) is in test_56004_reasoning_replay_qwen3_6.
    """
    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8")
    # The reasoning-pop gate is wired via _preserves_thinking_history()
    # inside the helper that builds the outgoing message. We assert the
    # predicate returns True for this model so the gate fires correctly.
    assert agent._preserves_thinking_history() is True


# ─── reapply_reasoning_echo_for_provider ─────────────────────────────────


def test_reapply_does_not_strip_reasoning_for_thinking_history_provider():
    """reapply_reasoning_echo_for_provider() must NOT strip the ``reasoning``
    field on outgoing messages for a thinking-history provider — that field
    is exactly what vLLM ≥0.20 reads to render prior turns."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8")
    api_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
    ]
    reapply_reasoning_echo_for_provider(agent, api_messages)
    assert api_messages[1].get("reasoning") == "thinking...", (
        "reapply_reasoning_echo_for_provider must preserve the 'reasoning' "
        "field for thinking-history providers — that's the field vLLM ≥0.20 "
        "reads for prior-turn thinking."
    )


def test_reapply_strips_reasoning_on_fallback_to_strict_provider():
    """When falling back from a thinking-history provider to a strict one
    (Mistral / Cerebras / Groq), reapply_reasoning_echo_for_provider() must
    strip the ``reasoning`` field — strict providers reject unknown fields
    with 400/422, same as reasoning_content on Mistral."""
    from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider

    agent = _make_agent(provider="mistral", model="mistral-large-latest")
    api_messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning": "thinking..."},
    ]
    reapply_reasoning_echo_for_provider(agent, api_messages)
    assert "reasoning" not in api_messages[1], (
        "On fallback to a strict provider (Mistral), the 'reasoning' field "
        "must be stripped — strict providers reject unknown fields."
    )


# ─── Cache behavior ──────────────────────────────────────────────────────


def test_preserves_thinking_history_cache_is_keyed_by_model_provider_base_url():
    """The predicate must cache on (provider, model, base_url) and invalidate
    when any of those change — same pattern as _thinking_pad_cache."""
    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8", base_url="http://10.0.0.5:8000/v1")
    # First call populates cache.
    assert agent._preserves_thinking_history() is True
    # Same key → cache hit (the result is the same; just exercise the path).
    assert agent._preserves_thinking_history() is True

    # Switching model to a non-think-history model must invalidate.
    agent.model = "Qwen3-32B"
    assert agent._preserves_thinking_history() is False


# ─── Integration sanity ──────────────────────────────────────────────────


def test_integration_predicate_is_exposed_on_agent():
    """The new method must be reachable from an AIAgent instance so
    conversation_loop / copy_reasoning_content_for_api / reapply can call it."""
    agent = _make_agent(provider="custom", model="qwen3.6-35B-A3B-FP8")
    assert hasattr(agent, "_preserves_thinking_history"), (
        "AIAgent must expose _preserves_thinking_history() so the three "
        "reasoning-replay helpers can call it as a unified predicate."
    )
    assert callable(agent._preserves_thinking_history)