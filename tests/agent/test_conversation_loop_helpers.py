"""Unit tests for the pure helper functions in ``agent.conversation_loop``.

Context: issue #45161 flags ``agent/conversation_loop.py`` (~5K lines) as the
largest zero-coverage file in the tree. The module is dominated by the giant
``run_conversation()`` orchestrator that is hard to unit test in isolation, but
several small, side-effect-light helpers sit at the top of the file and are
trivially testable:

* ``_ollama_context_limit_error`` — builds the "Ollama context too small"
  user-facing error (and returns ``None`` when it does not apply).
* ``_is_nous_inference_route`` — provider/base-url classifier used to pick
  Nous-specific billing guidance.
* ``_billing_or_entitlement_message`` — assembles the credits/billing hint,
  with a Nous branch and an OpenRouter credits-link branch.
* ``_get_continuation_prompt`` — selects the right continuation instruction
  for truncated / partial-stream / oversized-tool-call resumes.

These guards and user-facing strings are exactly the kind of thing a future
refactor can silently break, so we lock their behavior here. This is a focused
first slice of coverage for the file, not an attempt to test the whole loop.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import conversation_loop
from agent.conversation_loop import (
    _billing_or_entitlement_message,
    _get_continuation_prompt,
    _is_nous_inference_route,
    _ollama_context_limit_error,
)
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# _ollama_context_limit_error
# ---------------------------------------------------------------------------

def _ollama_agent(**overrides):
    """Build a minimal agent stub for the Ollama-context guard."""
    base = dict(
        tools=["tool_a", "tool_b"],
        _ollama_num_ctx=4096,
        model="qwen3",
        base_url="http://localhost:11434",
        provider="ollama",
        session_id="sess-1",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_ollama_guard_returns_none_without_tools():
    # No tools loaded → the guard does not apply (Ollama can run tool-less).
    assert _ollama_context_limit_error(SimpleNamespace(tools=None), 100) is None
    assert _ollama_context_limit_error(SimpleNamespace(tools=[]), 100) is None


def test_ollama_guard_returns_none_without_runtime_ctx():
    # Unknown / non-positive runtime context → can't assert it's too small.
    assert _ollama_context_limit_error(_ollama_agent(_ollama_num_ctx=None), 100) is None
    assert _ollama_context_limit_error(_ollama_agent(_ollama_num_ctx=0), 100) is None
    assert _ollama_context_limit_error(_ollama_agent(_ollama_num_ctx=-1), 100) is None


def test_ollama_guard_returns_none_when_ctx_sufficient():
    # At or above the minimum → no warning.
    agent = _ollama_agent(_ollama_num_ctx=MINIMUM_CONTEXT_LENGTH)
    assert _ollama_context_limit_error(agent, 100) is None
    agent = _ollama_agent(_ollama_num_ctx=MINIMUM_CONTEXT_LENGTH + 1)
    assert _ollama_context_limit_error(agent, 100) is None


def test_ollama_guard_message_when_ctx_too_small():
    agent = _ollama_agent(_ollama_num_ctx=4096, model="qwen3")
    msg = _ollama_context_limit_error(agent, 5000)
    assert msg is not None
    # Surfaces the actual runtime context and the required minimum.
    assert "4,096" in msg
    assert f"{MINIMUM_CONTEXT_LENGTH:,}" in msg
    # Names the model and points at the concrete config knob to fix it.
    assert "qwen3" in msg
    assert "ollama_num_ctx" in msg


def test_ollama_guard_tolerates_missing_optional_attrs():
    # model/base_url/provider absent → falls back to placeholders, no crash.
    agent = SimpleNamespace(tools=["t"], _ollama_num_ctx=2048)
    msg = _ollama_context_limit_error(agent, 1000)
    assert msg is not None
    assert "the selected model" in msg


# ---------------------------------------------------------------------------
# _is_nous_inference_route
# ---------------------------------------------------------------------------

def test_nous_route_true_for_provider_name():
    assert _is_nous_inference_route("nous", "") is True
    # Case-insensitive / whitespace-tolerant.
    assert _is_nous_inference_route("  NOUS  ", "") is True


def test_nous_route_true_for_known_hosts():
    assert _is_nous_inference_route("openai", "https://inference-api.nousresearch.com/v1") is True
    assert _is_nous_inference_route("custom", "https://inference.nousresearch.com/v1") is True


def test_nous_route_false_for_other_providers():
    assert _is_nous_inference_route("openai", "https://api.openai.com/v1") is False
    assert _is_nous_inference_route("", "") is False
    assert _is_nous_inference_route(None, None) is False


# ---------------------------------------------------------------------------
# _billing_or_entitlement_message
# ---------------------------------------------------------------------------

def test_billing_message_openrouter_includes_credits_link():
    msg = _billing_or_entitlement_message(
        capability="chat",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="some/model",
    )
    assert "openrouter.ai/settings/credits" in msg
    # Generic guidance still present.
    assert "Add credits" in msg


def test_billing_message_generic_fallback():
    msg = _billing_or_entitlement_message(
        capability="chat",
        provider="",
        base_url="",
        model="",
    )
    # Empty provider/model collapse to readable placeholders.
    assert "the selected provider" in msg
    assert "the selected model" in msg
    # No OpenRouter-specific line when the base url isn't OpenRouter.
    assert "openrouter.ai/settings/credits" not in msg


def test_billing_message_names_provider_and_model():
    msg = _billing_or_entitlement_message(
        capability="chat",
        provider="deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
    )
    assert "deepseek" in msg
    assert "deepseek-chat" in msg
    assert "/model" in msg  # mentions the provider-switch escape hatch


# ---------------------------------------------------------------------------
# _get_continuation_prompt
# ---------------------------------------------------------------------------

def test_continuation_prompt_length_truncation():
    msg = _get_continuation_prompt(is_partial_stub=False)
    assert "output" in msg.lower() and "length" in msg.lower()
    assert "Continue exactly where you left off" in msg


def test_continuation_prompt_partial_stream():
    msg = _get_continuation_prompt(is_partial_stub=True)
    assert "network error" in msg
    assert "Continue exactly where" in msg


def test_continuation_prompt_oversized_tool_call_lists_tools():
    msg = _get_continuation_prompt(is_partial_stub=True, dropped_tools=["browser", "read_file"])
    assert "browser" in msg
    assert "read_file" in msg
    # Advises splitting into smaller calls rather than retrying.
    assert "smaller" in msg


def test_continuation_prompt_truncates_dropped_tool_list_to_three():
    msg = _get_continuation_prompt(
        is_partial_stub=True,
        dropped_tools=["alpha", "bravo", "charlie", "delta", "echo"],
    )
    # Only the first three are listed in the parenthetical tool list.
    assert "(alpha, bravo, charlie)" in msg
    # The 4th and 5th tools are dropped from the rendered list.
    assert "delta" not in msg
    assert "echo" not in msg


def test_continuation_prompt_partial_without_tools_uses_network_variant():
    # Partial stub but empty dropped-tools list → network-error variant, not the
    # oversized-tool-call variant.
    msg = _get_continuation_prompt(is_partial_stub=True, dropped_tools=[])
    assert "network error" in msg


# ---------------------------------------------------------------------------
# Importability guard
# ---------------------------------------------------------------------------

def test_helpers_are_exposed_on_module():
    # Cheap guard against an accidental rename/removal that would silently
    # drop these user-facing guards.
    for name in (
        "_ollama_context_limit_error",
        "_is_nous_inference_route",
        "_billing_or_entitlement_message",
        "_get_continuation_prompt",
    ):
        assert hasattr(conversation_loop, name)
