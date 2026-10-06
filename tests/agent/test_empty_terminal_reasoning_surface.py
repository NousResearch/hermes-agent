"""Tests for reasoning-only final responses.

A clean-stop response (``finish_reason == "stop"``) with no ordinary content but
structured reasoning is promoted only for an explicitly trusted route; private reasoning
stays in the recovery path and never becomes the visible reply. Idea credit: PR #48795
(@ligl0325).

Invariants pinned here:
- Trusted clean-stop reasoning-only → returned and persisted after ONE API call.
- ``finish_reason == "length"`` reasoning is unfinished: never promoted, the
  continuation path still owns it.
- A truly empty response (no reasoning either) still reaches the ladder terminal.
- Untrusted-route private reasoning is never promoted nor echoed on exhaustion.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


# Provider-level ``capabilities:`` opt-ins: the fixture's own route and a second custom provider
# (plus an un-opted sibling on that provider's endpoint).
_OPTED_IN_PROVIDERS = [
    {"name": "local", "base_url": "https://example.invalid/v1", "capabilities": {"answer_in_reasoning": True}},
    {"name": "acme", "base_url": "https://llm.example.com/v1", "model": "acme/reasoner",
     "capabilities": {"answer_in_reasoning": True}},
    {"name": "acme-plain", "base_url": "https://llm.example.com/v1"},  # same endpoint, no opt-in
]


def _build_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy",
        base_url="https://example.invalid/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # Route through the non-streaming _interruptible_api_call path so the
    # monkeypatched fake responses are what the loop consumes.
    agent._disable_streaming = True
    # The fixture route is opted in through its custom_providers entry (no constructor
    # ``capabilities=``, as on CLI/TUI). Private reasoning tests move to an untrusted route.
    agent._custom_providers = _OPTED_IN_PROVIDERS
    return agent


def _reasoning_only_response(finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                reasoning="The answer is 42 because of the calculation above.",
                reasoning_content=None,
                reasoning_details=None,
                tool_calls=None,
            ),
            finish_reason=finish_reason,
        )],
        usage=None,
        model="test-model",
    )


def _truly_empty_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="",
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )],
        usage=None,
        model="test-model",
    )


def _private_reasoning_only_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=None,
                reasoning="private thoughts that must not be shown",
                reasoning_content="private thoughts that must not be shown",
                reasoning_details=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )],
        usage=None,
        model="deepseek/deepseek-v4.1",
    )


def test_clean_stop_reasoning_only_returns_on_first_call(tmp_path, monkeypatch):
    """A clean stop promotes structured reasoning without a recovery call."""
    agent = _build_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: _reasoning_only_response(),
    )

    result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == "The answer is 42 because of the calculation above."
    assert result["api_calls"] == 1
    # The promoted text replays as a real answer through the api_content sidecar; the row's
    # own content stays empty so chain-of-thought is never persisted as an ordinary reply.
    row = result["messages"][-1]
    assert row["role"] == "assistant"
    assert not row.get("content")
    assert row["api_content"] == "The answer is 42 because of the calculation above."


def test_exhausted_truly_empty_keeps_existing_behavior(tmp_path, monkeypatch):
    """No reasoning anywhere → behavior unchanged from main: the '(empty)'
    terminal (possibly rewritten by the downstream turn-completion explainer)
    is delivered, and no reasoning excerpt appears."""
    agent = _build_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: _truly_empty_response(),
    )

    result = agent.run_conversation("hello?")

    final = result["final_response"]
    # Either the raw sentinel (explainer off) or the explainer's rewrite —
    # never the reasoning-excerpt frame, which requires reasoning to exist.
    assert final == "(empty)" or final.startswith("⚠️ No reply:")
    assert "only internal reasoning" not in final


def test_length_cut_reasoning_is_not_promoted(tmp_path, monkeypatch):
    """``finish_reason == "length"`` means the model was cut off mid-thought: the reasoning
    is not an answer, so the continuation path runs and the model's real text wins."""
    agent = _build_agent(tmp_path, monkeypatch)
    responses = [
        _reasoning_only_response(finish_reason="length"),
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="42.",
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
            model="test-model",
        ),
    ]
    monkeypatch.setattr(
        agent, "_interruptible_api_call",
        lambda api_kwargs: responses.pop(0),
    )

    result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == "42."
    assert result["api_calls"] == 2


@pytest.mark.parametrize("provider, base_url, model, final, calls", [
    ("openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4.1", "the visible answer", 2),
    ("vllm", "http://127.0.0.1:8000/v1", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
     "private thoughts that must not be shown", 1),
    ("custom", "https://llm.example.com/v1", "acme/reasoner", "private thoughts that must not be shown", 1),
    ("custom", "https://llm.example.com/v1", "acme/after-model-switch", "private thoughts that must not be shown", 1),
    ("custom", "https://fallback.example.com/v1", "acme/reasoner", "the visible answer", 2),
    ("custom:acme-plain", "https://llm.example.com/v1", "acme/reasoner", "the visible answer", 2),
])
def test_reasoning_promotion_requires_a_trusted_route(tmp_path, monkeypatch, provider, base_url, model, final, calls):
    """Private reasoning on an untrusted route retries to the visible answer and never
    surfaces; the local Nemotron parser route (#109205) and a provider-level ``capabilities:``
    opt-in promote in one call, re-read on the live route (any model on that provider, never a
    fallback on another base_url), with no constructor ``capabilities=`` (CLI/TUI)."""
    agent = _build_agent(tmp_path, monkeypatch)
    agent.provider, agent.base_url, agent.model = provider, base_url, model
    responses = [
        _private_reasoning_only_response(),
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="the visible answer",
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=None,
            model=model,
        ),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == final
    assert result["api_calls"] == calls
    assert all("private thoughts" not in str(message.get("content", "")) for message in result["messages"])


def test_private_reasoning_is_not_echoed_when_recovery_exhausts(tmp_path, monkeypatch):
    """Exhausted recovery returns the empty sentinel without exposing a private preview."""
    agent = _build_agent(tmp_path, monkeypatch)
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _private_reasoning_only_response())

    result = agent.run_conversation("hello?")

    assert "private thoughts" not in result["final_response"]
    assert all("private thoughts" not in str(message.get("content", "")) for message in result["messages"])
