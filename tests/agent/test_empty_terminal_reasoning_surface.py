"""Tests for reasoning-only final responses.

A clean-stop response (``finish_reason == "stop"``) with no ordinary content but
structured reasoning IS the answer: the reasoning is promoted to the visible reply and
persisted as assistant content without entering the empty-response recovery ladder
(every rung re-bills the full prompt). Idea credit: PR #48795 (@ligl0325).

Invariants pinned here:
- Clean-stop reasoning-only → returned and persisted after ONE API call.
- ``finish_reason == "length"`` reasoning is unfinished: never promoted, the
  continuation path still owns it.
- A truly empty response (no reasoning either) still reaches the ladder terminal.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


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
