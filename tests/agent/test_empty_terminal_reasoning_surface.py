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
    # This fixture represents the explicit parser-compatible route. Private reasoning
    # tests clear the capability and use an OpenRouter route instead.
    agent.runtime_capabilities["answer_in_reasoning"] = True
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


def test_answer_in_reasoning_requires_a_trusted_route():
    """A generic reasoning field is not enough to opt into answer promotion."""
    from agent.agent_runtime_helpers import answer_in_reasoning_capability

    agent = SimpleNamespace(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        base_url="http://127.0.0.1:8000/v1",
        provider="vllm",
        api_mode="chat_completions",
        runtime_capabilities={},
        _custom_providers=[],
    )
    assert answer_in_reasoning_capability(agent) is True

    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    assert answer_in_reasoning_capability(agent) is False

    agent.provider = "vllm"
    agent.base_url = "http://127.0.0.1:8000/v1"
    agent.model = "deepseek/deepseek-v4.1"
    assert answer_in_reasoning_capability(agent) is False


def test_answer_in_reasoning_capability_survives_route_map_rebuild():
    """Route refreshes add the decision without carrying it to another route."""
    from agent.agent_runtime_helpers import _ensure_answer_in_reasoning_capability

    agent = SimpleNamespace(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        base_url="http://127.0.0.1:8000/v1",
        provider="vllm",
        api_mode="chat_completions",
        runtime_capabilities={"native_compaction": False},
        _custom_providers=[],
    )
    _ensure_answer_in_reasoning_capability(agent)
    assert agent.runtime_capabilities["answer_in_reasoning"] is True

    agent.model = "deepseek/deepseek-v4.1"
    agent.runtime_capabilities = {"native_compaction": False}
    _ensure_answer_in_reasoning_capability(agent)
    assert agent.runtime_capabilities["answer_in_reasoning"] is False

    agent.runtime_capabilities["answer_in_reasoning"] = True
    _ensure_answer_in_reasoning_capability(agent)
    assert agent.runtime_capabilities["answer_in_reasoning"] is True


def test_constructor_capability_reaches_runtime_map_and_primary_snapshot(tmp_path, monkeypatch):
    """An explicit startup opt-in must drive the live route and its restore snapshot."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy",
        provider="vllm",
        api_mode="chat_completions",
        base_url="http://127.0.0.1:8000/v1",
        capabilities={"answer_in_reasoning": True},
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert agent.runtime_capabilities["answer_in_reasoning"] is True
    assert agent._primary_runtime["runtime_capabilities"]["answer_in_reasoning"] is True


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


def test_private_reasoning_retries_to_visible_answer(tmp_path, monkeypatch):
    """A provider's private reasoning stays out of the answer, even when both generic
    reasoning fields contain the same text."""
    agent = _build_agent(tmp_path, monkeypatch)
    agent.runtime_capabilities["answer_in_reasoning"] = False
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
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
            model="deepseek/deepseek-v4.1",
        ),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == "the visible answer"
    assert result["api_calls"] == 2  # the visible retry follows one thinking-prefill call
    assert all("private thoughts" not in str(message.get("content", "")) for message in result["messages"])


def test_private_reasoning_is_not_echoed_when_recovery_exhausts(tmp_path, monkeypatch):
    """Exhausted recovery returns the empty sentinel without exposing a private preview."""
    agent = _build_agent(tmp_path, monkeypatch)
    agent.runtime_capabilities["answer_in_reasoning"] = False
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: _private_reasoning_only_response())

    result = agent.run_conversation("hello?")

    assert "private thoughts" not in result["final_response"]
    assert all("private thoughts" not in str(message.get("content", "")) for message in result["messages"])
