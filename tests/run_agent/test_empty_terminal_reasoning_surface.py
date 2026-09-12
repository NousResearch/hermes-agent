"""Tests for reasoning-only and truly empty final responses.

When a provider reports a clean stop with structured reasoning but no visible
text, the reasoning is the completed response and must bypass the expensive
empty-response recovery ladder. Idea credit: PR #48795 (@ligl0325).

Invariants pinned here:
- Clean-stop reasoning is returned and persisted without another API call.
- Length-limited reasoning still goes through continuation instead of being
  promoted as a complete answer.
- A truly empty exhaustion (no reasoning either) still returns "(empty)".
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


def _reasoning_only_response(*, finish_reason="stop", reasoning=None):
    reasoning = reasoning or "The answer is 42 because of the calculation above."
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="",
                reasoning=reasoning,
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


def _text_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
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


def _tool_call_response():
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="todo", arguments="{}"),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="",
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
                tool_calls=[tool_call],
            ),
            finish_reason="tool_calls",
        )],
        usage=None,
        model="test-model",
    )


def test_clean_stop_reasoning_is_promoted_without_retry(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch)
    calls = 0

    def respond(api_kwargs):
        nonlocal calls
        calls += 1
        return _reasoning_only_response()

    monkeypatch.setattr(agent, "_interruptible_api_call", respond)

    result = agent.run_conversation("what is the answer?")

    expected = "The answer is 42 because of the calculation above."
    assert result["final_response"] == expected
    assert result["turn_exit_reason"] == "reasoning_response(clean_stop)"
    assert calls == 1
    assert result["messages"][-1]["content"] == expected
    assert result["messages"][-1]["reasoning"] == expected


def test_clean_stop_promotion_survives_stall_guard_continuation(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch)
    agent.valid_tool_names = {"todo"}
    promoted = "The tests pass. I will now run the linter."
    responses = [
        _reasoning_only_response(reasoning=promoted),
        _text_response("The linter passes."),
    ]
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: responses.pop(0),
    )

    result = agent.run_conversation("verify the change")

    assert result["final_response"] == "The linter passes."
    assert not responses
    assistant_messages = [
        message for message in result["messages"] if message["role"] == "assistant"
    ]
    assert assistant_messages[0]["content"] == promoted
    assert assistant_messages[-1]["content"] == "The linter passes."
    roles = [message["role"] for message in result["messages"]]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def test_clean_stop_promotion_removes_prior_thinking_prefill(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch)
    responses = [
        _reasoning_only_response(
            finish_reason="tool_calls",
            reasoning="Still working through the request.",
        ),
        _reasoning_only_response(reasoning="The completed answer."),
    ]
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: responses.pop(0),
    )

    result = agent.run_conversation("what is the answer?")

    assert result["final_response"] == "The completed answer."
    assert result["turn_exit_reason"] == "reasoning_response(clean_stop)"
    assert not responses
    assert not any(message.get("_thinking_prefill") for message in result["messages"])
    roles = [message["role"] for message in result["messages"]]
    assert all(left != right for left, right in zip(roles, roles[1:]))


def test_clean_stop_promotion_preserves_completed_tool_exchange(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch)
    responses = [
        _tool_call_response(),
        _truly_empty_response(),
        _reasoning_only_response(reasoning="The tool result is complete."),
    ]
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: responses.pop(0),
    )

    def execute_tool(assistant_message, messages, effective_task_id, api_call_count=0):
        messages.append({
            "role": "tool",
            "name": "todo",
            "tool_call_id": "call_1",
            "content": "tool result",
        })

    monkeypatch.setattr(agent, "_execute_tool_calls", execute_tool)

    result = agent.run_conversation("run the tool")

    assert result["final_response"] == "The tool result is complete."
    assert not responses
    assert any(message.get("tool_calls") for message in result["messages"])
    assert any(message.get("role") == "tool" for message in result["messages"])
    assert not any(
        message.get("_empty_recovery_synthetic") for message in result["messages"]
    )


def test_clean_stop_promotion_runs_kanban_stop_gate(tmp_path, monkeypatch):
    agent = _build_agent(tmp_path, monkeypatch)
    responses = [
        _reasoning_only_response(reasoning="I am done."),
        _reasoning_only_response(reasoning="Lifecycle action completed."),
    ]
    monkeypatch.setattr(
        agent,
        "_interruptible_api_call",
        lambda api_kwargs: responses.pop(0),
    )
    monkeypatch.setattr(
        "agent.kanban_stop.build_kanban_stop_nudge",
        lambda messages, attempts=0: "Call the terminal lifecycle tool." if attempts == 0 else None,
    )

    result = agent.run_conversation("finish the task")

    assert result["final_response"] == "Lifecycle action completed."
    assert result["turn_exit_reason"] == "reasoning_response(clean_stop)"
    assert agent._kanban_stop_nudges == 1
    assert not responses


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


def test_length_limited_reasoning_still_uses_continuation(tmp_path, monkeypatch):
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
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert not responses
