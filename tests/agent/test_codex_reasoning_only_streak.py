"""Codex Responses reasoning-only stall recovery (#67321).

Encrypted reasoning items replay byte-for-byte, so a bare continuation of a
reasoning-only ``status=incomplete`` response repeats the stall. After three
consecutive reasoning-only responses the turn must reach the configured
fallback provider (with one bounded grace call when the trigger consumed the
iteration budget) instead of ending on the internal incomplete sentinel; a
visible partial resets the local streak; a cross-protocol fallback drops the
Codex-only nudge from the wire.
"""

from __future__ import annotations

from types import SimpleNamespace

import run_agent
from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE, _CODEX_INCOMPLETE_NUDGE
from agent.error_classifier import FailoverReason
from agent.turn_truncation import continue_codex_incomplete
from tests.agent.test_run_agent_codex_responses import (
    _build_agent,
    _codex_commentary_message_response,
    _codex_incomplete_message_response,
    _codex_message_response,
    _codex_reasoning_only_response,
)


def _spy_fallback(agent, monkeypatch):
    """Record fallback activations; keep ``api_mode`` on codex_responses so the
    stub fallback answer still parses through the Codex path."""
    calls = []

    def _fake(reason=None):
        calls.append(reason)
        return True

    monkeypatch.setattr(agent, "_try_activate_fallback", _fake)
    return calls


def _drive(agent, monkeypatch, responses):
    api_calls = {"n": 0}

    def _fake_api_call(api_kwargs):
        api_calls["n"] += 1
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)
    return api_calls


def test_reasoning_only_streak_reaches_fallback_with_one_grace_call(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.max_iterations = 3
    agent.iteration_budget = run_agent.IterationBudget(3)
    calls = _spy_fallback(agent, monkeypatch)
    api_calls = _drive(agent, monkeypatch, [
        _codex_reasoning_only_response(encrypted_content="enc_a"),
        _codex_reasoning_only_response(encrypted_content="enc_b"),
        _codex_reasoning_only_response(encrypted_content="enc_c"),
        _codex_message_response("Fallback answered."),
    ])

    result = agent.run_conversation("keep thinking")

    assert result["completed"] is True
    assert result["final_response"] == "Fallback answered."
    assert calls == [FailoverReason.incomplete_response]
    # Three budgeted calls + exactly one grace call; the grace flag is consumed.
    assert api_calls["n"] == 4
    assert agent._budget_grace_call is False


def test_visible_partial_resets_reasoning_only_streak(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent.max_iterations = 6
    agent.iteration_budget = run_agent.IterationBudget(6)
    calls = _spy_fallback(agent, monkeypatch)
    _drive(agent, monkeypatch, [
        _codex_incomplete_message_response("Partial visible progress."),
        _codex_reasoning_only_response(encrypted_content="enc_a"),
        _codex_reasoning_only_response(encrypted_content="enc_b"),
        _codex_reasoning_only_response(encrypted_content="enc_c"),
        _codex_message_response("Recovered."),
    ])

    result = agent.run_conversation("partial then stall")

    assert result["completed"] is True
    assert result["final_response"] == "Recovered."
    assert calls == [FailoverReason.incomplete_response]


def test_commentary_only_is_not_reasoning_stall_and_uses_action_nudge(monkeypatch):
    """Public commentary is progress, so it must not enter the #67321 internal-reasoning
    ladder or receive the nudge that tells the model to produce a final answer now."""
    from agent.codex_responses_adapter import _normalize_codex_response

    agent = _build_agent(monkeypatch)
    messages = [{"role": "user", "content": "audit it"}]

    for api_call_count, text in enumerate(
        ("I found another parity defect; continuing.", "Still auditing the remaining paths."),
        start=1,
    ):
        response = _codex_commentary_message_response(text)
        assistant_message, finish_reason = _normalize_codex_response(response)
        assert finish_reason == "incomplete"

        result = continue_codex_incomplete(
            agent,
            assistant_message,
            finish_reason,
            messages=messages,
            conversation_history=[],
            api_call_count=api_call_count,
            response=response,
        )

        assert result is None
        assert agent._codex_reasoning_only_streak == 0

    assert not any(m.get("content") == _CODEX_INCOMPLETE_NUDGE for m in messages)
    assert messages[-1] == {"role": "user", "content": _CODEX_ACK_CONTINUATION_NUDGE}


def test_commentary_max_output_does_not_disable_reasoning(monkeypatch):
    """A max-output response with public commentary did produce visible progress; the
    reasoning-off/output-cap recovery is for reasoning-only exhaustion, not commentary."""
    from agent.codex_responses_adapter import _normalize_codex_response

    agent = _build_agent(monkeypatch)
    response = _codex_commentary_message_response("Checking one more path before the fix.")
    response.status = "incomplete"
    response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    response.output[0].status = "incomplete"

    assistant_message, finish_reason = _normalize_codex_response(response)
    assert finish_reason == "incomplete"

    result = continue_codex_incomplete(
        agent,
        assistant_message,
        finish_reason,
        messages=[{"role": "user", "content": "keep auditing"}],
        conversation_history=[],
        api_call_count=1,
        response=response,
    )

    assert result is None
    assert agent._codex_reasoning_only_streak == 0
    assert agent._ephemeral_reasoning_off is False


def test_cross_protocol_fallback_wire_drops_codex_nudge_and_replay_state(monkeypatch):
    """The nudge and encrypted reasoning are Codex-only: once the stall falls over to a
    Chat Completions provider the assembled request must carry neither, with roles alternating."""
    agent = _build_agent(monkeypatch)
    agent.max_iterations = 6
    agent.iteration_budget = run_agent.IterationBudget(6)

    def _flip_to_chat(reason=None):
        agent.api_mode = "chat_completions"
        agent._disable_streaming = True  # the stub answer is a plain object, not a stream
        return True

    monkeypatch.setattr(agent, "_try_activate_fallback", _flip_to_chat)
    chat_answer = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Fallback answered.", tool_calls=None),
                                 finish_reason="stop")],
        model="fallback/model", usage=None,
    )
    responses = [
        _codex_reasoning_only_response(encrypted_content="enc_a"),
        _codex_reasoning_only_response(encrypted_content="enc_b"),
        _codex_reasoning_only_response(encrypted_content="enc_c"),
        chat_answer,
    ]
    wires = []

    def _fake_api_call(api_kwargs):
        wires.append((agent.api_mode, api_kwargs))
        return responses.pop(0)

    monkeypatch.setattr(agent, "_interruptible_api_call", _fake_api_call)

    result = agent.run_conversation("do it")

    assert result["final_response"] == "Fallback answered."
    assert [mode for mode, _ in wires] == ["codex_responses"] * 3 + ["chat_completions"]
    # The pre-fallback transcript did carry the nudge (replay + nudge before the third stall).
    assert any(m.get("content") == _CODEX_INCOMPLETE_NUDGE for m in agent._session_messages)
    wire = [m for m in wires[-1][1]["messages"] if m["role"] not in ("system", "developer")]
    # Thinking-only rows are dropped and adjacent users merged, so the nudge would
    # survive as a fragment of the merged user row rather than as its own row.
    assert not any(_CODEX_INCOMPLETE_NUDGE in str(m.get("content") or "") for m in wire)
    assert not any(m.get("codex_reasoning_items") for m in wire)
    roles = [m["role"] for m in wire]
    assert roles and all(a != b for a, b in zip(roles, roles[1:]))
