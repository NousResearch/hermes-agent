"""End-to-end loop test: autopilot drives run_conversation to completion.

Uses the loop's supported Mock-client seam (conversation_loop.py special-cases
``isinstance(agent.client, Mock)``) to script two model turns, with the goal gate
mocked to say CONTINUE then COMPLETE. Proves the Seam-B wiring actually injects a
directive and re-enters the loop, rather than delivering the first answer.
"""

import types
from unittest.mock import Mock

import pytest


def _completion(content, finish="stop"):
    """An OpenAI ChatCompletion-shaped object the chat_completions transport accepts."""
    msg = types.SimpleNamespace(
        content=content, tool_calls=None, role="assistant",
        reasoning_content=None, reasoning_details=None, function_call=None,
        refusal=None,
    )
    choice = types.SimpleNamespace(message=msg, finish_reason=finish, index=0)
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return types.SimpleNamespace(choices=[choice], usage=usage, model="mock", id="cmpl-x")


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        model="gpt-4o", provider="openrouter", api_key="sk-dummy",
        base_url="https://example.invalid/v1",
        quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli",
    )
    agent.api_mode = "chat_completions"
    return agent


def _final(result):
    return result["final_response"] if isinstance(result, dict) else result


def test_autopilot_continues_then_completes(monkeypatch):
    monkeypatch.setenv("HERMES_AUTOPILOT", "1")
    from agent.autopilot import driver
    from agent.autopilot.council_gate import CompletionVerdict

    agent = _make_agent()
    client = Mock()
    client.chat.completions.create = Mock(side_effect=[
        _completion("Work in progress, partial result."),
        _completion("DONE: the task is fully implemented and verified."),
    ])
    agent.client = client

    calls = {"n": 0}

    def fake_judge(goal, work_summary, final_response, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return CompletionVerdict(complete=False, directive="finish step 2", verdict="deny")
        return CompletionVerdict(complete=True, summary="verified")

    monkeypatch.setattr(driver, "judge_completion", fake_judge)

    result = agent.run_conversation("build the thing")

    # The loop made TWO model calls — it did not stop at the first answer.
    assert client.chat.completions.create.call_count == 2
    assert "DONE" in _final(result)
    # A synthetic autopilot directive was injected between the turns.
    assert any(
        isinstance(m, dict) and m.get("_autopilot_synthetic") for m in result["messages"]
    ), "expected an injected autopilot directive in the transcript"


def test_autopilot_off_delivers_first_answer(monkeypatch):
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    from agent.autopilot import driver

    agent = _make_agent()
    agent.autopilot_mode = False
    client = Mock()
    client.chat.completions.create = Mock(side_effect=[
        _completion("Here is the answer."),
        _completion("SHOULD NOT BE REACHED"),
    ])
    agent.client = client

    # Judge must never be consulted when autopilot is off.
    def fail_judge(*a, **k):
        raise AssertionError("judge_completion must not run when autopilot is off")

    monkeypatch.setattr(driver, "judge_completion", fail_judge)

    result = agent.run_conversation("answer this")
    assert client.chat.completions.create.call_count == 1
    assert "Here is the answer." in _final(result)


def _empty(finish="stop"):
    """A completion with no usable content (model 'choked')."""
    msg = types.SimpleNamespace(
        content=None, tool_calls=None, role="assistant",
        reasoning_content=None, reasoning_details=None, function_call=None,
        refusal=None,
    )
    choice = types.SimpleNamespace(message=msg, finish_reason=finish, index=0)
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10)
    return types.SimpleNamespace(choices=[choice], usage=usage, model="mock", id="cmpl-e")


def test_autopilot_reenters_on_empty_response_abnormal_exit(monkeypatch):
    """Belt-and-suspenders: a turn that exits via the (empty) break (which
    BYPASSES Seam B) must still re-enter under autopilot instead of silently
    stopping mid-goal. The model 'chokes' (empty) until the autopilot re-entry
    injects a directive, then delivers a real answer that the gate verifies."""
    monkeypatch.setenv("HERMES_AUTOPILOT", "1")
    from agent.autopilot import driver
    from agent.autopilot.council_gate import CompletionVerdict

    agent = _make_agent()
    agent._fallback_chain = []
    agent._current_streamed_assistant_text = ""

    state = {"n": 0}

    def model_side_effect(*a, **k):
        state["n"] += 1
        # 4 empties: 3 internal empty-retries + the 4th that hits the (empty)
        # break (the abnormal exit our guard catches). Then a real answer.
        if state["n"] <= 4:
            return _empty()
        return _completion("DONE: fully implemented and verified.")

    client = Mock()
    client.chat.completions.create = Mock(side_effect=model_side_effect)
    agent.client = client

    judged = {"n": 0}

    def fake_judge(goal, work_summary, final_response, **kw):
        judged["n"] += 1
        if "DONE" in (final_response or ""):
            return CompletionVerdict(complete=True, summary="verified")
        return CompletionVerdict(complete=False, directive="finish the task", verdict="deny")

    monkeypatch.setattr(driver, "judge_completion", fake_judge)

    result = agent.run_conversation("build the thing")

    # The loop pushed PAST the empty break (call 5 happened) rather than
    # stopping at "(empty)".
    assert client.chat.completions.create.call_count >= 5
    assert "DONE" in _final(result)
    # The re-entry injected a synthetic autopilot directive.
    assert any(
        isinstance(m, dict) and m.get("_autopilot_synthetic") for m in result["messages"]
    ), "expected an injected autopilot directive after the abnormal (empty) exit"
    # The gate was consulted at the abnormal exit AND at final completion.
    assert judged["n"] >= 2


def test_autopilot_off_stops_at_empty_response(monkeypatch):
    """With autopilot OFF, the (empty) break must still deliver immediately —
    the new guard must not change non-autopilot behavior."""
    monkeypatch.delenv("HERMES_AUTOPILOT", raising=False)
    from agent.autopilot import driver

    agent = _make_agent()
    agent.autopilot_mode = False
    agent._fallback_chain = []
    agent._current_streamed_assistant_text = ""

    client = Mock()
    client.chat.completions.create = Mock(side_effect=[_empty()] * 8)
    agent.client = client

    def fail_judge(*a, **k):
        raise AssertionError("judge must not run when autopilot is off")

    monkeypatch.setattr(driver, "judge_completion", fail_judge)

    result = agent.run_conversation("answer this")
    # 3 empty-retries + the 4th that breaks with "(empty)"; no re-entry.
    assert client.chat.completions.create.call_count == 4
    # The "(empty)" sentinel is surfaced as the user-facing no-reply message.
    assert "No reply" in _final(result) or _final(result) == "(empty)"
    assert not any(
        isinstance(m, dict) and m.get("_autopilot_synthetic") for m in result["messages"]
    )
