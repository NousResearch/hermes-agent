"""Loop-level integration test for the leaked-tool-call detect+retry guard.

Drives the REAL ``run_conversation`` loop (real transport normalization, real
turn-context reset, real scaffolding cleanup) and mocks only the network call
seam ``AIAgent._interruptible_api_call``. Proves the full lifecycle:

  turn 1: model leaks a complete <invoke> block as plain text (no tool_calls)
          → guard fires, NOT surfaced, re-prompt appended, loop continues
  turn 2: model does it again
          → guard fires again (budget now exhausted at 2/2)
  turn 3: model returns a normal answer
          → delivered as the final response; NO synthetic scaffolding persists

Also proves the budget cap: if the model leaks forever, the loop does not spin
indefinitely — after the 2-retry budget it surfaces the raw text honestly
rather than looping (anti-deadlock contract).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import run_agent
from run_agent import AIAgent


# ── OpenAI ChatCompletion fakes (real transport normalizes these) ──────────

def _completion(content=None, tool_calls=None, finish_reason="stop"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        refusal=None,
    )
    choice = SimpleNamespace(index=0, message=msg, finish_reason=finish_reason)
    return SimpleNamespace(
        id="cmpl-test",
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test/model",
    )


_LEAK_TEXT = (
    'course\n<invoke name="terminal">'
    '<parameter name="command">ls</parameter></invoke>'
)


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-7890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    # Avoid real streaming; force the non-streaming dispatch branch.
    agent._disable_streaming = True
    return agent


def _run_turn(agent, responses):
    """Drive one full run_conversation turn with a queued response list."""
    seq = list(responses)

    def _fake_api_call(api_kwargs, *a, **kw):
        return seq.pop(0)

    with (
        patch.object(AIAgent, "_interruptible_api_call", side_effect=_fake_api_call),
        patch.object(AIAgent, "_interruptible_streaming_api_call", side_effect=_fake_api_call),
    ):
        return agent.run_conversation(user_message="please list the directory")


# ── Tests ──────────────────────────────────────────────────────────────────

def test_leak_then_real_answer_converges_without_surfacing_xml():
    """Two leaks followed by a real answer → user sees only the real answer,
    and no <invoke> XML or synthetic scaffolding remains in the transcript."""
    agent = _make_agent()
    result = _run_turn(agent, [
        _completion(content=_LEAK_TEXT),        # turn 1: leak  → retry 1/2
        _completion(content=_LEAK_TEXT),        # turn 2: leak  → retry 2/2
        _completion(content="The directory contains a, b and c."),  # turn 3: real
    ])

    final = result.get("final_response") or ""
    assert "<invoke" not in final, f"leaked XML surfaced to user: {final!r}"
    assert final.strip() == "The directory contains a, b and c."

    # The retry budget was consumed exactly twice.
    assert agent._leaked_tool_call_retries == 2

    # No synthetic recovery scaffolding survives in the persisted messages.
    msgs = result.get("messages") or []
    assert not any(
        isinstance(m, dict) and m.get("_leaked_tool_call_synthetic") for m in msgs
    ), "synthetic leak-recovery scaffolding leaked into durable transcript"
    # And the raw leaked XML is not parked in any assistant message content.
    assert not any(
        isinstance(m, dict)
        and m.get("role") == "assistant"
        and "<invoke" in (m.get("content") or "")
        for m in msgs
    )


def test_budget_exhaustion_surfaces_text_instead_of_looping():
    """If the model leaks on every turn, the loop must NOT spin forever — once
    the 2-retry budget is spent it surfaces the (raw) text honestly and ends."""
    agent = _make_agent()
    # Provide MANY leaks; a correct implementation calls the API at most
    # 3 times (initial + 2 retries) before giving up. Provide a few extra so
    # an accidental infinite loop would still terminate the test via pop()
    # raising IndexError rather than hanging.
    result = _run_turn(agent, [_completion(content=_LEAK_TEXT) for _ in range(6)])

    # Budget capped at 2 — the guard stopped re-prompting.
    assert agent._leaked_tool_call_retries == 2
    # The turn ended (did not raise / hang). A final response exists.
    assert "final_response" in result


def test_normal_answer_never_triggers_guard():
    """A plain answer with no tool-call XML must pass straight through with the
    retry budget untouched."""
    agent = _make_agent()
    result = _run_turn(agent, [
        _completion(content="Here is the summary you asked for. All good."),
    ])
    assert agent._leaked_tool_call_retries == 0
    assert result.get("final_response", "").strip() == (
        "Here is the summary you asked for. All good."
    )


def test_real_tool_call_is_unaffected():
    """A genuine tool_calls response must execute normally — the guard only
    looks at the no-tool-calls branch, so a real call is never intercepted."""
    agent = _make_agent()
    real_tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'),
        extra_content=None,
    )
    with patch.object(AIAgent, "_execute_tool_calls", return_value=None) as exec_mock:
        _run_turn(agent, [
            _completion(tool_calls=[real_tc], finish_reason="tool_calls"),
            _completion(content="Done — listed the directory."),
        ])
    assert exec_mock.called, "a real tool_calls response must reach _execute_tool_calls"
    assert agent._leaked_tool_call_retries == 0
