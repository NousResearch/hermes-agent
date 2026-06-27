"""Regression tests for the chat_completions intermediate-ack recovery.

Some models (notably quantized local models like Qwen served over an
OpenAI-compatible endpoint) sometimes emit a short "I'll go do that" preamble
and then stop with ``finish_reason=stop`` and NO tool call — the turn ends with
an empty promise and the requested action never happens. Hermes already
recovered from this for Codex (``codex_responses`` api_mode) via
``looks_like_codex_intermediate_ack``; these tests cover extending the same
"Continue now, execute the tools" nudge to the ``chat_completions`` path via
``looks_like_intermediate_tool_ack``.

The integration test reproduces the real-world failure that motivated this:
a "who are my top friends?" chat where the model kept answering
"Let me dig deeper..." / "Let me pull a much broader list and check the RAG
store too." without ever calling a tool.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.agent_runtime_helpers import (
    looks_like_codex_intermediate_ack,
    looks_like_intermediate_tool_ack,
)


# --------------------------------------------------------------------------- #
# Unit tests: detector logic
# --------------------------------------------------------------------------- #

# The detectors only call ``agent._strip_think_blocks``; a tiny stub avoids
# standing up a full AIAgent for the pure-function checks.
_FAKE_AGENT = SimpleNamespace(_strip_think_blocks=lambda s: s)


# Preambles from a real stalled Open WebUI chat plus the variants reproduced
# live against the gateway. The "try again" / "retry"
# group is the degraded vocabulary the model drifts into once its history is
# poisoned with preamble-only turns — it must be caught too, or recovery only
# works until the model changes phrasing.
@pytest.mark.parametrize(
    "preamble",
    [
        # Original reported chat
        "Let me dig deeper — let me check how many people are actually indexed and pull a broader list.",
        "Let me check the memory health and pull a broader list of people.",
        "You're right, I need to actually run these and show you the results. Let me do that now.",
        # Reproduced live against the gateway
        "Let me pull a much broader list and check the RAG store too.",
        # Degraded "try again" / "retry" vocabulary observed in the live replay
        "Let me try again - the calls seem to have failed.",
        "Let me try again - something seems to be failing silently.",
        "Let me try again properly.",
        "Let me try again - looks like those calls may have stalled.",
        "Good question - let me retry these calls. The previous ones may have timed out.",
    ],
)
def test_general_ack_detects_real_stalled_preambles(preamble):
    assert (
        looks_like_intermediate_tool_ack(_FAKE_AGENT, "why are you stopping?", preamble, [])
        is True
    )


def test_general_ack_ignores_substantive_final_answer():
    # A real answer is long and data-bearing — not a stall, even though it
    # happens to contain action words like "check".
    answer = (
        "Here are your top people by email interaction volume:\n"
        + "\n".join(f"- Person {i} (score {20 - i}.0)" for i in range(12))
        + "\nCheck back if you want me to re-rank by recency instead."
    )
    assert (
        looks_like_intermediate_tool_ack(_FAKE_AGENT, "who are my friends?", answer, [])
        is False
    )


def test_general_ack_ignores_clarifying_question():
    # Handing control back to the user with a question is not a stalled action.
    q = "Let me check — do you want friends ranked by email volume or by recency?"
    assert looks_like_intermediate_tool_ack(_FAKE_AGENT, "who are my friends?", q, []) is False


def test_general_ack_requires_future_ack_and_action_verb():
    # No future-tense ack -> not a stall (e.g. a completed-action report).
    assert (
        looks_like_intermediate_tool_ack(
            _FAKE_AGENT, "do it", "Done. I archived the three emails.", []
        )
        is False
    )
    # Future ack but no concrete action verb -> not our signature.
    assert (
        looks_like_intermediate_tool_ack(
            _FAKE_AGENT, "thanks", "Sure, I'll help however I can.", []
        )
        is False
    )
    # "let me know" is a hand-back idiom ("let me" ack but no action verb).
    assert (
        looks_like_intermediate_tool_ack(
            _FAKE_AGENT, "thanks", "Let me know if you need anything else.", []
        )
        is False
    )


def test_general_ack_suppressed_once_a_tool_has_run():
    # If a tool already executed this conversation, a trailing ack is treated
    # as legitimate commentary, not a stall.
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    assert (
        looks_like_intermediate_tool_ack(
            _FAKE_AGENT, "dig deeper", "Let me check one more source.", msgs
        )
        is False
    )


def test_general_ack_does_not_require_workspace_target():
    # The friends stall targets the memory graph / RAG, not a filesystem
    # workspace — the Codex detector (which requires a workspace target) would
    # MISS it, which is exactly why the general detector exists.
    preamble = "Let me pull a much broader list and check the RAG store too."
    assert looks_like_codex_intermediate_ack(_FAKE_AGENT, "go further", preamble, []) is False
    assert looks_like_intermediate_tool_ack(_FAKE_AGENT, "go further", preamble, []) is True


# --------------------------------------------------------------------------- #
# Integration test: the nudge fires through run_conversation (chat_completions)
# --------------------------------------------------------------------------- #


def _mock_response(content="", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="qwen-test", usage=None)


def _mock_tool_call(name="memory", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _build_chat_completions_agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "description": "memory tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    # Target the chat_completions recovery path explicitly.
    agent.api_mode = "chat_completions"
    agent._cleanup_task_resources = lambda task_id: None
    agent._persist_session = lambda messages, history=None: None
    agent._save_trajectory = lambda messages, user_message, completed: None
    return agent


def test_run_conversation_chat_completions_continues_after_ack_stall(monkeypatch):
    agent = _build_chat_completions_agent()
    assert agent.api_mode == "chat_completions"

    # Turn 1: preamble + stop, no tool call (the stall).
    # Turn 2 (after the nudge): a real tool call.
    # Turn 3: the substantive final answer.
    responses = [
        _mock_response(
            content="Let me pull a much broader list and check the RAG store too.",
            finish_reason="stop",
        ),
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call()]),
        _mock_response(content="Here are your top 10 friends: ...", finish_reason="stop"),
    ]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    def _fake_execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count=0):
        for call in assistant_message.tool_calls:
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": '{"people": 25}'}
            )

    monkeypatch.setattr(agent, "_execute_tool_calls", _fake_execute_tool_calls)

    result = agent.run_conversation("are you sure you don't have more? dig deeper")

    assert result["completed"] is True
    assert result["final_response"] == "Here are your top 10 friends: ..."
    # The interim preamble is preserved (as an "incomplete" assistant turn)...
    assert any(
        msg.get("role") == "assistant"
        and msg.get("finish_reason") == "incomplete"
        and "check the RAG store" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    # ...the continuation nudge was injected...
    assert any(
        msg.get("role") == "user"
        and "Continue now. Execute the required tool calls" in (msg.get("content") or "")
        for msg in result["messages"]
    )
    # ...and a tool actually ran.
    assert any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1" for msg in result["messages"])


def test_run_conversation_chat_completions_no_nudge_for_real_answer(monkeypatch):
    """A substantive first answer must not trigger a spurious continuation."""
    agent = _build_chat_completions_agent()

    answer = (
        "Here are your top people:\n"
        + "\n".join(f"- Person {i}" for i in range(12))
    )
    responses = [_mock_response(content=answer, finish_reason="stop")]
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda api_kwargs: responses.pop(0))

    result = agent.run_conversation("who are my top friends?")

    assert result["completed"] is True
    assert result["final_response"] == answer
    assert not any(
        msg.get("role") == "user"
        and "Continue now. Execute the required tool calls" in (msg.get("content") or "")
        for msg in result["messages"]
    )
