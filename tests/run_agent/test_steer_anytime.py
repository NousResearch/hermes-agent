"""Regression tests for the steer-anytime fix (system-note fallback).

Problem: a /steer queued while the agent is mid-long-turn (the message tail is a
user or assistant message, not a tool result, and the agent is sitting inside one
long-running tool) could not inject. The loop-top drain only attached steers to a
``role:"tool"`` message, and if none was in the tail it re-parked the steer and
waited for a future tool batch — which during a multi-hour unattended turn could
starve for a very long time.

Fix: when the loop-top drain finds no tool message to attach to, it surfaces the
steer as a one-shot ephemeral SYSTEM note (``_pending_steer_systemnote``) folded
into the next API call's system message and then cleared. System is a separate
role, so this never breaks user/assistant alternation, and the steer lands on the
very next iteration regardless of message-tail shape.

These tests model the two code blocks in conversation_loop.py directly (the
drain-fallback that sets the note, and the system-message fold that consumes it),
since the full loop is integration-level.
"""

import threading
import types

from agent.prompt_builder import format_steer_marker


def _bare_agent():
    agent = types.SimpleNamespace()
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer_systemnote = None
    return agent


def _drain(agent):
    with agent._pending_steer_lock:
        text = agent._pending_steer
        agent._pending_steer = None
    return text


def _loop_top_drain_no_tool_message(agent, messages):
    """Mirror conversation_loop.py: drain, try to attach to a tool msg in the
    tail, and on failure stash a one-shot system note instead of re-parking."""
    pre = _drain(agent)
    if not pre:
        return
    injected = False
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "tool":
            m["content"] = (m.get("content", "") or "") + format_steer_marker(pre)
            injected = True
            break
    if not injected:
        note = format_steer_marker(pre).strip()
        existing = agent._pending_steer_systemnote
        agent._pending_steer_systemnote = (existing + "\n" + note) if existing else note


def _build_system_message(agent, base_system):
    """Mirror conversation_loop.py: fold a pending system note into the system
    message and clear it (one-shot)."""
    effective = base_system or ""
    note = agent._pending_steer_systemnote
    if note:
        effective = (effective + "\n\n" + note).strip()
        agent._pending_steer_systemnote = None
    return effective


def test_steer_with_tool_in_tail_attaches_to_tool_not_systemnote():
    agent = _bare_agent()
    agent._pending_steer = "check the redis config"
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "result", "tool_call_id": "a"},
    ]
    _loop_top_drain_no_tool_message(agent, messages)
    # Attached to the tool message; no system note needed.
    assert "check the redis config" in messages[-1]["content"]
    assert agent._pending_steer_systemnote is None
    assert agent._pending_steer is None


def test_steer_with_no_tool_in_tail_becomes_systemnote():
    agent = _bare_agent()
    agent._pending_steer = "you can install boto3 if you need it"
    # Tail is an assistant message (mid-long-turn, no tool result to attach to).
    messages = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "working on it"},
    ]
    _loop_top_drain_no_tool_message(agent, messages)
    # The steer did NOT silently re-park; it became a one-shot system note.
    assert agent._pending_steer is None
    assert agent._pending_steer_systemnote is not None
    assert "boto3" in agent._pending_steer_systemnote
    # Messages were not mutated (no role-alternation break).
    assert messages[-1]["content"] == "working on it"


def test_systemnote_folds_into_system_message_and_clears():
    agent = _bare_agent()
    agent._pending_steer_systemnote = format_steer_marker("install boto3").strip()
    sysmsg = _build_system_message(agent, "You are a helpful agent.")
    assert "You are a helpful agent." in sysmsg
    assert "install boto3" in sysmsg
    # One-shot: cleared after folding so it lands exactly once.
    assert agent._pending_steer_systemnote is None


def test_systemnote_lands_within_one_iteration_end_to_end():
    agent = _bare_agent()
    agent._pending_steer = "feed me this unblocking info"
    messages = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "deep in a long turn"},
    ]
    # Iteration N: drain finds no tool msg -> system note.
    _loop_top_drain_no_tool_message(agent, messages)
    # Same iteration: building the system message surfaces it.
    sysmsg = _build_system_message(agent, "base")
    assert "feed me this unblocking info" in sysmsg
    assert agent._pending_steer_systemnote is None


def test_multiple_steers_concatenate_into_systemnote():
    agent = _bare_agent()
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    agent._pending_steer = "first"
    _loop_top_drain_no_tool_message(agent, messages)
    agent._pending_steer = "second"
    _loop_top_drain_no_tool_message(agent, messages)
    note = agent._pending_steer_systemnote
    assert "first" in note and "second" in note


def test_no_steer_no_systemnote():
    agent = _bare_agent()
    messages = [{"role": "assistant", "content": "y"}]
    _loop_top_drain_no_tool_message(agent, messages)
    assert agent._pending_steer_systemnote is None
    sysmsg = _build_system_message(agent, "base")
    assert sysmsg == "base"
