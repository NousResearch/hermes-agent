"""Regression tests for #27178 — kanban worker checkpoint reminder.

When a dispatcher-spawned kanban worker exits with a text response
(finish_reason=stop) without having called kanban_complete or
kanban_block, the dispatcher flags it as protocol_violation and
immediately trips the failure breaker. Many of these cases are the
agent having done substantive work but forgetting to checkpoint
(reporter @Jonnybooyy's "Concern 3" repro shape in #27178).

The fix injects ONE synthetic user-role reminder before the worker
exits, giving the agent a chance to call kanban_complete/block on
a final turn. Capped at one nudge per turn so a non-compliant agent
falls through to the existing protocol_violation path.

These tests are intentionally light: the main contract is exercised
via a source-level invariant check (catches refactors that drop the
nudge) plus a message-list unit test that exercises the
detection logic directly. A full end-to-end run_conversation test
would require mocking the LLM client, which is heavyweight and
covered by the broader run_agent test suite.
"""
from __future__ import annotations

import pathlib

import pytest


def test_conversation_loop_has_kanban_checkpoint_reminder():
    """Source-level invariant: agent/conversation_loop.py must contain
    the kanban-worker checkpoint nudge block. Catches refactors that
    drop the env-gated reminder injection.
    """
    cl_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "agent" / "conversation_loop.py"
    )
    src = cl_path.read_text()
    # The nudge must be env-gated on HERMES_KANBAN_TASK so normal chat
    # sessions never see it.
    assert "HERMES_KANBAN_TASK" in src, (
        "conversation_loop.py must reference HERMES_KANBAN_TASK to gate "
        "the kanban-worker checkpoint nudge — see #27178"
    )
    # The reminder block must check for prior kanban_complete or
    # kanban_block calls before injecting (idempotency / don't nudge
    # tasks that already checkpointed).
    assert "kanban_complete" in src and "kanban_block" in src, (
        "conversation_loop.py must check for kanban_complete / "
        "kanban_block in messages before injecting reminder"
    )
    # The cap-at-one-nudge flag is what prevents reminder loops.
    assert "_kanban_checkpoint_reminder_sent" in src, (
        "conversation_loop.py must track _kanban_checkpoint_reminder_sent "
        "to cap the nudge at one per turn — see #27178"
    )


def test_kanban_terminal_call_detection_logic():
    """Verify the message-scan logic correctly identifies whether a
    worker has already called kanban_complete or kanban_block.

    This mirrors the inline scan in conversation_loop.py. If the scan
    pattern there changes, this test should be updated to match.
    """
    # Case 1: messages with a kanban_complete tool call → detected
    msgs_with_complete = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "kanban_complete",
                          "arguments": '{"task_id": "t_x"}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "kanban_complete",
         "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    assert _scan_for_kanban_terminal(msgs_with_complete) is True

    # Case 2: messages with kanban_block → detected
    msgs_with_block = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "tool_calls": [
            {"id": "b1", "type": "function",
             "function": {"name": "kanban_block",
                          "arguments": '{"task_id": "t_x", "reason": "need input"}'}}
        ]},
        {"role": "tool", "tool_call_id": "b1", "name": "kanban_block",
         "content": "ok"},
        {"role": "assistant", "content": "blocked"},
    ]
    assert _scan_for_kanban_terminal(msgs_with_block) is True

    # Case 3: only non-terminal kanban tool calls (kanban_show, heartbeat,
    # comment) — not detected. These are NOT terminal.
    msgs_non_terminal_only = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "tool_calls": [
            {"id": "s1", "type": "function",
             "function": {"name": "kanban_show", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "s1", "content": "..."},
        {"role": "assistant", "tool_calls": [
            {"id": "h1", "type": "function",
             "function": {"name": "kanban_heartbeat",
                          "arguments": '{"task_id": "t_x"}'}}
        ]},
        {"role": "tool", "tool_call_id": "h1", "content": "ok"},
        {"role": "assistant", "content": "I'm done writing the file"},
    ]
    assert _scan_for_kanban_terminal(msgs_non_terminal_only) is False

    # Case 4: mixed — kanban_show + kanban_complete → detected
    msgs_mixed = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "tool_calls": [
            {"id": "s1", "type": "function",
             "function": {"name": "kanban_show", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "s1", "content": "..."},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "kanban_complete",
                          "arguments": '{"task_id": "t_x"}'}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "complete"},
    ]
    assert _scan_for_kanban_terminal(msgs_mixed) is True

    # Case 5: empty / no tool calls → not detected
    msgs_empty = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "I think it's done"},
    ]
    assert _scan_for_kanban_terminal(msgs_empty) is False

    # Case 6: tool call with None arguments dict shouldn't crash
    msgs_malformed = [
        {"role": "assistant", "tool_calls": [None, "garbage",
                                             {"function": None}]},
        {"role": "assistant", "content": "done"},
    ]
    assert _scan_for_kanban_terminal(msgs_malformed) is False


def _scan_for_kanban_terminal(messages: list) -> bool:
    """Reimplementation of the message-scan logic in conversation_loop.py.

    Kept here as a test-side copy because the production scan is inlined
    inside run_conversation and not easily importable. If the production
    scan changes (e.g. adds more terminal tool names), update this too.
    """
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        for tc in (m.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = (tc.get("function") or {}).get("name", "")
            if fn in ("kanban_complete", "kanban_block"):
                return True
    return False
