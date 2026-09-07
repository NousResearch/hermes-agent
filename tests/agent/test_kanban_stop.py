"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.






# ── 2026-09-07: the checkpoint machinery is retired ──────────────────


def test_kanban_checkpoint_module_is_gone():
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent.kanban_checkpoint")


def test_no_live_import_of_the_retired_module():
    """Read as source: the loop is 9k lines and importing it for this
    assertion would drag half the agent in. A stale import is what breaks."""
    import pathlib, re, agent
    for rel in ("conversation_loop.py",):
        src = (pathlib.Path(agent.__file__).parent / rel).read_text()
        assert not re.search(r"^\s*from\s+agent\.kanban_checkpoint\s+import", src, re.M)
        assert not re.search(r"^\s*from\s+agent\s+import\s+kanban_checkpoint", src, re.M)
    assert "build_kanban_stop_nudge" in src, "the guard that replaced it must still be wired"


def test_four_tool_terminal_set_is_our_deliberate_divergence():
    """Upstream recognises only {complete, block}. A worker that correctly
    hands off with kanban_request_review has closed its run and must not be
    steered into completing an unreviewed card (Rodge, 2026-09-01)."""
    from agent.kanban_stop import _TERMINAL_KANBAN_TOOLS
    assert _TERMINAL_KANBAN_TOOLS == {
        "kanban_complete", "kanban_block",
        "kanban_request_review", "kanban_request_changes",
    }


def test_nudge_still_fires_at_turn_end_and_is_bounded(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_probe")
    monkeypatch.delenv("HERMES_KANBAN_STOP_NUDGE", raising=False)
    working = [{"role": "user", "content": "go"},
               {"role": "tool", "name": "read_file", "tool_call_id": "1", "content": "ok"}]
    assert build_kanban_stop_nudge(messages=working, attempts=0) is not None
    assert build_kanban_stop_nudge(messages=working, attempts=1) is not None
    assert build_kanban_stop_nudge(messages=working, attempts=2) is None
