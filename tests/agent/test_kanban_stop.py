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
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env, tmp_path):
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
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


# ── Live board-status suppression (stale-nag fix) ────────────────────
#
# The guard used to fire on HERMES_KANBAN_TASK alone. A session that
# inherited the task env from an already-terminal card (e.g. a child
# process spawned by a worker, or a session resumed after the task
# completed) was nagged to "call kanban_complete / kanban_block" against
# a done card — the nudge's "still `running`" claim was a hardcoded
# template string, never checked against the board. These tests pin the
# fix: the nudge is suppressed ONLY on a positive live read of a
# terminal status, and fails open (nudge still fires) whenever the
# board cannot be consulted.


def _make_board(clear_kanban_env, tmp_path, status):
    """Create a real board with one task in ``status``; return its id."""
    from hermes_cli import kanban_db

    db = tmp_path / "board.db"
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db))
    conn = kanban_db.connect()
    try:
        tid = kanban_db.create_task(
            conn, title="probe", assignee="w", initial_status="running",
        )
        if status != "running":
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?", (status, tid),
                )
    finally:
        conn.close()
    return tid


@pytest.mark.parametrize("terminal", ["done", "archived"])
def test_no_nudge_when_task_is_terminal_live(
    clear_kanban_env, tmp_path, terminal
):
    tid = _make_board(clear_kanban_env, tmp_path, terminal)
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", tid)
    # No terminal tool call in-session: the OLD code nagged here.
    assert build_kanban_stop_nudge(messages=[], attempts=0) is None


def test_nudge_still_fires_when_task_is_running_live(clear_kanban_env, tmp_path):
    tid = _make_board(clear_kanban_env, tmp_path, "running")
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", tid)
    nudge = build_kanban_stop_nudge(messages=[], attempts=0)
    assert nudge is not None
    assert tid in nudge


def test_nudge_fails_open_when_board_unreadable(clear_kanban_env, tmp_path):
    # A directory is not a database: connect() must raise, and the
    # guard must fall back to nudging (never over-suppress on error).
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_zzz")
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None


def test_nudge_fails_open_when_task_missing(clear_kanban_env, tmp_path):
    from hermes_cli import kanban_db

    db = tmp_path / "board.db"
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db))
    conn = kanban_db.connect()
    conn.close()
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_nope")
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.
