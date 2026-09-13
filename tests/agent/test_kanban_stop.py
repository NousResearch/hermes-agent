"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)
from hermes_cli import kanban_db, kanban_db_connect


def _create_task(monkeypatch, db_path, *, status, current_run_id, task_id=None):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kanban_db_connect.connect_closing() as conn:
        created_id = kanban_db.create_task(conn, title="Stop guard task", assignee="default")
        task_id = task_id or created_id
        conn.execute(
            "UPDATE tasks SET id = ?, status = ?, current_run_id = ? WHERE id = ?",
            (task_id, status, current_run_id, created_id),
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    return task_id


@pytest.fixture
def clear_kanban_env(monkeypatch, tmp_path):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_RUN_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    _create_task(
        monkeypatch,
        tmp_path / "kanban.db",
        status="running",
        current_run_id=42,
        task_id="t_46be8aa5",
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    return monkeypatch


@pytest.fixture
def kanban_task(clear_kanban_env, tmp_path):
    def create(*, status, current_run_id):
        return _create_task(
            clear_kanban_env,
            tmp_path / "kanban.db",
            status=status,
            current_run_id=current_run_id,
        )

    return create






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


def test_blocked_card_suppresses_nudge(kanban_task):
    kanban_task(status="blocked", current_run_id=None)
    assert build_kanban_stop_nudge(messages=[]) is None


def test_done_card_suppresses_nudge(kanban_task):
    kanban_task(status="done", current_run_id=42)
    assert build_kanban_stop_nudge(messages=[]) is None


def test_matching_running_card_gets_nudge(kanban_task, monkeypatch):
    task_id = kanban_task(status="running", current_run_id=42)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")

    nudge = build_kanban_stop_nudge(messages=[])

    assert nudge is not None
    assert task_id in nudge
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge


def test_run_mismatch_suppresses_nudge(kanban_task, monkeypatch):
    kanban_task(status="running", current_run_id=99)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    assert build_kanban_stop_nudge(messages=[]) is None


def test_unparseable_run_id_does_not_suppress_nudge(kanban_task, monkeypatch):
    kanban_task(status="running", current_run_id=42)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "invalid")
    assert build_kanban_stop_nudge(messages=[]) is not None


def test_missing_card_suppresses_nudge(clear_kanban_env, tmp_path):
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "empty.db"))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_missing")
    assert build_kanban_stop_nudge(messages=[]) is None


def test_db_lookup_failure_suppresses_nudge(clear_kanban_env, monkeypatch):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_failure")

    def fail_connect():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(kanban_db_connect, "connect_closing", fail_connect)
    assert build_kanban_stop_nudge(messages=[]) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.



