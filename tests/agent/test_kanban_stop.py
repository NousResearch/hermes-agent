"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)
from hermes_cli import kanban_db as kb


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_STOP_NUDGE",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def kanban_conn(tmp_path: Path, clear_kanban_env):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def _claim_worker(conn, monkeypatch, title: str = "Guarded worker"):
    task_id = kb.create_task(conn, title=title, assignee="builder")
    task = kb.claim_task(conn, task_id, claimer=f"builder:{title}")
    assert task is not None
    assert task.current_run_id is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task_id, task.current_run_id






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


def test_nudge_while_original_run_is_genuinely_active(
    kanban_conn, clear_kanban_env
):
    task_id, run_id = _claim_worker(
        kanban_conn, clear_kanban_env, "Still active"
    )

    nudge = build_kanban_stop_nudge(messages=[])

    assert nudge is not None
    assert task_id in nudge
    assert kb.get_run(kanban_conn, run_id).outcome is None


def test_no_nudge_for_review_requested_original_run_with_running_successor(
    kanban_conn, clear_kanban_env
):
    task_id, original_run_id = _claim_worker(
        kanban_conn, clear_kanban_env, "Implementation"
    )
    assert kb.request_review(
        kanban_conn,
        task_id,
        reviewer="reviewer",
        summary="ready for review",
        expected_run_id=original_run_id,
    )
    successor = kb.claim_review_task(
        kanban_conn, task_id, claimer="reviewer:successor"
    )
    assert successor is not None
    assert successor.current_run_id != original_run_id

    assert build_kanban_stop_nudge(messages=[]) is None
    assert kb.get_run(kanban_conn, original_run_id).outcome == "review_requested"
    assert (
        kb.get_task(kanban_conn, task_id).current_run_id
        == successor.current_run_id
    )


def test_no_nudge_for_changes_requested_original_run_with_running_successor(
    kanban_conn, clear_kanban_env
):
    task_id = kb.create_task(
        kanban_conn, title="Review changes", assignee="builder"
    )
    implementation = kb.claim_task(
        kanban_conn, task_id, claimer="builder:implementation"
    )
    assert implementation is not None
    assert kb.request_review(
        kanban_conn,
        task_id,
        reviewer="reviewer",
        summary="ready for review",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(
        kanban_conn, task_id, claimer="reviewer:original"
    )
    assert review is not None
    assert kb.request_changes(
        kanban_conn,
        task_id,
        reason="fix the finding",
        expected_run_id=review.current_run_id,
    ) == (True, "builder")
    successor = kb.claim_task(
        kanban_conn, task_id, claimer="builder:successor"
    )
    assert successor is not None
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    assert build_kanban_stop_nudge(messages=[]) is None
    assert (
        kb.get_run(kanban_conn, review.current_run_id).outcome
        == "changes_requested"
    )
    assert (
        kb.get_task(kanban_conn, task_id).current_run_id
        == successor.current_run_id
    )


def test_no_nudge_when_run_id_belongs_to_another_task(
    kanban_conn, clear_kanban_env
):
    task_id, _ = _claim_worker(kanban_conn, clear_kanban_env, "First task")
    other_id = kb.create_task(
        kanban_conn, title="Other task", assignee="builder"
    )
    other = kb.claim_task(kanban_conn, other_id, claimer="builder:other")
    assert other is not None
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(other.current_run_id))

    assert build_kanban_stop_nudge(messages=[]) is None


def test_missing_run_id_preserves_legacy_nudge(
    kanban_conn, clear_kanban_env
):
    task_id, _ = _claim_worker(kanban_conn, clear_kanban_env, "Legacy worker")
    clear_kanban_env.delenv("HERMES_KANBAN_RUN_ID")

    nudge = build_kanban_stop_nudge(messages=[])

    assert nudge is not None
    assert task_id in nudge






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




