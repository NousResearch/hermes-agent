"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    record_context_recovery_exhausted,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACE",
        "HERMES_SESSION_ID",
    ):
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


def test_recovery_fallback_is_kanban_only(clear_kanban_env):
    assert record_context_recovery_exhausted(messages=[], attempts=1) is False


def test_recovery_fallback_rejects_non_dispatcher_context(clear_kanban_env):
    from agent.delegation_context import non_dispatcher_owned_context

    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", "1")
    with non_dispatcher_owned_context():
        assert record_context_recovery_exhausted(
            messages=[],
            attempts=1,
        ) is False


def _claimed_task(tmp_path, *, goal_mode=False):
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path=db_path)
    task_id = kb.create_task(
        conn,
        title="terminal recovery",
        assignee="worker",
        goal_mode=goal_mode,
    )
    claimed = kb.claim_task(conn, task_id, claimer="worker:fixture")
    assert claimed is not None
    run_id = claimed.current_run_id
    conn.close()
    return kb, db_path, task_id, run_id


@pytest.mark.parametrize("terminal_tool", ["kanban_complete", "kanban_block"])
def test_recovery_fallback_ignores_committed_terminal_state(
    clear_kanban_env, tmp_path, terminal_tool,
):
    kb, db_path, task_id, run_id = _claimed_task(tmp_path)
    conn = kb.connect(db_path=db_path)
    try:
        if terminal_tool == "kanban_complete":
            assert kb.complete_task(
                conn,
                task_id,
                result="done",
                expected_run_id=run_id,
            )
            expected_status = "done"
            expected_outcome = "completed"
        else:
            assert kb.block_task(
                conn,
                task_id,
                reason="explicit block",
                expected_run_id=run_id,
            )
            expected_status = "blocked"
            expected_outcome = "blocked"
    finally:
        conn.close()

    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": terminal_tool, "arguments": "{}"},
                }
            ],
        }
    ]

    assert record_context_recovery_exhausted(
        messages=messages,
        attempts=1,
    ) is False

    conn = kb.connect(db_path=db_path)
    try:
        assert kb.get_task(conn, task_id).status == expected_status
        assert kb.latest_run(conn, task_id).outcome == expected_outcome
    finally:
        conn.close()


def test_recovery_fallback_blocks_after_rejected_terminal_tool_call(
    clear_kanban_env, tmp_path,
):
    kb, db_path, task_id, run_id = _claimed_task(tmp_path)
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
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
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "1",
            "content": "completion rejected; task remains in-flight",
        },
    ]

    assert session_called_kanban_terminal(messages) is True
    assert record_context_recovery_exhausted(
        messages=messages,
        attempts=1,
    ) is True

    conn = kb.connect(db_path=db_path)
    try:
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.latest_run(conn, task_id).outcome == "blocked"
    finally:
        conn.close()


@pytest.mark.parametrize("goal_mode", [False, True])
def test_recovery_fallback_records_run_fenced_diagnostic_block(
    clear_kanban_env, tmp_path, goal_mode,
):
    kb, db_path, task_id, run_id = _claimed_task(
        tmp_path,
        goal_mode=goal_mode,
    )
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    clear_kanban_env.setenv("HERMES_SESSION_ID", "session-fixture")
    clear_kanban_env.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))

    assert record_context_recovery_exhausted(
        messages=[{"role": "assistant", "content": "work is done"}],
        attempts=1,
    ) is True

    conn = kb.connect(db_path=db_path)
    try:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task.status == "blocked"
        assert run.id == run_id
        assert run.outcome == "blocked"
        assert "context overflow" in (run.summary or "").lower()
        assert "clearing gate" in (run.summary or "").lower()
        assert not any(event.kind == "completed" for event in kb.list_events(conn, task_id))
    finally:
        conn.close()


def test_recovery_fallback_cannot_mutate_successor_run(
    clear_kanban_env, tmp_path,
):
    kb, db_path, task_id, stale_run_id = _claimed_task(tmp_path)
    conn = kb.connect(db_path=db_path)
    try:
        assert kb.block_task(
            conn,
            task_id,
            reason="first run ended",
            expected_run_id=stale_run_id,
        )
        assert kb.unblock_task(conn, task_id)
        successor = kb.claim_task(conn, task_id, claimer="worker:successor")
        assert successor is not None
        assert successor.current_run_id != stale_run_id
    finally:
        conn.close()

    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", task_id)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(stale_run_id))

    assert record_context_recovery_exhausted(
        messages=[{"role": "assistant", "content": "stale run"}],
        attempts=1,
    ) is False

    conn = kb.connect(db_path=db_path)
    try:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert task.status == "running"
        assert run.id == successor.current_run_id
        assert run.outcome is None
    finally:
        conn.close()






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




