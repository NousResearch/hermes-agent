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
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE", "HERMES_KANBAN_RUN_ID"):
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






@pytest.fixture
def claimed_worker(clear_kanban_env, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Review handoff", assignee="builder")
        task = kb.claim_task(conn, tid, claimer="builder:1")
        assert task is not None
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", tid)
    clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task


def test_handed_off_run_stays_finished_after_compaction_and_successor_claim(
    claimed_worker, monkeypatch,
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_worker.id
    with kbc.connect_closing() as conn:
        assert kb.request_review(
            conn, tid, summary="Ready", reviewer="reviewer",
            expected_run_id=claimed_worker.current_run_id,
        )
        assert build_kanban_stop_nudge(messages=[]) is None
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
        # The task is running again, but the original worker has finished.
        assert build_kanban_stop_nudge(messages=[]) is None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None
        assert kb.request_changes(
            conn, tid, reason="Fix boundary case", expected_run_id=review.current_run_id,
        )[0]
        assert build_kanban_stop_nudge(messages=[]) is None
        successor = kb.claim_task(conn, tid, claimer="builder:2")
        assert successor is not None
        assert build_kanban_stop_nudge(messages=[]) is None
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(successor.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None
        assert kb.get_task(conn, tid) == successor


@pytest.mark.parametrize("evidence", ["failed", "other_task", "missing_db", "invalid_run"])
def test_pinned_worker_requires_its_own_durable_handoff(
    claimed_worker, monkeypatch, tmp_path, evidence,
):
    import json

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_worker.id
    missing = tmp_path / "missing.db"
    if evidence == "other_task":
        with kbc.connect_closing() as conn:
            tid = kb.create_task(conn, title="Unrelated", assignee="builder")
            assert kb.claim_task(conn, tid, claimer="other:1") is not None
            assert kb.complete_task(conn, tid, summary="Done")
    elif evidence == "missing_db":
        monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))
    elif evidence == "invalid_run":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "invalid")
    else:
        with kbc.connect_closing() as conn:
            # A stale-run completion attempt must not end the current run.
            assert not kb.complete_task(conn, tid, expected_run_id=-1)

    messages = [
        {"role": "assistant", "tool_calls": [{
            "id": "terminal", "type": "function",
            "function": {"name": "kanban_complete", "arguments": json.dumps({"task_id": tid})},
        }]},
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "terminal",
         "content": json.dumps({"ok": evidence == "other_task", "task_id": tid})},
    ]
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    assert "kanban_request_review" in nudge
    assert "kanban_request_changes" in nudge
    assert build_kanban_stop_nudge(messages=messages, attempts=2) is None
    if evidence == "missing_db":
        assert not missing.exists(), "The stop guard must not initialize a missing board"


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.




