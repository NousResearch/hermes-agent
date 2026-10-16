"""Run-scoped regression coverage for the kanban stop guard."""

from __future__ import annotations

import json

import pytest

from agent.kanban_stop import build_kanban_stop_nudge


@pytest.fixture
def claimed_worker(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_DB",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="Review handoff", assignee="builder")
        task = kb.claim_task(conn, tid, claimer="builder:1")
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task


def test_handed_off_run_stays_finished_after_compaction_and_successor_claim(
    claimed_worker, monkeypatch
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = claimed_worker.id
    with kbc.connect_closing() as conn:
        assert kb.request_review(
            conn,
            tid,
            summary="Ready",
            reviewer="reviewer",
            expected_run_id=claimed_worker.current_run_id,
        )
        assert build_kanban_stop_nudge(messages=[]) is None

        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
        # The card is running again, but the original worker has handed it off.
        assert build_kanban_stop_nudge(messages=[]) is None

        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None
        assert kb.request_changes(
            conn,
            tid,
            reason="Fix boundary case",
            expected_run_id=review.current_run_id,
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
    claimed_worker, monkeypatch, tmp_path, evidence
):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    missing = tmp_path / "missing.db"
    tool_task_id = claimed_worker.id
    if evidence == "other_task":
        with kbc.connect_closing() as conn:
            tool_task_id = kb.create_task(conn, title="Unrelated", assignee="builder")
            assert kb.claim_task(conn, tool_task_id, claimer="other:1") is not None
            assert kb.complete_task(conn, tool_task_id, summary="Done")
    elif evidence == "missing_db":
        monkeypatch.setenv("HERMES_KANBAN_DB", str(missing))
    elif evidence == "invalid_run":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "invalid")
    else:
        with kbc.connect_closing() as conn:
            assert not kb.complete_task(
                conn, claimed_worker.id, expected_run_id=-1, summary="Attempt from the wrong run"
            )

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "terminal",
                    "type": "function",
                    "function": {
                        "name": "kanban_complete",
                        "arguments": json.dumps({"task_id": tool_task_id}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "kanban_complete",
            "tool_call_id": "terminal",
            "content": json.dumps(
                {"ok": evidence == "other_task", "task_id": tool_task_id}
            ),
        },
    ]
    nudge = build_kanban_stop_nudge(messages=messages)
    assert nudge is not None
    assert "kanban_request_review" in nudge
    assert "kanban_request_changes" in nudge
    assert build_kanban_stop_nudge(messages=messages, attempts=2) is None
    if evidence == "missing_db":
        assert not missing.exists(), "The stop guard must not initialize a missing board"
