"""Run-bound completion-evidence gates for kanban task completion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn):
    task_id = kb.create_task(conn, title="child", assignee="worker")
    task = kb.claim_task(conn, task_id)
    assert task is not None
    assert task.current_run_id is not None
    return task_id, task.current_run_id


def _evidence(base_task_id: str, base_run_id: int, **overrides):
    evidence = {
        "receipt_id": 41,
        "source": "verification_evidence",
        "status": "passed",
        "task_id": base_task_id,
        "run_id": base_run_id,
        "session_id": "session-1",
        "root": "/repo",
    }
    evidence.update(overrides)
    return evidence


def test_matching_worker_prose_without_evidence_cannot_complete(kanban_home):
    """A perfect success claim is not independent acceptance evidence."""
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        with pytest.raises(kb.CompletionEvidenceError):
            kb.complete_task(
                conn,
                task_id,
                summary="All required checks passed and the feature is complete.",
                expected_run_id=run_id,
                require_completion_evidence=True,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_passed_evidence_bound_to_exact_task_and_run_completes(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        assert kb.complete_task(
            conn,
            task_id,
            summary="Implemented and verified.",
            expected_run_id=run_id,
            require_completion_evidence=True,
            completion_evidence=_evidence(task_id, run_id),
        )
        assert kb.get_task(conn, task_id).status == "done"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()
        assert event is not None
        receipt = json.loads(event["payload"])["completion_evidence"]
        assert receipt["receipt_id"] == 41
        assert receipt["run_id"] == run_id


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"task_id": "t_wrong"}, "task_mismatch"),
        ({"run_id": 999999}, "run_mismatch"),
        ({"status": "failed"}, "not_passed"),
        ({"receipt_id": None}, "missing_receipt"),
    ],
)
def test_stale_or_invalid_evidence_cannot_complete(kanban_home, overrides, reason):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        with pytest.raises(kb.CompletionEvidenceError) as exc_info:
            kb.complete_task(
                conn,
                task_id,
                summary="Implemented and verified.",
                expected_run_id=run_id,
                require_completion_evidence=True,
                completion_evidence=_evidence(task_id, run_id, **overrides),
            )
        assert exc_info.value.reason == reason
        assert kb.get_task(conn, task_id).status == "running"


def test_evidence_rejection_is_audited_without_storing_worker_prose(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        with pytest.raises(kb.CompletionEvidenceError):
            kb.complete_task(
                conn,
                task_id,
                summary="Untrusted secret-looking worker prose",
                expected_run_id=run_id,
                require_completion_evidence=True,
            )
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, "completion_blocked_missing_evidence"),
        ).fetchone()
        assert event is not None
        assert "Untrusted secret-looking" not in event["payload"]


def test_completion_without_evidence_requirement_is_backward_compatible(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _running_task(conn)
        assert kb.complete_task(conn, task_id, result="done", expected_run_id=run_id)
