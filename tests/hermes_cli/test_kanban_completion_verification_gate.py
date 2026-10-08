"""Completion gate integration: reject before done and persist accepted evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def _valid_verification() -> dict:
    return {
        "task": {
            "status": "done",
            "summary": "implemented",
            "evidence": [{"command": "pytest", "exit_code": 0, "output": "passed"}],
        },
        "repo": {
            "dirty": False,
            "commit": "abc1234",
            "pushed": True,
            "pr": {"number": 1, "state": "open"},
            "ci": {"status": "success"},
            "review": {"approved": True},
        },
        "preflight": {
            "requested_skills": [],
            "available_skills": [],
            "review_checks": [],
            "supported_flags": [],
            "dependency_copies": 0,
            "max_dependency_copies": 0,
        },
        "scope": {"changed_paths": [], "allowed_roots": [], "secret_findings": []},
    }


def _running_task(conn) -> str:
    tid = kb.create_task(conn, title="gated completion", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def test_rejected_completion_remains_running_and_records_failure(conn, monkeypatch):
    monkeypatch.setenv("HERMES_COMPLETION_GATE", "1")
    tid = _running_task(conn)
    payload = _valid_verification()
    payload["repo"]["dirty"] = True

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="blocked", metadata={"verification": payload})

    assert error.value.report["ok"] is False
    assert kb.get_task(conn, tid).status == "running"
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    assert events[-1]["kind"] == "completion_blocked_verification"
    assert "unsaved_work" in events[-1]["payload"]


def test_accepted_completion_preserves_stable_verification_report(conn, monkeypatch):
    monkeypatch.setenv("HERMES_COMPLETION_GATE", "1")
    tid = _running_task(conn)
    payload = _valid_verification()

    assert kb.complete_task(conn, tid, result="shipped", metadata={"verification": payload}) is True

    assert kb.get_task(conn, tid).status == "done"
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert run is not None
    assert '"ok": true' in run["metadata"]
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert event is not None
    assert '"verification_report"' in event["payload"]
