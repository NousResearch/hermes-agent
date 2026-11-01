from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _connect(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return kbc.connect_closing(kb.kanban_db_path(board="completion-states"))


def test_green_tests_do_not_implicitly_satisfy_live_verification(tmp_path, monkeypatch):
    """A test result is evidence for ``tested`` only, never ``verified``."""
    with _connect(tmp_path, monkeypatch) as conn:
        task_id = kb.create_task(
            conn,
            title="Aggregate release request",
            requires_live_verification=True,
        )

        assert not kb.complete_task(
            conn,
            task_id,
            summary="tests passed",
            metadata={"tests_passed": 42},
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.completion_state is None


def test_each_completion_state_requires_explicit_evidence_and_never_auto_upgrades(tmp_path, monkeypatch):
    with _connect(tmp_path, monkeypatch) as conn:
        task_id = kb.create_task(conn, title="Release")

        kb.record_completion_state(conn, task_id, "written", {"proof": "commit abc123"})
        task = kb.get_task(conn, task_id)
        assert task.completion_state == "written"

        kb.record_completion_state(conn, task_id, "tested", {"proof": "42 passed"})
        task = kb.get_task(conn, task_id)
        assert task.completion_state == "tested"

        events = kb.list_events(conn, task_id)
        states = [event.payload["state"] for event in events if event.kind == "completion_state_recorded"]
        assert states == ["written", "tested"]
        assert "deployed" not in states
        assert "verified" not in states


def test_live_verification_requires_explicit_verified_evidence(tmp_path, monkeypatch):
    with _connect(tmp_path, monkeypatch) as conn:
        task_id = kb.create_task(
            conn,
            title="Production release",
            requires_live_verification=True,
        )

        kb.record_completion_state(conn, task_id, "tested", {"proof": "42 passed"})
        assert not kb.complete_task(conn, task_id, summary="tests passed")

        kb.record_completion_state(
            conn,
            task_id,
            "verified",
            {"proof": "2026-09-17T20:00Z live endpoint returned expected version"},
        )
        assert kb.complete_task(conn, task_id, summary="live proof recorded")
        assert kb.get_task(conn, task_id).completion_state == "verified"
