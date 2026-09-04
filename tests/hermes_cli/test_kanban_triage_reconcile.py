"""Administrative reconciliation of resolved triage tasks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_reconcile_triage_task_marks_done_and_records_admin_event(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="verified prerequisite")
        assert kb.complete_task(conn, parent_id)
        task_id = kb.create_task(
            conn,
            title="resolved elsewhere",
            body="original evidence",
            assignee="specifier",
            created_by="operator-import",
            workspace_kind="dir",
            workspace_path=str(kanban_home),
            parents=[parent_id],
            triage=True,
        )
        kb.add_comment(conn, task_id, "alice", "keep this history")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, status, started_at, ended_at, outcome, summary) "
                "VALUES (?, 'legacy-worker', 'done', 1, 2, 'completed', 'old run')",
                (task_id,),
            )

        outcome = kb.reconcile_triage_task(
            conn,
            task_id,
            reason="Underlying deployment was verified separately.",
            operator="admin-profile",
        )

        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        comments = kb.list_comments(conn, task_id)
        runs = kb.list_runs(conn, task_id)
        parents = kb.parent_ids(conn, task_id)

    assert outcome == "reconciled"
    assert task.status == "done"
    assert task.completed_at is not None
    assert task.assignee == "specifier"
    assert task.body == "original evidence"
    assert task.created_by == "operator-import"
    assert task.workspace_kind == "dir"
    assert task.workspace_path == str(kanban_home)
    assert parents == [parent_id]
    assert [comment.body for comment in comments] == ["keep this history"]
    assert [(run.profile, run.summary) for run in runs] == [("legacy-worker", "old run")]
    event = events[-1]
    assert event.kind == "administratively_reconciled"
    assert event.payload == {
        "operator": "admin-profile",
        "reason": "Underlying deployment was verified separately.",
        "from_status": "triage",
        "to_status": "done",
    }


def test_reconcile_triage_task_is_idempotent(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)

        first = kb.reconcile_triage_task(
            conn, task_id, reason="Verified externally.", operator="admin"
        )
        second = kb.reconcile_triage_task(
            conn, task_id, reason="Verified externally.", operator="admin"
        )
        events = [
            event
            for event in kb.list_events(conn, task_id)
            if event.kind == "administratively_reconciled"
        ]

    assert first == "reconciled"
    assert second == "already_reconciled"
    assert len(events) == 1


@pytest.mark.parametrize("status", ["ready", "blocked", "review", "archived"])
def test_reconcile_triage_task_rejects_other_source_states(kanban_home, status):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="wrong state", triage=True)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))

        with pytest.raises(ValueError, match=f"cannot reconcile task {task_id} from {status}"):
            kb.reconcile_triage_task(
                conn, task_id, reason="Verified externally.", operator="admin"
            )

        assert kb.get_task(conn, task_id).status == status


def test_reconcile_triage_task_rejects_open_parent(kanban_home):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="open parent")
        task_id = kb.create_task(
            conn, title="resolved child", parents=[parent_id], triage=True
        )

        with pytest.raises(ValueError, match="parent dependencies are not terminal"):
            kb.reconcile_triage_task(
                conn, task_id, reason="Verified externally.", operator="admin"
            )

        assert kb.get_task(conn, task_id).status == "triage"


def test_reconcile_triage_task_rejects_live_run(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="inconsistent active task", triage=True)
        with kb.write_txn(conn):
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'worker', 'running', 1)",
                (task_id,),
            )
            conn.execute(
                "UPDATE tasks SET current_run_id = ?, claim_lock = 'live' WHERE id = ?",
                (cur.lastrowid, task_id),
            )

        with pytest.raises(ValueError, match="active worker or run"):
            kb.reconcile_triage_task(
                conn, task_id, reason="Verified externally.", operator="admin"
            )

        assert kb.get_task(conn, task_id).status == "triage"


@pytest.mark.parametrize("field", ["reason", "operator"])
def test_reconcile_triage_task_requires_audit_identity_and_reason(kanban_home, field):
    kwargs = {"reason": "Verified externally.", "operator": "admin"}
    kwargs[field] = "   "
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)
        with pytest.raises(ValueError, match=field):
            kb.reconcile_triage_task(conn, task_id, **kwargs)
        assert kb.get_task(conn, task_id).status == "triage"


def test_reconcile_cli_exposes_operator_transition(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)

    output = kc.run_slash(
        f'reconcile {task_id} --reason "Resolved by the release operator."'
    )

    assert f"Reconciled {task_id}" in output
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "done"
        event = kb.list_events(conn, task_id)[-1]
    assert event.kind == "administratively_reconciled"


def test_reconcile_tool_is_available_to_orchestrators(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "ops-orchestrator")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)

    from tools import kanban_tools as kt

    result = json.loads(
        kt._handle_reconcile(
            {"task_id": task_id, "reason": "Resolved by external automation."}
        )
    )

    assert result == {
        "ok": True,
        "task_id": task_id,
        "status": "done",
        "outcome": "reconciled",
    }


def test_reconcile_tool_audits_active_session_profile(kanban_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "default")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)

    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import kanban_tools as kt

    tokens = set_session_vars(profile="ops-orchestrator")
    try:
        result = json.loads(
            kt._handle_reconcile(
                {"task_id": task_id, "reason": "Resolved by external automation."}
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["ok"] is True
    with kb.connect() as conn:
        event = kb.list_events(conn, task_id)[-1]
    assert event.kind == "administratively_reconciled"
    assert event.payload["operator"] == "ops-orchestrator"


def test_reconcile_tool_refuses_dispatcher_workers(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="resolved", triage=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    from tools import kanban_tools as kt

    result = json.loads(
        kt._handle_reconcile(
            {"task_id": task_id, "reason": "Worker must not authorize this."}
        )
    )

    assert "orchestrator-only" in result["error"]
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "triage"
