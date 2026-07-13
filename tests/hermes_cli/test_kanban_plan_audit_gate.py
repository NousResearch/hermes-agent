"""Tests for the Kanban plan-audit claim gate."""

from __future__ import annotations

import sqlite3
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


def _event_kinds(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return [event.kind for event in kb.list_events(conn, task_id)]


def test_default_task_claims_without_plan_audit(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="worker")
        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)
        runs = kb.list_runs(conn, tid)

    assert claimed is not None
    assert task is not None
    assert task.status == "running"
    assert task.plan_audit_required is False
    assert len(runs) == 1


def test_required_plan_audit_blocks_claim_until_requested(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="needs plan audit",
            assignee="worker",
            plan_audit_required=True,
        )

        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)
        events_after_first = _event_kinds(conn, tid)

        # A second dispatcher tick should not spam duplicate request events.
        claimed_again = kb.claim_task(conn, tid, claimer="test-worker")
        events_after_second = _event_kinds(conn, tid)
        runs = kb.list_runs(conn, tid)

    assert claimed is None
    assert claimed_again is None
    assert task is not None
    assert task.status == "ready"
    assert task.plan_audit_required is True
    assert events_after_first.count("plan_audit_requested") == 1
    assert events_after_second.count("plan_audit_requested") == 1
    assert runs == []


def test_approved_plan_audit_allows_claim(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="approved plan",
            assignee="worker",
            plan_audit_required=True,
        )
        kb.record_plan_audit_verdict(
            conn,
            tid,
            approved=True,
            reviewer="auditor",
            reason="plan is specific enough",
        )

        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)
        events = _event_kinds(conn, tid)

    assert claimed is not None
    assert task is not None
    assert task.status == "running"
    assert "plan_audit_approved" in events
    assert "claimed" in events


def test_rejected_plan_below_limit_stays_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="needs another plan pass",
            assignee="worker",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )
        kb.record_plan_audit_verdict(conn, tid, approved=False, reason="too vague")

        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)

    assert claimed is None
    assert task is not None
    assert task.status == "ready"
    assert [event.kind for event in events].count("plan_audit_rejected") == 1
    requested = [event for event in events if event.kind == "plan_audit_requested"]
    assert requested[-1].payload == {"rejected_rounds": 1, "limit": 2}


def test_rejected_plan_at_limit_blocks_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="audit exhausted",
            assignee="worker",
            plan_audit_required=True,
            plan_audit_max_rounds=2,
        )
        kb.record_plan_audit_verdict(conn, tid, approved=False, reason="missing files")
        kb.record_plan_audit_verdict(conn, tid, approved=False, reason="still missing")

        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        runs = kb.list_runs(conn, tid)

    assert claimed is None
    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert "plan_audit_exhausted" in [event.kind for event in events]
    exhausted = [event for event in events if event.kind == "plan_audit_exhausted"]
    assert exhausted[-1].payload == {"rejected_rounds": 2, "limit": 2}
    assert runs == []


def test_approval_after_rejection_allows_claim(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="repaired plan",
            assignee="worker",
            plan_audit_required=True,
            plan_audit_max_rounds=1,
        )
        kb.record_plan_audit_verdict(conn, tid, approved=False, reason="needs detail")
        kb.record_plan_audit_verdict(conn, tid, approved=True, reason="now concrete")

        claimed = kb.claim_task(conn, tid, claimer="test-worker")
        task = kb.get_task(conn, tid)

    assert claimed is not None
    assert task is not None
    assert task.status == "running"


def test_legacy_db_migrates_plan_audit_columns(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    kb.init_db()
    with kb.connect() as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        task = kb.get_task(conn, "legacy1")

    assert "plan_audit_required" in cols
    assert "plan_audit_max_rounds" in cols
    assert task is not None
    assert task.plan_audit_required is False
    assert task.plan_audit_max_rounds is None


def test_dispatch_once_does_not_spawn_gated_task(kanban_home, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists",
        lambda _name: True,
        raising=False,
    )
    spawned: list[str] = []

    def _spawn(task, _workspace_path):
        spawned.append(task.id)
        return 1234

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="gated dispatch",
            assignee="worker",
            plan_audit_required=True,
        )

        result = kb.dispatch_once(conn, spawn_fn=_spawn)
        task = kb.get_task(conn, tid)
        events = _event_kinds(conn, tid)

    assert spawned == []
    assert result.spawned == []
    assert task is not None
    assert task.status == "ready"
    assert "plan_audit_requested" in events
