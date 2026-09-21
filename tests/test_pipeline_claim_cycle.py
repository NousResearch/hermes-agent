#!/usr/bin/env python3
"""Tests for claim_pipeline_task + complete_pipeline_task cycle.

Exercises the full gate-fail -> spawn -> worker-done -> return-to-stage
-> re-gate-check -> advance cycle. This cycle was never exercised before,
which allowed the claim_task(status='ready') bug to slip through.
"""
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch

from hermes_cli import kanban_db as kb


@pytest.fixture
def fresh_db():
    """Fresh in-memory database with tasks, task_runs, task_events tables."""
    os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE tasks (
        id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
        status TEXT, priority INTEGER DEFAULT 0, created_by TEXT,
        created_at INTEGER, started_at INTEGER, completed_at INTEGER,
        workspace_kind TEXT DEFAULT 'scratch', workspace_path TEXT,
        claim_lock TEXT, claim_expires INTEGER, tenant TEXT,
        branch_name TEXT, result TEXT, idempotency_key TEXT,
        consecutive_failures INTEGER DEFAULT 0, worker_pid INTEGER,
        last_failure_error TEXT, max_runtime_seconds INTEGER,
        last_heartbeat_at INTEGER, current_run_id INTEGER,
        workflow_template_id TEXT, current_step_key TEXT,
        pipeline_stage TEXT, pipeline_mode TEXT, skills TEXT,
        model_override TEXT, max_retries INTEGER, goal_mode INTEGER DEFAULT 0,
        goal_max_turns INTEGER, session_id TEXT, theme TEXT, tier TEXT
    );
    CREATE TABLE task_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, profile TEXT,
        step_key TEXT, status TEXT, claim_lock TEXT,
        claim_expires INTEGER, max_runtime_seconds INTEGER, started_at INTEGER
    );
    CREATE TABLE task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, run_id INTEGER,
        kind TEXT, payload TEXT, created_at INTEGER
    );
    """)
    return conn


def _insert_task(conn, **kw):
    defaults = {
        "id": "t_test",
        "title": "Test",
        "status": "research",
        "pipeline_stage": "research",
        "tier": "full",
        "assignee": "remii-deep",
        "created_at": 1717800000,
    }
    defaults.update(kw)
    cols = ", ".join(defaults.keys())
    vals = ", ".join("?" * len(defaults))
    conn.execute(f"INSERT INTO tasks ({cols}) VALUES ({vals})", tuple(defaults.values()))
    conn.commit()


class TestClaimPipelineTask:
    """claim_pipeline_task: research/prd/spec/council -> running."""

    def test_claims_research_to_running(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is not None, "should claim research task"
        assert claimed.status == "running"

    def test_claims_prd_to_running(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="prd", pipeline_stage="prd")
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is not None
        assert claimed.status == "running"

    def test_claims_spec_to_running(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="spec", pipeline_stage="spec")
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is not None
        assert claimed.status == "running"

    def test_claims_council_to_running(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="council", pipeline_stage="council")
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is not None
        assert claimed.status == "running"

    def test_rejects_ready_task(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="ready", pipeline_stage=None)
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is None, "should not claim a non-pipeline task"

    def test_rejects_non_pipeline_status(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="todo", pipeline_stage=None)
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is None

    def test_rejects_already_claimed(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        first = kb.claim_pipeline_task(conn, "t_test")
        assert first is not None
        second = kb.claim_pipeline_task(conn, "t_test")
        assert second is None, "already claimed — should be None"

    def test_stores_pipeline_stage_in_claim_event(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        kb.claim_pipeline_task(conn, "t_test")
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='claimed' "
            "AND json_extract(payload,'$.source_status')='pipeline' "
            "ORDER BY id DESC LIMIT 1",
            ("t_test",),
        ).fetchone()
        assert ev is not None
        import json
        payload = json.loads(ev["payload"])
        assert payload["pipeline_stage"] == "research"
        assert payload["source_status"] == "pipeline"


class TestCompletePipelineTask:
    """complete_pipeline_task: running -> original pipeline stage."""

    def test_returns_to_research(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        kb.claim_pipeline_task(conn, "t_test")
        ok = kb.complete_pipeline_task(conn, "t_test", result="artifact written")
        assert ok
        row = conn.execute(
            "SELECT status, pipeline_stage FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["status"] == "research"
        assert row["pipeline_stage"] == "research"

    def test_returns_to_prd(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="prd", pipeline_stage="prd")
        kb.claim_pipeline_task(conn, "t_test")
        kb.complete_pipeline_task(conn, "t_test")
        row = conn.execute(
            "SELECT status, pipeline_stage FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["status"] == "prd"

    def test_returns_to_spec(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="spec", pipeline_stage="spec")
        kb.claim_pipeline_task(conn, "t_test")
        kb.complete_pipeline_task(conn, "t_test")
        row = conn.execute(
            "SELECT status, pipeline_stage FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["status"] == "spec"

    def test_clears_claim_lock(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        kb.claim_pipeline_task(conn, "t_test")
        kb.complete_pipeline_task(conn, "t_test")
        row = conn.execute(
            "SELECT claim_lock, claim_expires FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["claim_lock"] is None
        assert row["claim_expires"] is None

    def test_rejects_non_running_task(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        ok = kb.complete_pipeline_task(conn, "t_test")
        assert not ok, "should not complete a non-running task"

    def test_falls_back_to_column_when_event_missing(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="research", pipeline_stage="research")
        # Manually set to running WITHOUT going through claim_pipeline_task
        conn.execute("UPDATE tasks SET status='running' WHERE id='t_test'")
        conn.commit()
        ok = kb.complete_pipeline_task(conn, "t_test")
        assert ok
        row = conn.execute(
            "SELECT status, pipeline_stage FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["status"] == "research"


class TestFullCycle:
    """End-to-end: gate-fail -> spawn -> worker-done -> re-gate-check."""

    def test_research_claim_complete_recheck(self, fresh_db):
        conn = fresh_db
        # Simulate: task at research, gate failed last tick.
        # Dispatcher spawns remii-deep to write artifact.
        _insert_task(conn, status="research", pipeline_stage="research",
                     assignee="remii-deep")

        # Step 1: claim_pipeline_task (the spawn)
        claimed = kb.claim_pipeline_task(conn, "t_test")
        assert claimed is not None, "spawn should succeed"
        assert claimed.status == "running"

        # Step 2: worker writes artifact, calls complete_pipeline_task
        ok = kb.complete_pipeline_task(conn, "t_test", result="research-brief.md")
        assert ok, "completion should succeed"

        # Step 3: task is back at research stage, claim cleared
        row = conn.execute(
            "SELECT status, pipeline_stage, claim_lock FROM tasks WHERE id=?",
            ("t_test",),
        ).fetchone()
        assert row["status"] == "research"
        assert row["pipeline_stage"] == "research"
        assert row["claim_lock"] is None

        # Step 4: next tick, dispatcher checks gate — if artifact exists,
        # gate passes and advances to prd. If not, gate fails and cycle
        # repeats. The claim is clear so a fresh spawn can happen.
        # (Gate check itself is tested in test_phase_a_pipeline.py)

    def test_prd_claim_complete_recheck(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="prd", pipeline_stage="prd", assignee="kensei-review")
        kb.claim_pipeline_task(conn, "t_test")
        kb.complete_pipeline_task(conn, "t_test")
        row = conn.execute(
            "SELECT status, pipeline_stage FROM tasks WHERE id=?", ("t_test",)
        ).fetchone()
        assert row["status"] == "prd"

    def test_completion_event_has_returned_to_stage(self, fresh_db):
        conn = fresh_db
        _insert_task(conn, status="spec", pipeline_stage="spec",
                     assignee="octacon-frontend")
        kb.claim_pipeline_task(conn, "t_test")
        kb.complete_pipeline_task(conn, "t_test")
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='completed' "
            "AND json_extract(payload,'$.source_status')='pipeline' "
            "ORDER BY id DESC LIMIT 1",
            ("t_test",),
        ).fetchone()
        assert ev is not None
        import json
        payload = json.loads(ev["payload"])
        assert payload["returned_to_stage"] == "spec"
