"""Independent preservation and prepared-generation contracts."""

import hashlib
import json
import sqlite3
from pathlib import Path

from hermes_cli.kanban_generation import validate_generation
from hermes_cli.kanban_reconcile import reconcile


def _db(path: Path, task: str, attachment: Path | None = None) -> None:
    with sqlite3.connect(path) as conn:
        from hermes_cli.kanban_db import SCHEMA_SQL
        conn.executescript(SCHEMA_SQL)
        conn.execute("ALTER TABLE tasks ADD COLUMN completion_contract TEXT")
        for sql in (
            "CREATE INDEX idx_tasks_assignee_status ON tasks(assignee, status)",
            "CREATE INDEX idx_tasks_tenant ON tasks(tenant)",
            "CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)",
            "CREATE INDEX idx_tasks_session_id ON tasks(session_id)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ):
            conn.execute(sql)
        conn.execute("INSERT INTO tasks(id, title, status, assignee, created_at, workspace_path, claim_lock, project_id, branch_name, session_id) VALUES (?, 'task', 'blocked', 'owner', 1, '/work', 'lock', 'project', 'branch', 'session')", (task,))
        conn.execute("INSERT INTO task_runs(id, task_id, status, started_at) VALUES (10, ?, 'running', 1)", (task,))
        conn.execute("INSERT INTO task_events(id, task_id, kind, run_id, created_at) VALUES (10, ?, 'started', 10, 1)", (task,))
        conn.execute("INSERT INTO task_comments(id, task_id, author, body, created_at) VALUES (10, ?, 'owner', 'history', 1)", (task,))
        if attachment:
            conn.execute("INSERT INTO task_attachments(id, task_id, filename, stored_path, size, created_at) VALUES (10, ?, 'a.txt', ?, ?, 1)", (task, str(attachment), attachment.stat().st_size))
        conn.commit()


def test_reopened_target_and_candidate_preserve_every_original_row_and_blob(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    blob = tmp_path / "old.txt"
    blob.write_bytes(b"preserved")
    _db(source, "source-task")
    _db(target, "target-task", blob)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"task_ids": ["source-task"]}))
    generation = tmp_path / "generation"
    report = reconcile(source, target, seeds, generation, apply=True)
    assert validate_generation(generation)["manifest_digest"] == report["manifest_digest"]
    with sqlite3.connect(generation / "kanban.db") as conn:
        conn.row_factory = sqlite3.Row
        assert tuple(conn.execute("SELECT title, workspace_path, claim_lock FROM tasks WHERE id='target-task'").fetchone()) == ("task", "/work", "lock")
        assert conn.execute("SELECT status FROM tasks WHERE id='source-task'").fetchone()[0] == "triage"
        assert conn.execute("SELECT status FROM task_runs WHERE task_id='source-task'").fetchone()[0] == "released"
        assert conn.execute("SELECT run_id FROM task_events WHERE task_id='source-task'").fetchone()[0] > 10
        custody = conn.execute("SELECT blob_sha256 FROM recovery_attachment_resolution WHERE attachment_id=10").fetchone()[0]
        assert custody == hashlib.sha256(b"preserved").hexdigest()


def test_legacy_transfer_control_is_not_used_for_recovery(tmp_path):
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    _db(source, "source-task")
    _db(target, "target-task")
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps(["source-task"]))
    result = reconcile(source, target, seeds, tmp_path / "candidate", apply=False)
    assert result["dry_run"] is True
