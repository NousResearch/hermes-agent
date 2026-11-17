"""Native offline board reconciliation behavior contracts."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _board(path: Path, *, task_id: str, child_id: int) -> None:
    conn = sqlite3.connect(path)
    from hermes_cli.kanban_db import SCHEMA_SQL
    conn.executescript(SCHEMA_SQL)
    conn.execute("ALTER TABLE tasks ADD COLUMN completion_contract TEXT")
    for sql in ("CREATE INDEX idx_tasks_assignee_status ON tasks(assignee, status)",
                "CREATE INDEX idx_tasks_tenant ON tasks(tenant)",
                "CREATE INDEX idx_tasks_idempotency ON tasks(idempotency_key)",
                "CREATE INDEX idx_tasks_session_id ON tasks(session_id)",
                "CREATE INDEX idx_events_run ON task_events(run_id, id)"):
        conn.execute(sql)
    conn.execute("INSERT INTO tasks(id, title, body, status, assignee, workspace_kind, workspace_path, claim_lock, project_id, branch_name, created_at) VALUES (?, ?, ?, 'running', 'alice', 'worktree', '/repo', 'lock', NULL, 'main', 1)",
                 (task_id, "seed", ""))
    conn.execute("INSERT INTO task_comments(id, task_id, author, body, created_at) VALUES (?, ?, 'a', 'incident', 1)",
                 (child_id, task_id))
    conn.commit()
    conn.close()


def test_reconcile_dry_run_reports_seed_closure_without_writing(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    seed = tmp_path / "seed.json"
    output = tmp_path / "candidate"
    _board(source, task_id="t_seed", child_id=7)
    _board(target, task_id="t_other", child_id=7)
    seed.write_text(json.dumps({"task_ids": ["t_seed"]}), encoding="utf-8")
    env = {"HERMES_HOME": str(tmp_path / "home"), "PYTHONPATH": str(Path.cwd())}
    result = subprocess.run(
        [str(Path.cwd() / ".venv/bin/hermes"), "kanban", "boards", "reconcile",
         "--source-db", str(source), "--target-db", str(target), "--seed-file", str(seed),
         "--output-generation", str(output), "--json"],
        env={**__import__("os").environ, **env}, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["dry_run"] is True
    assert report["seeds"] == ["t_seed"]
    assert report["closure"] == ["t_seed"]
    assert report["remaps"]["task_comments"]["7"] != 7
    assert not output.exists()


def test_all_named_seeds_are_supported_as_explicit_scope(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    seed = tmp_path / "seed.json"
    output = tmp_path / "candidate"
    _board(source, task_id="seed-00", child_id=7)
    _board(target, task_id="unrelated", child_id=7)
    with sqlite3.connect(source) as conn:
        for index in range(1, 16):
            conn.execute("INSERT INTO tasks(id, title, body, status, created_at, workspace_kind, idempotency_key, block_kind, block_recurrences) VALUES (?, ?, '', 'todo', 1, 'scratch', ?, 'incident', 0)", (f"seed-{index:02d}", f"seed {index}", f"seed-key-{index}"))
        conn.commit()
    seeds = [f"seed-{index:02d}" for index in range(16)]
    seed.write_text(json.dumps({"task_ids": seeds}), encoding="utf-8")
    result = subprocess.run(
        [str(Path.cwd() / ".venv/bin/hermes"), "kanban", "boards", "reconcile", "--source-db", str(source),
         "--target-db", str(target), "--seed-file", str(seed), "--output-generation", str(output), "--json"],
        env={**__import__("os").environ, "HERMES_HOME": str(tmp_path / "home"), "PYTHONPATH": str(Path.cwd())},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["seeds"] == sorted(seeds)
