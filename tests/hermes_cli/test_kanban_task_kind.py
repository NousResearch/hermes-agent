"""Durable free-form task kinds (#23959), independent of workspace/block kinds."""

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


ROOT = Path(__file__).parents[2]


def _cli(home, *args):
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("HERMES_"):
            env.pop(name)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        env[name] = str(home)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args, "--json"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=90, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cli_kind_survives_reopen_and_preserves_existing_create_flags(tmp_path):
    created = _cli(
        tmp_path, "create", "Repair CI", "--kind", "ci-fix",
        "--workspace", "scratch", "--initial-status", "blocked",
        "--tenant", "acme", "--priority", "7", "--body", "Keep this body",
        "--created-by", "operator",
    )
    assert created["kind"] == "ci-fix"
    shown = _cli(tmp_path, "show", created["id"])["task"]
    assert shown["kind"] == "ci-fix"
    assert shown["workspace_kind"] == "scratch"
    assert shown["status"] == "blocked"
    assert shown["tenant"] == "acme"
    assert shown["priority"] == 7
    assert shown["body"] == "Keep this body"
    assert shown["created_by"] == "operator"
    with sqlite3.connect(tmp_path / "kanban.db") as conn:
        assert conn.execute("SELECT kind FROM tasks WHERE id = ?", (created["id"],)).fetchone() == ("ci-fix",)


def test_legacy_tasks_default_to_task_after_migration(tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                     "status TEXT NOT NULL, created_at INTEGER NOT NULL)")
        conn.execute("INSERT INTO tasks VALUES ('legacy', 'Existing task', 'blocked', 123)")
    kbc.init_db(path)
    kbc.init_db(path)  # The guarded migration is safe to repeat.
    with kbc.connect_closing(path) as conn:
        task = kb.get_task(conn, "legacy")
        assert (task.kind, task.title, task.status, task.created_at) == (
            "task", "Existing task", "blocked", 123,
        )
        default_id = kb.create_task(conn, title="Default kind")
        assert kb.get_task(conn, default_id).kind == "task"
        assert conn.execute("SELECT kind FROM tasks WHERE id = 'legacy'").fetchone()[0] == "task"


def test_cli_kind_filter_is_exact_and_combines_with_existing_filters(tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing(tmp_path / "kanban.db") as conn:
        ids = {}
        for kind in ("ci-fix", "CI-fix", "ci-fix-more", "ci%fix", "task"):
            ids[kind] = kb.create_task(conn, title=kind, kind=kind, tenant="acme", initial_status="blocked")
        kb.create_task(conn, title="Other tenant", kind="ci-fix", tenant="other", initial_status="blocked")
    tasks = _cli(tmp_path, "list", "--kind", "ci-fix", "--tenant", "acme", "--status", "blocked")
    assert [(t["id"], t["kind"]) for t in tasks] == [(ids["ci-fix"], "ci-fix")]
    literal = _cli(tmp_path, "ls", "--kind", "ci%fix")
    assert [t["id"] for t in literal] == [ids["ci%fix"]]
    all_tasks = _cli(tmp_path, "list", "--tenant", "acme")
    assert {t["id"] for t in all_tasks} == set(ids.values())
