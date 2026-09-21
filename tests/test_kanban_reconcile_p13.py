"""P13 isolation proof for scripts/kanban-reconcile.py.

Verifies:
- HERMES_HOME parameterisation: HERMES and DRIFT_STATE resolve under
  HERMES_HOME, not /home/kensei/.hermes.
- --dry-run suppresses every write path: no reopen_task SQLite UPDATE
  or event INSERT, no save_state drift file write. Read paths
  (board_dbs, scan_board, detect_profile_drift) run unchanged so drift
  is still detected and reported in the returned alert text.
- import-safe: importing the module does not create the drift state dir.
"""
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kanban-reconcile.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "kanban_reconcile_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_paths_resolve_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.HERMES).startswith(str(fake_home))
    assert mod.DRIFT_STATE == fake_home / "governance" / "reconcile-drift-state.json"


def test_import_is_side_effect_free(monkeypatch, fake_home):
    assert not (fake_home / "governance").exists()
    _load_module(monkeypatch, fake_home)
    assert not (fake_home / "governance").exists()


def test_dry_run_reopen_task_returns_false(monkeypatch, fake_home):
    """reopen_task under --dry-run must not execute the UPDATE/INSERT."""
    mod = _load_module(monkeypatch, fake_home)
    mod._DRY_RUN = True
    # Build an in-memory DB with a done task.
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT, status_reason TEXT, "
        "claim_lock TEXT, claim_expires TEXT, worker_pid TEXT, current_run_id TEXT, "
        "updated_at INTEGER)"
    )
    conn.execute(
        "CREATE TABLE task_events (task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER)"
    )
    conn.execute("INSERT INTO tasks (id, status) VALUES ('t1', 'done')")
    conn.commit()
    result = mod.reopen_task(conn, "t1", "test")
    assert result is False, "dry-run reopen_task returned True"
    # Task must still be 'done'.
    row = conn.execute("SELECT status FROM tasks WHERE id = 't1'").fetchone()
    assert row[0] == "done", "dry-run mutated the task status"
    # No event inserted.
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    assert events == 0, "dry-run inserted an event"
    conn.close()


def test_dry_run_save_state_writes_nothing(monkeypatch, fake_home):
    """save_state under --dry-run must not write the drift file."""
    mod = _load_module(monkeypatch, fake_home)
    mod._DRY_RUN = True
    mod.save_state({"last_run": 123, "drift_keys": [], "fixed_count": 0})
    assert not (fake_home / "governance" / "reconcile-drift-state.json").exists()


def test_dry_run_main_detects_drift_no_fix(monkeypatch, fake_home):
    """With a board DB containing a drifted task, --dry-run must detect
    the drift (alert text non-empty) but NOT reopen the task. The task
    status stays 'done' and no event is written."""
    mod = _load_module(monkeypatch, fake_home)
    # Build a kanban board DB with a drifted task.
    board_dir = fake_home / "kanban" / "boards" / "ops"
    board_dir.mkdir(parents=True)
    db_path = board_dir / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT, "
        "assignee TEXT, result TEXT, status_reason TEXT)"
    )
    conn.execute(
        "CREATE TABLE task_runs (id TEXT, task_id TEXT, outcome TEXT, status TEXT, "
        "error TEXT, started_at INTEGER, ended_at INTEGER)"
    )
    conn.execute(
        "CREATE TABLE task_events (task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status) VALUES ('t1', 'build thing', 'done')"
    )
    conn.execute(
        "INSERT INTO task_runs (id, task_id, outcome, started_at) "
        "VALUES ('r1', 't1', 'crashed', 100)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(sys, "argv", ["kanban-reconcile.py", "--dry-run"])
    alert = mod.main()
    # Alert text should mention the drifted task (detection ran).
    assert "t1" in alert or "drift" in alert.lower(), (
        f"dry-run did not report drift: {alert!r}"
    )
    # Task must still be 'done' (not reopened).
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT status FROM tasks WHERE id = 't1'").fetchone()
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    conn.close()
    assert row[0] == "done", "dry-run reopened the task"
    assert events == 0, "dry-run wrote an event"
    # No drift state file.
    assert not (fake_home / "governance" / "reconcile-drift-state.json").exists()
