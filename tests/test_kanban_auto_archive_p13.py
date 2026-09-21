"""P13 isolation proof for scripts/kanban-auto-archive.py.

Verifies:
- HERMES_HOME parameterisation: REPORT_DIR and find_all_board_dbs resolve
  under HERMES_HOME, not ~/.hermes.
- --dry-run (already present) leaves DB state unchanged: no status
  UPDATE to 'archived', no task_events INSERT, no commit. Read paths
  (SELECT candidates) run so the eligible set is reported.
- The HTML report is still written under the fake home (not the live
  one) so dry-run output goes to a throwaway dir.
- import-safe under a fake HERMES_HOME.
"""
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kanban-auto-archive.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("kaa_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def _build_db(db_path: Path) -> None:
    """Build a kanban.db with a stale done task (>14d old)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT, "
        "assignee TEXT, done_at INTEGER, archived_at INTEGER)"
    )
    conn.execute(
        "CREATE TABLE task_events (task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER)"
    )
    conn.execute("PRAGMA table_info(tasks)")
    # done_at 20 days ago (stale)
    import time
    old = int(time.time()) - (20 * 86400)
    conn.execute("INSERT INTO tasks (id, title, status, done_at) VALUES ('t1', 'stale', 'done', ?)", (old,))
    conn.commit()
    conn.close()


def test_report_dir_resolves_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.REPORT_DIR).startswith(str(fake_home))


def test_find_all_board_dbs_resolves_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    dbs = mod.find_all_board_dbs()
    # Default db path must be under fake home.
    assert any(str(fake_home) in p for _, p in dbs)


def test_import_is_side_effect_free(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert not (fake_home / "kanban.db").exists()
    assert not (fake_home / "kanban" / "reports").exists()


def test_dry_run_leaves_state_unchanged(monkeypatch, fake_home):
    """--dry-run must detect the stale task but NOT archive it: status
    stays 'done', no event inserted."""
    db_path = fake_home / "kanban.db"
    _build_db(db_path)
    mod = _load_module(monkeypatch, fake_home)
    results = mod.archive_stale_tasks("default", str(db_path), dry_run=True)
    assert len(results) == 1, "dry-run should detect the stale candidate"
    assert results[0]["task_id"] == "t1"
    # DB unchanged
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT status FROM tasks WHERE id='t1'").fetchone()
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    conn.close()
    assert row[0] == "done", "dry-run archived the task"
    assert events == 0, "dry-run inserted an event"
