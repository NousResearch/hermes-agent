"""P13 isolation proof for scripts/kanban-task-notifier.py.

Verifies:
- HERMES_HOME parameterisation: KANBAN_BASE, DEFAULT_DB, STATE_FILE
  resolve under HERMES_HOME, not ~/.hermes.
- --dry-run suppresses Discord delivery: no save_state (state file not
  written), no stdout message block. Read paths (board scan) run.
- import-safe under a fake HERMES_HOME.
"""
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kanban-task-notifier.py"


def _load_module(monkeypatch, fake_home: Path, dry_run: bool = False):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    argv = ["kanban-task-notifier.py"] + (["--dry-run"] if dry_run else [])
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("ktn_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def _build_board(fake_home: Path) -> Path:
    """Build a default board DB with one ready+assigned task."""
    db_path = fake_home / "kanban" / "kanban.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT, "
        "assignee TEXT, priority INTEGER, status TEXT)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, priority, status) "
        "VALUES ('t1', 'do thing', 'desc', 'wesker', 2, 'ready')"
    )
    conn.commit()
    conn.close()
    return db_path


def test_paths_resolve_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.KANBAN_BASE).startswith(str(fake_home))
    assert str(mod.DEFAULT_DB).startswith(str(fake_home))
    assert str(mod.STATE_FILE).startswith(str(fake_home))


def test_import_is_side_effect_free(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert not (fake_home / "kanban").exists()
    assert mod._DRY_RUN is False


def test_dry_run_suppresses_delivery(monkeypatch, fake_home, capsys):
    """With a ready task and --dry-run, main() must NOT print the
    message block and must NOT save state (no state file written)."""
    _build_board(fake_home)
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    assert mod._DRY_RUN is True
    mod.main()
    out = capsys.readouterr().out
    assert "DRY-RUN" in out, f"no DRY-RUN banner: {out!r}"
    # Must NOT have printed the task message block (no Discord delivery).
    assert "Task t1 assigned" not in out, "dry-run delivered the Discord message"
    # State file must NOT exist (save_state suppressed).
    assert not mod.STATE_FILE.exists(), "dry-run wrote the state file"


def test_non_dry_run_delivers_and_saves(monkeypatch, fake_home, capsys):
    """Without --dry-run, the ready task is printed and state is saved
    (proves the dry-run guard is the only thing preventing delivery)."""
    _build_board(fake_home)
    mod = _load_module(monkeypatch, fake_home, dry_run=False)
    mod.main()
    out = capsys.readouterr().out
    assert "Task t1 assigned" in out, "non-dry-run did not deliver the message"
    assert mod.STATE_FILE.exists(), "non-dry-run did not save state"
