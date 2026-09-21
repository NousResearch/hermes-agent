"""P13 isolation proof for scripts/kensei-triage-processor.py.

Verifies:
- HERMES_HOME parameterisation: STATE_FILE / PENDING_FILE resolve under
  HERMES_HOME so a temp home is used instead of /home/kensei/.hermes
- --dry-run suppresses every write path: no SQLite UPDATEs, no
  `hermes kanban` CLI calls, no state/pending file writes. The script
  still runs end-to-end (classification, routing) and exits 0.
- import-safe: importing the module does not scan boards or write state
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "kensei-triage-processor.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "kensei_triage_processor_under_test", str(SCRIPT)
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


def test_state_and_pending_files_resolve_under_hermes_home(monkeypatch, fake_home):
    """STATE_FILE and PENDING_FILE must live under HERMES_HOME, not the
    hardcoded /home/kensei/.hermes path."""
    mod = _load_module(monkeypatch, fake_home)
    assert mod.STATE_FILE.startswith(str(fake_home))
    assert mod.PENDING_FILE.startswith(str(fake_home))
    assert "triage-state.json" in mod.STATE_FILE
    assert "pending-investigation.json" in mod.PENDING_FILE


def test_import_is_side_effect_free(monkeypatch, fake_home):
    """Importing the module must not create the state/pending dirs."""
    fake_data = fake_home / "data"
    assert not fake_data.exists()
    _load_module(monkeypatch, fake_home)
    assert not fake_data.exists(), "import created the data dir"


def test_dry_run_writes_nothing_and_exits_zero(monkeypatch, fake_home, tmp_path):
    """--dry-run must not write STATE_FILE or PENDING_FILE, must not call
    the hermes CLI, and must exit 0 even with no boards present."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(fake_home)
    # Ensure the `hermes` CLI is not on PATH so any accidental CLI call
    # would fail loudly (FileNotFoundError) rather than silently mutating.
    env["PATH"] = "/usr/bin:/bin"
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"dry-run failed: rc={proc.returncode} stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )
    # No state or pending file written.
    assert not (fake_home / "data" / "triage-state.json").exists()
    assert not (fake_home / "data" / "pending-investigation.json").exists()
    assert not (fake_home / "data").exists() or not any((fake_home / "data").iterdir())


def test_dry_run_does_not_mutate_existing_board_db(monkeypatch, fake_home, tmp_path):
    """With a board DB containing triage tasks, --dry-run must NOT flip
    any task status or write any state file. The DB hash is unchanged."""
    import sqlite3
    db_dir = fake_home / "kanban" / "boards" / "test_board"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
            status TEXT, tier TEXT, priority INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT 0, updated_at INTEGER DEFAULT 0
        )"""
    )
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status) "
        "VALUES ('t1', 'orphan process cleanup', 'fix typo', '', 'triage')"
    )
    conn.commit()
    conn.close()
    import hashlib
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    env = dict(os.environ)
    env["HERMES_HOME"] = str(fake_home)
    env["PATH"] = "/usr/bin:/bin"
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=30,
    )
    assert proc.returncode == 0, proc.stderr

    after = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert after == before, "dry-run mutated the board DB"
    # No state file written.
    assert not (fake_home / "data" / "triage-state.json").exists()
