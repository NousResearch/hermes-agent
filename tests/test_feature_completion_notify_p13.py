"""P13 isolation proof for scripts/feature-completion-notify.py.

Verifies --dry-run:
- proposes parent IDs in stdout (the parent whose every child is done)
- does NOT mutate the board kanban.db (hash unchanged before/after)
- does NOT write save_state (state file not created)
- does NOT touch the quarantine sentinel
- does NOT quarantine a corrupt DB (no copy/unlink/rename)
Live mode (no --dry-run):
- auto-completes the parent (status -> done)
- writes the state file with the parent id
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "feature-completion-notify.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_db(db_path: Path, *, parents_done=False, corrupt=False):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        # Write a file that's not a valid SQLite DB (header wrong).
        db_path.write_bytes(b"not-a-sqlite-db" * 16)
        return
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, status TEXT,
            assignee TEXT, priority INTEGER, created_at INTEGER,
            updated_at INTEGER, done_at INTEGER, completed_at INTEGER,
            epic_id TEXT, body TEXT, result TEXT, started_at INTEGER,
            archived_at INTEGER, status_reason TEXT, pipeline_stage TEXT,
            tier TEXT
        )"""
    )
    conn.execute(
        "CREATE TABLE task_links (parent_id TEXT, child_id TEXT, PRIMARY KEY(parent_id, child_id))"
    )
    # Parent P1 (todo) with two done children; parent P2 (running) with one done + one todo
    now = 1700000000
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('P1','parent one','todo',1,?)", (now,))
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('P2','parent two','running',1,?)", (now,))
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('C1','child one','done',1,?)", (now,))
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('C2','child two','done',1,?)", (now,))
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('C3','child three','done',1,?)", (now,))
    conn.execute("INSERT INTO tasks (id, title, status, priority, created_at) VALUES ('C4','child four','todo',1,?)", (now,))
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES ('P1','C1')")
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES ('P1','C2')")
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES ('P2','C3')")
    conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES ('P2','C4')")
    conn.commit()
    conn.close()


def _run(hermes_home: Path, *extra):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    env.pop("HERMES_KANBAN_DB", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *extra],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def _make_home(tmp_path: Path, *, corrupt_default=False) -> Path:
    home = tmp_path / "hermes"
    home.mkdir()
    # default board at <home>/kanban.db
    _make_db(home / "kanban.db", corrupt=corrupt_default)
    # one sub-board
    _make_db(home / "kanban" / "boards" / "test" / "kanban.db")
    return home


def test_dry_run_proposes_parent_and_does_not_mutate(tmp_path):
    home = _make_home(tmp_path)
    board_db = home / "kanban.db"
    sub_db = home / "kanban" / "boards" / "test" / "kanban.db"
    state_file = home / "data" / "feature-completion-state.json"

    before_board = _sha(board_db)
    before_sub = _sha(sub_db)

    r = _run(home, "--dry-run")
    assert r.returncode == 0, r.stderr

    # The sub-board's parent P1 (all children done) should be proposed.
    out = r.stdout
    assert "P1" in out, f"proposed parent P1 not in stdout: {out!r}"
    # P2 (one child still todo) must NOT be proposed.
    # Use a robust check: P2 should not appear as an auto-closed entry.
    assert "P2" not in out, f"non-completable P2 should not be proposed: {out!r}"

    assert _sha(board_db) == before_board, "default board DB mutated in dry-run"
    assert _sha(sub_db) == before_sub, "sub-board DB mutated in dry-run"
    assert not state_file.exists(), "state file created in dry-run"


def test_dry_run_does_not_quarantine_corrupt_db(tmp_path):
    home = _make_home(tmp_path, corrupt_default=True)
    default_db = home / "kanban.db"
    before = _sha(default_db)
    sentinel = default_db.parent / ".quarantine-cooldown"

    r = _run(home, "--dry-run")
    assert r.returncode == 0, r.stderr
    # Corrupt DB left in place (no rename/copy/unlink).
    assert default_db.exists(), "corrupt DB was removed/quarantined in dry-run"
    assert _sha(default_db) == before, "corrupt DB mutated in dry-run"
    assert not sentinel.exists(), "sentinel touched in dry-run"


def test_live_mode_completes_parent_and_writes_state(tmp_path):
    home = _make_home(tmp_path)
    sub_db = home / "kanban" / "boards" / "test" / "kanban.db"
    state_file = home / "data" / "feature-completion-state.json"

    r = _run(home)
    assert r.returncode == 0, r.stderr
    # P1 should now be done in the sub-board DB.
    conn = sqlite3.connect(sub_db)
    row = conn.execute("SELECT status FROM tasks WHERE id='P1'").fetchone()
    conn.close()
    assert row is not None and row[0] == "done", f"P1 not completed: {row}"
    # State file written with P1.
    assert state_file.exists(), "state file not written in live mode"
    state = json.loads(state_file.read_text())
    assert "P1" in state, f"P1 not in state: {state}"
