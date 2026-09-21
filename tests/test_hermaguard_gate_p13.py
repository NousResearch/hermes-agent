"""P13 isolation proof for scripts/hermaguard-gate.py.

Verifies:
- defers OUT_DIR.mkdir() until immediately before the live log write
  (importing the module under a temp HERMES_HOME does NOT create the
  logboard dir)
- --dry-run suppresses audit-log creation entirely
- --dry-run still prints flagged task IDs to stdout
- disposable board with review/evidence/no-evidence rows behaves:
    evidence task  → compliant (not flagged)
    no-evidence   → flagged in stdout
- live mode (no --dry-run) writes exactly one logboard JSON
- existing test contract still honoured (current schema without updated_at)
"""
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "hermaguard-gate.py"


def _load_module(hermes_home: Path):
    """Import the script fresh with HERMES_HOME pointed at a temp dir.

    Mirrors the existing test_hermaguard_gate.py loader so import-time
    side effects (OUT_DIR.mkdir) are observable.
    """
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)
    # board_compat imports hermes_cli.kanban_db which honours HERMES_HOME.
    scripts_dir = str(SCRIPT.parent)
    sys.path.insert(0, scripts_dir)
    import importlib
    # drop any cached module so the env takes effect on re-import
    for name in list(sys.modules):
        if name == "hermaguard_gate" or name.startswith("hermaguard_gate."):
            del sys.modules[name]
    # set env BEFORE exec
    old_env = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(hermes_home)
    try:
        spec = importlib.util.spec_from_file_location("hermaguard_gate_p13", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_env is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_env


def _make_board_db(db_path: Path, rows: list[dict], events: list[dict] | None = None):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
            status TEXT, tier TEXT, pipeline_stage TEXT,
            created_at INTEGER, started_at INTEGER, completed_at INTEGER,
            done_at INTEGER, archived_at INTEGER, result TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER
        )"""
    )
    for r in rows:
        conn.execute(
            """INSERT INTO tasks
            (id, title, body, assignee, status, tier, pipeline_stage,
             created_at, started_at, completed_at, done_at, archived_at, result)
            VALUES (:id, :title, :body, :assignee, :status, :tier, :pipeline_stage,
             :created_at, :started_at, :completed_at, :done_at, :archived_at, :result)""",
            r,
        )
    for e in events or []:
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,?)",
            (e["task_id"], e["kind"], e["payload"], e.get("created_at", 0)),
        )
    conn.commit()
    conn.close()


def _row(tid, title, status, *, started=None, completed=None, result=None, body=""):
    now = int(datetime(2026, 7, 27, 10, 0, tzinfo=timezone(timedelta(hours=1))).timestamp())
    return {
        "id": tid, "title": title, "body": body, "assignee": "octacon",
        "status": status, "tier": "full", "pipeline_stage": status,
        "created_at": now - 600, "started_at": started or now - 60,
        "completed_at": completed, "done_at": completed, "archived_at": None,
        "result": result,
    }


def test_import_does_not_create_logboard_dir(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    module = _load_module(home)
    logboard = home / "governance" / "logboard"
    assert not logboard.exists(), (
        "OUT_DIR.mkdir() ran at import time; must be deferred until live log write"
    )


def test_dry_run_suppresses_logboard_and_reports_flagged_ids(tmp_path, capsys):
    home = tmp_path / "hermes"
    home.mkdir()
    module = _load_module(home)

    # observed_now inside the 3h window of the task timestamps
    observed_now = datetime(2026, 7, 27, 11, 0, tzinfo=timezone(timedelta(hours=1)))
    module.now = observed_now

    db = home / "kanban" / "boards" / "test" / "kanban.db"
    rows = [
        _row("ev-1", "Backend API change", "review", result=None),
        _row("noev-1", "Frontend component update", "review", result=None),
        _row("skip-1", "Content post edit", "review", result=None),  # content→skip
    ]
    events = [
        {"task_id": "ev-1", "kind": "hermaguard_review", "payload": "hermaguard passed"},
    ]
    _make_board_db(db, rows, events)
    module.BOARDS = {"test": db}

    rc = module.main(["--dry-run"])

    logboard = home / "governance" / "logboard"
    assert not logboard.exists(), "dry-run must not create any logboard files"
    out = capsys.readouterr().out
    # no-evidence backend task flagged; evidence task not; content task skipped
    assert "`noev-1`" in out, "no-evidence task must be flagged"
    assert "`ev-1`" not in out, "evidence task must not be flagged"
    assert "`skip-1`" not in out
    assert rc == 0 or rc is None  # main may return None; check stdout only


def test_live_mode_writes_one_logboard_json(tmp_path, capsys):
    home = tmp_path / "hermes"
    home.mkdir()
    module = _load_module(home)
    observed_now = datetime(2026, 7, 27, 11, 0, tzinfo=timezone(timedelta(hours=1)))
    module.now = observed_now

    db = home / "kanban" / "boards" / "test" / "kanban.db"
    rows = [
        _row("noev-1", "Backend API change", "review", result=None),
        _row("ev-1", "Backend other API", "review", result=None),
    ]
    events = [
        {"task_id": "ev-1", "kind": "hermaguard_review", "payload": "hermaguard passed"},
    ]
    _make_board_db(db, rows, events)
    module.BOARDS = {"test": db}

    rc = module.main([])

    logboard = home / "governance" / "logboard"
    files = list(logboard.glob("hermaguard-gate-*.json"))
    assert len(files) == 1, f"expected exactly one logboard file, got {files}"
    payload = json.loads(files[0].read_text())
    assert payload["gate_name"] == "hermaguard-gate"
    assert payload["flagged"] == 1
    assert payload["compliant"] == 1
    out = capsys.readouterr().out
    assert "`noev-1`" in out


def test_existing_schema_without_updated_at_still_works(tmp_path, capsys):
    """Regression guard for the pre-existing test contract."""
    home = tmp_path / "hermes"
    home.mkdir()
    module = _load_module(home)
    observed_now = datetime(2026, 7, 27, 11, 0, tzinfo=timezone(timedelta(hours=1)))
    module.now = observed_now

    db = home / "kanban" / "boards" / "t" / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
            status TEXT, tier TEXT, pipeline_stage TEXT,
            created_at INTEGER, started_at INTEGER, completed_at INTEGER,
            done_at INTEGER, archived_at INTEGER, result TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER
        )"""
    )
    ts = int(observed_now.timestamp())
    conn.execute(
        """INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?)""",
        ("task-1", "Backend API change", "", "octacon", "review", "full", "review",
         ts - 60, ts, None, None, None),
    )
    conn.commit()
    conn.close()
    module.BOARDS = {"t": db}
    tasks = module._scan_boards()
    assert [t["id"] for t in tasks] == ["task-1"]
    assert tasks[0]["updated_at"] == ts
