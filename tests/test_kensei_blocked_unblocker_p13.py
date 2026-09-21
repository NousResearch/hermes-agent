"""P13 isolation proof for scripts/kensei-blocked-unblocker.py.

Verifies the --dry-run flag suppresses the dedup state file write while
the structured JSON output and stderr diagnostic are still emitted
unchanged. The script already uses HERMES_HOME so this test focuses on
the dry-run write suppression.
"""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SCRIPT = REPO_ROOT / "scripts" / "kensei-blocked-unblocker.py"


def _populate_board_db(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """CREATE TABLE tasks (
                id TEXT PRIMARY KEY, title TEXT, assignee TEXT, priority INTEGER,
                status TEXT, status_reason TEXT, body TEXT,
                created_at INTEGER, updated_at INTEGER,
                consecutive_failures INTEGER, max_retries INTEGER,
                escalation_target TEXT, block_kind TEXT
            )"""
        )
        for t in tasks:
            conn.execute(
                """INSERT INTO tasks (
                    id, title, assignee, priority, status, status_reason,
                    body, created_at, updated_at, consecutive_failures,
                    max_retries, escalation_target, block_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t["id"], t["title"], t.get("assignee", "tester"),
                    t.get("priority", 0), t["status"], t.get("status_reason", ""),
                    t.get("body", ""), t.get("created_at", 0), t.get("updated_at", 0),
                    t.get("consecutive_failures", 0), t.get("max_retries"),
                    t.get("escalation_target"), t.get("block_kind"),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _run_script(hermes_home: Path, *, dry_run: bool = False):
    env = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "KENSEI_UNBLOCKER_AGE": "3600",
        "KENSEI_UNBLOCKER_MAX": "50",
        "KENSEI_UNBLOCKER_STUCK_FAILURES": "3",
        "KENSEI_UNBLOCKER_DEDUP_HOURS": "12",
        "PYTHONPATH": "",
    }
    argv = [sys.executable, str(REPO_SCRIPT)]
    if dry_run:
        argv.append("--dry-run")
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)


def test_dry_run_emits_output_but_writes_no_state(tmp_path):
    """A board with a stuck task: dry-run emits the escalation JSON on
    stdout (wakeAgent=True) but does NOT create the dedup state file."""
    fake_home = tmp_path / "fake_hermes"
    db_path = fake_home / "kanban" / "boards" / "test_board" / "kanban.db"
    now = 1_700_000_000
    _populate_board_db(
        db_path,
        [{
            "id": "t_stuck", "title": "Stuck", "status": "blocked",
            "escalation_target": None, "block_kind": "transient",
            "updated_at": now - 7200, "consecutive_failures": 3,
        }],
    )
    proc = _run_script(fake_home, dry_run=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["wakeAgent"] is True
    assert [t["id"] for t in out["escalate"]] == ["t_stuck"]
    # The dedup state file must NOT have been written in dry-run.
    state_file = fake_home / "state" / "blocked-unblocker-dedup.json"
    assert not state_file.exists(), "dry-run wrote the dedup state file"


def test_dry_run_empty_board_still_silent(tmp_path):
    """No boards: dry-run emits {"wakeAgent": false} and writes no state."""
    fake_home = tmp_path / "fake_hermes"
    proc = _run_script(fake_home, dry_run=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"wakeAgent": False}
    assert not (fake_home / "state").exists()


def test_live_run_writes_state_for_comparison(tmp_path):
    """Live mode (no --dry-run) DOES write the dedup state file for the
    same stuck task — proves the dry-run guard is what suppresses it."""
    fake_home = tmp_path / "fake_hermes"
    db_path = fake_home / "kanban" / "boards" / "test_board" / "kanban.db"
    now = 1_700_000_000
    _populate_board_db(
        db_path,
        [{
            "id": "t_stuck", "title": "Stuck", "status": "blocked",
            "escalation_target": None, "block_kind": "transient",
            "updated_at": now - 7200, "consecutive_failures": 3,
        }],
    )
    proc = _run_script(fake_home, dry_run=False)
    assert proc.returncode == 0, proc.stderr
    state_file = fake_home / "state" / "blocked-unblocker-dedup.json"
    assert state_file.exists(), "live mode did not write the dedup state file"
    assert "t_stuck" in json.loads(state_file.read_text())
