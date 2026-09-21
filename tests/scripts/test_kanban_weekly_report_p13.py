"""P13 observer-proof test for scripts/kanban-weekly-report.py.

Verifies env-derived HERMES_HOME drives both board DB discovery and report
output. A temp board fixture supplies known task counts; the test asserts
exactly one HTML file lands under the temp reports dir, the MEDIA: path
points at it, the rendered summary carries the expected counts, and the
fixture board DBs are byte-identical before/after (report is read-only).
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "kanban-weekly-report.py"


def _sha_files(*paths: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _make_board_db(db_path: Path, rows: list[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Minimal schema matching what the script queries.
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_at INTEGER,
            completed_at INTEGER,
            updated_at INTEGER,
            done_at INTEGER,
            epic_id TEXT,
            status_reason TEXT
        );
        CREATE TABLE epics (
            id TEXT PRIMARY KEY,
            title TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO tasks (id, title, assignee, status, priority, created_at, "
        "completed_at, updated_at, done_at, epic_id, status_reason) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["id"], r["title"], r.get("assignee"), r["status"],
                r.get("priority", 0), r.get("created_at", 0),
                r.get("completed_at"), r.get("updated_at"),
                r.get("done_at"), r.get("epic_id"), r.get("status_reason"),
            )
            for r in rows
        ],
    )
    conn.execute("INSERT INTO epics (id, title) VALUES (?,?)", ("ep1", "Epic One"))
    conn.commit()
    conn.close()


def _build_fixture(home: Path) -> dict[str, Path]:
    """Default board at <home>/kanban.db + a sub-board under kanban/boards."""
    now = int(time.time())
    default_db = home / "kanban.db"
    sub_db = home / "kanban" / "boards" / "apps" / "kanban.db"
    _make_board_db(
        default_db,
        [
            # shipped in last 7 days
            {"id": "T-SHIP", "title": "Shipped task", "assignee": "octacon",
             "status": "done", "done_at": now - 100, "epic_id": "ep1"},
            # active
            {"id": "T-ACT", "title": "Active task", "assignee": "misa-misa",
             "status": "running", "priority": 5, "epic_id": "ep1",
             "created_at": now - 2000},
            # blocked
            {"id": "T-BLK", "title": "Blocked task", "assignee": "remii",
             "status": "blocked", "priority": 3, "epic_id": "ep1",
             "created_at": now - 3000, "status_reason": "waiting"},
            # backlog
            {"id": "T-BKLG", "title": "Backlog task", "assignee": None,
             "status": "backlog", "priority": 2, "epic_id": "ep1",
             "created_at": now - 4000},
            # shipped older than 7 days — must NOT appear
            {"id": "T-OLD", "title": "Old shipped", "assignee": "octacon",
             "status": "done", "done_at": now - (8 * 86400), "epic_id": "ep1"},
        ],
    )
    _make_board_db(
        sub_db,
        [
            {"id": "S-1", "title": "Sub shipped", "assignee": "wesker",
             "status": "done", "done_at": now - 50, "epic_id": None},
        ],
    )
    return {"default": default_db, "apps": sub_db}


def _run_script(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_weekly_report_uses_hermes_home_and_preserves_fixture(tmp_path: Path) -> None:
    home = tmp_path / "hermes_home"
    home.mkdir()
    dbs = _build_fixture(home)
    before = _sha_files(*dbs.values())

    r = _run_script(home)
    assert r.returncode == 0, r.stderr

    # Exactly one HTML file under <home>/kanban/reports
    reports_dir = home / "kanban" / "reports"
    html_files = sorted(reports_dir.glob("kanban-weekly-*.html"))
    assert len(html_files) == 1, f"expected 1 report, got {html_files}"

    # MEDIA: line points at that file
    media_lines = [l for l in r.stdout.splitlines() if l.startswith("MEDIA:")]
    assert len(media_lines) == 1, f"expected one MEDIA line, got {media_lines}"
    media_path = Path(media_lines[0].split("MEDIA:", 1)[1].strip())
    assert media_path == html_files[0], f"MEDIA path {media_path} != {html_files[0]}"

    # Summary line carries expected counts: 2 shipped (T-SHIP + S-1), 1 active,
    # 1 blocked, 1 backlog
    summary = [l for l in r.stdout.splitlines() if "shipped" in l and "active" in l]
    assert summary, f"no summary line in stdout: {r.stdout!r}"
    assert "2 shipped" in summary[0], summary[0]
    assert "1 active" in summary[0], summary[0]
    assert "1 blocked" in summary[0], summary[0]
    assert "1 backlog" in summary[0], summary[0]

    # HTML mentions the recent shipped task but not the old one
    html_text = html_files[0].read_text()
    assert "T-SHIP" in html_text
    assert "S-1" in html_text
    assert "T-OLD" not in html_text

    # Fixture DBs unchanged
    after = _sha_files(*dbs.values())
    assert after == before, "board DBs mutated by report"
