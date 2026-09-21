"""P13 observer-proof test for scripts/kanban-daily-report.py.

Verifies env-derived HERMES_HOME drives board DB discovery and report
output. A temp fixture covers each report section (shipped, in-progress,
blocked, new); the test asserts exactly one HTML attachment, the MEDIA:
path points at it, the summary carries exact counts, and the fixture
board DBs are byte-identical before/after (read-only).

This script is distinct from kanban-daily-digest (a shell aggregator) —
do not merge them.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "kanban-daily-report.py"


def _sha_files(*paths: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _make_board_db(db_path: Path, rows: list[dict]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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
        CREATE TABLE epics (id TEXT PRIMARY KEY, title TEXT);
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
    now = int(time.time())
    default_db = home / "kanban.db"
    _make_board_db(
        default_db,
        [
            # shipped within 24h
            {"id": "D-SHIP", "title": "Shipped today", "assignee": "octacon",
             "status": "done", "done_at": now - 100, "created_at": now - 90000, "epic_id": "ep1"},
            # in-progress updated within 24h
            {"id": "D-PROG", "title": "In progress now", "assignee": "misa-misa",
             "status": "running", "priority": 5, "updated_at": now - 200,
             "created_at": now - 90000, "epic_id": "ep1"},
            # blocked (no cutoff — always listed)
            {"id": "D-BLK", "title": "Blocked item", "assignee": "remii",
             "status": "blocked", "priority": 3, "status_reason": "dep",
             "created_at": now - 90000, "epic_id": "ep1"},
            # newly created within 24h
            {"id": "D-NEW", "title": "Fresh task", "assignee": "wesker",
             "status": "backlog", "priority": 1, "created_at": now - 50},
            # shipped older than 24h — must NOT appear in shipped
            {"id": "D-OLD", "title": "Old shipped", "assignee": "octacon",
             "status": "done", "done_at": now - (30 * 3600), "epic_id": "ep1"},
            # in-progress NOT updated within 24h — must NOT appear
            {"id": "D-PROG-OLD", "title": "Stale progress", "assignee": "misa-misa",
             "status": "running", "priority": 2, "updated_at": now - (30 * 3600),
             "created_at": now - 90000, "epic_id": "ep1"},
        ],
    )
    return {"default": default_db}


def _run_script(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_daily_report_uses_hermes_home_and_preserves_fixture(tmp_path: Path) -> None:
    home = tmp_path / "hermes_home"
    home.mkdir()
    dbs = _build_fixture(home)
    before = _sha_files(*dbs.values())

    r = _run_script(home)
    assert r.returncode == 0, r.stderr

    # Exactly one HTML attachment under <home>/kanban/reports
    reports_dir = home / "kanban" / "reports"
    html_files = sorted(reports_dir.glob("kanban-daily-*.html"))
    assert len(html_files) == 1, f"expected 1 report, got {html_files}"

    media_lines = [l for l in r.stdout.splitlines() if l.startswith("MEDIA:")]
    assert len(media_lines) == 1, f"expected one MEDIA line, got {media_lines}"
    media_path = Path(media_lines[0].split("MEDIA:", 1)[1].strip())
    assert media_path == html_files[0]

    # Summary line: 1 shipped, 1 in progress, 1 blocked, 1 new
    summary = [l for l in r.stdout.splitlines() if "shipped" in l and "in progress" in l]
    assert summary, f"no summary line: {r.stdout!r}"
    assert "1 shipped" in summary[0], summary[0]
    assert "1 in progress" in summary[0], summary[0]
    assert "1 blocked" in summary[0], summary[0]
    assert "1 new" in summary[0], summary[0]

    # HTML contains the fresh entries, not the old ones
    html_text = html_files[0].read_text()
    assert "D-SHIP" in html_text
    assert "D-PROG" in html_text
    assert "D-BLK" in html_text
    assert "D-NEW" in html_text
    assert "D-OLD" not in html_text
    assert "D-PROG-OLD" not in html_text

    # Fixture DBs unchanged
    after = _sha_files(*dbs.values())
    assert after == before, "board DBs mutated by daily report"
