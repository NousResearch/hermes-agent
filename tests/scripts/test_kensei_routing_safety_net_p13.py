"""P13 observer-proof test for scripts/kensei-routing-safety-net.py.

TEST ONLY — no source change. The script already resolves board DBs via
HERMES_HOME through _board_compat. This test builds temp canonical board
DBs with task rows covering: recent vs stale ready, updated_at fallback,
unassigned, human, known-bot, and unknown assignee. Asserts stdout
reports and unchanged DB hashes (read-only).
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
SCRIPT = REPO_ROOT / "scripts" / "kensei-routing-safety-net.py"


def _sha_files(*paths: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def _make_board(db_path: Path, rows: list[dict], *, with_updated: bool = True) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cols = "created_at INTEGER"
    if with_updated:
        cols += ", updated_at INTEGER"
    conn.executescript(
        f"CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        f"assignee TEXT, status TEXT NOT NULL, priority INTEGER DEFAULT 0, {cols});"
    )
    if with_updated:
        conn.executemany(
            "INSERT INTO tasks (id, title, assignee, status, priority, created_at, "
            "updated_at) VALUES (?,?,?,?,?,?,?)",
            [(r["id"], r["title"], r.get("assignee"), r["status"], r.get("priority", 0),
              r.get("created_at", 0), r.get("updated_at")) for r in rows],
        )
    else:
        conn.executemany(
            "INSERT INTO tasks (id, title, assignee, status, priority, created_at) "
            "VALUES (?,?,?,?,?,?)",
            [(r["id"], r["title"], r.get("assignee"), r["status"], r.get("priority", 0),
              r.get("created_at", 0)) for r in rows],
        )
    conn.commit()
    conn.close()


def _build_fixture(home: Path) -> dict[str, Path]:
    """Build canonical board DBs under <home>/kanban/boards/<slug>/kanban.db."""
    now = int(time.time())
    boards_dir = home / "kanban" / "boards"
    # core (legacy 'default'), security-ops (legacy 'ops'), content (legacy
    # 'content-lead'), research, apps — canonical slugs per _board_compat.
    core_db = boards_dir / "core" / "kanban.db"
    ops_db = boards_dir / "security-ops" / "kanban.db"
    _make_board(
        core_db,
        [
            # stale ready (>4h), unassigned -> DEAD ASSIGNEE (unassigned)
            {"id": "STALE-UNASSIGNED", "title": "Unassigned stale ready",
             "assignee": None, "status": "ready",
             "created_at": now - (10 * 3600), "updated_at": now - (10 * 3600)},
            # stale ready, human (sahil) -> awaiting human
            {"id": "STALE-HUMAN", "title": "Human stale ready",
             "assignee": "sahil", "status": "ready",
             "created_at": now - (10 * 3600), "updated_at": now - (10 * 3600)},
            # stale ready, known bot -> hasn't picked up
            {"id": "STALE-BOT", "title": "Bot stale ready",
             "assignee": "octacon", "status": "ready",
             "created_at": now - (10 * 3600), "updated_at": now - (10 * 3600)},
            # stale ready, unknown assignee -> DEAD ASSIGNEE (ghost)
            {"id": "STALE-UNKNOWN", "title": "Ghost stale ready",
             "assignee": "ghost", "status": "ready",
             "created_at": now - (10 * 3600), "updated_at": now - (10 * 3600)},
            # recent ready (<4h) -> NOT flagged
            {"id": "RECENT-READY", "title": "Recent ready",
             "assignee": None, "status": "ready",
             "created_at": now - 100, "updated_at": now - 100},
        ],
    )
    _make_board(
        ops_db,
        [
            # stale blocked (>24h) with updated_at
            {"id": "STALE-BLOCKED", "title": "Long blocked",
             "assignee": "remii", "status": "blocked",
             "created_at": now - (30 * 3600), "updated_at": now - (30 * 3600)},
            # recent blocked (<24h) -> NOT flagged
            {"id": "RECENT-BLOCKED", "title": "Recent blocked",
             "assignee": "misa-misa", "status": "blocked",
             "created_at": now - 100, "updated_at": now - 100},
        ],
    )
    # research board: NO updated_at column — staleness falls back to created_at
    research_db = boards_dir / "research" / "kanban.db"
    _make_board(
        research_db,
        [
            # stale blocked (>24h) via created_at fallback
            {"id": "BLOCKED-NO-UPDATE", "title": "Blocked no updated_at col",
             "assignee": "wesker", "status": "blocked",
             "created_at": now - (30 * 3600)},
            # recent blocked via created_at fallback -> NOT flagged
            {"id": "RECENT-NO-UPDATE", "title": "Recent blocked no col",
             "assignee": "gojo", "status": "blocked", "created_at": now - 100},
        ],
        with_updated=False,
    )
    return {"core": core_db, "security-ops": ops_db, "research": research_db}


def _run_script(home: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_routing_safety_net_reports_and_preserves_fixture(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    dbs = _build_fixture(home)
    before = _sha_files(*dbs.values())

    r = _run_script(home)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert out, "expected stdout report, got empty"

    # Header
    assert "## Routing Safety-Net Scan" in out
    # 4 stale ready (STALE-UNASSIGNED, STALE-HUMAN, STALE-BOT, STALE-UNKNOWN)
    # 2 stale blocked (STALE-BLOCKED via updated_at, BLOCKED-NO-UPDATE via created_at fallback)
    assert "4 stale ready, 2 stale blocked" in out, out

    # Assignee-note variants
    assert "DEAD ASSIGNEE (unassigned) — reassign" in out
    assert "awaiting human (sahil)" in out
    assert "octacon hasn't picked up" in out
    assert "DEAD ASSIGNEE (ghost) — reassign" in out
    assert "wesker hasn't picked up" in out
    # Recent rows absent
    assert "RECENT-READY" not in out
    assert "RECENT-BLOCKED" not in out
    assert "RECENT-NO-UPDATE" not in out

    # Fixture DBs unchanged
    after = _sha_files(*dbs.values())
    assert after == before, "board DBs mutated by safety-net"


def test_routing_safety_net_silent_when_no_stale(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    boards_dir = home / "kanban" / "boards"
    core_db = boards_dir / "core" / "kanban.db"
    now = int(time.time())
    _make_board(
        core_db,
        [
            # recent ready — not stale
            {"id": "RECENT", "title": "Recent", "assignee": "octacon",
             "status": "ready", "created_at": now - 100, "updated_at": now - 100},
        ],
    )
    before = _sha_files(core_db)

    r = _run_script(home)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "", f"expected silent, got: {r.stdout!r}"
    after = _sha_files(core_db)
    assert after == before, "board DBs mutated by safety-net"
