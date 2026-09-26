#!/usr/bin/env python3
"""phantom_assignee_probe.py — structural monitor for kanban cards assigned
to profiles that do not exist (task-local copy to enable repository-relative tests).
This is a near-copy of the fleet-installed probe, intended for integration tests
that import the tracked script in-repo. Keep behavior identical to the fleet copy.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import FrozenSet

HERMES_HOME = Path(".hermes")  # tests will set HERMES_HOME env and Path.home()


def _fleet_home(home: Path) -> Path:
    for candidate in (home, home.parent.parent):
        if (candidate / "kanban" / "boards").is_dir() and (candidate / "profiles").is_dir():
            return candidate
    return home


def _canonical_external_assignees() -> FrozenSet[str]:
    source_roots = (
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[1] / "hermes-agent",
        Path(__file__).resolve().parents[2] / "hermes-agent",
        Path(__file__).resolve().parents[3] / "hermes-agent",
    )
    for root in source_roots:
        if (root / "hermes_cli" / "kanban_db.py").is_file():
            sys.path.insert(0, str(root))
            break
    try:
        return frozenset(importlib.import_module("hermes_cli.kanban_db").external_assignees())
    except Exception:
        return frozenset()


def known_profiles(profiles_dir: Path) -> set[str]:
    if not profiles_dir.is_dir():
        return set()
    out = set()
    for p in profiles_dir.iterdir():
        if p.is_dir() and not (p / ".deleted").exists():
            out.add(p.name)
    return out


CLOSED_STATUSES = ("done", "archived", "cancelled")


def is_phantom(assignee: str | None, profiles: set[str], external_seats: FrozenSet[str] | set[str] = frozenset()) -> bool:
    if assignee is None:
        return False
    name = assignee.strip()
    if not name:
        return False
    low = name.casefold()
    if low in {profile.casefold() for profile in profiles}:
        return False
    return low not in {seat.casefold() for seat in external_seats}


def scan_board(db_path: Path, board: str, profiles: set[str], external_seats: FrozenSet[str] | set[str]) -> list[dict]:
    uri = f"file:{db_path}?mode=ro"
    hits: list[dict] = []
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error as e:
        return [{"board": board, "error": f"open failed: {e}"}]
    try:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(CLOSED_STATUSES))
        rows = conn.execute(
            f"SELECT id, assignee, status, title, created_at FROM tasks "
            f"WHERE status NOT IN ({placeholders}) AND assignee IS NOT NULL",
            CLOSED_STATUSES,
        ).fetchall()
    except sqlite3.Error as e:
        return [{"board": board, "error": f"query failed: {e}"}]
    finally:
        conn.close()
    for r in rows:
        if is_phantom(r["assignee"], profiles, external_seats):
            hits.append({
                "board": board,
                "id": r["id"],
                "assignee": r["assignee"],
                "status": r["status"],
                "created_at": r["created_at"],
                "title": (r["title"] or "")[:80],
            })
    return hits


def probe(boards_dir: Path, profiles_dir: Path, external_seats: FrozenSet[str] | set[str] | None = None) -> list[dict]:
    profiles = known_profiles(profiles_dir)
    if external_seats is None:
        external_seats = _canonical_external_assignees()
    hits: list[dict] = []
    if not boards_dir.is_dir():
        return [{"board": "*", "error": f"boards dir missing: {boards_dir}"}]
    for bdir in sorted(boards_dir.iterdir()):
        if not bdir.is_dir() or bdir.name.startswith(("_", ".")):
            continue
        db = bdir / "kanban.db"
        if not db.is_file():
            continue
        hits.extend(scan_board(db, bdir.name, profiles, external_seats))
    return hits


# Minimal CLI for local debug if run from repo
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "profiles" / "real-profile").mkdir(parents=True)
            bdir = root / "kanban" / "boards" / "selftest"
            bdir.mkdir(parents=True)
            db = bdir / "kanban.db"
            conn = sqlite3.connect(db)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, assignee TEXT, status TEXT, created_at TEXT)"
            )
            conn.executemany(
                "INSERT INTO tasks VALUES (?,?,?,?,?)",
                [
                    ("t_phantom1", "stranded", "reviewer", "review", "2026-09-01"),
                    ("t_real1", "fine", "real-profile", "ready", "2026-09-01"),
                    ("t_ext1", "fine", "external-claude-x", "ready", "2026-09-01"),
                    ("t_allow1", "fine", "fable", "ready", "2026-09-01"),
                    ("t_closed1", "old", "writer", "done", "2026-09-01"),
                ],
            )
            conn.commit()
            conn.close()
            hits = probe(root / "kanban" / "boards", root / "profiles", frozenset({"external-claude-x","fable"}))
            print(json.dumps(hits))
    else:
        print(json.dumps([]))
