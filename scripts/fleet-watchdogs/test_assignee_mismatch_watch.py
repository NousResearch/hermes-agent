#!/usr/bin/env python3
"""Regression tests for the assignee-mismatch auditor watchdog (fix-C2, t_9efe82a4).

Run:  scripts/run_tests.sh scripts/fleet-watchdogs/test_assignee_mismatch_watch.py
or:   python3 -m pytest scripts/fleet-watchdogs/test_assignee_mismatch_watch.py -q
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_TARGET = _HERE / "assignee-mismatch-watch.py"

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("amw", _TARGET)
aw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aw)


@pytest.fixture
def db(tmp_path):
    con = sqlite3.connect(str(tmp_path / "t.db"))
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT,
            status TEXT, created_at INTEGER, completed_at INTEGER, block_kind TEXT);
        CREATE TABLE task_events(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            run_id INTEGER, kind TEXT, payload TEXT, created_at INTEGER);
        CREATE TABLE task_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            profile TEXT, outcome TEXT, summary TEXT);
        CREATE TABLE task_comments(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT,
            author TEXT, body TEXT, created_at INTEGER);
    """)
    yield con
    con.close()


def mint(db, tid, title, payload, status="todo", minted_at=None):
    now = minted_at or int(time.time())
    db.execute("INSERT INTO tasks(id,title,body,assignee,status,created_at) VALUES(?,?,?,?,?,?)",
               (tid, title, "", json.loads(payload).get("assignee"), status, now))
    db.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES(?,?,?,?)",
               (tid, "created", payload, now))
    db.commit()


def flagged_ids(db, days=30):
    flags, _ = aw.scan(db, days)
    return {c["id"] for c in flags}


def test_ac1_confirmed_build_lane_misassignments_flagged(db):
    """A [Bob]/build-lane card minted to rodge/axel/steve-o is a genuine mismatch."""
    mint(db, "t_a", "[Bob] Implement orgagent health router", '{"assignee": "rodge"}')
    mint(db, "t_b", "[Bob] Fix brain search grounding", '{"assignee": "axel"}')
    mint(db, "t_c", "Implement Slack draft launcher", '{"assignee": "steve-o"}')
    assert {"t_a", "t_b", "t_c"} <= flagged_ids(db)


def test_ac2_zero_false_positive_on_legit_lanes(db):
    """Legit review->rodge, verify->steve-o, triage->jobsy, deploy->default never flag."""
    cases = [
        ("t1", "Review orgagent PR against AC 1-7", '{"assignee": "rodge"}'),
        ("t2", "Verify AC 3-6 against served backend", '{"assignee": "steve-o"}'),
        ("t3", "[triage] Implement the Slack draft launcher", '{"assignee": "jobsy"}'),
        ("t4", "Deploy: squash-merge orgagent PR + restart", '{"assignee": "default"}'),
        ("t5", "Adjudicate cost-cap for test card", '{"assignee": "steve-o"}'),
        ("t6", "Wire proof: karl turn on the restored config", '{"assignee": "karl"}'),
        ("t7", "DECIDE pre-review-gate configuration + commission", '{"assignee": "steve-o"}'),
        ("t8", "[Rodge] Review E1 promotion manifest", '{"assignee": "rodge"}'),
        ("t9", "[Steve-o] QA E1 promotion", '{"assignee": "steve-o"}'),
    ]
    for tid, title, payload in cases:
        mint(db, tid, title, payload)
    assert flagged_ids(db) == set()


def test_ac3_auto_decomposer_gap_never_flagged(db):
    """Auto-decomposer children carry no mint assignee (recording gap) - never a flag."""
    mint(db, "t_g", "Implement Gmail draft launcher",
         '{"by": "auto-decomposer", "from_decompose_of": "x"}')
    _, gaps = aw.scan(db, 30)
    assert flagged_ids(db) == set()
    assert any(g["id"] == "t_g" and g["mint_src"] == "gap" for g in gaps)


def test_selftest_passes():
    ok, msg = aw.run_selftest()
    assert ok, msg


def test_apply_blocks_actionable_and_is_idempotent(db):
    """A live (todo, never-run) build-lane->rodge card is commented+blocked ONCE;
    a second apply run is a no-op (no duplicate comment)."""
    mint(db, "t_live", "[Bob] Implement live-flag check", '{"assignee": "rodge"}')
    flags, _ = aw.scan(db, 30)
    acted = aw.apply_actions(db, flags, commit=True)

    assert len(acted) == 1 and acted[0]["id"] == "t_live"
    row = db.execute("SELECT status, block_kind FROM tasks WHERE id='t_live'").fetchone()
    assert row["status"] == "blocked" and row["block_kind"] == "needs_input"
    n = db.execute("SELECT COUNT(*) n FROM task_comments WHERE task_id='t_live'").fetchone()["n"]
    assert n == 1
    n_ev = db.execute("SELECT COUNT(*) n FROM task_events WHERE task_id='t_live' AND kind='blocked'").fetchone()["n"]
    assert n_ev == 1

    # idempotent: already flagged => skipped
    acted2 = aw.apply_actions(db, flags, commit=True)
    assert acted2 == []
    n2 = db.execute("SELECT COUNT(*) n FROM task_comments WHERE task_id='t_live'").fetchone()["n"]
    assert n2 == 1