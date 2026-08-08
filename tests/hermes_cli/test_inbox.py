"""Tests for ``hermes inbox`` (hermes_cli/inbox.py)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import inbox as inbox_mod


DELEGATIONS_DDL = """CREATE TABLE async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at REAL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    task_json TEXT,
    delivery_claim TEXT,
    delivery_claimed_at REAL,
    origin_session_id TEXT NOT NULL DEFAULT ''
)"""


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    (tmp_path / "runtime").mkdir()
    (tmp_path / "cron").mkdir()
    return tmp_path


def _write_delegations(home: Path, rows):
    conn = sqlite3.connect(home / "state.db")
    conn.execute(DELEGATIONS_DDL)
    for r in rows:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, state, dispatched_at,
                completed_at, updated_at, delivery_state, delivery_attempts,
                task_json)
               VALUES (?, 's', ?, ?, ?, ?, ?, ?, ?)""",
            r,
        )
    conn.commit()
    conn.close()


class TestCollectProcesses:
    def test_missing_file_returns_empty(self, home):
        assert inbox_mod.collect_processes(home) == []

    def test_corrupt_file_returns_empty(self, home):
        (home / "processes.json").write_text("{not json")
        assert inbox_mod.collect_processes(home) == []

    def test_live_and_dead_pids_classified(self, home):
        now = time.time()
        (home / "processes.json").write_text(
            json.dumps(
                [
                    {
                        "session_id": "live",
                        "command": "sleep 5",
                        "pid": os.getpid(),
                        "pid_scope": "host",
                        "started_at": now - 60,
                    },
                    {
                        "session_id": "dead",
                        "command": "gone",
                        "pid": 2**22 + 12345,  # implausible PID
                        "pid_scope": "host",
                        "started_at": now - 60,
                    },
                    {
                        "session_id": "sandboxed",
                        "command": "in docker",
                        "pid": 42,
                        "pid_scope": "env",
                        "started_at": now - 60,
                    },
                ]
            )
        )
        procs = {p["id"]: p for p in inbox_mod.collect_processes(home)}
        assert procs["live"]["alive"] is True
        assert procs["dead"]["alive"] is False
        # Non-host scope PIDs can't be probed — unknown, not dead.
        assert procs["sandboxed"]["alive"] is None


class TestCollectDelegations:
    def test_missing_db_returns_empty(self, home):
        assert inbox_mod.collect_delegations(home) == []

    def test_running_pending_and_recent_dropped_surface(self, home):
        now = time.time()
        _write_delegations(
            home,
            [
                ("run", "running", now - 60, None, now, "pending", 0,
                 '{"goal": "active work"}'),
                ("done", "completed", now - 600, now - 30, now, "pending", 0,
                 '{"goal": "finished work"}'),
                ("dropped_new", "stalled", now - 7200, now - 3600, now,
                 "dropped", 5, '{"goal": "recent drop"}'),
                # Dropped 30 days ago — must be filtered as historical noise.
                ("dropped_old", "stalled", now - 30 * 86400,
                 now - 30 * 86400, now, "dropped", 5, '{"goal": "old drop"}'),
                # Already delivered — not inbox material.
                ("delivered", "completed", now - 600, now - 500, now,
                 "delivered", 1, '{"goal": "old news"}'),
            ],
        )
        ids = {d["id"] for d in inbox_mod.collect_delegations(home)}
        assert ids == {"run", "done", "dropped_new"}

    def test_goal_extracted_from_task_json(self, home):
        now = time.time()
        _write_delegations(
            home,
            [("run", "running", now, None, now, "pending", 0,
              '{"goal": "research pricing"}')],
        )
        (item,) = inbox_mod.collect_delegations(home)
        assert item["goal"] == "research pricing"


class TestBuildInbox:
    def test_sections_classified(self, home, monkeypatch):
        now = time.time()
        (home / "processes.json").write_text(
            json.dumps(
                [
                    {"session_id": "live", "command": "x", "pid": os.getpid(),
                     "pid_scope": "host", "started_at": now},
                    {"session_id": "dead", "command": "y",
                     "pid": 2**22 + 12345, "pid_scope": "host",
                     "started_at": now},
                ]
            )
        )
        _write_delegations(
            home,
            [
                ("run", "running", now, None, now, "pending", 0, "{}"),
                ("done", "completed", now - 60, now - 10, now, "pending", 0,
                 "{}"),
            ],
        )
        monkeypatch.setattr(inbox_mod, "collect_cron", lambda home=None: [
            {"kind": "cron", "id": "j1", "name": "ok", "enabled": True,
             "paused": False, "paused_reason": None, "last_status": "success",
             "last_error": None, "last_run_at": None, "next_run_at": "soon",
             "needs_attention": False},
            {"kind": "cron", "id": "j2", "name": "bad", "enabled": True,
             "paused": False, "paused_reason": None, "last_status": "error",
             "last_error": "boom", "last_run_at": None, "next_run_at": None,
             "needs_attention": True},
        ])
        data = inbox_mod.build_inbox(home)

        attention_ids = {i.get("id") for i in data["attention"]}
        assert "dead" in attention_ids  # orphaned checkpoint entry
        assert "j2" in attention_ids  # failed cron job
        in_progress_ids = {i.get("id") for i in data["in_progress"]}
        assert {"live", "run"} <= in_progress_ids
        assert [i["id"] for i in data["finished_undelivered"]] == ["done"]

    def test_render_smoke_and_json_roundtrip(self, home):
        data = inbox_mod.build_inbox(home)
        text = inbox_mod.render_inbox(data)
        assert "Needs attention" in text
        assert "In progress" in text
        # Must be JSON-serializable for --json.
        json.dumps(data, default=str)

    def test_empty_home_all_clear(self, home):
        data = inbox_mod.build_inbox(home)
        assert data["attention"] == []
        assert data["in_progress"] == []
        assert data["finished_undelivered"] == []


class TestActiveSessions:
    def test_dead_pid_leases_excluded(self, home):
        (home / "runtime" / "active_sessions.json").write_text(
            json.dumps(
                [
                    {"lease_id": "a", "session_id": "s1", "surface": "cli",
                     "pid": os.getpid(), "started_at": time.time()},
                    {"lease_id": "b", "session_id": "s2", "surface": "tg",
                     "pid": 2**22 + 12345, "started_at": time.time()},
                ]
            )
        )
        sessions = inbox_mod.collect_active_sessions(home)
        assert len(sessions) == 1
        assert sessions[0]["surface"] == "cli"
