#!/usr/bin/env python3
"""Regression test for profile-name leak in kanban task skills.

This guards against the t_47727c99 / t_ca595dd2 / t_6e6c889a crash-loop
class. The structural fix added profile-name validation in three places:

  1. create_task() in kanban_db.py — raises ValueError on profile leak
  2. _cmd_create in kanban.py — strips profile names with stderr warning
  3. dispatcher spawn loop in kanban_db.py — silently skips profile names

This test exercises path (1) — the deepest guard.

Run: python3 tests/test_skills_profile_leak.py
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Locate the kanban_db module
KANBAN_DB_PATH = Path(__file__).resolve().parent.parent / "hermes_cli" / "kanban_db.py"


class TestSkillsProfileLeak(unittest.TestCase):
    """create_task must reject profile names in the skills list."""

    def test_profile_name_in_skills_raises(self):
        from hermes_cli import kanban_db as kb
        # Use the real create_task with a tmp DB to exercise validation
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            _create_minimal_schema(conn)

            with self.assertRaises(ValueError) as ctx:
                kb.create_task(
                    conn,
                    title="Test ticket that should not be created",
                    assignee="code-craftsman",
                    skills=["code-craftsman", "totum-platform-audit"],
                )
            err = str(ctx.exception)
            self.assertIn("profile name", err)
            self.assertIn("code-craftsman", err)

    def test_real_skill_names_accepted(self):
        from hermes_cli import kanban_db as kb
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            _create_minimal_schema(conn)

            task_id = kb.create_task(
                conn,
                title="Test ticket that should be created",
                assignee="code-craftsman",
                skills=["code-craftsman-toolkit", "totum-platform-audit"],
            )
            self.assertTrue(task_id)
            row = conn.execute(
                "SELECT skills FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            stored = json.loads(row["skills"])
            self.assertEqual(
                sorted(stored),
                sorted(["code-craftsman-toolkit", "totum-platform-audit"]),
            )

    def test_mixed_profile_and_real_skill_raises(self):
        from hermes_cli import kanban_db as kb
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            _create_minimal_schema(conn)

            # Both profile and real skill — must still raise (atomic)
            with self.assertRaises(ValueError):
                kb.create_task(
                    conn,
                    title="Test mixed",
                    assignee="ideas-capture",
                    skills=["ideas-capture", "real-skill-name"],
                )


def _create_minimal_schema(conn):
    """Create minimal schema (tasks + task_events) for create_task() to succeed."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER,
            goal_mode INTEGER NOT NULL DEFAULT 0,
            goal_max_turns INTEGER,
            session_id TEXT,
            branch_name TEXT,
            board TEXT
        );
        CREATE TABLE IF NOT EXISTS task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL,
            run_id     INTEGER,
            kind       TEXT NOT NULL,
            payload    TEXT,
            created_at INTEGER NOT NULL
        );
    """)
    conn.commit()


if __name__ == "__main__":
    unittest.main()
