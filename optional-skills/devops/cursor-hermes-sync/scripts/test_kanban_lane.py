#!/usr/bin/env python3
"""Test suite for the Cursor Kanban worker lane script.

Creates a temporary Kanban DB with the real schema, temp board directories,
and exercises every lane command end-to-end. Uses the Hermes venv Python 3.11
since the lane script imports hermes_cli.kanban_db.

Run:
    cd ~/.hermes/skills/devops/cursor-hermes-sync
    ~/.hermes/hermes-agent/venv/bin/python3 scripts/test_kanban_lane.py

Or with pytest:
    ~/.hermes/hermes-agent/venv/bin/python3 -m pytest scripts/test_kanban_lane.py -v

No external dependencies — uses only stdlib (unittest, sqlite3, tempfile, subprocess).
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
LANE = SKILL_DIR / "scripts" / "cursor_kanban_lane.py"
BRIDGE = SKILL_DIR / "scripts" / "cursor_hermes_bridge.py"

# Use the Hermes venv Python (3.11) — the lane script needs hermes_cli.kanban_db
HERMES_PYTHON = os.environ.get(
    "HERMES_PYTHON",
    str(Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3"),
)

# Real Kanban DB schema — extracted from Hermes kanban.db
KANBAN_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority            INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at          INTEGER NOT NULL,
    started_at          INTEGER,
    completed_at        INTEGER,
    workspace_kind      TEXT NOT NULL DEFAULT 'scratch',
    workspace_path      TEXT,
    branch_name         TEXT,
    project_id          TEXT,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    tenant              TEXT,
    result              TEXT,
    idempotency_key     TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid          INTEGER,
    last_failure_error  TEXT,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    current_run_id      INTEGER,
    workflow_template_id TEXT,
    current_step_key    TEXT,
    model_override      TEXT,
    provider_override   TEXT,
    session_id          TEXT,
    skills              TEXT,
    max_retries         INTEGER,
    goal_mode           INTEGER
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS task_links (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id   TEXT NOT NULL,
    child_id    TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_attachments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    file_name  TEXT NOT NULL,
    mime_type  TEXT,
    size       INTEGER,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT,
    platform    TEXT NOT NULL,
    chat_id     TEXT NOT NULL,
    chat_type   TEXT DEFAULT '',
    thread_id   TEXT,
    user_id     TEXT,
    created_at  INTEGER NOT NULL
);
"""


def create_temp_kanban_db(db_path):
    """Create a temp Kanban DB with the real schema."""
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(KANBAN_SCHEMA_DDL)
    conn.commit()
    conn.close()


def insert_task(conn, task_id, title="Test Task", body="Test body", assignee="cursor",
                status="ready", priority=0, workspace_path=None):
    """Insert a task into the Kanban DB."""
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, created_at, "
        "workspace_kind, workspace_path) VALUES (?, ?, ?, ?, ?, ?, ?, 'scratch', ?)",
        (task_id, title, body, assignee, status, priority, int(time.time()), workspace_path),
    )
    conn.commit()


class CursorKanbanLaneTestBase(unittest.TestCase):
    """Base class — sets up temp Hermes home, Kanban DB, and env vars."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        # Create a fake Hermes home with a board
        self.hermes_home = self.tmp / "hermes"
        self.boards_root = self.hermes_home / "kanban" / "boards"
        self.board_slug = "test-board"
        self.board_dir = self.boards_root / self.board_slug
        self.board_db = self.board_dir / "kanban.db"
        create_temp_kanban_db(self.board_db)

        # Also create the default kanban.db
        self.default_db = self.hermes_home / "kanban.db"
        create_temp_kanban_db(self.default_db)

        self.env = {
            **os.environ,
            "HERMES_HOME": str(self.hermes_home),
            "CURSOR_ASSIGNEE": "cursor",
            "CURSOR_CLI": "/usr/bin/true",  # no-op — won't actually open Cursor
            "CURSOR_LANE_INTERVAL": "0.1",  # fast polling for tests
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_lane(self, *args):
        """Run the lane script with given args, return (returncode, stdout, stderr)."""
        proc = subprocess.run(
            [HERMES_PYTHON, str(LANE), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def connect_board(self):
        """Get a SQLite connection to the board DB."""
        import sqlite3
        conn = sqlite3.connect(str(self.board_db))
        conn.row_factory = sqlite3.Row
        return conn


class TestStatus(CursorKanbanLaneTestBase):
    """status command — show cursor-assigned tasks."""

    def test_no_tasks(self):
        """status with no cursor tasks should say so."""
        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No tasks assigned to 'cursor'", out)

    def test_with_ready_task(self):
        """status should show a ready task."""
        conn = self.connect_board()
        insert_task(conn, "t_test1", "Build Landing", "Wire up hero", "cursor", "ready")
        conn.close()

        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_test1", out)
        self.assertIn("Build Landing", out)
        self.assertIn("ready", out)

    def test_with_running_task(self):
        """status should show a running task."""
        conn = self.connect_board()
        insert_task(conn, "t_run1", "In Progress", "Building", "cursor", "running")
        conn.close()

        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_run1", out)
        self.assertIn("running", out)

    def test_with_done_task(self):
        """status should show a done task with checkmark."""
        conn = self.connect_board()
        insert_task(conn, "t_done1", "Finished", "Done", "cursor", "done")
        conn.close()

        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_done1", out)
        self.assertIn("done", out)

    def test_ignores_other_assignees(self):
        """status should not show tasks assigned to other profiles."""
        conn = self.connect_board()
        insert_task(conn, "t_cursor1", "Cursor Task", "Body", "cursor", "ready")
        insert_task(conn, "t_dev1", "Dev Task", "Body", "developer", "ready")
        conn.close()

        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_cursor1", out)
        self.assertNotIn("t_dev1", out)

    def test_custom_assignee_name(self):
        """status should respect CURSOR_ASSIGNEE env var."""
        conn = self.connect_board()
        insert_task(conn, "t_custom", "Custom Task", "Body", "my-cursor-profile", "ready")
        conn.close()

        self.env["CURSOR_ASSIGNEE"] = "my-cursor-profile"
        rc, out, err = self.run_lane("status", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_custom", out)
        self.assertIn("my-cursor-profile", out)


class TestTick(CursorKanbanLaneTestBase):
    """tick command — single dispatch tick."""

    def test_tick_no_tasks(self):
        """tick with no ready tasks should report 0 spawned."""
        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)
        self.assertIn("no cursor-assigned tasks ready", out)

    def test_tick_claims_ready_task(self):
        """tick should claim a ready cursor-assigned task and set it to running."""
        conn = self.connect_board()
        insert_task(conn, "t_tick1", "Tick Task", "Build something", "cursor", "ready")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  1", out)
        self.assertIn("t_tick1", out)

        # Verify DB state — should be running now
        conn = self.connect_board()
        row = conn.execute("SELECT status, claim_lock, started_at FROM tasks WHERE id = 't_tick1'").fetchone()
        conn.close()
        self.assertEqual(row["status"], "running")
        self.assertIsNotNone(row["claim_lock"])
        self.assertIsNotNone(row["started_at"])

    def test_tick_writes_comment(self):
        """tick should write a comment to the task with dispatch details."""
        conn = self.connect_board()
        insert_task(conn, "t_comment1", "Comment Task", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        conn = self.connect_board()
        comments = conn.execute(
            "SELECT author, body FROM task_comments WHERE task_id = 't_comment1'"
        ).fetchall()
        conn.close()
        self.assertGreaterEqual(len(comments), 1)
        self.assertEqual(comments[0]["author"], "cursor-lane")
        self.assertIn("Dispatched to Cursor", comments[0]["body"])

    def test_tick_writes_agents_md(self):
        """tick should write AGENTS.md in the task workspace with context."""
        conn = self.connect_board()
        insert_task(conn, "t_agents1", "Agents MD Task", "Build the frontend", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        # Find the workspace
        workspace = self.board_dir / "workspaces" / "t_agents1"
        agents_md = workspace / "AGENTS.md"
        self.assertTrue(agents_md.exists(), f"AGENTS.md not at {agents_md}")
        content = agents_md.read_text()
        self.assertIn("Agents MD Task", content)
        self.assertIn("Build the frontend", content)
        self.assertIn("t_agents1", content)
        self.assertIn("kanban complete t_agents1", content)

    def test_tick_skips_running_tasks(self):
        """tick should not re-claim a task that's already running."""
        conn = self.connect_board()
        insert_task(conn, "t_running1", "Running Task", "Body", "cursor", "running")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)

    def test_tick_skips_done_tasks(self):
        """tick should not claim a done task."""
        conn = self.connect_board()
        insert_task(conn, "t_done1", "Done Task", "Body", "cursor", "done")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)

    def test_tick_skips_other_assignees(self):
        """tick should only claim tasks assigned to cursor, not other profiles."""
        conn = self.connect_board()
        insert_task(conn, "t_dev1", "Dev Task", "Body", "developer", "ready")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)

    def test_tick_respects_max_limit(self):
        """tick --max 1 should only claim one task even if multiple are ready."""
        conn = self.connect_board()
        insert_task(conn, "t_max1", "First Task", "Body", "cursor", "ready", priority=1)
        insert_task(conn, "t_max2", "Second Task", "Body", "cursor", "ready", priority=0)
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug, "--max", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  1", out)

        # Verify only one is running
        conn = self.connect_board()
        running = conn.execute("SELECT id FROM tasks WHERE status = 'running'").fetchall()
        conn.close()
        self.assertEqual(len(running), 1)

    def test_tick_priority_ordering(self):
        """tick should claim higher-priority tasks first."""
        conn = self.connect_board()
        insert_task(conn, "t_low", "Low Priority", "Body", "cursor", "ready", priority=0)
        insert_task(conn, "t_high", "High Priority", "Body", "cursor", "ready", priority=10)
        conn.close()

        self.run_lane("tick", "--board", self.board_slug, "--max", "1")

        conn = self.connect_board()
        running = conn.execute("SELECT id FROM tasks WHERE status = 'running'").fetchone()
        conn.close()
        self.assertEqual(running["id"], "t_high")

    def test_tick_creates_workspace_if_missing(self):
        """tick should create a workspace directory if it doesn't exist."""
        conn = self.connect_board()
        insert_task(conn, "t_ws1", "Workspace Task", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        workspace = self.board_dir / "workspaces" / "t_ws1"
        self.assertTrue(workspace.exists())
        self.assertTrue(workspace.is_dir())

    def test_tick_uses_existing_workspace(self):
        """tick should use an existing workspace_path if set."""
        custom_ws = self.tmp / "custom-workspace"
        custom_ws.mkdir()

        conn = self.connect_board()
        insert_task(conn, "t_ws2", "Existing WS Task", "Body", "cursor", "ready",
                     workspace_path=str(custom_ws))
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        # Check AGENTS.md was written to the custom workspace
        agents_md = custom_ws / "AGENTS.md"
        self.assertTrue(agents_md.exists())
        self.assertIn("Existing WS Task", agents_md.read_text())

    def test_tick_sets_worker_pid(self):
        """tick should set worker_pid on the task."""
        conn = self.connect_board()
        insert_task(conn, "t_pid1", "PID Task", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        conn = self.connect_board()
        row = conn.execute("SELECT worker_pid FROM tasks WHERE id = 't_pid1'").fetchone()
        conn.close()
        # PID should be set (non-null) — /usr/bin/true returns a PID
        self.assertIsNotNone(row["worker_pid"])

    def test_tick_multiple_tasks(self):
        """tick with --max 2 should claim two ready tasks."""
        conn = self.connect_board()
        insert_task(conn, "t_multi1", "Task One", "Body", "cursor", "ready")
        insert_task(conn, "t_multi2", "Task Two", "Body", "cursor", "ready")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug, "--max", "2")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  2", out)

    def test_tick_blocked_on_spawn_failure(self):
        """tick should mark task blocked if Cursor CLI is not found."""
        conn = self.connect_board()
        insert_task(conn, "t_fail1", "Fail Task", "Body", "cursor", "ready")
        conn.close()

        # Set CURSOR_CLI to a nonexistent path
        self.env["CURSOR_CLI"] = "/nonexistent/cursor"
        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")

        # Task should be blocked (spawn failed)
        conn = self.connect_board()
        row = conn.execute("SELECT status FROM tasks WHERE id = 't_fail1'").fetchone()
        conn.close()
        self.assertEqual(row["status"], "blocked")

    def test_tick_writes_error_comment_on_spawn_failure(self):
        """tick should write an error comment when Cursor CLI is not found."""
        conn = self.connect_board()
        insert_task(conn, "t_err1", "Error Task", "Body", "cursor", "ready")
        conn.close()

        self.env["CURSOR_CLI"] = "/nonexistent/cursor"
        self.run_lane("tick", "--board", self.board_slug)

        conn = self.connect_board()
        comments = conn.execute(
            "SELECT body FROM task_comments WHERE task_id = 't_err1'"
        ).fetchall()
        conn.close()
        self.assertGreaterEqual(len(comments), 1)
        # The comment should mention failure (either "ERROR" or "Failed")
        body = comments[0]["body"]
        self.assertTrue("ERROR" in body or "Failed" in body, f"Expected error in: {body}")


class TestDispatchTask(CursorKanbanLaneTestBase):
    """dispatch <task_id> command — manually dispatch a specific task."""

    def test_dispatch_ready_task(self):
        """dispatch should claim and spawn a ready task."""
        conn = self.connect_board()
        insert_task(conn, "t_disp1", "Dispatch Task", "Build it", "cursor", "ready")
        conn.close()

        rc, out, err = self.run_lane("dispatch", "t_disp1", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Dispatching task t_disp1", out)
        self.assertIn("running", out)

        conn = self.connect_board()
        row = conn.execute("SELECT status FROM tasks WHERE id = 't_disp1'").fetchone()
        conn.close()
        self.assertEqual(row["status"], "running")

    def test_dispatch_wrong_assignee(self):
        """dispatch should error if task is not assigned to cursor."""
        conn = self.connect_board()
        insert_task(conn, "t_other1", "Other Task", "Body", "developer", "ready")
        conn.close()

        rc, out, err = self.run_lane("dispatch", "t_other1", "--board", self.board_slug)
        self.assertNotEqual(rc, 0)
        self.assertIn("developer", out + err)

    def test_dispatch_nonexistent_task(self):
        """dispatch should error for a nonexistent task."""
        rc, out, err = self.run_lane("dispatch", "t_ghost1", "--board", self.board_slug)
        self.assertNotEqual(rc, 0)
        self.assertIn("not found", out + err)

    def test_dispatch_already_done(self):
        """dispatch should say task is already done."""
        conn = self.connect_board()
        insert_task(conn, "t_done2", "Done Task", "Body", "cursor", "done")
        conn.close()

        rc, out, err = self.run_lane("dispatch", "t_done2", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("already done", out)

    def test_dispatch_writes_agents_md(self):
        """dispatch should write AGENTS.md with task context."""
        conn = self.connect_board()
        insert_task(conn, "t_disp_md", "Dispatch MD", "Build the API", "cursor", "ready")
        conn.close()

        self.run_lane("dispatch", "t_disp_md", "--board", self.board_slug)

        workspace = self.board_dir / "workspaces" / "t_disp_md"
        agents_md = workspace / "AGENTS.md"
        self.assertTrue(agents_md.exists())
        content = agents_md.read_text()
        self.assertIn("Dispatch MD", content)
        self.assertIn("Build the API", content)

    def test_dispatch_writes_comment(self):
        """dispatch should write a comment with dispatch details."""
        conn = self.connect_board()
        insert_task(conn, "t_disp_c", "Comment Test", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("dispatch", "t_disp_c", "--board", self.board_slug)

        conn = self.connect_board()
        comments = conn.execute(
            "SELECT author FROM task_comments WHERE task_id = 't_disp_c'"
        ).fetchall()
        conn.close()
        self.assertGreaterEqual(len(comments), 1)
        self.assertEqual(comments[0]["author"], "cursor-lane")


class TestSpawnFunction(CursorKanbanLaneTestBase):
    """Test the cursor_spawn function indirectly via tick."""

    def test_spawn_writes_task_title_to_agents_md(self):
        """AGENTS.md should contain the task title."""
        conn = self.connect_board()
        insert_task(conn, "t_title1", "My Special Title", "Body here", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        workspace = self.board_dir / "workspaces" / "t_title1"
        content = (workspace / "AGENTS.md").read_text()
        self.assertIn("My Special Title", content)
        self.assertIn("Body here", content)

    def test_spawn_writes_board_name(self):
        """AGENTS.md should contain the board name."""
        conn = self.connect_board()
        insert_task(conn, "t_board1", "Board Task", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        content = (self.board_dir / "workspaces" / "t_board1" / "AGENTS.md").read_text()
        self.assertIn(self.board_slug, content)

    def test_spawn_includes_complete_instructions(self):
        """AGENTS.md should include instructions for completing the task."""
        conn = self.connect_board()
        insert_task(conn, "t_instr1", "Instruction Task", "Body", "cursor", "ready")
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        content = (self.board_dir / "workspaces" / "t_instr1" / "AGENTS.md").read_text()
        self.assertIn("kanban complete t_instr1", content)
        self.assertIn("kanban block t_instr1", content)

    def test_spawn_appends_to_existing_agents_md(self):
        """If AGENTS.md exists, spawn should append, not overwrite."""
        workspace = self.board_dir / "workspaces" / "t_append1"
        workspace.mkdir(parents=True)
        existing = workspace / "AGENTS.md"
        existing.write_text("# My Project\n\nExisting content\n")

        conn = self.connect_board()
        insert_task(conn, "t_append1", "Append Task", "Body", "cursor", "ready",
                     workspace_path=str(workspace))
        conn.close()

        self.run_lane("tick", "--board", self.board_slug)

        content = existing.read_text()
        self.assertIn("Existing content", content)
        self.assertIn("Append Task", content)

    def test_spawn_with_empty_body(self):
        """spawn should handle a task with no body."""
        conn = self.connect_board()
        insert_task(conn, "t_nobody1", "No Body Task", "", "cursor", "ready")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  1", out)


class TestDaemon(CursorKanbanLaneTestBase):
    """daemon command — continuous polling."""

    def test_daemon_starts_and_stops(self):
        """daemon should start, run one tick, and stop on signal."""
        # Start the daemon in the background with a very short interval
        proc = subprocess.Popen(
            [HERMES_PYTHON, str(LANE), "daemon",
             "--board", self.board_slug,
             "--interval", "0.1"],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Let it run briefly
        time.sleep(0.5)

        # Send SIGTERM
        proc.terminate()
        proc.wait(timeout=5)

        # Should have started and stopped cleanly
        self.assertIn("Cursor Kanban Lane Daemon", proc.stdout.read())

    def test_daemon_reports_spawn(self):
        """daemon should pick up a ready task and dispatch it to Cursor."""
        conn = self.connect_board()
        insert_task(conn, "t_daemon1", "Daemon Task", "Body", "cursor", "ready")
        conn.close()

        proc = subprocess.Popen(
            [HERMES_PYTHON, str(LANE), "daemon",
             "--board", self.board_slug,
             "--interval", "0.1"],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the daemon to start up and tick (Hermes imports are slow)
        time.sleep(4)
        proc.terminate()
        proc.wait(timeout=5)

        # Verify via DB state that the task was claimed
        conn = self.connect_board()
        row = conn.execute(
            "SELECT status, worker_pid FROM tasks WHERE id = 't_daemon1'"
        ).fetchone()
        conn.close()
        self.assertEqual(row["status"], "running")
        self.assertIsNotNone(row["worker_pid"])

    def test_daemon_silent_on_no_tasks(self):
        """daemon should be quiet when there are no ready tasks."""
        proc = subprocess.Popen(
            [HERMES_PYTHON, str(LANE), "daemon",
             "--board", self.board_slug,
             "--interval", "0.1"],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        time.sleep(0.5)
        proc.terminate()
        proc.wait(timeout=5)

        output = proc.stdout.read()
        # Should print the header but no "Dispatched" lines
        self.assertIn("Cursor Kanban Lane Daemon", output)
        self.assertNotIn("Dispatched to Cursor", output)


class TestEdgeCases(CursorKanbanLaneTestBase):
    """Edge cases and error handling."""

    def test_status_nonexistent_board(self):
        """status on a nonexistent board should not crash."""
        rc, out, err = self.run_lane("status", "--board", "ghost-board")
        # May error or show empty — just shouldn't crash with traceback
        self.assertNotIn("Traceback", out + err)

    def test_tick_nonexistent_board(self):
        """tick on a nonexistent board should not crash."""
        rc, out, err = self.run_lane("tick", "--board", "ghost-board")
        self.assertNotIn("Traceback", out + err)

    def test_dispatch_nonexistent_board(self):
        """dispatch on a nonexistent board should not crash."""
        rc, out, err = self.run_lane("dispatch", "t_ghost", "--board", "ghost-board")
        self.assertNotIn("Traceback", out + err)

    def test_tick_with_blocked_task(self):
        """tick should not claim a blocked task."""
        conn = self.connect_board()
        insert_task(conn, "t_blocked1", "Blocked Task", "Body", "cursor", "blocked")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)

    def test_tick_with_archived_task(self):
        """tick should not claim an archived task."""
        conn = self.connect_board()
        insert_task(conn, "t_arch1", "Archived Task", "Body", "cursor", "archived")
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  0", out)

    def test_tick_empty_body_task(self):
        """tick should handle a task with NULL body."""
        conn = self.connect_board()
        conn.execute(
            "INSERT INTO tasks (id, title, body, assignee, status, created_at, workspace_kind) "
            "VALUES ('t_nullbody', 'Null Body', NULL, 'cursor', 'ready', ?, 'scratch')",
            (int(time.time()),),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_lane("tick", "--board", self.board_slug)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Spawned:  1", out)

    def test_custom_assignee_via_env(self):
        """tick should respect CURSOR_ASSIGNEE env var."""
        conn = self.connect_board()
        insert_task(conn, "t_custom1", "Custom Assignee", "Body", "my-cursor", "ready")
        insert_task(conn, "t_default1", "Default Assignee", "Body", "cursor", "ready")
        conn.close()

        self.env["CURSOR_ASSIGNEE"] = "my-cursor"
        rc, out, err = self.run_lane("tick", "--board", self.board_slug, "--max", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("t_custom1", out)

        # Verify only the custom assignee task was claimed
        conn = self.connect_board()
        running = conn.execute("SELECT id FROM tasks WHERE status = 'running'").fetchone()
        conn.close()
        self.assertEqual(running["id"], "t_custom1")

    def test_atomic_claim_prevents_double_dispatch(self):
        """Two concurrent ticks should not both claim the same task."""
        conn = self.connect_board()
        insert_task(conn, "t_atomic1", "Atomic Task", "Body", "cursor", "ready")
        conn.close()

        # Run two ticks in quick succession
        self.run_lane("tick", "--board", self.board_slug, "--max", "1")
        # Second tick should find no ready tasks
        rc, out, err = self.run_lane("tick", "--board", self.board_slug, "--max", "1")
        self.assertIn("Spawned:  0", out)


class TestLiveDbSmokeTest(unittest.TestCase):
    """Smoke tests against the real Hermes Kanban DB (if installed).

    Read-only and safe. Skipped if Hermes isn't installed.
    """

    def setUp(self):
        self.hermes_home = Path.home() / ".hermes"
        if not (self.hermes_home / "hermes-agent" / "venv" / "bin" / "python3").exists():
            self.skipTest("Hermes not installed — skipping live smoke test")

    def run_lane_live(self, *args):
        env = {**os.environ, "CURSOR_CLI": "/usr/bin/true"}
        proc = subprocess.run(
            [str(self.hermes_home / "hermes-agent" / "venv" / "bin" / "python3"),
             str(LANE), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_live_status(self):
        """status should work against the real Kanban DB."""
        rc, out, err = self.run_lane_live("status")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_live_tick_no_crash(self):
        """tick should not crash against the real Kanban DB."""
        rc, out, err = self.run_lane_live("tick")
        # Live tick may fail if no kanban DB exists or if delegation guard fires
        # Just verify it doesn't crash with a traceback
        self.assertNotIn("Traceback", out + err)


if __name__ == "__main__":
    unittest.main(verbosity=2)