#!/usr/bin/env python3
"""Test suite for the Cursor ↔ Hermes two-way sync bridge script.

Creates temporary project dirs, a temp Cursor AI tracking DB, and a temp
Hermes state DB with the real schemas, then exercises every bridge command
end-to-end.

Run:
    cd ~/.hermes/skills/devops/cursor-hermes-sync
    python3 scripts/test_bridge.py

Or with pytest:
    pytest scripts/test_bridge.py -v

No external dependencies — uses only stdlib (unittest, sqlite3, tempfile, subprocess).
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
BRIDGE = SKILL_DIR / "scripts" / "cursor_hermes_bridge.py"

# Import REGISTRY_FILE from the bridge module for test verification
import importlib.util
_spec = importlib.util.spec_from_file_location("bridge", BRIDGE)
_bridge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bridge_mod)
REGISTRY_FILE = _bridge_mod.REGISTRY_FILE

# Real Cursor AI tracking DB schema — extracted from Cursor 3.15.1
CURSOR_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS ai_code_hashes (
    hash TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    fileExtension TEXT,
    fileName TEXT,
    requestId TEXT,
    conversationId TEXT,
    timestamp INTEGER,
    model TEXT,
    createdAt INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scored_commits (
    commitHash TEXT NOT NULL,
    branchName TEXT NOT NULL,
    scoredAt INTEGER NOT NULL,
    linesAdded INTEGER,
    linesDeleted INTEGER,
    tabLinesAdded INTEGER,
    tabLinesDeleted INTEGER,
    composerLinesAdded INTEGER,
    composerLinesDeleted INTEGER,
    humanLinesAdded INTEGER,
    humanLinesDeleted INTEGER,
    blankLinesAdded INTEGER,
    blankLinesDeleted INTEGER,
    commitMessage TEXT,
    commitDate TEXT,
    v1AiPercentage TEXT,
    v2AiPercentage TEXT,
    PRIMARY KEY (commitHash, branchName)
);

CREATE TABLE IF NOT EXISTS tracking_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversationId TEXT PRIMARY KEY,
    title TEXT,
    tldr TEXT,
    overview TEXT,
    summaryBullets TEXT,
    model TEXT,
    mode TEXT,
    updatedAt INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_file_content (
    hash TEXT PRIMARY KEY,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_deleted_files (
    hash TEXT PRIMARY KEY,
    fileName TEXT,
    deletedAt INTEGER NOT NULL
);
"""

# Minimal Hermes state.db sessions table schema
HERMES_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    cwd TEXT,
    created_at TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0
);
"""


def create_temp_cursor_db(db_path):
    """Create a temp Cursor AI tracking DB with the real schema."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CURSOR_SCHEMA_DDL)
    conn.commit()
    conn.close()


def create_temp_hermes_db(db_path):
    """Create a temp Hermes state.db with minimal sessions table."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.executescript(HERMES_SCHEMA_DDL)
    conn.commit()
    conn.close()


class CursorHermesBridgeTestBase(unittest.TestCase):
    """Base class — sets up temp dirs, DBs, and env vars."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)

        # Temp Cursor DB
        self.cursor_db = self.tmp / "ai-code-tracking.db"
        create_temp_cursor_db(self.cursor_db)

        # Temp Hermes state DB
        self.hermes_db = self.tmp / "state.db"
        create_temp_hermes_db(self.hermes_db)

        # Temp workspace for projects
        self.workspace = self.tmp / "workspace"
        self.workspace.mkdir()

        self.env = {
            **os.environ,
            "CURSOR_DB_PATH": str(self.cursor_db),
            "HERMES_STATE_DB_PATH": str(self.hermes_db),
            "SYNC_SCAN_PATHS": str(self.workspace),
            "HERMES_HOME": str(self.tmp),  # registry writes under HERMES_HOME
            # Suppress actual app launching during tests
            "CURSOR_CLI_PATH": "/usr/bin/true",  # no-op command
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def run_bridge(self, *args, stdin_data=None):
        """Run the bridge script with given args, return (returncode, stdout, stderr)."""
        proc = subprocess.run(
            [sys.executable, str(BRIDGE), *args],
            env=self.env,
            capture_output=True,
            text=True,
            input=stdin_data,
            timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def make_project(self, name="test-project", with_git=True):
        """Create a temp project directory, optionally with git init."""
        proj = self.workspace / name
        proj.mkdir()
        if with_git:
            subprocess.run(["git", "init", "-q"], cwd=str(proj), check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=str(proj), check=True, capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=str(proj), check=True, capture_output=True,
            )
        return proj


class TestInit(CursorHermesBridgeTestBase):
    """init command — initialize two-way sync for a project."""

    def test_init_creates_all_files(self):
        """init should create AGENTS.md, .cursor/rules, .agent-sync/state.json, .gitignore."""
        proj = self.make_project("init-test")

        rc, out, err = self.run_bridge("init", str(proj), "--name", "Init Test")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Two-way sync initialized", out)
        self.assertIn("Init Test", out)

        # Check all files exist
        self.assertTrue((proj / "AGENTS.md").exists(), "AGENTS.md not created")
        self.assertTrue((proj / ".agent-sync" / "state.json").exists(), "state.json not created")
        self.assertTrue((proj / ".cursor" / "rules" / "agent-sync.mdc").exists(), "Cursor rules not created")
        self.assertTrue((proj / ".gitignore").exists(), ".gitignore not created")

    def test_init_state_json_content(self):
        """state.json should contain project name, path, and timestamps."""
        proj = self.make_project("state-test")

        rc, out, err = self.run_bridge("init", str(proj), "--name", "State Test")
        self.assertEqual(rc, 0, f"stderr: {err}")

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertEqual(state["project_name"], "State Test")
        self.assertEqual(state["project_path"], str(proj.resolve()))
        self.assertIsNotNone(state["created"])
        self.assertIsNone(state["last_sync"])
        self.assertEqual(state["handoffs"], [])
        self.assertEqual(state["tasks"], [])

    def test_init_agents_md_content(self):
        """AGENTS.md should contain project name and standard sections."""
        proj = self.make_project("agents-test")

        rc, out, err = self.run_bridge("init", str(proj), "--name", "Agents Test")
        self.assertEqual(rc, 0, f"stderr: {err}")

        content = (proj / "AGENTS.md").read_text()
        self.assertIn("Agents Test", content)
        self.assertIn("## Build & Run", content)
        self.assertIn("## Architecture", content)
        self.assertIn("## Conventions", content)
        self.assertIn("## Current Tasks", content)
        self.assertIn("## Handoff Log", content)

    def test_init_cursor_rules_content(self):
        """Cursor rules should contain sync instructions."""
        proj = self.make_project("rules-test")

        rc, out, err = self.run_bridge("init", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")

        content = (proj / ".cursor" / "rules" / "agent-sync.mdc").read_text()
        self.assertIn("Hermes", content)
        self.assertIn("Cursor", content)
        self.assertIn("alwaysApply: true", content)
        self.assertIn(".agent-sync/state.json", content)

    def test_init_gitignore_includes_agent_sync(self):
        """.gitignore should include .agent-sync/."""
        proj = self.make_project("gitignore-test")

        rc, out, err = self.run_bridge("init", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")

        gitignore = (proj / ".gitignore").read_text()
        self.assertIn(".agent-sync/", gitignore)

    def test_init_appends_to_existing_gitignore(self):
        """init should append to an existing .gitignore, not overwrite it."""
        proj = self.make_project("existing-gitignore")
        (proj / ".gitignore").write_text("node_modules/\ndist/\n")

        rc, out, err = self.run_bridge("init", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")

        gitignore = (proj / ".gitignore").read_text()
        self.assertIn("node_modules/", gitignore)
        self.assertIn(".agent-sync/", gitignore)

    def test_init_with_context(self):
        """init --context should include extra context in AGENTS.md."""
        proj = self.make_project("context-test")

        rc, out, err = self.run_bridge(
            "init", str(proj), "--name", "Context Test",
            "--context", "This is a Next.js 15 app with InsForge backend.",
        )
        self.assertEqual(rc, 0, f"stderr: {err}")

        content = (proj / "AGENTS.md").read_text()
        self.assertIn("Next.js 15 app with InsForge backend", content)

    def test_init_without_git(self):
        """init should work without a git repo."""
        proj = self.make_project("no-git-test", with_git=False)

        rc, out, err = self.run_bridge("init", str(proj), "--name", "No Git")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue((proj / "AGENTS.md").exists())

    def test_init_resolves_git_root(self):
        """init from a subdirectory should resolve to the git root."""
        proj = self.make_project("git-root-test")
        subdir = proj / "src" / "components"
        subdir.mkdir(parents=True)

        rc, out, err = self.run_bridge("init", str(subdir), "--name", "Git Root Test")
        self.assertEqual(rc, 0, f"stderr: {err}")
        # Files should be at the git root, not the subdir
        self.assertTrue((proj / "AGENTS.md").exists(), "AGENTS.md should be at git root")
        self.assertFalse((subdir / "AGENTS.md").exists(), "AGENTS.md should not be in subdir")


class TestStatus(CursorHermesBridgeTestBase):
    """status command — show sync status."""

    def test_status_before_init(self):
        """status on a project without sync should show empty state."""
        proj = self.make_project("status-noinit")

        rc, out, err = self.run_bridge("status", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        # Should show project name (from directory) and no handoffs/tasks
        self.assertIn("status-noinit", out)
        self.assertIn("none", out.lower())

    def test_status_after_init(self):
        """status after init should show initialized state."""
        proj = self.make_project("status-init")
        self.run_bridge("init", str(proj), "--name", "Status After Init")

        rc, out, err = self.run_bridge("status", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Status After Init", out)
        self.assertIn("AGENTS.md: ✓ exists", out)
        self.assertIn("Cursor rules: ✓ exists", out)

    def test_status_shows_handoffs(self):
        """status should show recorded handoffs."""
        proj = self.make_project("status-handoffs")
        self.run_bridge("init", str(proj), "--name", "Handoff Status")

        # Add a handoff (use --to hermes to avoid opening Cursor)
        self.run_bridge("handoff", str(proj), "--to", "hermes", "--message", "Test handoff")

        rc, out, err = self.run_bridge("status", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Handoffs (1)", out)
        self.assertIn("Test handoff", out)


class TestSync(CursorHermesBridgeTestBase):
    """sync command — reconcile state between agents."""

    def test_sync_updates_timestamp(self):
        """sync should update last_sync timestamp."""
        proj = self.make_project("sync-test")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("sync", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Synced at", out)

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertIsNotNone(state["last_sync"])

    def test_sync_shows_counts(self):
        """sync should show Cursor chat and Hermes session counts."""
        proj = self.make_project("sync-counts")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("sync", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Cursor chats:", out)
        self.assertIn("Hermes sessions:", out)

    def test_sync_reads_cursor_chats(self):
        """sync should count Cursor chats from the DB."""
        proj = self.make_project("sync-cursor")
        self.run_bridge("init", str(proj))

        # Insert a chat summary into the Cursor DB
        import sqlite3
        conn = sqlite3.connect(str(self.cursor_db))
        conn.execute(
            "INSERT INTO conversation_summaries (conversationId, title, tldr, mode, model, updatedAt) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("conv1", "Test Chat", "A test", "agent", "gpt-4", int(time.time() * 1000),
        ))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("sync", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Cursor chats: 1", out)

    def test_sync_reads_hermes_sessions(self):
        """sync should count Hermes sessions from state.db."""
        proj = self.make_project("sync-hermes")
        self.run_bridge("init", str(proj))

        # Insert a session into the Hermes DB with resolved path
        import sqlite3
        conn = sqlite3.connect(str(self.hermes_db))
        resolved = str(proj.resolve())
        conn.execute(
            "INSERT INTO sessions (id, title, cwd, created_at, updated_at, message_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("sess1", "Test Session", resolved, "2026-01-01", "2026-01-01", 5),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("sync", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Hermes sessions: 1", out)


class TestWatch(CursorHermesBridgeTestBase):
    """watch command — detect and log file changes."""

    def test_watch_baseline(self):
        """First watch run should set baseline (no changes detected)."""
        proj = self.make_project("watch-baseline")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("watch", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertIsNotNone(state.get("last_watch_time"))

    def test_watch_detects_new_files(self):
        """Second watch run should detect new files."""
        proj = self.make_project("watch-new")
        self.run_bridge("init", str(proj))

        # Baseline watch
        self.run_bridge("watch", str(proj))

        # Create a file
        (proj / "src.ts").write_text('export function hello() { return "world"; }')

        # Second watch
        rc, out, err = self.run_bridge("watch", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Changed files: 1", out)

    def test_watch_detects_git_commits(self):
        """Watch should detect git commits and log them."""
        proj = self.make_project("watch-git")
        self.run_bridge("init", str(proj))

        # Initial commit + baseline watch
        (proj / "src.ts").write_text('export const x = 1;')
        subprocess.run(["git", "add", "-A"], cwd=str(proj), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=str(proj), check=True, capture_output=True)
        self.run_bridge("watch", str(proj))

        # Make changes + commit
        (proj / "src.ts").write_text('export const x = 2;')
        subprocess.run(["git", "add", "-A"], cwd=str(proj), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "update value"], cwd=str(proj), check=True, capture_output=True)

        # Second watch should detect the commit
        rc, out, err = self.run_bridge("watch", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")

        # Check changes.jsonl was written
        changes_file = proj / ".agent-sync" / "changes.jsonl"
        self.assertTrue(changes_file.exists())
        lines = changes_file.read_text().strip().split("\n")
        # Should have at least one git-type entry
        found_git = False
        for line in lines:
            entry = json.loads(line)
            if entry.get("type") == "git":
                found_git = True
                self.assertIn("update value", entry.get("commits", ""))
        self.assertTrue(found_git, "No git change entry found in changes.jsonl")

    def test_watch_ignores_agent_sync_dir(self):
        """Watch should not report changes in .agent-sync/."""
        proj = self.make_project("watch-ignore")
        self.run_bridge("init", str(proj))

        self.run_bridge("watch", str(proj))

        # Touch a file in .agent-sync (should be ignored)
        (proj / ".agent-sync" / "test.txt").write_text("test")

        rc, out, err = self.run_bridge("watch", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        # Should report 0 changed files (agent-sync is excluded)
        self.assertIn("Changed files: 0", out)

    def test_watch_ignores_node_modules(self):
        """Watch should not report changes in node_modules/."""
        proj = self.make_project("watch-nm")
        self.run_bridge("init", str(proj))

        self.run_bridge("watch", str(proj))

        # Create a file in node_modules
        nm = proj / "node_modules" / "some-pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {};")

        rc, out, err = self.run_bridge("watch", str(proj), "-v")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Changed files: 0", out)


class TestChanges(CursorHermesBridgeTestBase):
    """changes command — show recorded file changes."""

    def test_no_changes_file(self):
        """changes should say no changes if changes.jsonl doesn't exist."""
        proj = self.make_project("changes-empty")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("changes", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No changes recorded", out)

    def test_shows_recorded_changes(self):
        """changes should show entries from changes.jsonl."""
        proj = self.make_project("changes-show")
        self.run_bridge("init", str(proj))

        # Write changes directly
        changes_file = proj / ".agent-sync" / "changes.jsonl"
        changes_file.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "filesystem",
            "files": ["src/app.ts", "src/utils.ts"],
            "count": 2,
        }) + "\n")

        rc, out, err = self.run_bridge("changes", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("src/app.ts", out)
        self.assertIn("src/utils.ts", out)
        self.assertIn("Filesystem", out)

    def test_shows_git_changes(self):
        """changes should show git-type entries."""
        proj = self.make_project("changes-git")
        self.run_bridge("init", str(proj))

        changes_file = proj / ".agent-sync" / "changes.jsonl"
        changes_file.write_text(json.dumps({
            "timestamp": "2026-01-01T00:00:00+00:00",
            "type": "git",
            "from_sha": "abc1234",
            "to_sha": "def5678",
            "commits": "def5678 update feature",
            "diff_stat": "src/app.ts | 2 +-\n1 file changed",
        }) + "\n")

        rc, out, err = self.run_bridge("changes", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("abc1234", out)
        self.assertIn("def5678", out)
        self.assertIn("update feature", out)


class TestHandoff(CursorHermesBridgeTestBase):
    """handoff command — hand off work between agents."""

    def test_handoff_records_state(self):
        """handoff should record the handoff in state.json."""
        proj = self.make_project("handoff-state")
        self.run_bridge("init", str(proj))

        # Use --to hermes with a fake hermes binary to avoid actually launching
        rc, out, err = self.run_bridge(
            "handoff", str(proj), "--to", "hermes", "--message", "Built the API"
        )
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Handoff: cursor → hermes", out)
        self.assertIn("Built the API", out)

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertEqual(len(state["handoffs"]), 1)
        self.assertEqual(state["handoffs"][0]["from"], "cursor")
        self.assertEqual(state["handoffs"][0]["to"], "hermes")
        self.assertEqual(state["handoffs"][0]["message"], "Built the API")

    def test_handoff_appends_to_agents_md(self):
        """handoff should append to the Handoff Log in AGENTS.md."""
        proj = self.make_project("handoff-md")
        self.run_bridge("init", str(proj))

        self.run_bridge("handoff", str(proj), "--to", "hermes", "--message", "Test handoff note")

        content = (proj / "AGENTS.md").read_text()
        self.assertIn("cursor → hermes", content)
        self.assertIn("Test handoff note", content)

    def test_handoff_to_cursor(self):
        """handoff --to cursor should record direction as hermes → cursor."""
        proj = self.make_project("handoff-cursor")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge(
            "handoff", str(proj), "--to", "cursor", "--message", "Frontend done"
        )
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("hermes → cursor", out)

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertEqual(state["handoffs"][0]["from"], "hermes")
        self.assertEqual(state["handoffs"][0]["to"], "cursor")

    def test_handoff_multiple(self):
        """Multiple handoffs should accumulate in state.json."""
        proj = self.make_project("handoff-multi")
        self.run_bridge("init", str(proj))

        self.run_bridge("handoff", str(proj), "--to", "hermes", "--message", "First")
        self.run_bridge("handoff", str(proj), "--to", "cursor", "--message", "Second")

        import json
        state = json.loads((proj / ".agent-sync" / "state.json").read_text())
        self.assertEqual(len(state["handoffs"]), 2)
        self.assertEqual(state["handoffs"][0]["message"], "First")
        self.assertEqual(state["handoffs"][1]["message"], "Second")

    def test_handoff_default_to_cursor(self):
        """handoff without --to should default to cursor."""
        proj = self.make_project("handoff-default")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("handoff", str(proj), "--message", "Default direction")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("hermes → cursor", out)


class TestListProjects(CursorHermesBridgeTestBase):
    """list-projects command — list all synced projects."""

    def test_no_projects(self):
        """list-projects with no synced projects should say so."""
        rc, out, err = self.run_bridge("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No synced projects found", out)

    def test_with_project(self):
        """list-projects should show synced projects."""
        proj = self.make_project("list-test")
        self.run_bridge("init", str(proj), "--name", "Listable Project")

        rc, out, err = self.run_bridge("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Listable Project", out)
        self.assertIn(str(proj), out)

    def test_multiple_projects(self):
        """list-projects should show all synced projects."""
        proj1 = self.make_project("proj-a")
        self.run_bridge("init", str(proj1), "--name", "Project A")

        proj2 = self.make_project("proj-b")
        self.run_bridge("init", str(proj2), "--name", "Project B")

        rc, out, err = self.run_bridge("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Project A", out)
        self.assertIn("Project B", out)
        self.assertIn("Synced projects (2)", out)


class TestCursorChats(CursorHermesBridgeTestBase):
    """cursor-chats command — show Cursor AI chat history."""

    def test_no_chats(self):
        """cursor-chats with no chats should say so."""
        proj = self.make_project("cc-empty")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("cursor-chats", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No Cursor chats found", out)

    def test_with_chat(self):
        """cursor-chats should show chats from the Cursor DB."""
        proj = self.make_project("cc-with-chat")
        self.run_bridge("init", str(proj))

        import sqlite3
        conn = sqlite3.connect(str(self.cursor_db))
        conn.execute(
            "INSERT INTO conversation_summaries (conversationId, title, tldr, mode, model, updatedAt) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("conv1", "Build Dashboard", "Created a React dashboard", "agent", "claude-4", int(time.time() * 1000),
        ))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("cursor-chats", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Build Dashboard", out)
        self.assertIn("Created a React dashboard", out)

    def test_multiple_chats(self):
        """cursor-chats should show multiple chats ordered by recency."""
        proj = self.make_project("cc-multi")
        self.run_bridge("init", str(proj))

        import sqlite3
        conn = sqlite3.connect(str(self.cursor_db))
        conn.execute(
            "INSERT INTO conversation_summaries (conversationId, title, mode, updatedAt) VALUES (?, ?, ?, ?)",
            ("old", "Old Chat", "build", 1000),
        )
        conn.execute(
            "INSERT INTO conversation_summaries (conversationId, title, mode, updatedAt) VALUES (?, ?, ?, ?)",
            ("new", "New Chat", "agent", 2000),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("cursor-chats", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        # New chat should appear first (ordered by updatedAt DESC)
        self.assertLess(out.index("New Chat"), out.index("Old Chat"))


class TestHermesSessions(CursorHermesBridgeTestBase):
    """hermes-sessions command — show Hermes session history."""

    def test_no_sessions(self):
        """hermes-sessions with no sessions should say so."""
        proj = self.make_project("hs-empty")
        self.run_bridge("init", str(proj))

        rc, out, err = self.run_bridge("hermes-sessions", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No Hermes sessions found", out)

    def test_with_session(self):
        """hermes-sessions should show sessions from state.db."""
        proj = self.make_project("hs-with")
        self.run_bridge("init", str(proj))

        import sqlite3
        conn = sqlite3.connect(str(self.hermes_db))
        # Use the resolved path (project_root resolves symlinks)
        resolved = str(proj.resolve())
        conn.execute(
            "INSERT INTO sessions (id, title, cwd, created_at, updated_at, message_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", "Fix auth bug", resolved, "2026-01-01", "2026-01-02", 12),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("hermes-sessions", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Fix auth bug", out)
        self.assertIn("s1", out)


class TestEdgeCases(CursorHermesBridgeTestBase):
    """Edge cases and error handling."""

    def test_nonexistent_path(self):
        """init on a nonexistent path should still work (creates the dir)."""
        # project_root resolves the path; it doesn't need to exist first
        # but git root detection won't find anything — which is fine
        rc, out, err = self.run_bridge("init", str(self.tmp / "nonexistent"), "--name", "Ghost")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_status_on_nonexistent_project(self):
        """status on a nonexistent project should not crash."""
        rc, out, err = self.run_bridge("status", str(self.tmp / "ghost"))
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_watch_without_init(self):
        """watch without init should not crash (creates sync dir)."""
        proj = self.make_project("watch-noinit")
        rc, out, err = self.run_bridge("watch", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue((proj / ".agent-sync").exists())

    def test_changes_without_init(self):
        """changes without init should not crash."""
        proj = self.make_project("changes-noinit")
        rc, out, err = self.run_bridge("changes", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_reinit_overwrites_state(self):
        """Re-running init should not crash (files already exist)."""
        proj = self.make_project("reinit")
        self.run_bridge("init", str(proj), "--name", "First")
        rc, out, err = self.run_bridge("init", str(proj), "--name", "Second")
        self.assertEqual(rc, 0, f"stderr: {err}")
        content = (proj / "AGENTS.md").read_text()
        self.assertIn("Second", content)


class TestProjectResolution(CursorHermesBridgeTestBase):
    """Test project resolution by name, path, and registry — mirrors Dyad's resolve_project tests."""

    def test_resolve_by_existing_path(self):
        """resolve_project should use an existing path directly."""
        proj = self.make_project("resolve-path")
        self.run_bridge("init", str(proj), "--name", "Path Project")

        rc, out, err = self.run_bridge("status", str(proj))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Path Project", out)

    def test_resolve_by_registered_name(self):
        """resolve_project should resolve a bare name from the registry after init."""
        proj = self.make_project("resolve-name")
        self.run_bridge("init", str(proj), "--name", "Named Project")

        # Now use just the name, not the full path
        rc, out, err = self.run_bridge("status", "Named Project")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Named Project", out)
        self.assertIn(str(proj), out)

    def test_resolve_by_bare_name_under_scan_path(self):
        """resolve_project should resolve a bare name under the scan paths."""
        proj = self.make_project("bare-name")
        self.run_bridge("init", str(proj), "--name", "bare-name")

        # Use just the directory name (which is under workspace/)
        rc, out, err = self.run_bridge("status", "bare-name")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("bare-name", out)

    def test_resolve_falls_back_to_path_for_new_project(self):
        """resolve_project should fall back to treating identifier as a path for new projects."""
        # Path doesn't exist yet — should fall back gracefully
        rc, out, err = self.run_bridge("init", str(self.workspace / "brand-new"), "--name", "Brand New")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_handoff_by_name(self):
        """handoff should work with just the project name, not full path."""
        proj = self.make_project("handoff-by-name")
        self.run_bridge("init", str(proj), "--name", "Named Handoff")

        rc, out, err = self.run_bridge("handoff", "Named Handoff", "--to", "hermes", "--message", "By name")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("By name", out)

    def test_open_cursor_by_name(self):
        """open-cursor should resolve by registered name."""
        proj = self.make_project("open-by-name")
        self.run_bridge("init", str(proj), "--name", "Open By Name")

        rc, out, err = self.run_bridge("open-cursor", "Open By Name")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_watch_by_name(self):
        """watch should resolve by registered name."""
        proj = self.make_project("watch-by-name")
        self.run_bridge("init", str(proj), "--name", "Watch Named")

        rc, out, err = self.run_bridge("watch", "Watch Named")
        self.assertEqual(rc, 0, f"stderr: {err}")


class TestCreateProject(CursorHermesBridgeTestBase):
    """create-project command — like Dyad's create-project."""

    def test_create_default_path(self):
        """create-project with --name should create dir under first scan path + register."""
        rc, out, err = self.run_bridge("create-project", "--name", "new-app")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Created project 'new-app'", out)

        # Verify dir created under workspace (first scan path)
        proj = self.workspace / "new-app"
        self.assertTrue(proj.exists())
        self.assertTrue((proj / ".git").exists())
        self.assertTrue((proj / "AGENTS.md").exists())
        self.assertTrue((proj / ".agent-sync" / "state.json").exists())

    def test_create_custom_path(self):
        """create-project with --path should use the custom location."""
        custom = self.tmp / "custom-loc" / "my-app"
        rc, out, err = self.run_bridge("create-project", "--name", "custom-app", "--path", str(custom))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue(custom.exists())
        self.assertTrue((custom / ".git").exists())

    def test_create_with_context(self):
        """create-project --context should include context in AGENTS.md."""
        rc, out, err = self.run_bridge(
            "create-project", "--name", "ctx-app",
            "--context", "Next.js 15 with InsForge backend",
        )
        self.assertEqual(rc, 0, f"stderr: {err}")

        content = (self.workspace / "ctx-app" / "AGENTS.md").read_text()
        self.assertIn("Next.js 15 with InsForge backend", content)

    def test_create_registers_in_registry(self):
        """create-project should register the project so it's resolvable by name."""
        self.run_bridge("create-project", "--name", "reg-app")

        # Should be resolvable by name now
        rc, out, err = self.run_bridge("status", "reg-app")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("reg-app", out)

    def test_create_appears_in_list(self):
        """create-project should make the project appear in list-projects."""
        self.run_bridge("create-project", "--name", "listed-app")
        rc, out, err = self.run_bridge("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("listed-app", out)

    def test_create_with_custom_path_resolves_by_name(self):
        """create-project with --path should still register for name-based resolution."""
        custom = self.tmp / "far-away" / "named-app"
        self.run_bridge("create-project", "--name", "named-app", "--path", str(custom))

        # Should be resolvable by name even with custom path
        rc, out, err = self.run_bridge("status", "named-app")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn(str(custom), out)

    def test_create_does_not_overwrite_existing_git(self):
        """create-project on a dir that already has git should not re-init."""
        proj = self.workspace / "has-git"
        proj.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=str(proj), check=True, capture_output=True)

        rc, out, err = self.run_bridge("create-project", "--name", "has-git")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_create_writes_cursor_rules(self):
        """create-project should write .cursor/rules/agent-sync.mdc."""
        self.run_bridge("create-project", "--name", "rules-app")
        rules = self.workspace / "rules-app" / ".cursor" / "rules" / "agent-sync.mdc"
        self.assertTrue(rules.exists())
        self.assertIn("Hermes", rules.read_text())


class TestRegistry(CursorHermesBridgeTestBase):
    """Project registry — the JSON equivalent of Dyad's apps table."""

    def test_init_registers_project(self):
        """init should add the project to the registry."""
        proj = self.make_project("reg-init")
        self.run_bridge("init", str(proj), "--name", "Reg Init")

        # Check registry file (under temp HERMES_HOME)
        import json
        reg_path = self.tmp / "cursor-hermes-sync" / "projects.json"
        self.assertTrue(reg_path.exists(), f"Registry not at {reg_path}")
        registry = json.loads(reg_path.read_text())
        self.assertIn("Reg Init", registry)
        self.assertEqual(registry["Reg Init"]["path"], str(proj.resolve()))

    def test_create_project_registers(self):
        """create-project should add to the registry."""
        self.run_bridge("create-project", "--name", "reg-create")

        import json
        reg_path = self.tmp / "cursor-hermes-sync" / "projects.json"
        registry = json.loads(reg_path.read_text())
        self.assertIn("reg-create", registry)

    def test_registry_survives_across_commands(self):
        """Registry should persist between invocations."""
        proj1 = self.make_project("persist-1")
        self.run_bridge("init", str(proj1), "--name", "Persist One")

        proj2 = self.make_project("persist-2")
        self.run_bridge("init", str(proj2), "--name", "Persist Two")

        rc, out, err = self.run_bridge("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Persist One", out)
        self.assertIn("Persist Two", out)


class TestLiveDbSmokeTest(unittest.TestCase):
    """Smoke tests against real Cursor and Hermes DBs (if installed).

    Read-only and safe. Skipped if the apps aren't installed.
    """

    def setUp(self):
        self.cursor_db = Path.home() / ".cursor" / "ai-tracking" / "ai-code-tracking.db"
        self.hermes_db = Path.home() / ".hermes" / "state.db"
        if not self.cursor_db.exists() and not self.hermes_db.exists():
            self.skipTest("Neither Cursor nor Hermes installed — skipping live smoke tests")

    def run_bridge_live(self, *args):
        proc = subprocess.run(
            [sys.executable, str(BRIDGE), *args],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_live_cursor_chats(self):
        """cursor-chats should work against the live Cursor DB."""
        if not self.cursor_db.exists():
            self.skipTest("Cursor not installed")
        # Create a temp project to query
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, err = self.run_bridge_live("cursor-chats", tmp)
            self.assertEqual(rc, 0, f"stderr: {err}")

    def test_live_list_projects(self):
        """list-projects should work against real workspace."""
        rc, out, err = self.run_bridge_live("list-projects")
        self.assertEqual(rc, 0, f"stderr: {err}")


# Needed by the changes test for json.dumps
import json

if __name__ == "__main__":
    unittest.main(verbosity=2)