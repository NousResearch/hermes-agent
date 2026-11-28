#!/usr/bin/env python3
"""Test suite for the Dyad ↔ Hermes bridge script.

Creates a temporary Dyad SQLite DB with the real schema, temp project dirs,
and exercises every bridge command end-to-end.

Run:
    cd ~/.hermes/skills/devops/dyad-integration
    python3 scripts/test_bridge.py

Or with pytest:
    pytest scripts/test_bridge.py -v

No external dependencies — uses only stdlib (unittest, sqlite3, tempfile, subprocess).
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).resolve().parent.parent
BRIDGE = SKILL_DIR / "scripts" / "dyad_bridge.py"
SCHEMA_SQL = SKILL_DIR / "references" / "dyad-db-schema.md"

# Real schema DDL — extracted from Dyad v1.6.2, matches references/dyad-db-schema.md
SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS "apps" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "name" text NOT NULL,
    "path" text NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    "github_org" text,
    "github_repo" text,
    "supabase_project_id" text,
    "chat_context" text,
    "github_branch" text,
    "vercel_project_id" text,
    "vercel_project_name" text,
    "vercel_team_id" text,
    "vercel_deployment_url" text,
    "neon_project_id" text,
    "neon_development_branch_id" text,
    "neon_preview_branch_id" text,
    "install_command" text,
    "start_command" text,
    "is_favorite" integer DEFAULT 0 NOT NULL,
    "supabase_parent_project_id" text,
    "supabase_organization_slug" text,
    "theme_id" text,
    "neon_active_branch_id" text,
    "needs_app_blueprint" integer DEFAULT 0 NOT NULL,
    "neon_production_auth_cookie_secret" text,
    "neon_development_auth_cookie_secret" text,
    "collection_id" integer,
    "selected_database_branch_type" text
);

CREATE TABLE IF NOT EXISTS "chats" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "app_id" integer NOT NULL,
    "title" text,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "initial_commit_hash" text,
    "compacted_at" integer,
    "compaction_backup_path" text,
    "pending_compaction" integer,
    "chat_mode" text,
    FOREIGN KEY ("app_id") REFERENCES "apps"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "messages" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "chat_id" integer NOT NULL,
    "role" text NOT NULL,
    "content" text NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "approval_state" text,
    "commit_hash" text,
    "request_id" text,
    "source_commit_hash" text,
    "max_tokens_used" integer,
    "ai_messages_json" text,
    "model" text,
    "using_free_agent_mode_quota" integer,
    "is_compaction_summary" integer,
    FOREIGN KEY ("chat_id") REFERENCES "chats"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "mcp_servers" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "name" text NOT NULL,
    "transport" text NOT NULL,
    "command" text,
    "args" text,
    "env_json" text,
    "url" text,
    "enabled" integer DEFAULT 0 NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    "headers_json" text,
    "oauth_enabled" integer DEFAULT 0 NOT NULL,
    "oauth_state" text,
    "oauth_client_id" text,
    "oauth_client_secret" text,
    "oauth_scope" text,
    "oauth_callback_port" integer
);

CREATE TABLE IF NOT EXISTS "mcp_tool_consents" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "server_id" integer NOT NULL,
    "tool_name" text NOT NULL,
    "consent" text DEFAULT 'ask' NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    FOREIGN KEY ("server_id") REFERENCES "mcp_servers"("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "uniq_mcp_consent" ON "mcp_tool_consents" ("server_id","tool_name");

CREATE TABLE IF NOT EXISTS "prompts" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "title" text NOT NULL,
    "description" text,
    "content" text NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    "slug" text
);
CREATE UNIQUE INDEX IF NOT EXISTS "prompts_slug_unique" ON "prompts" ("slug");

CREATE TABLE IF NOT EXISTS "versions" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "app_id" integer NOT NULL,
    "commit_hash" text NOT NULL,
    "neon_db_timestamp" text,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    "is_favorite" integer DEFAULT 0 NOT NULL,
    "note" text,
    FOREIGN KEY ("app_id") REFERENCES "apps"("id") ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS "versions_app_commit_unique" ON "versions" ("app_id","commit_hash");

CREATE TABLE IF NOT EXISTS "app_collections" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "name" text NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS "app_collections_name_unique" ON "app_collections" ("name");

CREATE TABLE IF NOT EXISTS "language_model_providers" (
    "id" text PRIMARY KEY NOT NULL,
    "name" text NOT NULL,
    "api_base_url" text NOT NULL,
    "env_var_name" text,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL
);

CREATE TABLE IF NOT EXISTS "language_models" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "display_name" text NOT NULL,
    "api_name" text NOT NULL,
    "builtin_provider_id" text,
    "custom_provider_id" text,
    "description" text,
    "max_output_tokens" integer,
    "context_window" integer,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL,
    FOREIGN KEY ("custom_provider_id") REFERENCES "language_model_providers"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "custom_themes" (
    "id" integer PRIMARY KEY AUTOINCREMENT NOT NULL,
    "name" text NOT NULL,
    "description" text,
    "prompt" text NOT NULL,
    "created_at" integer DEFAULT (unixepoch()) NOT NULL,
    "updated_at" integer DEFAULT (unixepoch()) NOT NULL
);

CREATE TABLE IF NOT EXISTS "__drizzle_migrations" (
    "id" SERIAL PRIMARY KEY,
    "hash" text NOT NULL,
    "created_at" numeric
);
"""


def create_temp_dyad_db(db_path):
    """Create a temp Dyad SQLite DB with the real schema + WAL mode."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_DDL)
    # Enable WAL mode like the real Dyad
    conn.execute("PRAGMA journal_mode=WAL")
    conn.commit()
    conn.close()


class DyadBridgeTestBase(unittest.TestCase):
    """Base class — sets up temp DB, temp projects dir, and env vars."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.db_path = self.tmp / "sqlite.db"
        self.projects_dir = self.tmp / "dyad-apps"
        self.projects_dir.mkdir()

        create_temp_dyad_db(self.db_path)

        self.env = {
            **os.environ,
            "DYAD_DB_PATH": str(self.db_path),
            "DYAD_PROJECTS_DIR": str(self.projects_dir),
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


class TestListProjects(DyadBridgeTestBase):
    """list command — list all Dyad projects."""

    def test_empty_db(self):
        """list with no projects should succeed and say so."""
        rc, out, err = self.run_bridge("list")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No Dyad projects found", out)

    def test_with_project(self):
        """list with a project in the DB should show it."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO apps (name, path, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("test-app", "test-app", int(time.time()), int(time.time())),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("list")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("test-app", out)
        self.assertIn("ID", out)


class TestChats(DyadBridgeTestBase):
    """chats command — list chats for a project."""

    def test_no_chats(self):
        """chats for a project with no chats should say so."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO apps (name, path, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("test-app", "test-app", int(time.time()), int(time.time())),
        )
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("chats", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No chats found", out)

    def test_with_chat(self):
        """chats should show a chat that exists."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('app1', 'app1', ?, ?)", (ts, ts))
        conn.execute("INSERT INTO chats (app_id, title, created_at, chat_mode) VALUES (1, 'Build app', ?, 'build')", (ts,))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("chats", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Build app", out)
        self.assertIn("build", out)


class TestMessages(DyadBridgeTestBase):
    """messages command — read messages from a chat."""

    def test_no_messages(self):
        """messages for a chat with no messages should say so."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('app1', 'app1', ?, ?)", (ts, ts))
        conn.execute("INSERT INTO chats (app_id, created_at) VALUES (1, ?)", (ts,))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("messages", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No messages found", out)

    def test_with_messages(self):
        """messages should show user and assistant messages."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('app1', 'app1', ?, ?)", (ts, ts))
        conn.execute("INSERT INTO chats (app_id, created_at) VALUES (1, ?)", (ts,))
        conn.execute("INSERT INTO messages (chat_id, role, content, created_at, model) VALUES (1, 'user', 'Build a dashboard', ?, 'user')", (ts,))
        conn.execute("INSERT INTO messages (chat_id, role, content, created_at, model) VALUES (1, 'assistant', 'I will create...', ?, 'auto')", (ts,))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("messages", "1", "--limit", "10")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("USER", out)
        self.assertIn("ASSISTANT", out)
        self.assertIn("Build a dashboard", out)
        self.assertIn("I will create", out)

    def test_message_truncation(self):
        """Very long messages should be truncated in terminal output."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('app1', 'app1', ?, ?)", (ts, ts))
        conn.execute("INSERT INTO chats (app_id, created_at) VALUES (1, ?)", (ts,))
        long_content = "x" * 3000
        conn.execute("INSERT INTO messages (chat_id, role, content, created_at) VALUES (1, 'user', ?, ?)", (long_content, ts))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("messages", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("truncated", out)


class TestWriteRules(DyadBridgeTestBase):
    """write-rules command — write AI_RULES.md to a Dyad project."""

    def test_write_from_file(self):
        """write-rules with --file should create AI_RULES.md in the project dir."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('test-app', 'test-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        # Create project dir
        proj_dir = self.projects_dir / "test-app"
        proj_dir.mkdir()

        # Write rules file
        rules_content = "# Test Project\n\n## Tech Stack\n- React\n"
        rules_file = self.tmp / "rules.md"
        rules_file.write_text(rules_content)

        rc, out, err = self.run_bridge("write-rules", "test-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Wrote", out)
        self.assertIn("AI_RULES.md", out)

        # Verify file on disk
        rules_path = proj_dir / "AI_RULES.md"
        self.assertTrue(rules_path.exists(), "AI_RULES.md was not created")
        self.assertEqual(rules_path.read_text(), rules_content)

    def test_write_from_stdin(self):
        """write-rules with --stdin should read content from stdin."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('stdin-app', 'stdin-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        proj_dir = self.projects_dir / "stdin-app"
        proj_dir.mkdir()

        content = "# From Stdin\n\n## Context\n- Test conventions\n"
        rc, out, err = self.run_bridge("write-rules", "stdin-app", "--stdin", stdin_data=content)
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Wrote", out)

        rules_path = proj_dir / "AI_RULES.md"
        self.assertTrue(rules_path.exists())
        self.assertEqual(rules_path.read_text(), content)

    def test_write_overwrites_existing(self):
        """write-rules should overwrite an existing AI_RULES.md."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('overwrite-app', 'overwrite-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        proj_dir = self.projects_dir / "overwrite-app"
        proj_dir.mkdir()
        existing = proj_dir / "AI_RULES.md"
        existing.write_text("# OLD CONTENT\n")

        new_content = "# NEW CONTENT\n"
        rules_file = self.tmp / "new_rules.md"
        rules_file.write_text(new_content)

        rc, out, err = self.run_bridge("write-rules", "overwrite-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")

        self.assertEqual(existing.read_text(), new_content)

    def test_write_creates_project_dir_if_missing(self):
        """write-rules should create the project dir if it doesn't exist."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('missing-dir-app', 'missing-dir-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        # Don't create the project dir — the script should
        rules_file = self.tmp / "rules.md"
        rules_file.write_text("# Test\n")

        rc, out, err = self.run_bridge("write-rules", "missing-dir-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")

        proj_dir = self.projects_dir / "missing-dir-app"
        self.assertTrue(proj_dir.exists())
        self.assertTrue((proj_dir / "AI_RULES.md").exists())

    def test_write_no_file_or_stdin_errors(self):
        """write-rules without --file or --stdin should error."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('err-app', 'err-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        proj_dir = self.projects_dir / "err-app"
        proj_dir.mkdir()

        rc, out, err = self.run_bridge("write-rules", "err-app")
        self.assertNotEqual(rc, 0)
        self.assertIn("Error", err)

    def test_write_nonexistent_project_errors(self):
        """write-rules for a nonexistent project should error."""
        rules_file = self.tmp / "rules.md"
        rules_file.write_text("# Test\n")

        rc, out, err = self.run_bridge("write-rules", "no-such-project", "--file", str(rules_file))
        self.assertNotEqual(rc, 0)
        self.assertIn("No Dyad project matching", err)


class TestReadRules(DyadBridgeTestBase):
    """read-rules command — read AI_RULES.md from a Dyad project."""

    def test_read_existing(self):
        """read-rules should show the AI_RULES.md content."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('read-app', 'read-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        proj_dir = self.projects_dir / "read-app"
        proj_dir.mkdir()
        content = "# Read Test\n\n## Context\n- Something\n"
        (proj_dir / "AI_RULES.md").write_text(content)

        rc, out, err = self.run_bridge("read-rules", "read-app")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Read Test", out)
        self.assertIn("Something", out)

    def test_read_missing(self):
        """read-rules for a project with no AI_RULES.md should say so."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('no-rules', 'no-rules', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        proj_dir = self.projects_dir / "no-rules"
        proj_dir.mkdir()

        rc, out, err = self.run_bridge("read-rules", "no-rules")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No AI_RULES.md found", out)


class TestCreateProject(DyadBridgeTestBase):
    """create-project command — create a new Dyad project."""

    def test_create_default_path(self):
        """create-project with default path should create dir + register in DB."""
        rc, out, err = self.run_bridge("create-project", "--name", "new-app")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Created Dyad project", out)

        # Verify dir
        proj_dir = self.projects_dir / "new-app"
        self.assertTrue(proj_dir.exists())
        self.assertTrue((proj_dir / ".git").exists())
        self.assertTrue((proj_dir / "AI_RULES.md").exists())
        self.assertTrue((proj_dir / "package.json").exists())
        self.assertTrue((proj_dir / ".gitignore").exists())

    def test_create_custom_path(self):
        """create-project with --path should use the custom path and store it in DB."""
        custom = self.tmp / "custom-location" / "my-app"
        rc, out, err = self.run_bridge("create-project", "--name", "custom-app", "--path", str(custom))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue(custom.exists())
        self.assertTrue((custom / ".git").exists())

        # Verify the DB stores the absolute path, not the project name
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT path FROM apps WHERE name = 'custom-app'").fetchone()
        conn.close()
        self.assertIsNotNone(row, "Project not found in DB")
        self.assertEqual(row[0], str(custom), f"DB path should be '{custom}' but got '{row[0]}'")

    def test_create_custom_path_then_write_rules(self):
        """write-rules should resolve to the custom path after create-project."""
        custom = self.tmp / "custom-loc" / "rules-app"
        self.run_bridge("create-project", "--name", "rules-app", "--path", str(custom))

        # Write rules to the project — should land in the custom dir, not ~/dyad-apps/
        rules_file = self.tmp / "rules.md"
        rules_file.write_text("# Custom Path Rules\n")
        rc, out, err = self.run_bridge("write-rules", "rules-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")

        # AI_RULES.md must be in the custom directory
        self.assertTrue((custom / "AI_RULES.md").exists(),
                        f"AI_RULES.md not in custom dir {custom}")
        # And NOT in ~/dyad-apps/rules-app/
        self.assertFalse((self.projects_dir / "rules-app" / "AI_RULES.md").exists(),
                         "AI_RULES.md wrongly created in default dyad-apps dir")

    def test_create_default_path_stores_name(self):
        """create-project without --path should store the bare name in apps.path."""
        self.run_bridge("create-project", "--name", "default-path-app")

        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT path FROM apps WHERE name = 'default-path-app'").fetchone()
        conn.close()
        self.assertEqual(row[0], "default-path-app",
                         f"Default path should store bare name, got '{row[0]}'")

    def test_create_registers_in_db(self):
        """create-project should insert a row into the apps table."""
        rc, out, err = self.run_bridge("create-project", "--name", "db-app")
        self.assertEqual(rc, 0, f"stderr: {err}")

        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT name, path FROM apps WHERE name = 'db-app'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "db-app")
        self.assertEqual(row[1], "db-app")

    def test_create_starter_ai_rules(self):
        """create-project should write a starter AI_RULES.md with project name."""
        rc, out, err = self.run_bridge("create-project", "--name", "starter-app")
        self.assertEqual(rc, 0, f"stderr: {err}")

        rules = (self.projects_dir / "starter-app" / "AI_RULES.md").read_text()
        self.assertIn("starter-app", rules)
        self.assertIn("Project Context", rules)

    def test_create_package_json(self):
        """create-project should write a valid package.json."""
        import json
        rc, out, err = self.run_bridge("create-project", "--name", "pkg-app")
        self.assertEqual(rc, 0, f"stderr: {err}")

        pkg = json.loads((self.projects_dir / "pkg-app" / "package.json").read_text())
        self.assertEqual(pkg["name"], "pkg-app")
        self.assertIn("dev", pkg["scripts"])


class TestMcpServer(DyadBridgeTestBase):
    """MCP server commands — add, list, remove."""

    def test_add_stdio_server(self):
        """add-mcp with stdio transport should register in DB."""
        rc, out, err = self.run_bridge(
            "add-mcp", "--name", "Test Bridge",
            "--transport", "stdio",
            "--command", "python3",
            "--args", "/path/to/server.py",
            "--env", '{"TEST": "true"}',
        )
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Registered MCP server", out)
        self.assertIn("stdio", out)

        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT name, transport, command FROM mcp_servers").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "Test Bridge")
        self.assertEqual(row[1], "stdio")

    def test_add_sse_server(self):
        """add-mcp with sse transport should register with URL."""
        rc, out, err = self.run_bridge(
            "add-mcp", "--name", "Remote SSE",
            "--transport", "sse",
            "--url", "http://localhost:8082/sse",
        )
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Remote SSE", out)
        self.assertIn("http://localhost:8082/sse", out)

    def test_add_stdio_without_command_errors(self):
        """add-mcp stdio without --command should error."""
        rc, out, err = self.run_bridge(
            "add-mcp", "--name", "Bad", "--transport", "stdio",
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("command is required", err)

    def test_add_sse_without_url_errors(self):
        """add-mcp sse without --url should error."""
        rc, out, err = self.run_bridge(
            "add-mcp", "--name", "Bad", "--transport", "sse",
        )
        self.assertNotEqual(rc, 0)
        self.assertIn("url is required", err)

    def test_list_mcp_empty(self):
        """list-mcp with no servers should say so."""
        rc, out, err = self.run_bridge("list-mcp")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No MCP servers registered", out)

    def test_list_mcp_with_server(self):
        """list-mcp should show registered servers."""
        self.run_bridge("add-mcp", "--name", "Visible", "--transport", "sse", "--url", "http://x")
        rc, out, err = self.run_bridge("list-mcp")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Visible", out)
        self.assertIn("sse", out)

    def test_remove_mcp(self):
        """remove-mcp should delete the server from DB."""
        self.run_bridge("add-mcp", "--name", "To Remove", "--transport", "sse", "--url", "http://x")

        rc, out, err = self.run_bridge("remove-mcp", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Removed MCP server", out)

        # Verify it's gone
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM mcp_servers").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_remove_nonexistent_mcp_errors(self):
        """remove-mcp with invalid ID should error."""
        rc, out, err = self.run_bridge("remove-mcp", "999")
        self.assertNotEqual(rc, 0)
        self.assertIn("No MCP server with id", err)


class TestPrompts(DyadBridgeTestBase):
    """Prompt commands — add, list."""

    def test_add_prompt(self):
        """add-prompt should insert into the prompts table."""
        rc, out, err = self.run_bridge(
            "add-prompt", "--title", "Build Dashboard",
            "--description", "Standard dashboard",
            "--content", "Create a React dashboard...",
        )
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("Added prompt", out)

        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute("SELECT title, content FROM prompts").fetchone()
        conn.close()
        self.assertEqual(row[0], "Build Dashboard")
        self.assertEqual(row[1], "Create a React dashboard...")

    def test_list_prompts_empty(self):
        """list-prompts with no prompts should say so."""
        rc, out, err = self.run_bridge("list-prompts")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No reusable prompts found", out)

    def test_list_prompts_with_data(self):
        """list-prompts should show existing prompts."""
        self.run_bridge("add-prompt", "--title", "My Prompt", "--content", "Do stuff")
        rc, out, err = self.run_bridge("list-prompts")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("My Prompt", out)


class TestVersions(DyadBridgeTestBase):
    """versions command — list git versions for a project."""

    def test_no_versions(self):
        """versions for a project with no versions should say so."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('ver-app', 'ver-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("versions", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("No versions found", out)

    def test_with_version(self):
        """versions should show existing version entries."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('ver-app', 'ver-app', ?, ?)", (ts, ts))
        conn.execute("INSERT INTO versions (app_id, commit_hash, note, created_at, updated_at) VALUES (1, 'abc123def456', 'Initial version', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        rc, out, err = self.run_bridge("versions", "1")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("abc123def", out)
        self.assertIn("Initial version", out)


class TestSchema(DyadBridgeTestBase):
    """schema command — print the Dyad DB schema."""

    def test_schema_output(self):
        """schema should print CREATE TABLE statements."""
        rc, out, err = self.run_bridge("schema")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("CREATE TABLE", out)
        self.assertIn("apps", out)
        self.assertIn("mcp_servers", out)
        self.assertIn("messages", out)


class TestProjectResolution(DyadBridgeTestBase):
    """Test project resolution by name vs ID."""

    def test_resolve_by_name(self):
        """write-rules should resolve a project by name."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('named-app', 'named-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        (self.projects_dir / "named-app").mkdir()

        rules_file = self.tmp / "r.md"
        rules_file.write_text("# Test\n")
        rc, out, err = self.run_bridge("write-rules", "named-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue((self.projects_dir / "named-app" / "AI_RULES.md").exists())

    def test_resolve_by_id(self):
        """write-rules should resolve a project by numeric ID."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('id-app', 'id-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        (self.projects_dir / "id-app").mkdir()

        rules_file = self.tmp / "r.md"
        rules_file.write_text("# Test\n")
        rc, out, err = self.run_bridge("write-rules", "1", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue((self.projects_dir / "id-app" / "AI_RULES.md").exists())

    def test_resolve_by_path(self):
        """write-rules should resolve a project by path value."""
        import sqlite3, time
        conn = sqlite3.connect(str(self.db_path))
        ts = int(time.time())
        conn.execute("INSERT INTO apps (name, path, created_at, updated_at) VALUES ('path-app', 'path-app', ?, ?)", (ts, ts))
        conn.commit()
        conn.close()

        (self.projects_dir / "path-app").mkdir()

        rules_file = self.tmp / "r.md"
        rules_file.write_text("# Test\n")
        rc, out, err = self.run_bridge("write-rules", "path-app", "--file", str(rules_file))
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertTrue((self.projects_dir / "path-app" / "AI_RULES.md").exists())


class TestLiveDbSmokeTest(unittest.TestCase):
    """Smoke test against the real live Dyad DB (if installed).

    These are read-only and safe to run while Dyad is open.
    Skipped if Dyad is not installed.
    """

    def setUp(self):
        self.db_path = Path.home() / "Library" / "Application Support" / "dyad" / "sqlite.db"
        if not self.db_path.exists():
            self.skipTest("Dyad not installed — skipping live DB smoke test")

    def run_bridge_live(self, *args):
        proc = subprocess.run(
            [sys.executable, str(BRIDGE), *args],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def test_live_list(self):
        """list should work against the live DB."""
        rc, out, err = self.run_bridge_live("list")
        self.assertEqual(rc, 0, f"stderr: {err}")

    def test_live_schema(self):
        """schema should work against the live DB."""
        rc, out, err = self.run_bridge_live("schema")
        self.assertEqual(rc, 0, f"stderr: {err}")
        self.assertIn("CREATE TABLE", out)

    def test_live_list_mcp(self):
        """list-mcp should work against the live DB."""
        rc, out, err = self.run_bridge_live("list-mcp")
        self.assertEqual(rc, 0, f"stderr: {err}")


if __name__ == "__main__":
    unittest.main(verbosity=2)