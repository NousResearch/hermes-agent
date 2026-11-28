#!/usr/bin/env python3
"""Dyad ↔ Hermes Bridge — manage Dyad projects, context, and MCP servers from Hermes.

Usage:
    python3 dyad_bridge.py list
    python3 dyad_bridge.py chats <app_id>
    python3 dyad_bridge.py messages <chat_id> [--limit N]
    python3 dyad_bridge.py write-rules <project_name_or_id> [--file PATH | --stdin]
    python3 dyad_bridge.py read-rules <project_name_or_id>
    python3 dyad_bridge.py create-project --name NAME [--path PATH]
    python3 dyad_bridge.py add-mcp --name NAME --transport stdio --command CMD [--args ARGS] [--env JSON]
    python3 dyad_bridge.py add-mcp --name NAME --transport sse --url URL [--headers JSON]
    python3 dyad_bridge.py list-mcp
    python3 dyad_bridge.py remove-mcp <server_id>
    python3 dyad_bridge.py add-prompt --title T --content C [--description D]
    python3 dyad_bridge.py list-prompts
    python3 dyad_bridge.py versions <app_id>
    python3 dyad_bridge.py schema

All read operations are safe to run while Dyad is open (WAL mode).
Write operations (create-project, add-mcp, remove-mcp, add-prompt) should be done
with Dyad closed to avoid lock conflicts. If you get "database is locked", close
Dyad and retry.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# Env overrides allow tests to point at a temp DB + temp projects dir.
DYAD_DB = Path(os.environ.get("DYAD_DB_PATH", str(Path.home() / "Library" / "Application Support" / "dyad" / "sqlite.db")))
DYAD_PROJECTS_DIR = Path(os.environ.get("DYAD_PROJECTS_DIR", str(Path.home() / "dyad-apps")))


def connect_db():
    """Connect to the Dyad SQLite DB in read-only mode by default."""
    if not DYAD_DB.exists():
        print(f"Error: Dyad DB not found at {DYAD_DB}", file=sys.stderr)
        print("Is Dyad installed? Download from https://dyad.sh", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(DYAD_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def connect_db_write():
    """Connect to the Dyad SQLite DB in read-write mode with retry."""
    if not DYAD_DB.exists():
        print(f"Error: Dyad DB not found at {DYAD_DB}", file=sys.stderr)
        sys.exit(1)
    for attempt in range(3):
        try:
            conn = sqlite3.connect(str(DYAD_DB), timeout=15)
            conn.row_factory = sqlite3.Row
            # Test we can write
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
            return conn
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                print(f"Database locked (attempt {attempt+1}/3), retrying in 2s...", file=sys.stderr)
                time.sleep(2)
            else:
                print(f"Error: Cannot acquire write lock on Dyad DB: {e}", file=sys.stderr)
                print("Close Dyad and retry.", file=sys.stderr)
                sys.exit(1)


def resolve_project(identifier):
    """Resolve a project name or numeric ID to a row dict."""
    conn = connect_db()
    try:
        if identifier.isdigit():
            row = conn.execute("SELECT * FROM apps WHERE id = ?", (int(identifier),)).fetchone()
        else:
            row = conn.execute("SELECT * FROM apps WHERE name = ? OR path = ?", (identifier, identifier)).fetchone()
    finally:
        conn.close()
    if not row:
        print(f"Error: No Dyad project matching '{identifier}'", file=sys.stderr)
        print(f"Available projects:", file=sys.stderr)
        cmd_list()
        sys.exit(1)
    return dict(row)


def project_dir(app_row):
    """Get the absolute filesystem path for a Dyad project.

    Dyad stores either a bare project name (relative, resolved under
    DYAD_PROJECTS_DIR) or an absolute path (when --path was used).
    """
    path = app_row["path"]
    p = Path(path)
    # If it's already absolute, use it directly
    if p.is_absolute():
        return p
    # Otherwise resolve under DYAD_PROJECTS_DIR
    return DYAD_PROJECTS_DIR / path


def fmt_ts(ts):
    """Format a Unix timestamp to ISO date string."""
    if ts is None:
        return ""
    try:
        from datetime import datetime
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_list():
    """List all Dyad projects."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, name, path, created_at, updated_at, github_repo, vercel_deployment_url "
            "FROM apps ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No Dyad projects found. Create one with: dyad_bridge.py create-project --name <name>")
        return

    print(f"{'ID':>4}  {'Name':<24}  {'Path':<20}  {'Created':<18}  {'GitHub':<20}  {'Deployed'}")
    print("-" * 100)
    for r in rows:
        print(f"{r['id']:>4}  {r['name']:<24}  {r['path']:<20}  {fmt_ts(r['created_at']):<18}  "
              f"{(r['github_repo'] or ''):<20}  {r['vercel_deployment_url'] or ''}")


def cmd_chats(app_id):
    """List chats for a project."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, app_id, title, created_at, chat_mode, pending_compaction "
            "FROM chats WHERE app_id = ? ORDER BY created_at DESC",
            (app_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"No chats found for app_id={app_id}")
        return

    print(f"{'ID':>4}  {'Title':<30}  {'Created':<18}  {'Mode':<10}  {'Pending Compaction'}")
    print("-" * 85)
    for r in rows:
        title = (r["title"] or "(untitled)")[:30]
        print(f"{r['id']:>4}  {title:<30}  {fmt_ts(r['created_at']):<18}  "
              f"{r['chat_mode'] or '':<10}  {bool(r['pending_compaction'])}")


def cmd_messages(chat_id, limit=50):
    """Read messages from a chat."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, role, content, created_at, model, approval_state, commit_hash "
            "FROM messages WHERE chat_id = ? ORDER BY created_at ASC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"No messages found for chat_id={chat_id}")
        return

    for r in rows:
        print(f"\n{'─' * 80}")
        print(f"[{r['id']}] {r['role'].upper()} · {fmt_ts(r['created_at'])} · model={r['model'] or '?'}")
        if r["approval_state"]:
            print(f"  approval: {r['approval_state']}")
        if r["commit_hash"]:
            print(f"  commit: {r['commit_hash']}")
        print()
        # Truncate very long messages for terminal display
        content = r["content"]
        if len(content) > 2000:
            print(content[:1800])
            print(f"\n... ({len(content)} chars total, truncated)")
        else:
            print(content)


def cmd_write_rules(identifier, file_path=None, use_stdin=False):
    """Write AI_RULES.md into a Dyad project."""
    app = resolve_project(identifier)
    pdir = project_dir(app)

    if use_stdin:
        content = sys.stdin.read()
    elif file_path:
        content = Path(file_path).read_text()
    else:
        print("Error: provide --file PATH or --stdin", file=sys.stderr)
        sys.exit(1)

    rules_path = pdir / "AI_RULES.md"

    # Create project dir if it doesn't exist
    pdir.mkdir(parents=True, exist_ok=True)

    # Write the file
    rules_path.write_text(content)

    print(f"✓ Wrote {len(content)} bytes to {rules_path}")
    print(f"  Project: {app['name']} (id={app['id']})")
    print(f"  Dyad will load this as authoritative context on the next chat turn.")


def cmd_read_rules(identifier):
    """Read the current AI_RULES.md from a Dyad project."""
    app = resolve_project(identifier)
    pdir = project_dir(app)
    rules_path = pdir / "AI_RULES.md"

    if not rules_path.exists():
        print(f"No AI_RULES.md found in {pdir}")
        print(f"Create one with: dyad_bridge.py write-rules {app['name']} --file <path>")
        return

    content = rules_path.read_text()
    print(f"┌─ AI_RULES.md — {app['name']} ({len(content)} bytes) ─────────────────────────")
    print(content)
    print(f"└─────────────────────────────────────────────────────────────")


def cmd_create_project(name, path=None):
    """Create a new Dyad project directory and register it in the DB."""
    if path is None:
        path = DYAD_PROJECTS_DIR / name
    else:
        path = Path(path).expanduser()

    # Create directory
    path.mkdir(parents=True, exist_ok=True)

    # Init git repo
    subprocess.run(["git", "init"], cwd=str(path), capture_output=True, check=True)

    # Create .gitignore
    gitignore = path / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("node_modules/\ndist/\n.env\n*.local\n")

    # Create starter AI_RULES.md
    rules = path / "AI_RULES.md"
    if not rules.exists():
        rules.write_text(f"# {name}\n\n## Project Context\n\n<!-- This file is loaded by Dyad as authoritative project context. -->\n<!-- Keep it concise: specs, decisions, conventions. No scratchpads. -->\n\n## Architecture\n\n<!-- Describe the tech stack and key architectural decisions -->\n\n## Conventions\n\n<!-- Naming, file structure, coding conventions -->\n")

    # Create package.json stub
    pkg = path / "package.json"
    if not pkg.exists():
        pkg.write_text(json.dumps({
            "name": name,
            "version": "0.0.1",
            "private": True,
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview"
            }
        }, indent=2))

    # Determine what to store in apps.path:
    # - If using the default dir (~/dyad-apps/<name>), store the bare name (relative —
    #   Dyad's own convention, resolved by project_dir() under DYAD_PROJECTS_DIR).
    # - If using a custom --path, store the absolute path so project_dir() can find it.
    if path == DYAD_PROJECTS_DIR / name:
        stored_path = name
    else:
        stored_path = str(path)

    # Register in Dyad DB
    conn = connect_db_write()
    try:
        conn.execute(
            "INSERT INTO apps (name, path, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (name, stored_path, int(time.time()), int(time.time())),
        )
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"Error: Could not insert into Dyad DB: {e}", file=sys.stderr)
        print(f"  The project dir is created at {path}", file=sys.stderr)
        print(f"  You may need to add it manually in Dyad's UI, or a project named '{name}' already exists.", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    print(f"✓ Created Dyad project '{name}'")
    print(f"  Directory: {path}")
    print(f"  Registered in Dyad DB (apps table)")
    print(f"  AI_RULES.md: {rules}")
    print(f"  The project will appear in Dyad on next launch.")


def cmd_add_mcp(name, transport, command=None, args=None, env_json=None,
                url=None, headers_json=None):
    """Register an MCP server in Dyad's mcp_servers table."""
    if transport not in ("stdio", "sse"):
        print("Error: --transport must be 'stdio' or 'sse'", file=sys.stderr)
        sys.exit(1)

    if transport == "stdio" and not command:
        print("Error: --command is required for stdio transport", file=sys.stderr)
        sys.exit(1)
    if transport == "sse" and not url:
        print("Error: --url is required for sse transport", file=sys.stderr)
        sys.exit(1)

    conn = connect_db_write()
    try:
        conn.execute(
            """INSERT INTO mcp_servers
               (name, transport, command, args, env_json, url, enabled, created_at, updated_at, headers_json)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (name, transport, command, args, env_json, url,
             int(time.time()), int(time.time()), headers_json),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, name, transport, enabled FROM mcp_servers ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    print(f"✓ Registered MCP server '{name}' (id={row['id']})")
    print(f"  Transport: {transport}")
    if transport == "stdio":
        print(f"  Command: {command} {args or ''}")
    else:
        print(f"  URL: {url}")
    print(f"  Enabled: {bool(row['enabled'])}")
    print(f"  ⚠ Restart Dyad for it to connect to this server.")


def cmd_list_mcp():
    """List all registered MCP servers."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, name, transport, command, args, url, enabled, created_at "
            "FROM mcp_servers ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No MCP servers registered.")
        print("Add one with: dyad_bridge.py add-mcp --name <name> --transport <stdio|sse> ...")
        return

    print(f"{'ID':>4}  {'Name':<24}  {'Transport':<10}  {'Enabled':<8}  {'Command/URL'}")
    print("-" * 90)
    for r in rows:
        endpoint = r["command"] if r["transport"] == "stdio" else r["url"]
        print(f"{r['id']:>4}  {r['name']:<24}  {r['transport']:<10}  "
              f"{'✓' if r['enabled'] else '✗':<8}  {endpoint or ''}")


def cmd_remove_mcp(server_id):
    """Remove an MCP server from Dyad's DB."""
    conn = connect_db_write()
    try:
        row = conn.execute("SELECT name FROM mcp_servers WHERE id = ?", (server_id,)).fetchone()
        if not row:
            print(f"Error: No MCP server with id={server_id}", file=sys.stderr)
            sys.exit(1)
        conn.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
        conn.commit()
    finally:
        conn.close()
    print(f"✓ Removed MCP server '{row['name']}' (id={server_id})")
    print(f"  ⚠ Restart Dyad for the change to take effect.")


def cmd_add_prompt(title, content, description=None):
    """Add a reusable prompt to Dyad's prompts table."""
    conn = connect_db_write()
    try:
        conn.execute(
            "INSERT INTO prompts (title, description, content, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (title, description, content, int(time.time()), int(time.time())),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, title FROM prompts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    print(f"✓ Added prompt '{title}' (id={row['id']})")


def cmd_list_prompts():
    """List all reusable prompts."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, title, description, created_at FROM prompts ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No reusable prompts found.")
        return

    print(f"{'ID':>4}  {'Title':<30}  {'Description':<40}  {'Created'}")
    print("-" * 95)
    for r in rows:
        print(f"{r['id']:>4}  {r['title'][:30]:<30}  "
              f"{(r['description'] or '')[:40]:<40}  {fmt_ts(r['created_at'])}")


def cmd_versions(app_id):
    """List git version snapshots for a project."""
    conn = connect_db()
    try:
        rows = conn.execute(
            "SELECT id, commit_hash, note, is_favorite, created_at "
            "FROM versions WHERE app_id = ? ORDER BY created_at DESC",
            (app_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"No versions found for app_id={app_id}")
        return

    print(f"{'ID':>4}  {'Commit':<12}  {'Favorite':<10}  {'Created':<18}  {'Note'}")
    print("-" * 80)
    for r in rows:
        commit = (r["commit_hash"] or "")[:10]
        note = (r["note"] or "")[:30]
        print(f"{r['id']:>4}  {commit:<12}  {'★' if r['is_favorite'] else '':<10}  "
              f"{fmt_ts(r['created_at']):<18}  {note}")


def cmd_schema():
    """Print the Dyad DB schema."""
    conn = connect_db()
    try:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()

    for row in schema:
        print(row["sql"])
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dyad ↔ Hermes Bridge — manage Dyad from Hermes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand")

    # list
    sub.add_parser("list", help="List all Dyad projects")

    # chats
    p_chats = sub.add_parser("chats", help="List chats for a project")
    p_chats.add_argument("app_id", type=int)

    # messages
    p_msgs = sub.add_parser("messages", help="Read messages from a chat")
    p_msgs.add_argument("chat_id", type=int)
    p_msgs.add_argument("--limit", type=int, default=50, help="Max messages to show (default 50)")

    # write-rules
    p_wr = sub.add_parser("write-rules", help="Write AI_RULES.md to a Dyad project")
    p_wr.add_argument("identifier", help="Project name or numeric ID")
    p_wr.add_argument("--file", help="Path to file to write as AI_RULES.md")
    p_wr.add_argument("--stdin", action="store_true", help="Read content from stdin")

    # read-rules
    p_rr = sub.add_parser("read-rules", help="Read AI_RULES.md from a Dyad project")
    p_rr.add_argument("identifier", help="Project name or numeric ID")

    # create-project
    p_cp = sub.add_parser("create-project", help="Create a new Dyad project")
    p_cp.add_argument("--name", required=True, help="Project name")
    p_cp.add_argument("--path", help="Project directory (default: ~/dyad-apps/<name>)")

    # add-mcp
    p_amcp = sub.add_parser("add-mcp", help="Register an MCP server in Dyad")
    p_amcp.add_argument("--name", required=True, help="Server display name")
    p_amcp.add_argument("--transport", required=True, choices=["stdio", "sse"])
    p_amcp.add_argument("--command", help="Command to run (stdio transport)")
    p_amcp.add_argument("--args", help="Arguments for command (stdio transport)")
    p_amcp.add_argument("--env", help="JSON env vars (stdio transport)")
    p_amcp.add_argument("--url", help="Server URL (sse transport)")
    p_amcp.add_argument("--headers", help="JSON headers (sse transport)")

    # list-mcp
    sub.add_parser("list-mcp", help="List registered MCP servers")

    # remove-mcp
    p_rmcp = sub.add_parser("remove-mcp", help="Remove an MCP server")
    p_rmcp.add_argument("server_id", type=int)

    # add-prompt
    p_ap = sub.add_parser("add-prompt", help="Add a reusable prompt to Dyad")
    p_ap.add_argument("--title", required=True)
    p_ap.add_argument("--description", default=None)
    p_ap.add_argument("--content", required=True)

    # list-prompts
    sub.add_parser("list-prompts", help="List reusable prompts")

    # versions
    p_ver = sub.add_parser("versions", help="List versions for a project")
    p_ver.add_argument("app_id", type=int)

    # schema
    sub.add_parser("schema", help="Print the Dyad DB schema")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "list": lambda: cmd_list(),
        "chats": lambda: cmd_chats(args.app_id),
        "messages": lambda: cmd_messages(args.chat_id, args.limit),
        "write-rules": lambda: cmd_write_rules(args.identifier, args.file, args.stdin),
        "read-rules": lambda: cmd_read_rules(args.identifier),
        "create-project": lambda: cmd_create_project(args.name, args.path),
        "add-mcp": lambda: cmd_add_mcp(args.name, args.transport, args.command,
                                       args.args, args.env, args.url, args.headers),
        "list-mcp": lambda: cmd_list_mcp(),
        "remove-mcp": lambda: cmd_remove_mcp(args.server_id),
        "add-prompt": lambda: cmd_add_prompt(args.title, args.content, args.description),
        "list-prompts": lambda: cmd_list_prompts(),
        "versions": lambda: cmd_versions(args.app_id),
        "schema": lambda: cmd_schema(),
    }

    dispatch[args.subcommand]()


if __name__ == "__main__":
    main()