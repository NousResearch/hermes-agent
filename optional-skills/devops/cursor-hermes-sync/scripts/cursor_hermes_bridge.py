#!/usr/bin/env python3
"""
cursor_hermes_bridge.py — Two-way project sync between Hermes Agent and Cursor AI.

Usage:
  python3 cursor_hermes_bridge.py init <project-path> [--name "My Project"]
  python3 cursor_hermes_bridge.py status <project-path>
  python3 cursor_hermes_bridge.py sync <project-path>
  python3 cursor_hermes_bridge.py open-cursor <project-path> [--file path:line]
  python3 cursor_hermes_bridge.py open-hermes <project-path> [--prompt "message"]
  python3 cursor_hermes_bridge.py handoff <project-path> [--to cursor|hermes] [--message "context"]
  python3 cursor_hermes_bridge.py list-projects
  python3 cursor_hermes_bridge.py cursor-chats <project-path>
  python3 cursor_hermes_bridge.py cursor-transcripts <project-path> [--limit 5] [--full]
  python3 cursor_hermes_bridge.py hermes-sessions <project-path>
  python3 cursor_hermes_bridge.py remove <project-path> [--force]
  python3 cursor_hermes_bridge.py add-mcp --name "Server" --command python3 [--args ...] [--env '{}'] [--url URL] [--workspace]
  python3 cursor_hermes_bridge.py list-mcp [--workspace]
  python3 cursor_hermes_bridge.py remove-mcp <name> [--workspace]

Architecture:
  - Shared context file: AGENTS.md (both agents read it natively)
  - Sync state file: .agent-sync/state.json (tracks last sync, open tasks, handoffs)
  - Cursor integration: opens via `cursor` CLI, reads chat summaries from SQLite
  - Hermes integration: opens via `hermes` CLI, reads session list from state.db
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Constants ---

HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
CURSOR_APP = Path("/Applications/Cursor.app/Contents/Resources/app/bin/cursor")
CURSOR_DB = Path(os.environ.get("CURSOR_DB_PATH", str(Path.home() / ".cursor" / "ai-tracking" / "ai-code-tracking.db")))
HERMES_BIN = os.environ.get("HERMES_BIN", str(HOME / "hermes-agent" / "venv" / "bin" / "hermes"))
HERMES_STATE_DB = Path(os.environ.get("HERMES_STATE_DB_PATH", str(HOME / "state.db")))
SYNC_DIR = ".agent-sync"
SYNC_STATE = f"{SYNC_DIR}/state.json"
AGENTS_MD = "AGENTS.md"
CURSOR_RULES_DIR = ".cursor/rules"
CURSOR_MCP_CONFIG = Path.home() / ".cursor" / "mcp.json"
CURSOR_PROJECTS_DIR = Path.home() / ".cursor" / "projects"

# Project registry — maps project names to paths, like Dyad's apps table.
# Stored at ~/.hermes/cursor-hermes-sync/projects.json
REGISTRY_FILE = HOME / "cursor-hermes-sync" / "projects.json"

# Default scan paths for list-projects (colon-separated, override via SYNC_SCAN_PATHS env)
DEFAULT_SCAN_PATHS = [
    str(Path.home() / "workspace"),
    "/Volumes/Work/code",
]


def load_registry():
    """Load the project registry, return dict of {name: {path, initialized, last_sync}}."""
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_registry(registry):
    """Save the project registry."""
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2) + "\n")


def register_project(name, path):
    """Register a project in the registry."""
    registry = load_registry()
    registry[name] = {
        "path": str(path),
        "initialized": now_iso(),
        "last_sync": None,
    }
    save_registry(registry)


def unregister_project(name):
    """Remove a project from the registry."""
    registry = load_registry()
    if name in registry:
        del registry[name]
        save_registry(registry)


def resolve_project(identifier):
    """Resolve a project identifier to an absolute filesystem path.

    Resolution order (mirrors the Dyad skill's resolve_project):
    1. If it's a valid existing path, use it directly (absolute or relative).
    2. If it matches a registered project name, use the registered path.
    3. If it's a bare name, try resolving under known scan paths (like DYAD_PROJECTS_DIR).
    4. Fall back to treating it as a path (may not exist yet).
    """
    p = Path(identifier).expanduser()

    # 1. Existing absolute or relative path
    if p.exists():
        return p.resolve()

    # 2. Registered project name
    registry = load_registry()
    if identifier in registry:
        registered = Path(registry[identifier]["path"])
        if registered.exists():
            return registered.resolve()

    # 3. Bare name under scan paths (like Dyad's DYAD_PROJECTS_DIR resolution)
    scan_paths_str = os.environ.get("SYNC_SCAN_PATHS", "")
    if scan_paths_str:
        scan_paths = [Path(p) for p in scan_paths_str.split(":") if p]
    else:
        scan_paths = [Path(p) for p in DEFAULT_SCAN_PATHS if p]

    for sp in scan_paths:
        candidate = sp / identifier
        if candidate.exists():
            return candidate.resolve()

    # 4. Fall back — treat as path (may not exist yet, like init on a new dir)
    return p.resolve()

# --- Helpers ---

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def project_root(path):
    """Resolve a project path to absolute, find git root if inside a repo."""
    p = Path(path).expanduser().resolve()
    # Try to find git root
    current = p
    for _ in range(20):
        if (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return p

def ensure_sync_dir(root):
    """Create .agent-sync/ directory if it doesn't exist."""
    sync = root / SYNC_DIR
    sync.mkdir(parents=True, exist_ok=True)
    return sync

def load_state(root):
    """Load sync state, return empty dict if not present."""
    state_file = root / SYNC_STATE
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "created": now_iso(),
        "last_sync": None,
        "last_cursor_open": None,
        "last_hermes_open": None,
        "handoffs": [],
        "tasks": [],
    }

def save_state(root, state):
    """Save sync state to .agent-sync/state.json."""
    state_file = root / SYNC_STATE
    ensure_sync_dir(root)
    state_file.write_text(json.dumps(state, indent=2) + "\n")

def find_cursor_cli():
    """Find the Cursor CLI binary."""
    # Check PATH first
    try:
        result = subprocess.run(["which", "cursor"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # Fall back to known macOS location
    if CURSOR_APP.exists():
        return str(CURSOR_APP)
    return None

def find_hermes_cli():
    """Find the Hermes CLI binary."""
    # Check environment override
    if os.environ.get("HERMES_BIN"):
        return os.environ["HERMES_BIN"]
    # Check PATH
    try:
        result = subprocess.run(["which", "hermes"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    # Check known location
    if Path(HERMES_BIN).exists():
        return HERMES_BIN
    return None

def get_cursor_project_id(root):
    """Get Cursor's internal project ID for a given path."""
    # Cursor encodes project paths by replacing / with - and stripping leading -
    encoded = str(root).replace("/", "-").lstrip("-")
    # Also check for colon-separated encoding for some macOS paths
    # Cursor stores projects in ~/.cursor/projects/<encoded>/
    cursor_projects = Path.home() / ".cursor" / "projects"
    if not cursor_projects.exists():
        return None
    # Try to match by checking workspace.json
    for d in cursor_projects.iterdir():
        ws_file = d / "workspace.json"
        if ws_file.exists():
            try:
                ws = json.loads(ws_file.read_text())
                ws_folder = ws.get("folder", "")
                if Path(ws_folder).resolve() == root:
                    return d.name
            except (json.JSONDecodeError, IOError):
                pass
    # Fall back to encoded name
    encoded_dir = cursor_projects / encoded
    if encoded_dir.exists():
        return encoded
    return None

def get_cursor_chats(root, limit=10):
    """Read recent Cursor AI chat summaries from its SQLite DB."""
    if not CURSOR_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(CURSOR_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT conversationId, title, tldr, mode, model,
                   datetime(updatedAt/1000, 'unixepoch', 'localtime') as updated
            FROM conversation_summaries
            ORDER BY updatedAt DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]

def get_hermes_sessions(root, limit=10):
    """Read recent Hermes sessions for this workspace from state.db."""
    if not HERMES_STATE_DB.exists():
        return []
    try:
        conn = sqlite3.connect(str(HERMES_STATE_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Hermes stores sessions with workspace in metadata
        cursor.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at, s.message_count
            FROM sessions s
            WHERE s.cwd LIKE ? OR s.title LIKE ?
            ORDER BY s.updated_at DESC
            LIMIT ?
        """, (f"{root}%", f"%{root.name}%", limit))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]

def write_agents_md(root, name=None, extra_context=None):
    """Write or update the shared AGENTS.md file."""
    agents_file = root / AGENTS_MD
    project_name = name or root.name

    content = f"""# {project_name}

> Shared context file for Hermes Agent and Cursor AI.
> Both agents read this file automatically when working in this repository.
> Edit freely — changes are picked up on the next agent session.

## Project

- **Name:** {project_name}
- **Path:** `{root}`
- **Sync initialized:** {now_iso()}

## Build & Run

<!-- Add your build commands here -->
- `npm install`
- `npm run dev`

## Architecture

<!-- Describe the architecture so both agents have context -->

## Conventions

<!-- Coding conventions both agents should follow -->
- Use TypeScript strict mode
- Prefer functional components

## Current Tasks

<!-- Active task list — both agents read and update this -->
<!-- Format: - [ ] Task description @agent:cursor or @agent:hermes -->

## Handoff Log

<!-- When switching between agents, leave a note about what was done and what's next -->
"""

    if extra_context:
        content += f"\n## Additional Context\n\n{extra_context}\n"

    agents_file.write_text(content)
    return agents_file

def write_cursor_rules(root, name=None):
    """Write a .cursor/rules/agent-sync.mdc that points Cursor at the shared state."""
    rules_dir = root / CURSOR_RULES_DIR
    rules_dir.mkdir(parents=True, exist_ok=True)
    rules_file = rules_dir / "agent-sync.mdc"

    content = f"""---
description: "Two-way sync with Hermes Agent — read .agent-sync/state.json for handoffs and task state"
globs:
  - "**/*"
alwaysApply: true
---

# Hermes ↔ Cursor Sync

This project is synced between **Cursor AI** and **Hermes Agent**.

## Before starting work:
1. Read `.agent-sync/state.json` for the current task state and any handoff notes.
2. Read `AGENTS.md` for project context, conventions, and the current task list.
3. Check the **Handoff Log** in `AGENTS.md` — if the other agent left work for you, pick it up.

## After finishing work:
1. Update `.agent-sync/state.json` — set `last_sync` timestamp, add any completed tasks.
2. Update the **Current Tasks** section in `AGENTS.md` — check off completed items.
3. Add a **Handoff Log** entry in `AGENTS.md` with what you did and what's next.

## Sync state file
`.agent-sync/state.json` contains:
- `last_sync`: ISO timestamp of last sync
- `handoffs`: list of {{from, to, message, timestamp}} entries
- `tasks`: list of active tasks with status

## Rules
- Never delete `.agent-sync/` — it is the sync layer.
- Always read the handoff log before starting work.
- Always leave a handoff note when switching to the other agent.
"""

    rules_file.write_text(content)
    return rules_file

# --- Commands ---

def cmd_init(args):
    """Initialize two-way sync for a project."""
    root = project_root(resolve_project(args.project_path))
    name = args.name or root.name

    # Create sync directory
    sync = ensure_sync_dir(root)

    # Write shared AGENTS.md
    agents_file = write_agents_md(root, name, args.context)

    # Write Cursor rules
    rules_file = write_cursor_rules(root, name)

    # Initialize state
    state = load_state(root)
    state["project_name"] = name
    state["project_path"] = str(root)
    state["created"] = now_iso()
    save_state(root, state)

    # Register in the project registry (like Dyad's apps table)
    register_project(name, root)

    # Add .agent-sync to .gitignore (sync state is local, not committed)
    gitignore = root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".agent-sync/" not in content:
            gitignore.write_text(content.rstrip() + "\n\n# Agent sync state (local)\n.agent-sync/\n")
    else:
        gitignore.write_text("# Agent sync state (local)\n.agent-sync/\n")

    cursor_cli = find_cursor_cli()
    hermes_cli = find_hermes_cli()

    print(f"✓ Two-way sync initialized for '{name}'")
    print(f"  Project: {root}")
    print(f"  Shared context: {agents_file}")
    print(f"  Cursor rules: {rules_file}")
    print(f"  Sync state: {root / SYNC_STATE}")
    print(f"  Registry: {REGISTRY_FILE}")
    print(f"  Cursor CLI: {cursor_cli or 'NOT FOUND'}")
    print(f"  Hermes CLI: {hermes_cli or 'NOT FOUND'}")
    print()
    print("Both agents will now read AGENTS.md automatically when working in this repo.")
    print("Use 'handoff' to switch between them with context.")
    print()
    print(f"You can now use the project name '{name}' with all commands:")
    print(f"  python3 $SCRIPT status {name}")
    print(f"  python3 $SCRIPT open-cursor {name}")
    print(f"  python3 $SCRIPT handoff {name} --to cursor --message '...'")

def cmd_status(args):
    """Show sync status for a project."""
    root = project_root(resolve_project(args.project_path))
    state = load_state(root)

    print(f"Project: {state.get('project_name', root.name)}")
    print(f"Path: {root}")
    print(f"Initialized: {state.get('created', 'unknown')}")
    print(f"Last sync: {state.get('last_sync', 'never')}")
    print(f"Last Cursor open: {state.get('last_cursor_open', 'never')}")
    print(f"Last Hermes open: {state.get('last_hermes_open', 'never')}")
    print()

    # Handoffs
    handoffs = state.get("handoffs", [])
    if handoffs:
        print(f"Handoffs ({len(handoffs)}):")
        for h in handoffs[-5:]:
            ts = h.get("timestamp", "?")
            direction = f"{h.get('from', '?')} → {h.get('to', '?')}"
            msg = h.get("message", "")
            print(f"  [{ts}] {direction}: {msg}")
    else:
        print("Handoffs: none")

    # Tasks
    tasks = state.get("tasks", [])
    if tasks:
        print(f"\nTasks ({len(tasks)}):")
        for t in tasks:
            status = "✓" if t.get("done") else "○"
            assignee = t.get("assignee", "unassigned")
            print(f"  {status} {t.get('text', '?')} @{assignee}")
    else:
        print("\nTasks: none")

    # Check AGENTS.md exists
    agents_file = root / AGENTS_MD
    print(f"\nAGENTS.md: {'✓ exists' if agents_file.exists() else '✗ missing'}")
    rules_dir = root / CURSOR_RULES_DIR
    print(f"Cursor rules: {'✓ exists' if rules_dir.exists() else '✗ missing'}")

    # Check CLIs
    cursor_cli = find_cursor_cli()
    hermes_cli = find_hermes_cli()
    print(f"Cursor CLI: {cursor_cli or 'NOT FOUND'}")
    print(f"Hermes CLI: {hermes_cli or 'NOT FOUND'}")

def cmd_sync(args):
    """Sync project state — reads both agent states and reconciles."""
    root = project_root(resolve_project(args.project_path))
    state = load_state(root)

    # Read Cursor chats
    cursor_chats = get_cursor_chats(root)
    # Read Hermes sessions
    hermes_sessions = get_hermes_sessions(root)

    state["last_sync"] = now_iso()
    state["cursor_chats_count"] = len(cursor_chats)
    state["hermes_sessions_count"] = len(hermes_sessions)
    save_state(root, state)

    print(f"✓ Synced at {state['last_sync']}")
    print(f"  Cursor chats: {len(cursor_chats)} recent")
    print(f"  Hermes sessions: {len(hermes_sessions)} recent")

    if cursor_chats and args.verbose:
        print("\nRecent Cursor chats:")
        for c in cursor_chats[:5]:
            print(f"  [{c.get('updated', '?')}] {c.get('title', 'untitled')} ({c.get('mode', '?')})")
            if c.get('tldr'):
                print(f"    {c['tldr']}")

    if hermes_sessions and args.verbose:
        print("\nRecent Hermes sessions:")
        for s in hermes_sessions[:5]:
            print(f"  [{s.get('updated_at', '?')}] {s.get('title', 'untitled')}")

def cmd_open_cursor(args):
    """Open the project in Cursor."""
    root = project_root(resolve_project(args.project_path))
    cursor_cli = find_cursor_cli()

    if not cursor_cli:
        print("ERROR: Cursor CLI not found. Install Cursor or add to PATH.")
        sys.exit(1)

    cmd = [cursor_cli, str(root)]
    if args.new_window:
        cmd.insert(1, "-n")
    if args.file:
        cmd = [cursor_cli, "-g", args.file, str(root)]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    state = load_state(root)
    state["last_cursor_open"] = now_iso()
    save_state(root, state)

    print(f"✓ Opened Cursor at {root}")
    if args.file:
        print(f"  File: {args.file}")

def cmd_open_hermes(args):
    """Open the project in Hermes."""
    root = project_root(resolve_project(args.project_path))
    hermes_cli = find_hermes_cli()

    if not hermes_cli:
        print("ERROR: Hermes CLI not found. Install Hermes Agent.")
        sys.exit(1)

    if args.prompt:
        # One-shot mode — send a prompt and print the response
        cmd = [hermes_cli, "-z", args.prompt, "--workdir", str(root)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    else:
        # Interactive mode — launch in background terminal
        cmd = [hermes_cli, "chat", "--workdir", str(root)]
        subprocess.Popen(cmd)
        print(f"✓ Opened Hermes chat at {root}")

    state = load_state(root)
    state["last_hermes_open"] = now_iso()
    save_state(root, state)

def cmd_handoff(args):
    """Hand off work from one agent to the other with context."""
    root = project_root(resolve_project(args.project_path))
    state = load_state(root)

    direction = args.to or "cursor"
    message = args.message or "No message provided"

    handoff = {
        "from": "hermes" if direction == "cursor" else "cursor",
        "to": direction,
        "message": message,
        "timestamp": now_iso(),
    }
    state["handoffs"].append(handoff)
    save_state(root, state)

    # Update AGENTS.md handoff log
    agents_file = root / AGENTS_MD
    if agents_file.exists():
        content = agents_file.read_text()
        handoff_entry = f"\n### {now_iso()} — {handoff['from']} → {direction}\n{message}\n"
        # Find the Handoff Log section and append
        if "## Handoff Log" in content:
            content = content.rstrip() + handoff_entry
            agents_file.write_text(content)

    print(f"✓ Handoff: {handoff['from']} → {direction}")
    print(f"  Message: {message}")
    print()

    # Auto-watch: record file changes so the receiving agent knows what changed
    try:
        cmd_watch(argparse.Namespace(
            project_path=str(root), verbose=False
        ))
    except Exception as e:
        print(f"  (watch skipped: {e})", file=sys.stderr)

    # Open the target agent
    if direction == "cursor":
        cmd_open_cursor(argparse.Namespace(
            project_path=str(root), new_window=False, file=None
        ))
    else:
        cmd_open_hermes(argparse.Namespace(
            project_path=str(root),
            prompt=f"Continuing work from Cursor on {root.name}. Handoff note: {message}. Read AGENTS.md and .agent-sync/state.json for full context."
        ))

def cmd_watch(args):
    """
    Watch for file changes and log them to .agent-sync/changes.jsonl.
    Run as a background process. When the other agent starts work, it reads
    the changelog to see what changed since its last session.

    Both Hermes and Cursor already pick up file changes via their own file
    watchers — this adds a *semantic* changelog so the receiving agent knows
    WHAT changed and can surface a summary.
    """
    root = project_root(resolve_project(args.project_path))
    changes_file = root / SYNC_DIR / "changes.jsonl"
    ensure_sync_dir(root)

    # Use git diff if in a git repo, otherwise fall back to mtime scanning
    is_git = (root / ".git").exists()

    if is_git:
        # Track git changes — log new diffs since last run
        state = load_state(root)
        last_sha = state.get("last_git_sha")

        # Get current HEAD
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(root), timeout=10
            )
            current_sha = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            current_sha = None

        if current_sha and last_sha and last_sha != current_sha:
            # Get diff summary
            try:
                result = subprocess.run(
                    ["git", "diff", "--stat", f"{last_sha}..{current_sha}"],
                    capture_output=True, text=True, cwd=str(root), timeout=15
                )
                diff_stat = result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                diff_stat = ""

            # Get commit messages
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", f"{last_sha}..{current_sha}"],
                    capture_output=True, text=True, cwd=str(root), timeout=10
                )
                commits = result.stdout.strip() if result.returncode == 0 else ""
            except Exception:
                commits = ""

            change_entry = {
                "timestamp": now_iso(),
                "type": "git",
                "from_sha": last_sha,
                "to_sha": current_sha,
                "commits": commits,
                "diff_stat": diff_stat,
            }
            with open(changes_file, "a") as f:
                f.write(json.dumps(change_entry) + "\n")

        if current_sha:
            state["last_git_sha"] = current_sha
            save_state(root, state)

    # Also scan for uncommitted changes (works without git too)
    # Get files modified since last watch run
    state = load_state(root)
    last_watch = state.get("last_watch_time")

    import glob
    changed_files = []
    extensions = [".ts", ".tsx", ".js", ".jsx", ".py", ".css", ".scss", ".html",
                  ".json", ".yaml", ".yml", ".md", ".toml", ".go", ".rs", ".vue", ".svelte"]

    for ext in extensions:
        for f in root.rglob(f"*{ext}"):
            if SYNC_DIR in str(f) or ".git" in str(f) or "node_modules" in str(f):
                continue
            mtime = f.stat().st_mtime
            if last_watch and mtime > last_watch:
                rel = f.relative_to(root)
                changed_files.append(str(rel))

    if changed_files:
        change_entry = {
            "timestamp": now_iso(),
            "type": "filesystem",
            "files": changed_files[:50],  # cap at 50
            "count": len(changed_files),
        }
        with open(changes_file, "a") as f:
            f.write(json.dumps(change_entry) + "\n")

    state["last_watch_time"] = time.time()
    save_state(root, state)

    if args.verbose:
        print(f"Watch complete. Changed files: {len(changed_files)}")

def cmd_changes(args):
    """Show recent file changes recorded by the watcher."""
    root = project_root(resolve_project(args.project_path))
    changes_file = root / SYNC_DIR / "changes.jsonl"

    if not changes_file.exists():
        print("No changes recorded. Run 'watch' first.")
        return

    lines = changes_file.read_text().strip().split("\n")
    if not lines or lines == [""]:
        print("No changes recorded.")
        return

    limit = args.limit or 10
    recent = lines[-limit:]

    print(f"Recent changes ({len(recent)} of {len(lines)} total):")
    for line in recent:
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "?")
            etype = entry.get("type", "?")

            if etype == "git":
                print(f"\n  [{ts}] Git changes: {entry.get('from_sha', '?')[:7]} → {entry.get('to_sha', '?')[:7]}")
                if entry.get("commits"):
                    for c in entry["commits"].split("\n"):
                        print(f"    {c}")
                if entry.get("diff_stat"):
                    print(f"  Stats: {entry['diff_stat'][:200]}")
            elif etype == "filesystem":
                files = entry.get("files", [])
                print(f"\n  [{ts}] Filesystem changes ({entry.get('count', len(files))} files):")
                for f in files[:10]:
                    print(f"    {f}")
                if len(files) > 10:
                    print(f"    ... and {len(files) - 10} more")
        except json.JSONDecodeError:
            continue

def cmd_create_project(args):
    """Create a new project directory, git init, write starter files, and register it."""
    name = args.name

    # Resolve the path
    if args.path:
        path = Path(args.path).expanduser()
    else:
        # Default: under the first scan path
        scan_paths_str = os.environ.get("SYNC_SCAN_PATHS", "")
        if scan_paths_str:
            base = Path(scan_paths_str.split(":")[0])
        else:
            base = Path(DEFAULT_SCAN_PATHS[0])
        path = base / name

    # Create directory
    path.mkdir(parents=True, exist_ok=True)

    # Init git repo
    if not (path / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=str(path), capture_output=True, check=True)

    # Create .gitignore if missing
    gitignore = path / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("node_modules/\ndist/\n.env\n*.local\n")

    # Write starter AGENTS.md
    agents_file = path / AGENTS_MD
    if not agents_file.exists():
        write_agents_md(path, name, args.context)

    # Initialize sync state
    state = load_state(path)
    state["project_name"] = name
    state["project_path"] = str(path.resolve())
    state["created"] = now_iso()
    save_state(path, state)

    # Write Cursor rules
    write_cursor_rules(path, name)

    # Add .agent-sync to .gitignore
    if ".agent-sync/" not in gitignore.read_text():
        gitignore.write_text(gitignore.read_text().rstrip() + "\n\n# Agent sync state (local)\n.agent-sync/\n")

    # Register in the project registry
    register_project(name, path.resolve())

    print(f"✓ Created project '{name}'")
    print(f"  Path: {path}")
    print(f"  Git: initialized")
    print(f"  AGENTS.md: {agents_file}")
    print(f"  Cursor rules: {path / CURSOR_RULES_DIR / 'agent-sync.mdc'}")
    print(f"  Registry: {REGISTRY_FILE}")
    print()
    print(f"Use '{name}' with all bridge commands:")
    print(f"  python3 $SCRIPT status {name}")
    print(f"  python3 $SCRIPT open-cursor {name}")
    print(f"  python3 $SCRIPT open-hermes {name} --prompt 'Build something'")

def cmd_list_projects():
    """List all synced projects."""
    found = {}

    # 1. Check the registry first (authoritative — like Dyad's apps table)
    registry = load_registry()
    for name, info in registry.items():
        p = Path(info["path"])
        if p.exists() and (p / SYNC_STATE).exists():
            state = load_state(p)
            found[name] = {
                "name": name,
                "path": str(p),
                "last_sync": state.get("last_sync", "never"),
                "handoffs": len(state.get("handoffs", [])),
            }

    # 2. Also scan filesystem for projects with .agent-sync/ not in the registry
    scan_paths_str = os.environ.get("SYNC_SCAN_PATHS", "")
    if scan_paths_str:
        search_paths = [Path(p) for p in scan_paths_str.split(":") if p]
    else:
        search_paths = [Path(p) for p in DEFAULT_SCAN_PATHS if p]

    # Track resolved paths already found via registry to avoid duplicates
    seen_paths = {Path(info["path"]).resolve() for info in registry.values()}

    for sp in search_paths:
        if not sp.exists():
            continue
        for d in sp.iterdir():
            if not d.is_dir():
                continue
            sync_file = d / SYNC_STATE
            if sync_file.exists() and d.resolve() not in seen_paths:
                try:
                    state = json.loads(sync_file.read_text())
                    name = state.get("project_name", d.name)
                    found[name] = {
                        "name": name,
                        "path": str(d),
                        "last_sync": state.get("last_sync", "never"),
                        "handoffs": len(state.get("handoffs", [])),
                    }
                    seen_paths.add(d.resolve())
                except (json.JSONDecodeError, IOError):
                    pass

    if not found:
        print("No synced projects found. Use 'init <path>' or 'create-project --name <name>' to set one up.")
        return

    print(f"Synced projects ({len(found)}):")
    for p in sorted(found.values(), key=lambda x: x["name"]):
        print(f"  {p['name']:20s} {p['path']}")
        print(f"    last sync: {p['last_sync']}, handoffs: {p['handoffs']}")

def cmd_cursor_chats(args):
    """Show recent Cursor AI chat summaries."""
    root = project_root(resolve_project(args.project_path))
    chats = get_cursor_chats(root, limit=args.limit or 10)

    if not chats:
        print("No Cursor chats found.")
        return

    print(f"Recent Cursor AI chats ({len(chats)}):")
    for c in chats:
        print(f"\n  [{c.get('updated', '?')}] {c.get('title', 'untitled')}")
        print(f"  Mode: {c.get('mode', '?')} | Model: {c.get('model', '?')}")
        if c.get('tldr'):
            print(f"  TL;DR: {c['tldr']}")

def cmd_hermes_sessions(args):
    """Show recent Hermes sessions for this project."""
    root = project_root(resolve_project(args.project_path))
    sessions = get_hermes_sessions(root, limit=args.limit or 10)

    if not sessions:
        print("No Hermes sessions found for this project.")
        return

    print(f"Recent Hermes sessions ({len(sessions)}):")
    for s in sessions:
        print(f"  {s.get('id', '?')}: {s.get('title', 'untitled')}")
        print(f"    Updated: {s.get('updated_at', '?')} | Messages: {s.get('message_count', '?')}")

def cmd_cursor_transcripts(args):
    """List and summarize Cursor agent transcripts (JSONL) for a project.

    Cursor stores full agent transcripts as JSONL in:
      ~/.cursor/projects/<encoded-path>/agent-transcripts/<conversation-id>/subagents/*.jsonl

    For each transcript file, shows conversation ID, message count, first user
    message (truncated), and last assistant message (truncated).
    """
    root = project_root(resolve_project(args.project_path))
    project_id = get_cursor_project_id(root)

    if not project_id:
        print(f"No Cursor project directory found for: {root}")
        print("  Make sure the project has been opened in Cursor at least once.")
        return

    transcripts_base = CURSOR_PROJECTS_DIR / project_id / "agent-transcripts"
    if not transcripts_base.exists():
        print(f"No agent-transcripts directory found for project: {project_id}")
        print(f"  Expected: {transcripts_base}")
        return

    # Collect all JSONL transcript files
    transcript_files = []
    for conv_dir in sorted(transcripts_base.iterdir()):
        if not conv_dir.is_dir():
            continue
        subagents_dir = conv_dir / "subagents"
        if subagents_dir.exists():
            for f in sorted(subagents_dir.glob("*.jsonl")):
                transcript_files.append((conv_dir.name, f))
        # Also check for JSONL files directly in the conversation dir
        for f in sorted(conv_dir.glob("*.jsonl")):
            transcript_files.append((conv_dir.name, f))

    if not transcript_files:
        print(f"No transcript JSONL files found in {transcripts_base}")
        return

    limit = args.limit or 20
    full = getattr(args, "full", False)

    # Deduplicate by (conv_id, file path) and limit
    seen = set()
    unique_transcripts = []
    for conv_id, f in transcript_files:
        key = (conv_id, str(f))
        if key not in seen:
            seen.add(key)
            unique_transcripts.append((conv_id, f))

    transcripts_to_show = unique_transcripts[:limit]

    print(f"Cursor agent transcripts for '{root.name}' (project_id: {project_id})")
    print(f"  Found {len(unique_transcripts)} transcript file(s), showing {len(transcripts_to_show)}")
    print(f"  Location: {transcripts_base}")
    print()

    for conv_id, f in transcripts_to_show:
        messages = []
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except IOError as e:
            print(f"  [conv: {conv_id}] Error reading {f.name}: {e}")
            continue

        if not messages:
            print(f"  [conv: {conv_id}] {f.name} — 0 messages (empty or unreadable)")
            continue

        # Find first user message and last assistant message
        first_user_msg = None
        last_assistant_msg = None
        for msg in messages:
            role = msg.get("role") or msg.get("type") or ""
            content = msg.get("content") or msg.get("text") or ""
            if isinstance(content, list):
                # Some JSONL formats use content as list of blocks
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if role in ("user", "human") and not first_user_msg:
                first_user_msg = content
            if role in ("assistant", "ai", "model"):
                last_assistant_msg = content

        def truncate(text, maxlen=200):
            if not text:
                return "(none)"
            text = text.strip().replace("\n", " ")
            if len(text) > maxlen:
                return text[:maxlen] + "..."
            return text

        print(f"  ┌─ Conversation: {conv_id}")
        print(f"  │  File: {f.name}")
        print(f"  │  Messages: {len(messages)}")

        if full:
            print(f"  │  First user message:")
            if first_user_msg:
                for line in first_user_msg.splitlines():
                    print(f"  │  │  {line}")
            else:
                print(f"  │  │  (none)")
            print(f"  │  Last assistant message:")
            if last_assistant_msg:
                for line in last_assistant_msg.splitlines():
                    print(f"  │  │  {line}")
            else:
                print(f"  │  │  (none)")
        else:
            print(f"  │  First user: {truncate(first_user_msg)}")
            print(f"  │  Last assistant: {truncate(last_assistant_msg)}")
        print(f"  └─")
        print()


def cmd_remove(args):
    """Remove all sync artifacts for a project.

    - Removes the project from the registry
    - Removes .agent-sync/ directory
    - Removes .cursor/rules/agent-sync.mdc
    - Removes the handoff log from AGENTS.md (or the whole AGENTS.md if auto-generated)
    """
    root = project_root(resolve_project(args.project_path))
    name = root.name

    # Try to get the registered project name
    registry = load_registry()
    registered_name = None
    for rname, info in registry.items():
        if Path(info["path"]).resolve() == root:
            registered_name = rname
            break
    if registered_name:
        name = registered_name

    # Confirmation
    if not args.force:
        print(f"About to remove sync artifacts for project '{name}' at:")
        print(f"  {root}")
        print()
        print("This will remove:")
        print(f"  - Registry entry for '{name}'")
        print(f"  - {root / SYNC_DIR}/")
        print(f"  - {root / CURSOR_RULES_DIR / 'agent-sync.mdc'}")
        print(f"  - Handoff log in {root / AGENTS_MD} (or the whole file if auto-generated)")
        print()
        response = input("Proceed? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return

    removed = []

    # 1. Remove from registry
    if registered_name:
        unregister_project(registered_name)
        removed.append(f"registry entry for '{registered_name}'")
    elif name in registry:
        unregister_project(name)
        removed.append(f"registry entry for '{name}'")

    # 2. Remove .agent-sync/ directory
    sync_dir = root / SYNC_DIR
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
        removed.append(str(sync_dir))

    # 3. Remove .cursor/rules/agent-sync.mdc
    rules_file = root / CURSOR_RULES_DIR / "agent-sync.mdc"
    if rules_file.exists():
        rules_file.unlink()
        removed.append(str(rules_file))
        # Try to remove the .cursor/rules dir if now empty
        rules_dir = root / CURSOR_RULES_DIR
        try:
            if rules_dir.exists() and not any(rules_dir.iterdir()):
                rules_dir.rmdir()
        except OSError:
            pass

    # 4. Remove handoff log from AGENTS.md, or the whole file if auto-generated
    agents_file = root / AGENTS_MD
    if agents_file.exists():
        content = agents_file.read_text()
        # Check if this was auto-generated by our init/create-project
        is_auto_generated = "Shared context file for Hermes Agent and Cursor AI" in content

        if is_auto_generated:
            agents_file.unlink()
            removed.append(f"{agents_file} (auto-generated, removed entirely)")
        else:
            # Remove just the Handoff Log section
            lines = content.splitlines()
            new_lines = []
            in_handoff_section = False
            for line in lines:
                if line.strip().startswith("## Handoff Log"):
                    in_handoff_section = True
                    continue
                if in_handoff_section:
                    # Stop skipping when we hit the next ## section
                    if line.strip().startswith("## ") and not line.strip().startswith("## Handoff"):
                        in_handoff_section = False
                        new_lines.append(line)
                    # else: skip handoff log lines
                else:
                    new_lines.append(line)
            agents_file.write_text("\n".join(new_lines).rstrip() + "\n")
            removed.append(f"handoff log in {agents_file}")

    print(f"✓ Removed sync artifacts for '{name}'")
    for r in removed:
        print(f"  - {r}")
    if not removed:
        print("  (nothing to remove — no sync artifacts found)")


# --- MCP Server Management (Cursor's ~/.cursor/mcp.json) ---

def _load_cursor_mcp_config(config_path):
    """Load Cursor MCP config JSON, return dict with 'mcpServers' key."""
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
            if "mcpServers" not in data:
                data["mcpServers"] = {}
            return data
        except (json.JSONDecodeError, IOError):
            return {"mcpServers": {}}
    return {"mcpServers": {}}


def _save_cursor_mcp_config(config_path, data):
    """Save Cursor MCP config JSON, creating parent dirs as needed."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + "\n")


def cmd_add_mcp(args):
    """Add an MCP server to Cursor's ~/.cursor/mcp.json (or workspace .cursor/mcp.json)."""
    if args.workspace:
        config_path = Path.cwd() / ".cursor" / "mcp.json"
    else:
        config_path = CURSOR_MCP_CONFIG

    if not args.name:
        print("Error: --name is required", file=sys.stderr)
        sys.exit(1)

    # Determine server config based on whether URL (SSE) or command (stdio) is provided
    if args.url:
        server_config = {"url": args.url}
        transport = "sse"
    elif args.command:
        server_config = {"command": args.command}
        if args.args:
            # args can be a single string or JSON array
            if args.args.startswith("["):
                try:
                    server_config["args"] = json.loads(args.args)
                except json.JSONDecodeError:
                    server_config["args"] = args.args.split()
            else:
                server_config["args"] = args.args.split()
        transport = "stdio"
    else:
        print("Error: either --command (stdio) or --url (sse) is required", file=sys.stderr)
        sys.exit(1)

    if args.env:
        try:
            server_config["env"] = json.loads(args.env)
        except json.JSONDecodeError:
            print(f"Error: --env must be valid JSON, got: {args.env}", file=sys.stderr)
            sys.exit(1)

    data = _load_cursor_mcp_config(config_path)

    # Check if server already exists
    if args.name in data["mcpServers"]:
        print(f"Warning: MCP server '{args.name}' already exists in {config_path}")
        print(f"  Overwriting...")

    data["mcpServers"][args.name] = server_config
    _save_cursor_mcp_config(config_path, data)

    print(f"✓ Added MCP server '{args.name}' to {config_path}")
    print(f"  Transport: {transport}")
    if transport == "stdio":
        print(f"  Command: {server_config['command']} {' '.join(server_config.get('args', []))}")
    else:
        print(f"  URL: {server_config['url']}")
    if "env" in server_config:
        print(f"  Env: {list(server_config['env'].keys())}")
    print(f"  ⚠ Restart Cursor for the change to take effect.")


def cmd_list_mcp(args):
    """List all MCP servers in Cursor's ~/.cursor/mcp.json."""
    if args.workspace:
        config_path = Path.cwd() / ".cursor" / "mcp.json"
    else:
        config_path = CURSOR_MCP_CONFIG

    data = _load_cursor_mcp_config(config_path)
    servers = data.get("mcpServers", {})

    if not servers:
        print(f"No MCP servers found in {config_path}")
        return

    print(f"MCP servers in {config_path} ({len(servers)}):")
    print(f"{'Name':<30}  {'Transport':<10}  {'Command/URL'}")
    print("-" * 90)
    for name, cfg in servers.items():
        if "url" in cfg:
            transport = "sse"
            endpoint = cfg["url"]
        elif "command" in cfg:
            transport = "stdio"
            endpoint = f"{cfg['command']} {' '.join(cfg.get('args', []))}"
        else:
            transport = "?"
            endpoint = json.dumps(cfg)[:60]
        print(f"{name:<30}  {transport:<10}  {endpoint}")


def cmd_remove_mcp(args):
    """Remove an MCP server from Cursor's ~/.cursor/mcp.json by name."""
    if args.workspace:
        config_path = Path.cwd() / ".cursor" / "mcp.json"
    else:
        config_path = CURSOR_MCP_CONFIG

    data = _load_cursor_mcp_config(config_path)
    servers = data.get("mcpServers", {})

    if args.name not in servers:
        print(f"Error: No MCP server named '{args.name}' in {config_path}", file=sys.stderr)
        print(f"  Available servers: {', '.join(servers.keys()) or '(none)'}", file=sys.stderr)
        sys.exit(1)

    del data["mcpServers"][args.name]
    _save_cursor_mcp_config(config_path, data)

    print(f"✓ Removed MCP server '{args.name}' from {config_path}")
    print(f"  ⚠ Restart Cursor for the change to take effect.")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Two-way project sync between Hermes Agent and Cursor AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize sync for a new project
  %(prog)s init ~/workspace/my-app --name "My App"

  # Check sync status
  %(prog)s status ~/workspace/my-app

  # Open in Cursor
  %(prog)s open-cursor ~/workspace/my-app

  # Open in Hermes with a prompt
  %(prog)s open-hermes ~/workspace/my-app --prompt "Fix the auth bug"

  # Hand off from Hermes to Cursor with context
  %(prog)s handoff ~/workspace/my-app --to cursor --message "Built the API, need you to wire up the frontend"

  # List all synced projects
  %(prog)s list-projects
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p = sub.add_parser("init", help="Initialize two-way sync for a project")
    p.add_argument("project_path")
    p.add_argument("--name", help="Project display name")
    p.add_argument("--context", help="Additional context for AGENTS.md")
    p.set_defaults(func=cmd_init)

    # status
    p = sub.add_parser("status", help="Show sync status")
    p.add_argument("project_path")
    p.set_defaults(func=cmd_status)

    # sync
    p = sub.add_parser("sync", help="Sync project state between agents")
    p.add_argument("project_path")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_sync)

    # open-cursor
    p = sub.add_parser("open-cursor", help="Open project in Cursor")
    p.add_argument("project_path")
    p.add_argument("-n", "--new-window", action="store_true")
    p.add_argument("--file", help="Open a specific file (path:line:col)")
    p.set_defaults(func=cmd_open_cursor)

    # open-hermes
    p = sub.add_parser("open-hermes", help="Open project in Hermes")
    p.add_argument("project_path")
    p.add_argument("--prompt", help="Send a one-shot prompt to Hermes")
    p.set_defaults(func=cmd_open_hermes)

    # handoff
    p = sub.add_parser("handoff", help="Hand off work to the other agent")
    p.add_argument("project_path")
    p.add_argument("--to", choices=["cursor", "hermes"], default="cursor")
    p.add_argument("--message", help="Handoff context message")
    p.set_defaults(func=cmd_handoff)

    # watch — detect and log file changes since last run
    p = sub.add_parser("watch", help="Detect and log file changes since last watch")
    p.add_argument("project_path")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_watch)

    # changes — show recorded changes
    p = sub.add_parser("changes", help="Show recent file changes from the watcher")
    p.add_argument("project_path")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_changes)

    # create-project — like Dyad's create-project
    p = sub.add_parser("create-project", help="Create a new project with sync enabled")
    p.add_argument("--name", required=True, help="Project name")
    p.add_argument("--path", help="Project directory (default: under first scan path)")
    p.add_argument("--context", help="Additional context for AGENTS.md")
    p.set_defaults(func=cmd_create_project)

    # list-projects
    p = sub.add_parser("list-projects", help="List all synced projects")
    p.set_defaults(func=lambda a: cmd_list_projects())

    # cursor-chats
    p = sub.add_parser("cursor-chats", help="Show recent Cursor AI chats")
    p.add_argument("project_path")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_cursor_chats)

    # hermes-sessions
    p = sub.add_parser("hermes-sessions", help="Show recent Hermes sessions")
    p.add_argument("project_path")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_hermes_sessions)

    # remove — remove all sync artifacts for a project
    p = sub.add_parser("remove", help="Remove all sync artifacts for a project")
    p.add_argument("project_path")
    p.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    p.set_defaults(func=cmd_remove)

    # cursor-transcripts — list and summarize Cursor agent JSONL transcripts
    p = sub.add_parser("cursor-transcripts", help="List Cursor agent transcripts (JSONL)")
    p.add_argument("project_path")
    p.add_argument("--limit", type=int, default=20, help="Max transcripts to show (default 20)")
    p.add_argument("--full", action="store_true", help="Print full message content, not just summaries")
    p.set_defaults(func=cmd_cursor_transcripts)

    # add-mcp — add an MCP server to Cursor's mcp.json
    p = sub.add_parser("add-mcp", help="Add an MCP server to Cursor's ~/.cursor/mcp.json")
    p.add_argument("--name", required=True, help="Server name (key in mcpServers)")
    p.add_argument("--command", help="Command to run (stdio transport)")
    p.add_argument("--args", help="Arguments for command (space-separated or JSON array)")
    p.add_argument("--env", help="JSON env vars (e.g. '{\"KEY\": \"val\"}')")
    p.add_argument("--url", help="Server URL (SSE transport)")
    p.add_argument("--workspace", action="store_true", help="Use .cursor/mcp.json in current dir instead of global")
    p.set_defaults(func=cmd_add_mcp)

    # list-mcp — list all MCP servers in Cursor's mcp.json
    p = sub.add_parser("list-mcp", help="List MCP servers in Cursor's mcp.json")
    p.add_argument("--workspace", action="store_true", help="Use .cursor/mcp.json in current dir instead of global")
    p.set_defaults(func=cmd_list_mcp)

    # remove-mcp — remove an MCP server by name
    p = sub.add_parser("remove-mcp", help="Remove an MCP server from Cursor's mcp.json")
    p.add_argument("name", help="Server name to remove")
    p.add_argument("--workspace", action="store_true", help="Use .cursor/mcp.json in current dir instead of global")
    p.set_defaults(func=cmd_remove_mcp)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()