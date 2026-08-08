---
name: cursor-hermes-sync
description: "Two-way project sync between Hermes Agent and Cursor AI."
version: 1.0.0
author: Arthur Talley (Ambientmuse Studios LLC)
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [cursor, hermes, sync, handoff, ide, two-way, project]
    category: devops
    related_skills: [dyad-integration, hermes-agent]
---

# Cursor ↔ Hermes Two-Way Sync

Start a project in one agent. Continue it in the other. Full context handoff, live file-change tracking, and one-command switching.

## When to Use

- User starts work in Hermes (or Cursor) and wants to continue in the other agent
- User wants to open the same project in either agent with full context preserved
- User wants file changes made in one environment to be visible to the other
- User asks about "syncing between Hermes and Cursor" or "switching between agents"
- User wants to hand off work from one agent to the other with a context note

## Architecture

Two agents, one shared layer:

| Layer | Mechanism |
|-------|-----------|
| **Shared context** | `AGENTS.md` — both Hermes and Cursor read this natively as project context |
| **Cursor rules** | `.cursor/rules/agent-sync.mdc` — tells Cursor to read sync state on startup |
| **Sync state** | `.agent-sync/state.json` — handoffs, tasks, timestamps, git SHA tracking |
| **Change log** | `.agent-sync/changes.jsonl` — append-only log of file changes between sessions |
| **Filesystem** | Both agents operate on the same repo. Git is the real sync layer. |
| **Bridge script** | `cursor_hermes_bridge.py` — the CLI that powers everything |

### How live change tracking works

Both Hermes and Cursor already have file watchers that detect filesystem changes in real time. The bridge adds a **semantic changelog** so the receiving agent knows *what* changed and *when*:

1. When you finish a session in one agent, run `watch` — it records git commits and file mtime changes to `.agent-sync/changes.jsonl`
2. When you start in the other agent, run `changes` — it shows you everything that changed since the last session
3. Both agents read `AGENTS.md` on startup for shared context, conventions, and the handoff log

## Prerequisites

- **Cursor** installed (`/Applications/Cursor.app` on macOS, or `cursor` on PATH on Linux)
- **Hermes Agent** installed (`hermes` CLI on PATH)
- **Python 3.9+** with stdlib only (no pip packages needed) — for the bridge script
- **Python 3.11** (Hermes venv) — for the Kanban lane script (`cursor_kanban_lane.py`), which imports `hermes_cli.kanban_db`. The Hermes venv Python is at `~/.hermes/hermes-agent/venv/bin/python3`
- **Hermes source tree** present at `~/.hermes/hermes-agent/` — the lane script imports from it. This is the default install location for git-installed Hermes
- **No external pip dependencies** — both scripts use only Python stdlib + Hermes internal modules

## Key Paths

```
Cursor CLI:     /Applications/Cursor.app/Contents/Resources/app/bin/cursor
Cursor chats:   ~/.cursor/ai-tracking/ai-code-tracking.db (SQLite)
Hermes CLI:     hermes (or ~/.hermes/hermes-agent/venv/bin/hermes)
Hermes sessions: ~/.hermes/state.db (SQLite + FTS5)
Bridge script:  ~/.hermes/skills/devops/cursor-hermes-sync/scripts/cursor_hermes_bridge.py
```

## Procedure

### 0. Quick reference

```bash
SCRIPT=~/.hermes/skills/devops/cursor-hermes-sync/scripts/cursor_hermes_bridge.py

python3 $SCRIPT init ~/workspace/my-project --name "My Project"
python3 $SCRIPT status ~/workspace/my-project
python3 $SCRIPT sync ~/workspace/my-project -v
python3 $SCRIPT watch ~/workspace/my-project
python3 $SCRIPT changes ~/workspace/my-project
python3 $SCRIPT open-cursor ~/workspace/my-project
python3 $SCRIPT open-hermes ~/workspace/my-project --prompt "Fix the auth bug"
python3 $SCRIPT handoff ~/workspace/my-project --to cursor --message "Built the API, wire up the frontend"
python3 $SCRIPT list-projects
```

### 1. Initialize sync for a project

```bash
python3 $SCRIPT init /path/to/project --name "My Project"
```

This creates:
- `AGENTS.md` — shared context file (both agents read automatically)
- `.cursor/rules/agent-sync.mdc` — tells Cursor to read sync state
- `.agent-sync/state.json` — sync state (handoffs, tasks, timestamps)
- Adds `.agent-sync/` to `.gitignore`

### 2. Open in Cursor

```bash
python3 $SCRIPT open-cursor /path/to/project
# Open a specific file:
python3 $SCRIPT open-cursor /path/to/project --file src/app.tsx:42
# Force new window:
python3 $SCRIPT open-cursor /path/to/project --new-window
```

### 3. Open in Hermes

```bash
# Interactive chat:
python3 $SCRIPT open-hermes /path/to/project

# One-shot prompt (prints response and exits):
python3 $SCRIPT open-hermes /path/to/project --prompt "Review the auth module for security issues"
```

### 4. Hand off from one agent to the other

```bash
# Hermes → Cursor:
python3 $SCRIPT handoff /path/to/project --to cursor --message "Built and tested the API. Frontend needs to wire up the login form to POST /api/auth/login."

# Cursor → Hermes:
python3 $SCRIPT handoff /path/to/project --to hermes --message "Frontend is done. Need backend to add rate limiting and refresh tokens."
```

Handoff does three things:
1. Records the handoff in `.agent-sync/state.json` with timestamp and message
2. Appends a note to `AGENTS.md` Handoff Log section
3. Opens the target agent (Cursor or Hermes) with the handoff context

### 5. Track file changes between sessions

```bash
# Record changes since last watch (run before switching agents):
python3 $SCRIPT watch /path/to/project

# View what changed since last session (run when starting in the other agent):
python3 $SCRIPT changes /path/to/project
```

The watcher tracks two things:
- **Git commits** — new commits since last `watch`, with diff stats and commit messages
- **Filesystem changes** — files modified by mtime since last `watch` (for uncommitted work)

### 6. Check sync status

```bash
python3 $SCRIPT status /path/to/project
```

Shows: last sync time, handoff count, active tasks, AGENTS.md status, CLI availability.

### 7. Read Cursor's AI chat history

```bash
python3 $SCRIPT cursor-chats /path/to/project
```

Reads recent Cursor AI conversation summaries from `~/.cursor/ai-tracking/ai-code-tracking.db` — titles, TL;DRs, mode, model.

### 8. Read Hermes session history

```bash
python3 $SCRIPT hermes-sessions /path/to/project
```

Reads recent Hermes sessions for this project from `~/.hermes/state.db`.

### 9. List all synced projects

```bash
python3 $SCRIPT list-projects
```

Scans `~/workspace` and `/Volumes/Work/code` for projects with `.agent-sync/state.json`.

## How the two-way sync works

```
┌──────────────┐                    ┌──────────────┐
│   Hermes     │                    │   Cursor     │
│   Agent      │                    │   AI         │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  reads/writes files               │  reads/writes files
       │                                   │
       ▼                                   ▼
  ┌─────────────────────────────────────────────┐
  │            Shared Project Repo               │
  │                                              │
  │  AGENTS.md ← shared context (both read)     │
  │  .cursor/rules/agent-sync.mdc ← Cursor rule │
  │  .agent-sync/state.json ← handoffs + tasks  │
  │  .agent-sync/changes.jsonl ← change log      │
  │                                              │
  │  Git ← the real file sync layer              │
  └─────────────────────────────────────────────┘
```

- **File changes** made in either agent are written to the same filesystem. Both agents' file watchers pick them up.
- **Context sync** happens through `AGENTS.md` — both agents read it on every session start.
- **Change tracking** happens through `.agent-sync/changes.jsonl` — run `watch` before switching, `changes` after arriving.
- **Handoffs** are explicit context notes — run `handoff --to cursor/hermes --message "..."` to switch with a note about what was done and what's next.

### 10. Assign Kanban tasks to Cursor (Hermes Kanban integration)

The Cursor worker lane lets you assign Kanban tasks to Cursor from the Hermes
Kanban board. The lane script polls for `cursor`-assigned tasks, claims them,
opens Cursor on the task's workspace, and writes the task context to `AGENTS.md`.

```bash
# Use Hermes' Python 3.11 (required for the kanban module)
PYTHON=~/.hermes/hermes-agent/venv/bin/python3
LANE=~/.hermes/skills/devops/cursor-hermes-sync/scripts/cursor_kanban_lane.py

# 1. Create a task assigned to cursor
hermes kanban create "Build landing page" --assignee cursor --body "Wire up the hero section"

# 2. Start the Cursor lane daemon (foreground, Ctrl+C to stop)
$PYTHON $LANE daemon --interval 30

# 3. Or run a single dispatch tick
$PYTHON $LANE tick

# 4. Or dispatch a specific task manually
$PYTHON $LANE dispatch <task_id>

# 5. Check status — show cursor-assigned tasks
$PYTHON $LANE status

# 6. When done in Cursor, complete the task in Hermes
hermes kanban complete <task_id> --summary "Done"
```

What happens when the lane dispatches a task:
1. Queries the Kanban DB for tasks with `assignee = 'cursor'` and `status = 'ready'`
2. Atomically claims the task (sets status to `running`, claim lock)
3. Creates a scratch workspace under the board's workspaces directory
4. Writes the task context to `AGENTS.md` in the workspace
5. Opens Cursor on the workspace
6. Writes a comment to the task with the Cursor PID
7. Cursor's AI reads `AGENTS.md` and sees the task instructions

The task lifecycle stays in Hermes Kanban: `ready → running → done/blocked`.
You complete or block the task from Hermes after finishing work in Cursor.

## Pitfalls

- **Cursor CLI not on PATH on macOS** — the `cursor` binary lives at `/Applications/Cursor.app/Contents/Resources/app/bin/cursor`. The bridge checks this fallback automatically.
- **`AGENTS.md` must stay concise** — it's the shared context file. Don't dump logs or scratchpads. Specs, decisions, conventions, and the handoff log only.
- **`.agent-sync/` is gitignored** — sync state is local to each machine, not committed. This is intentional.
- **Hermes `-z` one-shot mode has a 300s timeout** — for long tasks, use `hermes chat` interactive mode instead of `--prompt`.
- **Cursor SQLite reads are safe** — WAL mode means concurrent reads work while Cursor is running. Writes to Cursor's DB are never needed.
- **`watch` uses mtime scanning** — if the system clock changes, or files are touched without content changes, false positives may appear. Git-based tracking is more reliable when available.
- **Both agents share the same working directory** — there's no branching or worktree isolation. If both agents edit the same file simultaneously, git conflicts will occur. Use handoffs to coordinate, not parallel edits.

## Verification

```bash
# Run the bridge test suite (62 tests)
python3 ~/.hermes/skills/devops/cursor-hermes-sync/scripts/test_bridge.py

# Run the Kanban lane test suite (45 tests — requires Hermes venv Python 3.11)
~/.hermes/hermes-agent/venv/bin/python3 ~/.hermes/skills/devops/cursor-hermes-sync/scripts/test_kanban_lane.py

# Or with pytest:
pytest ~/.hermes/skills/devops/cursor-hermes-sync/scripts/test_bridge.py -v
~/.hermes/hermes-agent/venv/bin/python3 -m pytest ~/.hermes/skills/devops/cursor-hermes-sync/scripts/test_kanban_lane.py -v

# Manual smoke test
python3 $SCRIPT init /tmp/test-sync-project --name "Test"
python3 $SCRIPT status /tmp/test-sync-project
python3 $SCRIPT watch /tmp/test-sync-project
python3 $SCRIPT changes /tmp/test-sync-project
python3 $SCRIPT list-projects
rm -rf /tmp/test-sync-project
```

### Test coverage

107 tests across two suites covering every command in both scripts:

| Suite | Tests | Coverage |
|-------|-------|---------|
| **Bridge (test_bridge.py)** | 62 | init (9), status (3), sync (4), watch (5), changes (3), handoff (5), list-projects (3), cursor-chats (3), hermes-sessions (2), create-project (8), project-resolution (7), registry (3), edge cases (5), live smoke (2) |
| **Lane (test_kanban_lane.py)** | 45 | status (6), tick (14), dispatch (6), spawn (5), daemon (3), edge cases (8), live smoke (2) |

## References

- `references/cursor-storage-schema.md` — Cursor's SQLite tables, JSONL transcript format, MCP config format, CLI path resolution.
- `references/sync-state-schema.md` — `.agent-sync/state.json`, `changes.jsonl`, project registry, resolution order, shared context files.
- `references/architecture.md` — Component map, data flow diagrams, design principles, file layout.
- `scripts/cursor_hermes_bridge.py` — Main bridge script (15 commands).
- `scripts/cursor_kanban_lane.py` — Kanban worker lane for Cursor (5 commands).
- `scripts/test_bridge.py` — 62 tests for the bridge (Python 3.9+ stdlib).
- `scripts/test_kanban_lane.py` — 45 tests for the lane (Python 3.11 / Hermes venv).