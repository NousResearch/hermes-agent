---
name: aoe
description: Orchestrate parallel AI coding agents in tmux sessions.
version: 1.0.1
author: Nathan Brake (njbrake) + Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Coding-Agent, tmux, Orchestration, Sessions, Worktrees, Automation]
    category: autonomous-ai-agents
    related_skills: [claude-code, codex, opencode]
---

# Agent of Empires (aoe) Skill

Drive the `aoe` CLI (Agent of Empires) to create, monitor, and control AI
coding-agent sessions (Claude Code, Codex, OpenCode, and others) running inside
tmux. Each session is a tracked agent process with an ID, title, tool, project
path, and live status. This skill covers session orchestration only — it is not
a general tmux window/pane manager.

## When to Use

- Launching one or more AI coding agents on project directories.
- Monitoring agent progress (waiting vs running vs idle).
- Capturing agent pane output for review.
- Running parallel git-worktree agents on separate branches.
- Organizing agents into groups or profiles.

**Don't use for:** general tmux window/pane management unrelated to coding
agents, or on Windows — `aoe` and tmux are POSIX-only.

## Prerequisites

- **Platform:** Linux or macOS. tmux has no native Windows build, so this skill
  is gated `[linux, macos]`.
- **Binaries:** `aoe` and `tmux` must be on `PATH`. Install aoe from
  <https://github.com/agent-of-empires/agent-of-empires>.
- **Optional:** `jq` for parsing `--json` output in shell pipelines.

## How to Run

Run every command below through the native `terminal` tool — `aoe` is a CLI, so
there is no MCP server or wrapped Hermes tool for it. Prefer `--json` output and
parse it with `terminal` pipelines (or read captured files with `read_file`)
rather than scraping human-readable text.

```bash
aoe add . -t "feature X" -l          # create + launch a session
aoe status --json                    # live counts across sessions
aoe session capture "feature X" --json   # read an agent's pane
```

Always use `aoe session start/stop/restart` — never raw `tmux` — so status and
metadata stay in sync.

## Quick Reference

| Task | Command |
| --- | --- |
| Add + launch a session | `aoe add . -t "title" -l` |
| Add in a group | `aoe add /repo -t "title" -g backend -l` |
| Add with a specific tool | `aoe add . -t "title" -c codex` |
| Add a worktree agent | `aoe add . -t "fix-123" -w fix/issue-123 -l` |
| Add in a Docker sandbox | `aoe add . -t "sandboxed" -s -l` |
| Add a sub-session | `aoe add . -t "sub" -P <parent-id>` |
| YOLO mode (skip prompts) | `aoe add . -t "yolo" -y -l` |
| List sessions | `aoe list --json` / `aoe list --all` |
| Start/stop/restart | `aoe session start\|stop\|restart <id-or-title>` |
| Attach (interactive) | `aoe session attach <id-or-title>` |
| Session metadata | `aoe session show <id-or-title> --json` |
| Capture pane | `aoe session capture <id-or-title> -n 100 --strip-ansi` |
| Live status summary | `aoe status --json` / `aoe status -q` |
| Rename / regroup | `aoe session rename <id> -t "…"` / `-g "…"` |
| Groups | `aoe group create\|move\|list\|delete …` |
| Profiles | `aoe profile list\|create\|delete\|default …` / `-p <name>` |
| Worktrees | `aoe worktree list\|info\|cleanup` |
| Remove | `aoe remove <id-or-title> [--delete-worktree --force]` |

### Core concepts

- **Session:** an agent process in a tmux session (ID, title, tool, path).
- **Group:** a named folder for sessions; nests with `/` (e.g. `backend/api`).
- **Profile:** an isolated workspace with its own sessions and config. Select
  with `-p <name>` or `AGENT_OF_EMPIRES_PROFILE`.
- **Status:** one of `running`, `waiting`, `idle`, `stopped`, `error`,
  `starting`, `unknown`.

### JSON shapes

`aoe list --json` — static metadata, **no live status**:

```json
[
  {
    "id": "a1b2c3d4-...",
    "title": "my feature",
    "path": "/home/user/project",
    "group": "backend",
    "tool": "claude",
    "command": "claude",
    "profile": "default",
    "created_at": "2025-01-01T00:00:00Z",
    "workspace_repos": []
  }
]
```

`command` is omitted when empty; `worktree` appears only for worktree-backed
sessions.

`aoe status --json` — live counters:

```json
{ "waiting": 1, "running": 2, "idle": 1, "stopped": 1, "error": 0, "total": 5 }
```

`aoe session capture --json` — pane content plus live status:

```json
{ "id": "a1b2c3d4-...", "title": "my feature", "status": "waiting",
  "tool": "claude", "content": "... pane text ...", "lines": 50 }
```

`aoe session show --json` — full metadata; `parent_session_id` appears only for
sub-sessions:

```json
{ "id": "a1b2c3d4-...", "title": "my feature", "path": "/home/user/project",
  "group": "backend", "tool": "claude", "command": "claude",
  "status": "running", "profile": "default" }
```

### Auto-detection (inside an aoe-managed pane)

When run from within an aoe session, the identifier can be omitted:
`aoe session show`, `aoe session capture`, `aoe session current --json`.

## Procedure

**Single agent**

```bash
aoe add /path/to/repo -t "feature X" -l
# ... let it work ...
aoe session capture "feature X" --json
```

**Parallel worktree agents** — one branch per session:

```bash
aoe add . -t "issue-100" -w fix/issue-100 -l
aoe add . -t "issue-101" -w fix/issue-101 -l
aoe add . -t "issue-102" -w fix/issue-102 -l
aoe status --json   # check all at once
```

**Monitoring loop** — poll until nothing is running or waiting:

```bash
while true; do
  status=$(aoe status --json)
  running=$(echo "$status" | jq '.running')
  waiting=$(echo "$status" | jq '.waiting')
  if [ "$running" -eq 0 ] && [ "$waiting" -eq 0 ]; then
    echo "All agents finished"; break
  fi
  echo "Running: $running, Waiting: $waiting"; sleep 30
done
```

**Capture and review** every session:

```bash
for id in $(aoe list --json | jq -r '.[].id'); do
  echo "=== $id ==="
  aoe session capture "$id" -n 100 --strip-ansi
done
```

## Pitfalls

1. **`aoe list --json` carries no live status.** Its fields are static metadata
   (`path`, `group`, `tool`, `command`, …). Get status from `aoe status --json`
   or `aoe session capture --json`.
2. **Raw `tmux` to start/stop agents bypasses tracking.** Session status and
   metadata go stale. Always use `aoe session start/stop/restart`.
3. **Forgetting `-l`/`--launch`.** `aoe add` creates a session but does not
   start it until you pass `-l` or run `aoe session start`.
4. **Wrong profile.** Sessions are profile-scoped; use `-p <name>` or set
   `AGENT_OF_EMPIRES_PROFILE`, and `aoe list --all` to see everything.

## Verification

- [ ] `aoe` and `tmux` are on `PATH` (Linux or macOS).
- [ ] `aoe add` was followed by `-l` or an explicit `aoe session start`.
- [ ] JSON parsing reads `path`/`group` (not `project_path`/`group_path`) and
      pulls status from `aoe status`/`aoe session capture`, not `aoe list`.
- [ ] Scripted polling exits when both `running` and `waiting` reach 0.
