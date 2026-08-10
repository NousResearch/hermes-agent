---
name: safety-guard
description: 'Prevent destructive operations when working on production systems or running agents autonomously. Use for cron jobs, kanban workers, and unattended agent sessions.'
version: 1.0.0
author: Hermes Agent (adapted from ECC)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [safety, security, production, autonomous, guardrails]
    related_skills: [verification-loop, systematic-debugging]
    config:
      safety_guard.enabled:
        description: 'Enable safety guard for autonomous agent sessions'
        type: boolean
        default: false
      safety_guard.frozen_dir:
        description: 'Restrict file writes to this directory tree'
        type: string
        default: ''
---

# Safety Guard — Prevent Destructive Operations

## When to Use

- When working on production systems
- When agents are running autonomously (cron jobs, kanban workers, full-auto mode)
- When you want to restrict edits to a specific directory
- During sensitive operations (migrations, deploys, data changes)

## How It Works

Three modes of protection:

### Mode 1: Careful Mode

Intercepts destructive commands before execution and warns:

**Watched patterns:**

- `rm -rf` (especially `/`, `~`, or project root)
- `git push --force`
- `git reset --hard`
- `git checkout .` (discard all changes)
- `DROP TABLE` / `DROP DATABASE`
- `docker system prune`
- `kubectl delete`
- `chmod 777`
- `sudo rm`
- `npm publish` (accidental publishes)
- Any command with `--no-verify`

When detected: shows what the command does, asks for confirmation, suggests safer alternative.

### Mode 2: Freeze Mode

Locks file edits to a specific directory tree. Any Write/Edit outside the frozen directory is blocked with an explanation. Useful when you want an agent to focus on one area without touching unrelated code.

### Mode 3: Guard Mode (Careful + Freeze combined)

Both protections active. Maximum safety for autonomous agents. Agents can read anything but only write to the frozen directory. Destructive commands are blocked everywhere.

## Hermes Integration

### For Cron Jobs

Enable safety guard for all cron sessions by default. Cron agents run unattended — a destructive command could cause real damage before anyone notices.

In `config.yaml`:

```yaml
cron:
  safety_guard_enabled: true
```

### For Kanban Workers

Kanban workers process tasks autonomously. Enable guard mode to restrict writes to the project directory:

```yaml
kanban:
  safety_guard_enabled: true
  safety_guard_frozen_dir: '/path/to/project'
```

### For Delegation

Subagents inherit the parent's safety guard settings. Orchestrator agents should enable guard mode before spawning leaf workers.

### Using the terminal tool

When using Hermes's `terminal` tool, apply these checks before executing:

1. **Before `rm`**: Check if path is outside project directory. If so, require explicit confirmation.
2. **Before `git push --force`**: Verify the remote is `origin`, not `upstream`. Check if branch is `main`/`master`.
3. **Before destructive DB operations**: Require confirmation with the exact command spelled out.
4. **Before `docker` prune/rm**: Check if containers are running. Require listing what will be removed.

### Using the write_file tool

When using Hermes's `write_file` tool in freeze mode:

1. Check if target path is within the frozen directory
2. If outside, block and explain: "Write blocked: path is outside frozen directory [dir]"
3. Suggest moving the file or expanding the frozen directory

## Destructive Command Checklist

Before executing ANY of these, pause and verify:

| Command                        | Check                                                       |
| ------------------------------ | ----------------------------------------------------------- |
| `rm -rf <path>`                | Is `<path>` within the project? Is it a critical directory? |
| `git push --force`             | Is this `origin/main` or `origin/master`?                   |
| `git reset --hard`             | Are there uncommitted changes?                              |
| `DROP TABLE` / `DROP DATABASE` | Is this production? Is there a backup?                      |
| `docker system prune -a`       | Will this remove running containers?                        |
| `kubectl delete`               | Is this a production namespace?                             |
| `chmod 777`                    | Why does this need world-writable?                          |
| `npm publish`                  | Is this intentional? What's the version bump?               |

## Autonomous Agent Safety Rules

When running as an autonomous agent (cron, kanban, unattended):

1. **Never** run destructive commands without explicit user pre-approval
2. **Never** modify files outside the project directory
3. **Never** push to `main`/`master` without explicit permission
4. **Never** modify database schemas without confirmation
5. **Always** create backups before data migrations
6. **Always** use `--dry-run` flags when available before destructive operations
7. **Always** log what you're about to do before doing it

## Recovery

If a destructive operation was blocked:

1. The agent explains what was blocked and why
2. The agent suggests a safer alternative
3. The user can explicitly approve with `/approve`

If a destructive operation was accidentally executed:

1. Check git reflog for recoverable state
2. Check database backups
3. Report what happened and what was lost
