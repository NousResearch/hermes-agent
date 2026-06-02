---
name: wave-role-board
description: Use when configuring or operating a Wave Terminal T2 role board that displays Hermes progress for Coda, Clara, Mira, and Nova via the bundled wave-role-board plugin.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [wave, progress, plugins, role-board, coda, clara, mira, nova]
    related_skills: [hermes-agent, macos-computer-use]
---

# Wave Role Board

## Overview

The Wave role board is a lightweight progress surface for Hermes workflows.
It writes role events to `$HERMES_HOME/wave-hub/messages.jsonl` and stores the latest per-role state in `$HERMES_HOME/wave-hub/agents.json`.
A Wave Terminal tab can render those files as four panes:

- Coda: implementation, Codex, file edits, proposals.
- Clara: review, Claude, tests, security, regressions.
- Mira: research, docs, context, user-facing meaning.
- Nova: ops, automation, terminal commands, restore/watchdog/Kanban.

This skill is not the implementation by itself. The implementation is the bundled `wave-role-board` plugin plus whatever local viewer/restore scripts the user chooses to run.

## When to Use

Use this when a user wants:

- T2-like role progress visibility during Hermes work.
- A Slack-like local progress board for Coda, Clara, Mira, and Nova.
- A structured way to emit progress events from tools, subagents, Codex/Claude terminals, or Kanban work.
- To verify whether Wave viewer panes are running and updating.

Do not use this for unrelated terminal layout recovery unless the requested outcome includes a role progress board.

## Plugin Tools

The bundled plugin registers the `wave_role_board` toolset with these tools:

- `wave_progress(role, message, kind, status, task, source, mode, scope)`
  - Appends a role event to `messages.jsonl`.
  - Updates `agents.json`.
- `wave_board_status()`
  - Returns the hub path, latest role state, viewer processes, and whether all roles appear to be running.
- `wave_board_restore(force=false)`
  - Runs `$HERMES_HOME/wave-hub/restore_wave.sh` if it exists.
  - Safe no-op when all viewers are already running.
  - Returns a missing-script reason when no local restore script exists.

Enable in config:

```yaml
plugins:
  enabled:
    - wave-role-board
toolsets:
  - wave_role_board
```

A new Hermes session or gateway restart may be required before the model sees the new tools.

## Event Model

Each message record is JSONL:

```json
{
  "ts": "22:35:39",
  "at": "2026-06-02T22:35:39+09:00",
  "role": "Coda",
  "text": "Implementation started",
  "kind": "progress",
  "status": "running",
  "task": "build",
  "source": "wave_progress",
  "mode": "chat",
  "scope": "global",
  "project_name": null,
  "project_path": null,
  "metadata": {}
}
```

Valid roles:
- Coda
- Clara
- Mira
- Nova

Valid kinds:
- progress
- agent
- log
- memo
- status
- system

Recommended statuses:
- planning
- running
- blocked
- needs_review
- done
- interrupted

## Hook Behavior

The plugin emits lightweight automatic events for high-signal tool calls:

- `delegate_task` -> Coda start, Clara completion status.
- `terminal` -> Nova by default.
- terminal command containing `codex` -> Coda.
- terminal command containing `claude` -> Clara.
- terminal command containing `kanban` -> Nova.
- `write_file` / `patch` -> Coda.
- `read_file` / `search_files` -> Mira.

For important milestones, call `wave_progress` explicitly rather than relying only on hooks.

## Local Viewer Pattern

The plugin is profile-aware and writes to `$HERMES_HOME/wave-hub/` but does not impose a specific terminal UI. A local viewer can tail/filter the messages into four panes.

Preferred Wave layout:

```text
T1: Hugo/Hermes conversation
T2: Coda | Clara | Mira | Nova
```

Each pane should render only its role's messages. For narrow panes, avoid long prefixed one-line formats. Use a short header and wrap the body by terminal display width, especially for Korean text.

## Operating Rules

1. T2 is not a T1 mirror. It is a role progress board.
2. Emit progress at stage boundaries: planning, implementation, review, verification, blocked, done.
3. Do not claim the board is working until `wave_board_status` shows the expected role viewers or the user visually confirms the four panes.
4. Keep T3 optional. Add extra logs only when the user asks or debugging requires it.
5. Do not kill tmux sessions, rewrite Wave DB, or click permission dialogs without explicit user direction.

## Verification Checklist

- [ ] `wave_progress` writes a message and returns `success: true`.
- [ ] `$HERMES_HOME/wave-hub/messages.jsonl` contains the new role event.
- [ ] `$HERMES_HOME/wave-hub/agents.json` updates the role's latest status/task/message.
- [ ] `wave_board_status` returns the hub path and role state.
- [ ] If using Wave viewers, all Coda/Clara/Mira/Nova panes are visible and updating.
- [ ] If a restore script is configured, `wave_board_restore` is a safe no-op when all viewers are running.

## Common Pitfalls

1. Treating the skill as the implementation. The skill is operating memory; the plugin and viewer runtime are the implementation.
2. Expecting newly enabled plugin tools to appear mid-session. Restart the CLI/gateway or start a new session.
3. Depending on Wave UI automation without Accessibility permission. The plugin is UI-agnostic; local restore scripts may need macOS TCC permissions.
4. Emitting too much noise. Use role events for meaningful stage changes, not every token or every log line.
5. Assuming all users have Sangkun's local `restore_wave.sh`. The bundled plugin handles missing restore scripts gracefully.
