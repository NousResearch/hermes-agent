# Hermes ↔ Cursor Sync State Schema

Reference for the `.agent-sync/` directory and the project registry,
used by the `cursor-hermes-sync` bridge script.

## .agent-sync/ directory

Created by `init` or `create-project`. Gitignored (local state, not committed).

### state.json

The primary sync state file. Tracks handoffs, tasks, and timestamps.

```json
{
  "created": "2026-08-07T22:59:03.107421+00:00",
  "last_sync": "2026-08-07T23:02:59.660591+00:00",
  "last_cursor_open": "2026-08-07T22:59:13.564087+00:00",
  "last_hermes_open": null,
  "handoffs": [
    {
      "from": "hermes",
      "to": "cursor",
      "message": "Built the API, wire up the frontend",
      "timestamp": "2026-08-07T22:59:13.559882+00:00"
    }
  ],
  "tasks": [],
  "project_name": "My Project",
  "project_path": "/Volumes/Work/code/my-project",
  "last_git_sha": "23e8502281f5964c0ca2e97a63c0a1e393dcd9ab",
  "last_watch_time": 1786143548.417494
}
```

| Field | Type | Description |
|-------|------|-------------|
| `created` | ISO timestamp | When sync was initialized |
| `last_sync` | ISO timestamp | Last `sync` command run |
| `last_cursor_open` | ISO timestamp | Last time `open-cursor` or `handoff --to cursor` ran |
| `last_hermes_open` | ISO timestamp | Last time `open-hermes` or `handoff --to hermes` ran |
| `handoffs` | array | List of handoff records (from, to, message, timestamp) |
| `tasks` | array | Active tasks with status and assignee |
| `project_name` | string | Display name from `--name` flag |
| `project_path` | string | Resolved absolute path to project root |
| `last_git_sha` | string | Last git HEAD SHA recorded by `watch` |
| `last_watch_time` | float | Unix timestamp of last `watch` run (for mtime comparison) |

### changes.jsonl

Append-only log of file changes between sessions. Written by `watch`,
read by `changes`.

Each line is a JSON object:

**Git change entry:**
```json
{
  "timestamp": "2026-08-07T22:59:08.403777+00:00",
  "type": "git",
  "from_sha": "c64bf6f",
  "to_sha": "23e8502",
  "commits": "23e8502 update hello, add goodbye",
  "diff_stat": "src.ts   | 2 +-\nutils.ts | 1 +\n2 files changed, 2 insertions(+), 1 deletion(-)"
}
```

**Filesystem change entry:**
```json
{
  "timestamp": "2026-08-07T22:59:08.417429+00:00",
  "type": "filesystem",
  "files": ["src.ts", "utils.ts", "extra.ts"],
  "count": 3
}
```

## Project Registry

**Location:** `~/.hermes/cursor-hermes-sync/projects.json`

Maps project names to paths, like Dyad's `apps` table.

```json
{
  "My Project": {
    "path": "/Volumes/Work/code/my-project",
    "initialized": "2026-08-07T22:59:03.107421+00:00",
    "last_sync": null
  },
  "another-project": {
    "path": "/Users/ambientmuse/workspace/another-project",
    "initialized": "2026-08-08T10:00:00.000000+00:00",
    "last_sync": "2026-08-08T10:05:00.000000+00:00"
  }
}
```

Projects are registered by `init` and `create-project`. The registry
enables name-based resolution — you can use `status my-project` instead
of `status /Volumes/Work/code/my-project`.

## Resolution order

`resolve_project(identifier)` tries these in order:

1. **Existing filesystem path** — if the identifier is a valid existing path, use it directly
2. **Registered project name** — if it matches a key in the registry, use the registered path
3. **Bare name under scan paths** — try `~/workspace/<name>` or `/Volumes/Work/code/<name>`
4. **Fall back to path** — treat as a path (may not exist yet, for `init` on new directories)

This mirrors the Dyad bridge's `resolve_project` which checks the SQLite
`apps` table by name, ID, or path.

## Shared context files

| File | Who reads it | Purpose |
|------|-------------|---------|
| `AGENTS.md` | Hermes, Cursor | Shared project context — conventions, build commands, task list, handoff log |
| `.cursor/rules/agent-sync.mdc` | Cursor only | Tells Cursor's agent to read `.agent-sync/state.json` and the handoff log |

Both files are auto-loaded by the respective agents when working in the
project directory. Hermes also reads `.cursorrules` and `.cursor/rules/*.mdc`
natively (same priority tier as `AGENTS.md`).