---
name: dyad-integration
description: Sync Hermes and Dyad projects, context, and MCP tools.
version: 1.0.0
author: Arthur Talley (Ambientmuse Studios LLC)
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [dyad, integration, mcp, ai-rules, sync, app-builder]
    category: devops
    related_skills: []
---

# Dyad ↔ Hermes Integration

Connect Hermes to [Dyad](https://dyad.sh) — the local, open-source AI app builder.
Sync project context, read Dyad's chat history, write `AI_RULES.md` files that Dyad
treats as authoritative, register MCP servers so Dyad can call Hermes tools, and
manage Dyad projects programmatically.

## When to Use

- The user wants to synchronize project work between Hermes and Dyad.
- The user wants to send context, specs, or task state to a Dyad project so Dyad's
  AI chat picks it up automatically.
- The user wants to list, inspect, or manage Dyad projects from Hermes.
- The user wants Dyad's AI to call Hermes tools (web search, file ops, terminal,
  memory) during its app-building chats — via MCP.
- The user asks about connecting Dyad to Hermes or "sending work to Dyad."

## Architecture Overview

Dyad is an **Electron app** (`com.electron.dyad`) that stores all state in a single
SQLite database. Projects live as plain directories on disk with git repos. Dyad
loads a per-project `AI_RULES.md` file and injects it as authoritative context into
every AI chat turn. Dyad is also an **MCP client** — it can connect to external MCP
servers.

Three integration paths:

| Path | Direction | Mechanism | Risk |
|------|-----------|-----------|------|
| 1. AI_RULES.md + filesystem | Hermes → Dyad | Write context files into `~/dyad-apps/<name>/AI_RULES.md`; Dyad auto-loads them | Very low |
| 2. MCP bridge | Dyad → Hermes | Register a Hermes MCP server in Dyad's `mcp_servers` table; Dyad calls Hermes tools | Medium |
| 3. SQLite + filesystem | Bidirectional | Read/write Dyad's DB and project dirs directly from Hermes | Low (reads safe with WAL) |

## Prerequisites

- Dyad installed (`/Applications/dyad.app` on macOS).
- Dyad SQLite DB at `~/Library/Application Support/dyad/sqlite.db` (WAL mode —
  concurrent reads are safe while Dyad runs; writes should be cautious).
- Dyad projects stored at `~/dyad-apps/<project-name>/`.
- Python 3 with `sqlite3` (stdlib) for the bridge script.

## Key Paths

```
Dyad app:           /Applications/dyad.app
Dyad Electron DB:   ~/Library/Application Support/dyad/sqlite.db
Dyad projects dir:  ~/dyad-apps/<project-name>/
AI_RULES.md:        ~/dyad-apps/<project-name>/AI_RULES.md  (per-project, Dyad auto-loads)
Dyad deep link:     dyad://  (registered URL scheme — OAuth callbacks only)
Dyad config env:   DYAD_LANGUAGE_MODEL_CATALOG_URL  (override model catalog)
```

## Procedure

### 1. List Dyad Projects

Run the bridge script to list all Dyad projects:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py list
```

Output: id, name, path, created_at, github_repo for each Dyad app.

### 2. Read Dyad Chat History

List chats for a project:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py chats <app_id>
```

Read messages from a chat:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py messages <chat_id>
```

### 3. Write AI_RULES.md (Hermes → Dyad context sync)

This is the primary integration. Write project context, specs, architecture
decisions, or task state into a Dyad project's `AI_RULES.md`. Dyad loads this file
and injects it as authoritative context into every AI chat turn.

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py write-rules \
  <project-name-or-id> --file /path/to/context.md
```

Or pipe content directly:

```bash
echo "# Project Context\n\nBuild a React dashboard with..." | \
  python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py write-rules \
    <project-name-or-id> --stdin
```

Dyad's own system instructions say:
> "Treat AI_RULES.md as authoritative project context, unless it conflicts with the
> user's current request. Edit AI_RULES.md only when the user explicitly asks to
> remember something across conversations. Keep AI_RULES.md concise and easy to
> scan."

So keep the file focused — specs, decisions, conventions — not scratchpads.

### 4. Sync Files Between Hermes and a Dyad Project

Since Dyad projects are plain git repos on disk, Hermes can read, write, and
diff them directly. Use `read_file`, `write_file`, `patch`, `search_files`, and
`terminal` (git) against `~/dyad-apps/<project-name>/`.

Example — see what Dyad changed:

```bash
cd ~/dyad-apps/<project-name> && git log --oneline -10
```

Example — write a file into a Dyad project from Hermes:

Use `write_file` with `path=~/dyad-apps/<project-name>/src/utils/api.ts`.

Dyad's file watcher picks up external changes. The next chat turn will see them.

### 5. Register an MCP Server (Dyad → Hermes tools)

Register an MCP server in Dyad's `mcp_servers` table so Dyad's AI can call
external tools during app-building chats. Dyad supports `stdio` and `sse`
transports.

For a stdio MCP server (e.g., a Hermes tool bridge):

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py add-mcp \
  --name "Hermes Bridge" \
  --transport stdio \
  --command python3 \
  --args "/path/to/mcp_server.py" \
  --env '{"HERMES_HOME": "/Users/ambientmuse/.hermes"}'
```

For an SSE/URL-based MCP server:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py add-mcp \
  --name "Remote Tools" \
  --transport sse \
  --url "http://localhost:8082/sse"
```

List registered MCP servers:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py list-mcp
```

Remove an MCP server:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py remove-mcp <server_id>
```

**Note:** After adding an MCP server via the DB, Dyad needs to be restarted (or the
MCP connections reloaded) to pick up the new server. Dyad's UI also has an MCP
settings page — changes there write to the same `mcp_servers` table.

### 6. Manage Reusable Prompts

Dyad stores reusable prompts in the `prompts` table. Add one from Hermes:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py add-prompt \
  --title "Build Dashboard" \
  --description "Standard dashboard scaffold prompt" \
  --content "Create a React dashboard with chart.js, dark theme, responsive layout..."
```

List prompts:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py list-prompts
```

### 7. Create a New Dyad Project

Create the project directory and register it in Dyad's DB:

```bash
python3 ~/.hermes/skills/devops/dyad-integration/scripts/dyad_bridge.py create-project \
  --name "my-new-app" \
  --path ~/dyad-apps/my-new-app
```

This creates the directory, initializes a git repo, writes a starter
`AI_RULES.md`, and inserts a row into Dyad's `apps` table. The project will appear
in Dyad's project list on next launch.

## Full Dyad DB Schema

See `references/dyad-db-schema.md` for the complete SQLite schema — all tables,
columns, types, and foreign keys. Use it for advanced queries or custom
integrations.

## Pitfalls

- **Dyad must be closed for DB writes** — WAL mode makes concurrent reads safe,
  but writes while Dyad's Electron process holds a write transaction can still
  conflict. If a write fails with "database is locked", close Dyad and retry.
- **`mcp_servers` inserts need restart** — Dyad loads MCP servers at startup.
  After inserting a row via the bridge script, restart Dyad for it to connect.
- **AI_RULES.md must be concise** — Dyad's instructions say to keep it scannable,
  not a scratchpad. Don't dump raw logs or chatty notes. Specs, decisions, and
  conventions only.
- **Project path is relative in DB** — the `apps.path` column stores the project
  name (relative to `~/dyad-apps/`), not an absolute path. The bridge script
  resolves it. If you query the DB directly, prepend `~/dyad-apps/`.
- **Dyad version drift** — schema was extracted from Dyad v1.6.2 (Aug 2026). Future
  versions may add columns or tables. The bridge script uses `SELECT *` and is
  resilient to new columns, but if Dyad changes the schema fundamentally,
  re-extract with `sqlite3 ~/Library/Application\ Support/dyad/sqlite.db .schema`.
- **`chat_mode` column** — Dyad chats have a `chat_mode` field (values seen in
  source: build, agent, etc.). This affects how Dyad processes the chat. When
  creating chats via the bridge, leave this null unless you know the mode.
- **No HTTP API in shipping Dyad** — the `rkendel1/Dyad` GitHub fork claims a REST
  API, but the official `dyad-sh/dyad` shipping build does NOT expose one. All
  integration is via SQLite + filesystem. Do not expect `curl localhost:PORT` to
  work.

## Verification

### Running the test suite

```bash
cd ~/.hermes/skills/devops/dyad-integration
python3 scripts/test_bridge.py
```

40 tests covering every bridge command: list, chats, messages, write-rules
(file + stdin + overwrite + missing dir + error cases), read-rules, create-project
(dir + DB registration + starter files + custom path), add/list/remove MCP servers
(stdio + sse + error cases), add/list prompts, versions, schema, and project
resolution by name/ID/path. Includes 3 live DB smoke tests that auto-skip if Dyad
isn't installed.

Also compatible with pytest:

```bash
pytest scripts/test_bridge.py -v
```

### Manual verification

- **List projects:** `python3 scripts/dyad_bridge.py list` — should show your
  Dyad apps.
- **AI_RULES.md:** write one, then open the project in Dyad and start a chat —
  Dyad should reference the context.
- **MCP server:** after registering one and restarting Dyad, check Dyad's MCP
  settings UI — your server should appear.
- **SQLite reads:** all bridge read commands work while Dyad is running (WAL mode).
- **Schema freshness:** `sqlite3 ~/Library/Application\ Support/dyad/sqlite.db .schema`
  should match `references/dyad-db-schema.md`.

## References

- `references/dyad-db-schema.md` — complete Dyad SQLite schema (all tables, columns,
  types, foreign keys, indexes).
- `scripts/dyad_bridge.py` — Python bridge implementing all operations (list,
  chats, messages, write-rules, create-project, add-mcp, list-mcp, remove-mcp,
  add-prompt, list-prompts).