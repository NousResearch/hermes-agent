---
name: hermes-ideas
description: Markdown Ideas (ideas_* tools, /ideas, dashboard) — not Kanban triage tasks. Load when the user says idea, ideas page, draft, or list/create ideas on a board.
version: 1.2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ideas, kanban, markdown, drafts]
    related_skills: [kanban-orchestrator, kanban-worker]
---

# Hermes Ideas

Markdown idea drafts live in the **Ideas** plugin (`~/.hermes/ideas`, one list per Kanban board slug). They are **not** Kanban tasks.

## When to load

Load this skill when the user mentions **ideas**, the **Ideas page/tab**, **markdown drafts**, or wants to **capture a rough thought** before it becomes work on the board.

Do **not** use Kanban tools for that intent — see pitfalls below.

## Use the tools (one call)

| User intent | Tool |
|-------------|------|
| List ideas on **all** boards | `ideas_list(all_boards=true)` |
| List ideas on one board | `ideas_list(board="roguelike-td")` |
| Board names + counts (find current `*`) | `ideas_boards()` |
| Read full draft | `ideas_show(idea_id="i_…")` |
| New draft | `ideas_create(title="…", body="…", board="…")` |
| Promote to Kanban | `ideas_convert(idea_id="i_…")` |

Do **not** `read_file` / `grep` `~/.hermes/ideas` or run `hermes ideas …` in the terminal unless `ideas_*` tools are **unavailable in this session** (not registered in the tool list at all).

**Never silently shell when MCP tools misbehave.** If `ideas_list` returns misleading empty results but `ideas_boards()` shows ideas elsewhere, diagnose the MCP argument layer first (see pitfall 5), explain to the user, then retry — do not fall back to `hermes ideas …` without saying why.

After code or toolset changes, the user may need a **new session** or `/reload` so `ideas_*` appears in the tool schema.

## Ideas vs Kanban (critical)

| | Ideas | Kanban |
|---|--------|--------|
| Storage | `.md` + SQLite index | `kanban.db` tasks |
| IDs | `i_…` | `t_…` |
| User says "create an **idea**" | `ideas_create` | — |
| Rough capture | Yes (`draft` status) | Use `triage` **only after** `ideas_convert` |
| Multi-board listing | `ideas_list(all_boards=true)` | `kanban_list` per board |

### Pitfalls

1. **"Idea" is not a triage task.** Never `kanban_create` / `hermes kanban create --triage` when the user asked for an idea on the Ideas page.
2. **"Default board" may not be slug `default`.** The user's *current* board (marked `*` in `ideas_boards()` or `/ideas boards`) may be e.g. `roguelike-td`. Pass `board=` explicitly when they name a project; otherwise use `ideas_boards()` once, then create/list on that slug.
3. **All boards = one tool call.** For "ideas on all boards", use `ideas_list(all_boards=true)` — not per-board loops, file reads, or repeated shell commands.
4. **Promotion is explicit.** Kanban work starts with `ideas_convert`, not by creating a task and hoping it links.
5. **Cursor / `hermes-tools` MCP kwargs nesting.** Symptom: `ideas_boards()` is correct but `ideas_list(all_boards=true)` or `ideas_list(board="…")` always lists only the empty **default** board. The MCP client schema may expose a single required `kwargs` object while the handler expects **top-level** keys (`all_boards`, `board`, `idea_id`). Calling `{"kwargs": {"all_boards": true}}` double-nests args so filters are ignored. Retry with flat top-level parameters when the schema allows; otherwise fix/unblock `agent/transports/hermes_tools_mcp_server.py` (pass Hermes `parameters` to `add_tool`, unwrap a lone `kwargs` dict in `_dispatch`). Details: `references/cursor-hermes-tools-mcp.md`.

More session-derived detail: `references/pitfalls-and-commands.md`.

## CLI / slash (humans)

- `hermes ideas list --all-boards`
- `/ideas list --all-boards`
- `/ideas boards` — lists boards with counts; bare `boards` works (no `list` subcommand required)

Agents should prefer `ideas_*` tools over slash/CLI.
