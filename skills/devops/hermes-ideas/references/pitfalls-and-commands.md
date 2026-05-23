# Ideas — pitfalls and command map

Condensed from real failures: agents confused Ideas with Kanban, read `~/.hermes/ideas` on disk, or used shell because tools were missing from the schema.

## Wrong path (avoid)

| User asked | Wrong approach | Why it fails |
|------------|----------------|--------------|
| Create an idea on the default board | `hermes kanban create "…" --triage` | Creates a **task** (`t_…`) in Kanban, not a markdown idea (`i_…`) |
| List ideas on all boards | `read_file` / `grep` under `~/.hermes/ideas` | Bypasses SQLite index; slow and board-scoped paths are easy to miss |
| List ideas on all boards | `ideas boards` + N× `ideas list --board X` | Works in shell but wastes turns; use `ideas_list(all_boards=true)` |
| `ideas_list` empty but `ideas_boards` shows ideas | `hermes ideas list` shell (silent fallback) | Data exists; MCP args likely nested under `kwargs` — see `cursor-hermes-tools-mcp.md` |
| Create idea | Assume board slug is `default` | Current board may differ (`ideas_boards()` shows `*` on current) |

## Right path (agents)

```
ideas_boards()                          # optional: resolve current board slug
ideas_create(title="…", board="…")    # rough draft
ideas_list(all_boards=true)             # cross-board listing
ideas_show(idea_id="i_…")             # full body
ideas_convert(idea_id="i_…")          # → Kanban task when ready
```

Tools ship in the default `hermes-cli` toolset via `includes: ["ideas"]` (no extra `toolsets: [ideas]` in config required).

## Slash / CLI quirks (humans + gateway)

- **`/ideas boards`**: Handler must not assume `args.json` exists on every subcommand namespace. Use `getattr(args, "json", False)` (or a `_wants_json` helper) and define `--json` on parent parsers when a subcommand can be omitted.
- **`--board` placement**: Match `hermes kanban` — parent `board` argparse on subcommands so `hermes ideas create "…" --board slug` works after the subcommand.
- **All boards flag**: CLI `--all-boards` ↔ tool `all_boards=true` ↔ `list_ideas_all_boards()` in `hermes_cli/ideas_db.py`.

## Cleanup after a mistake

If a Kanban triage task was created instead of an idea: archive/delete the `t_…` task, then `ideas_create` on the correct board slug.
