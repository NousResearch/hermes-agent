# Project Context Files

Hermes injects project-level instructions into the system prompt by reading context files anchored at the working directory. The discovery order is **first match wins** — only one project context source is loaded per session.

| File (in priority order) | Discovery | Use when |
|---|---|---|
| `.hermes.md` / `HERMES.md` | Walks parents up to the git root, stops at git root | You want hierarchical project rules (root + per-package overrides) |
| `AGENTS.md` / `agents.md` | **Merged directory chain** — git root down to cwd, deeper wins; per directory `AGENTS.override.md` > `AGENTS.md` > `agents.md`; cwd only outside a git repo | You want portable agent instructions that work the same in Hermes, Claude Code, Codex, etc. |
| `CLAUDE.md` / `claude.md` | Cwd only | Same as AGENTS.md, Claude-flavored |
| `.cursorrules` / `.cursor/rules/*.mdc` | Cwd only | Migrating from Cursor |

Files **below** the cwd are not ignored either: `agent/subdirectory_hints.py` discovers the same names lazily on first tool access to a directory and appends them to the tool result (first name wins per directory; 32k cap). `AGENTS.md` is tried before `CLAUDE.md`, so the session's `CLAUDE.md` loads only when no `AGENTS.md` / `AGENTS.override.md` exists between the git root and the cwd.

`SOUL.md` (in `$HERMES_HOME`) is independent and always loaded when present — it sets the agent's identity, not project rules.

### Pick the right one

- **Use `.hermes.md`** when you want Hermes-specific behavior that lives above the cwd (root + subtree), or when you want rules to inherit from a parent directory. The parent walk stops at the git root, so a home-level `.hermes.md` won't leak into every project (a git repo's root is the boundary).
- **Use `AGENTS.md`** when the same project will also be worked on by other agents (Codex, Claude Code, OpenCode) — it's their convention too, so one file serves them all. Rules inherit along the chain (git root → cwd), so repo-wide rules go at the root and package rules live next to the code.
- **Don't put project rules in `~/.hermes/AGENTS.md`** (or any other home-level location): it loads only when its directory is on the session's chain (the cwd or a parent, up to the git root), so it won't govern your projects when you run Hermes elsewhere — and if the home directory is itself a git repo, subdirectories under it inherit the file. For cross-project context, use `SOUL.md` (in `$HERMES_HOME`, identity-only) or install a skill via `hermes skills install`.

### Size and truncation

Each context file is capped at `context_file_max_chars` (config.yaml) when set; otherwise the cap scales with the model's context window — 20,000-char floor, 500,000-char ceiling (flat 20,000 when the window is unknown). Files over the cap get **head + tail** truncated (70% head / 20% tail, middle dropped with a `[...truncated...]` marker); the merged `AGENTS.md` chain is capped again as one unit, so a deep monorepo can't multiply the budget. For large project rules, prefer splitting into multiple skills over cramming one file.

### Security

All context files pass through the threat-pattern scanner before reaching the system prompt. Patterns matching prompt injection or promptware are replaced with a `[BLOCKED: ...]` placeholder. This means an `AGENTS.md` containing obvious injection attempts won't reach the model — the scanner blocks the content, not the file, so the rest of the file still loads.

### Disable for one session

`hermes --ignore-rules` skips auto-injection of all project context files (`.hermes.md`, `AGENTS.override.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`) **and** `SOUL.md` identity, plus user config, plugins, and MCP servers. Use it to isolate whether a problem is your setup or Hermes itself.

### Example: a small `.hermes.md`

```markdown
# My Project

Hermes: when working in this repo, follow these rules.

## Build
- Always run `make test` before declaring a change done.
- Use `uv run` for Python, not `pip install`.

## Style
- Prefer `pathlib.Path` over `os.path`.
- No `print()` in production code — use the `logger`.
```

That file at `/home/me/projects/myrepo/.hermes.md` is auto-loaded when Hermes runs in any subdirectory of `/home/me/projects/myrepo`, but not when it runs in `/home/me/other-project`.
