# Project Context Files

Hermes loads project instructions from the session's working directory. At
startup, project context uses a **first-match-wins** priority: `.hermes.md` →
`AGENTS.md` → `CLAUDE.md` → Cursor rules. `SOUL.md` is independent identity
content loaded from `$HERMES_HOME`.

For setup examples and the current user guide, see the
[Context Files documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files).

| File (in priority order) | Startup discovery | During the session |
|---|---|---|
| `.hermes.md` / `HERMES.md` | Nearest file from cwd up to the git root | Not progressively discovered |
| `AGENTS.md` / `agents.md` | Merged chain from git root to cwd; deeper files take precedence | Nested files are discovered when tools access their directories |
| `CLAUDE.md` / `claude.md` | Cwd only | Nested files are discovered when tools access their directories |
| `.cursorrules` / `.cursor/rules/*.mdc` | Cwd only | Nested `.cursorrules` files are discovered when tools access their directories |

Outside a Git repository, startup discovery does not walk above cwd. This
prevents a context file in a home or temporary directory from leaking into an
unrelated session.

## Pick the right file

- **Use `.hermes.md`** for Hermes-specific project instructions that should be
  inherited by sessions launched below that file. The nearest match wins; this
  is not a merged hierarchy.
- **Use `AGENTS.md`** for portable project instructions shared with other coding
  agents. In a Git repository, root-to-cwd files form a layered startup chain;
  nested files below cwd are loaded only when that subtree becomes relevant.
- **Use `CLAUDE.md`** when that is the project's existing convention. Hermes
  loads it only if no higher-priority project context type won at startup.
- **Use Cursor rules** as a compatibility fallback for projects already using
  Cursor conventions.
- **Use `SOUL.md`** for identity and tone, not project rules. Cross-project
  procedures belong in skills rather than a home-level `AGENTS.md`.

## Progressive discovery

When file or terminal tools access a nested path, Hermes checks that directory
and a bounded set of ordinary ancestor directories within the active workspace
for `AGENTS.md`, `CLAUDE.md`, or `.cursorrules` (first match per directory).
New instructions are appended to the tool result instead of changing the system
prompt, preserving prompt-cache stability. The current implementation checks at
most five parent levels and excludes dependency, cache, VCS, hidden, and other
generated directory classes. Each directory is checked at most once per session
and duplicate content is not injected again.

## Size and truncation

Startup files use `context_file_max_chars` when explicitly configured.
Otherwise the cap scales with the model context window, with a 20,000-character
floor and a 500,000-character ceiling. Oversized content is head/tail
truncated; merged `AGENTS.md` chains also receive an overall cap. Progressively
discovered files are capped at 8,000 characters each.

Keep startup context concise even when the model permits more: it is included
in the cached prompt and interpreted throughout the session. Put detailed,
reusable procedures in skills.

## Security

Context files pass through a bounded threat-pattern scan before loading. When a
pattern is detected in the scanned portion, that file is replaced wholesale
with a `[BLOCKED: ...]` notice. The scan intentionally caps how much input its
regular expressions inspect, so it is not a guarantee that every threat in a
large file will be detected. Review context files from repositories you do not
trust—the scanner is a safeguard, not a substitute for source review.

## Troubleshooting and isolation

- `hermes --ignore-rules` suppresses startup project context, `SOUL.md`, and
  built-in memory for that session. Explicit `--skills` arguments still load.
- Progressive nested context is currently independent of the startup flag and
  can still be discovered after file or terminal tools access a directory.
- It does **not** disable the rest of `config.yaml`, plugins, MCP servers, or
  credentials.
- `hermes --safe-mode` is the broader troubleshooting mode: it implies
  `--ignore-rules` and `--ignore-user-config`, and disables plugins and MCP
  servers. Environment credentials remain available, and the progressive-
  discovery caveat above still applies.

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

That file at `/home/me/projects/myrepo/.hermes.md` is auto-loaded when Hermes
runs in a subdirectory of that Git repository unless a nearer `.hermes.md`
exists. It is not loaded for `/home/me/other-project`.
