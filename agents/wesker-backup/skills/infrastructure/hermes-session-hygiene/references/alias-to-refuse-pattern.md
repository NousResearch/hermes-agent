# Alias-to-Refuse Implementation

## The Problem

Typing `hermes` with no subcommand in a terminal spawns a full Hermes CLI session, which in turn spawns 3 MCP child processes (workspace-mcp, ms-365-mcp, nanobanana-mcp). This is the upstream entry point — `hermes` without args is equivalent to `hermes chat`.

Users on this VPS type `hermes` out of muscle memory (checking version, checking status, or just pressing Enter after SSH login). Each invocation compounds MCP duplication.

## The Fix

Add to `~/.bashrc`:

```bash
hermes() {
  if [ $# -eq 0 ]; then
    echo "Bare 'hermes' forks a private MCP stack (~500MB). Use 'hermes chat --force' if you really want a new session, or talk to the gateway over Telegram." >&2
    return 1
  fi
  command hermes "$@"
}
```

## What it blocks

- `hermes` (bare) — blocked with message
- `hermes chat` — not blocked (explicit subcommand)
- `hermes --version` — not blocked (flag counts as arg)
- `hermes setup` — not blocked
- `hermes status` — not blocked
- `hermes gateway run` — not blocked
- `hermes logs --follow` — not blocked (actually logs gateway+history)

## What it doesn't fix

- `hermes chat` explicitly typed — still spawns MCPs. But this is intentional usage, not muscle memory.
- Running `hermes dashboard` — spawns a separate process, should not be blocked
- Running `hermes <skill-name>` — skill subcommands pass through

## Alternative: Soft Singleton (Option B from the analysis)

A proper architectural fix would add a pidfile-based soft singleton to `hermes_cli/main.py`:

```python
# ~/.hermes/run/cli-<profile-hash>.pid
# Second invocation in same key refuses unless --force/--allow-duplicate
# Stale detection: if pid gone or /proc/<pid>/comm doesn't match hermes, reclaim
```

Not yet implemented. Requires a small upstream patch. Worth doing when rebasing the 67-commit update.
