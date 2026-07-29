# disk-cleanup

Auto-tracks and cleans up ephemeral files created during Hermes Agent
sessions — test scripts, temp outputs, cron logs, stale chrome profiles.
Scoped strictly to `$HERMES_HOME` and `/tmp/hermes-*`.

Originally contributed by [@LVT382009](https://github.com/LVT382009) as a
skill in PR #12212.  Ported to the plugin system so the behaviour runs
automatically via `post_tool_call` and `on_session_end` hooks — the agent
never needs to remember to call a tool.

## How it works

| Hook | Behaviour |
|---|---|
| `post_tool_call` | When `write_file` / `terminal` / `patch` creates a file matching `test_*`, `tmp_*`, or `*.test.*` inside `HERMES_HOME`, track it only when it is outside a protected worktree/source path. Declared cache/cron-output roots and Git-ignored ephemeral paths in the root `HERMES_HOME` repo remain eligible. |
| `on_session_end` | If any test files were auto-tracked during this turn, run `quick` cleanup (no prompts). |

Deletion rules (same as the original PR):

| Category | Threshold | Confirmation |
|---|---|---|
| `test` | every session end, except non-explicit Git-protected entries | Never |
| `temp` | >7 days since tracked | Never |
| `cron-output` | >14 days since tracked | Never |
| unprotected empty dirs under HERMES_HOME | always | Never |
| `research` | >30 days, beyond 10 newest | Always (deep only) |
| `chrome-profile` | >14 days since tracked | Always (deep only) |
| files >500 MB | never auto | Always (deep only) |

## Slash command

```
/disk-cleanup status                     # breakdown + top-10 largest
/disk-cleanup dry-run                    # preview without deleting
/disk-cleanup quick                      # run safe cleanup now
/disk-cleanup deep                       # quick + list items needing prompt
/disk-cleanup track <path> <category>    # manual tracking
/disk-cleanup forget <path>              # stop tracking
```

## Safety

- `is_safe_path()` resolves paths before scope checks and rejects anything
  outside `HERMES_HOME` or `/tmp/hermes-*`, including symlink escapes
- Windows mounts (`/mnt/c` etc.) are rejected
- The state directory `$HERMES_HOME/disk-cleanup/` is itself excluded
- `$HERMES_HOME/logs/`, `memories/`, `sessions/`, `skills/`, `plugins/`,
  backup/profile state, and config files are never tracked or deleted, even
  by stale state or an explicit manual track request
- Git worktrees are never auto-tracked or traversed by automatic cleanup;
  explicit manual `/disk-cleanup track` may override protection for the selected
  path, but recursive deletion still refuses directories containing nested Git
  repositories/worktrees, including bare repositories
- In a Git-managed `HERMES_HOME`, tracked and non-ignored source paths are
  preserved; only declared `cache/` and `cron/output/` roots plus explicitly
  Git-ignored ephemeral paths remain eligible
- Stored paths, Git ownership, and pathname identity are revalidated at the
  deletion boundary; non-canonical paths (including `..` components) and stale
  out-of-scope entries are dropped without deletion
- Recursive cleanup and empty-directory sweeping refuse mount points and
  cross-device subtrees
- File and directory deletion is bound to a validated parent-directory file
  descriptor; platforms without the required `dir_fd`/symlink-safe primitives
  fail closed instead of falling back to pathname-only deletion
- Backup/restore is scoped to `tracked.json` — the plugin never touches
  agent logs
- Atomic writes: `.tmp` → backup → rename
