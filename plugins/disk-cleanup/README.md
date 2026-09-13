# disk-cleanup

Tracks and cleans generated media caches and cron run output inside fixed,
Hermes-owned roots. A directory name, terminal output, successful file creation,
manual category, or filename such as `test_*` / `tmp_*` never establishes
ownership for a platform-temp or workspace path.

Originally contributed by [@LVT382009](https://github.com/LVT382009) as a
skill in PR #12212.  Ported to the plugin system so the behaviour runs
automatically via `post_tool_call` and `on_session_end` hooks — the agent
never needs to remember to call a tool.

## How it works

| Hook | Behaviour |
|---|---|
| `post_tool_call` | Track regular files only when they are inside a fixed Hermes-owned cache/cron root. |
| `on_session_end` | After every completed turn, run aged cache/cron retention and delete only immediate-cleanup file generations owned by that turn. Long-running bot sessions therefore still clean incrementally. |

Deletion rules:

| Category | Threshold | Confirmation |
|---|---|---|
| `test` | end of the creating turn | Never |
| `temp` | >7 days since tracked | Never |
| `cron-output` | >14 days since tracked | Never |
| empty dirs in owned ephemeral roots | always | Never |
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

- Auto-deletion requires both an eligible category and current membership in an
  explicit owned root; stored tracking data is revalidated immediately before deletion
- Arbitrary workspace and durable Hermes files survive regardless of filename
- A cleanup obligation records its profile/turn, file generation, and owned-root generation;
  reusing the same pathname never transfers deletion authority to another turn
- Malformed or stale tracking entries are skipped fail-closed
- Platform-temp paths are never automatically owned. File creation proves who wrote
  a file, not that its lifetime is disposable; Git/worktree source and other durable
  workspace artifacts therefore remain untouched
- Automatic deletion traverses from verified owned-root directory handles and unlinks
  relative to the verified parent handle. Ancestor symlink swaps cannot redirect it
- Hosts without secure directory-handle unlink support skip automatic deletion and
  keep their tracking records for a future supported run
- Backup/restore is scoped to `tracked.json` — the plugin never touches
  agent logs
- Atomic writes: `.tmp` → backup → rename

The only owned roots are `$HERMES_HOME/cache/vision/temp_vision_images/`,
`$HERMES_HOME/cache/video/temp_video_files/`, `$HERMES_HOME/cron/output/`
(plus the legacy `cronjobs/output/` alias). Nothing outside those roots is
automatically deleted.
