# disk-cleanup

Tracks and cleans ephemeral files in roots Hermes explicitly owns:
process-registered `hermes-*` directories directly beneath the platform temp
directory, generated media caches, and cron run output. A directory name alone
does not establish ownership, and neither does a filename such as `test_*` or
`tmp_*`.

Originally contributed by [@LVT382009](https://github.com/LVT382009) as a
skill in PR #12212.  Ported to the plugin system so the behaviour runs
automatically via `post_tool_call` and `on_session_end` hooks — the agent
never needs to remember to call a tool.

## How it works

| Hook | Behaviour |
|---|---|
| `post_tool_call` | When `write_file` / `terminal` / `patch` creates a file inside a Hermes-owned ephemeral root, track it silently. |
| `on_session_end` | Delete only immediate-cleanup files tracked by that exact turn. Concurrent and long-running bot turns remain isolated. |

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
- One turn ending cannot delete files tracked by another active turn
- Malformed or stale tracking entries are skipped fail-closed
- System temp roots must be observed and registered in the current process;
  registration is bound to the root's filesystem identity, so a replaced root
  and records left by an earlier process are skipped
- Backup/restore is scoped to `tracked.json` — the plugin never touches
  agent logs
- Atomic writes: `.tmp` → backup → rename

The owned roots are `$HERMES_HOME/cache/vision/temp_vision_images/`,
`$HERMES_HOME/cache/video/temp_video_files/`, `$HERMES_HOME/cron/output/`
(plus the legacy `cronjobs/output/` alias), and process-registered platform temp
directories whose top-level name starts with `hermes-`.
