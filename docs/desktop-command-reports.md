# Desktop command reports

## Storage and isolation

Desktop sends a UUID `display_event_id` with each logical `slash.exec` or
`command.dispatch` invocation. After the command finishes, the gateway saves
its output in the owning profile's `session_display_events` table and returns
a stable `display:<uuid>` UI row. Only the command token is stored in the
header, not its arguments. The existing pending session title is applied so a
command-only draft can be reopened from the sidebar.

These events are not model messages. They do not change the model transcript,
message count, FTS index, skill routing, or prompt cache prefix. The REST
message endpoint includes them only with `include_display_events=true`;
gateway history/resume adds them to the outgoing display projection only.
Branch seeding excludes them on both the Desktop and gateway sides.

Pagination happens after merging messages and events. Compression descendants
can display their ancestor command reports; an explicit branch does not inherit
the original branch's reports. Numeric message row IDs remain reserved for real
messages. Ordinary empty-session cleanup does not delete command-only sessions;
an explicit session deletion still deletes its display events.

If saving fails, the original command result is returned with a persistence
warning. A storage failure must never become an RPC error that causes the UI
to execute the command again. The UUID makes the storage operation idempotent;
it is not permission to replay a command with external effects.

## Compatibility and updates

- Older clients omit the UUID and keep their existing behavior.
- Older backends return no display event; Desktop keeps the existing in-memory
  output behavior. This is compatibility, not durable-history support.
- Existing writable profile databases receive the additive table through the
  normal schema initializer. Read-only legacy databases without the table
  retain their ordinary message projection without forced migration.
- No skill files, project working directories, runtime pins, scheduler settings,
  updater configuration, or Membership records are changed by this feature.
- Reports from sessions that disappeared before this change are not recreated
  automatically. Recovering an old job requires its own verified ownership and
  source evidence; running the command again is not recovery.

Keep the renderer and command-persistence changes together as ordinary source
commits. They contain no FT-specific routing, machine paths, or release hashes.
They belong in the normal Hermes source and Desktop build, not a plugin that
rewrites core files or a patched application archive.

Until the upstream release includes them, use a maintained branch rather than
putting local commits on `main`. The updater's same-branch divergence recovery
can reset `main` to `origin/main`. Configure the existing updater instead:

```yaml
updates:
  parked_branch_strategy: update_in_place
  non_interactive_local_changes: stash
```

With the source checkout on a maintained branch, normal `hermes update` and the
Desktop Update action merge official changes into that branch, retaining its
local commits. The existing Desktop update flow rebuilds from the resulting
source and replaces the installed App. No separate binary patcher or custom
update daemon is needed. Keep the Desktop update target on `main`: it identifies
the upstream branch to merge, not the local maintenance branch.

Uncommitted files on an unmerged maintained branch use the existing autostash
without a branch switch. CLI updates normally restore that stash; Desktop's
`--keep-stash` retains it for explicit restoration. Committed report fixes are
not part of the stash and remain active. An actual merge conflict aborts the
merge, retains local commits and the stash, and reports failure. Explicit
`--switch-branch`, disabled auto-switch, and dirty branches whose commits are
already fully upstream keep their existing refusal/switch behavior.
An already conflicted index is refused before autostash, preserving the
unfinished merge or rebase for the user to resolve.

When all local patches are already upstream, a clean maintained branch returns
to the normal upstream branch automatically. Upstream acceptance remains an
external step, not something a local verification can establish. Before that
point, resolve real conflicts in an isolated checkout and run the focused checks
below. Do not disable updates or automatically edit a downloaded App archive.

## Focused verification

- Updates: `tests/hermes_cli/test_update_parked_branch_guard.py` uses real Git
  repositories for consecutive upstream upgrades, local commit retention,
  tracked/untracked edit recovery, and pre-existing or new conflict refusal.
- Database: `tests/hermes_state/test_command_display_events.py` exercises
  immutable ownership, concurrent writes, paging, compaction, branch isolation,
  read-only compatibility, deletion, and unchanged model history.
- Gateway: `tests/tui_gateway/test_command_display_persistence.py` exercises
  real temporary session storage, slash/dispatch routing, cold resume, REST
  opt-in, profile ownership, and failure without command re-execution.
- Desktop: command-display hydration, prompt actions, and session API tests
  cover stable IDs, refresh deduplication, model-seed exclusion, legacy replies,
  and persistence warnings without fallback execution.
- Also run the related resume ownership, profile sidebar, goal-command, and
  REST session regression files with `scripts/run_tests.sh`, plus the Desktop
  build. Use temporary state only; these checks need no live FT analysis,
  provider requests, Membership writes, or service restart.

After a separately approved installation, verify one harmless command in a new
Desktop conversation, close and reopen it, and confirm exactly one readable
report remains. Check that an ordinary chat still follows its normal path.
