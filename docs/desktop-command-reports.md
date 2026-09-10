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

Keep the renderer fix (`e00f7f01212f7807d783b9d6190352ddb7862879`) and the
command-persistence change together on a source-controlled maintenance branch.
Build the Desktop renderer and Python backend from that integrated source.
Updating only the installed application bundle is not a durable maintenance
strategy: a later release without these commits can replace the customized UI
or stop displaying the saved events.

Before an upstream update, merge/rebase the maintained changes in an isolated
checkout, resolve actual conflicts, and run the focused checks below. Publish
the reviewed change upstream or retain a maintained build until its release
contains the fix. Do not reapply a binary patch automatically or disable
updates. Local commits and tests do not mean the installed application or the
upstream release has been updated.

## Focused verification

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
