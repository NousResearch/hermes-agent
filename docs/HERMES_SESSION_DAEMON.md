# Hermes Session Daemon Architecture

Status: MVP implemented behind explicit `hermes daemon ...` commands.

## Why

jcode's most useful architectural idea is not its full Rust implementation; it is
the separation of long-lived session runtime from thin client surfaces. Hermes can
use that pattern to support Mission Control / Agent Room, CLI, TUI, and future
mobile/web clients without making every surface spawn a separate independent
agent process.

## MVP Scope

The daemon is intentionally small and local-first:

- one local daemon per Hermes profile
- Unix-domain socket only: `$HERMES_HOME/runtime/hermes-daemon.sock`
- PID file: `$HERMES_HOME/runtime/hermes-daemon.pid`
- log file: `$HERMES_HOME/runtime/hermes-daemon.log`
- JSON-lines protocol, one request and one response per line
- existing `SessionDB` remains the durable session/transcript store
- agent turns run in daemon-owned background threads after explicit `session.send`

Commands:

```bash
hermes daemon start
hermes daemon status [--json]
hermes daemon sessions [--limit N] [--json]
hermes daemon create-session [--title T] [--source S] [--model M]
hermes daemon send "message" [--session-id ID] [--title T] [--source S] [--model M]
hermes daemon events [--session-id ID] [--run-id ID] [--since N] [--limit N]
hermes daemon stop
hermes daemon serve      # foreground/debug mode
```

Protocol methods:

- `ping`
- `session.list`
- `session.create`
- `session.get`
- `session.send`
- `session.events`
- `run.get`
- `shutdown`

## Design Rules

1. Preserve current CLI behavior. The daemon is opt-in until it has enough
   runtime coverage to become the default.
2. Keep the socket local/profile-scoped. Do not expose a network API by default.
3. Keep sensitive operations approval-gated. The daemon should not become a way
   to bypass CLI/gateway safety checks.
4. Use existing `SessionDB` as the source of truth instead of creating a second
   session store.
5. Build toward `session != window/process`: clients are surfaces; sessions are
   durable server-owned records/runtimes.

## Next Slices

1. Add read-only Mission Control panel consuming `ping`, `session.list`, and
   `session.events`.
2. Expand event fidelity: tool start/complete, approval-wait, blocked, error,
   and cancellation states.
3. Add approval-gated controls: stop, fork, resume, attach.
4. Add durable event snapshots for long-running sessions while keeping
   transcripts in `SessionDB`.
5. Add memory graph retrieval as a separate local service path, not coupled to
   the daemon MVP.
