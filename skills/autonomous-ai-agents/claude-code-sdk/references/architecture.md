# Architecture Notes

Maintainer-facing reference for `skills/autonomous-ai-agents/claude-code-sdk/`. End users should read `SKILL.md`; this document explains why the skill is shaped the way it is and how to extend it.

## Why a per-call client, not a daemon

Hermes invokes the skill through the `terminal` tool, which spawns a fresh OS process for every command. That means the skill cannot keep a live `ClaudeSDKClient` connection in memory between calls; pickling a connected client is not supported by the SDK and would not survive a process boundary even if it were.

The implementation chooses a per-invocation client (open and close inside one `query` call) plus durable state on disk:

- `ClaudeSDKClient.connect()` at the start of each `_run_query`, `disconnect()` in a `try/finally`.
- All conversation continuity comes from the SDK's native session resume, not from any in-memory cache on our side.

A long-lived daemon was considered and rejected as overkill for v1.0.0. It buys faster successive queries (no `connect()` cost per call) but introduces lifecycle complexity (orphan reaping, crash recovery, IPC), all of which the SDK's resume already solves at the conversation layer.

## Session resume contract

The skill manages two distinct identifiers:

- **Handle** (`session_id` in the JSON output): a 12-character hex token we generate at `open` time. The handle is the abstraction users interact with.
- **Claude session ID** (`claude_session_id` in `sessions.json`): the UUID Claude Code itself assigns. Captured from `ResultMessage.session_id` after the first query and persisted on the record.

Flow:

1. `open <path>`: validate path, generate handle, persist record with `claude_session_id: null`. No SDK call yet.
2. First `query <handle>`: `ClaudeAgentOptions(cwd=path)` with no `resume`. After receiving the `ResultMessage`, store `claude_session_id` on the record.
3. Subsequent `query <handle>`: `ClaudeAgentOptions(cwd=path, resume=claude_session_id)`. Claude restores the conversation history.

This indirection is deliberate. Users get a stable handle that does not change; the underlying Claude session ID is an implementation detail that may rotate (for example, if a future version adds `fork_session=True` semantics behind the scenes).

## State layout

Both files live under `~/.hermes/skills/claude-code-sdk/` (override with `HERMES_CLAUDE_SDK_STATE_DIR`).

### `sessions.json`

```json
{
  "sessions": {
    "7c3a91fb22d4": {
      "handle": "7c3a91fb22d4",
      "project_path": "/Users/raghu/Projects/foo",
      "created_at": "2026-04-27T16:53:27Z",
      "last_activity": "2026-04-27T16:55:46Z",
      "message_count": 4,
      "total_cost_usd": 0.306472,
      "claude_session_id": "a33915a3-a49b-40f3-ab04-b362d447af61"
    }
  }
}
```

Writes are atomic via temp-file plus rename, the same pattern used in the upstream `perplexity-claude-agent` registry. Concurrent CLI invocations that touch the store (parallel `open`/`query`/`close`) coordinate through an advisory file lock on `.sessions.lock`. The lock is held only around the bookkeeping load-modify-save, never during the SDK call itself, so parallel queries on different handles still run concurrently inside the SDK and only serialise the (microsecond-scale) JSON writes.

Cross-platform locking primitive selection:

- POSIX (Linux, macOS): `fcntl.flock` with `LOCK_EX` / `LOCK_UN`.
- Windows: `msvcrt.locking` with `LK_LOCK` / `LK_UNLCK` on byte 0 of the lock file. The lock file is initialised with a single sentinel byte on first use so the byte to lock exists.
- Other (no `fcntl` and no `msvcrt`): the lock degrades to a no-op and a one-line JSON warning is written to stderr; concurrent writers may race. This branch is unreachable on supported platforms but ensures the skill imports cleanly anywhere Python 3.10+ runs.

### `cost.log`

Tab-separated, append-only:

```
2026-04-27T16:53:51Z	7c3a91fb22d4	0.200468
2026-04-27T16:54:12Z	7c3a91fb22d4	0.031739
```

Format: `<iso8601_utc>\t<handle>\t<cost_usd>`. Rows are appended only when `ResultMessage.total_cost_usd` is not `None`. The `costs` subcommand sums per-handle entries from this log.

## Idle reaper

Runs at the start of every command (`open`, `query`, `list`, `close`, `costs`). Drops records whose `last_activity` is older than `HERMES_CLAUDE_SDK_IDLE_TTL` seconds (default 3600, that is 1 hour). There is no background process; the next CLI invocation does the cleanup. A silent Hermes turn longer than the TTL means the next call sweeps stale records.

The TTL is bookkeeping only. It does not kill processes (none are running between calls), nor does it delete Claude's underlying session state on disk. It is unrelated to Claude's 5-hour subscription quota window. The default of 1 hour was chosen as a "human idle" threshold that survives normal interruptions while keeping `sessions.json` from accumulating stale handles indefinitely. Set `HERMES_CLAUDE_SDK_IDLE_TTL=18000` to align with the 5-hour subscription rhythm if that matches user expectations better.

This is intentionally a soft reap: the records are dropped from `sessions.json`, but the corresponding rows in `cost.log` remain for audit. If you need stricter cleanup, run `cost.log` rotation externally.

## Failure modes the dispatcher already surfaces

| Trigger | Output (stderr, JSON) | Exit |
|---|---|---|
| `claude-agent-sdk` not installed | `{"error": "claude-agent-sdk is not installed. Run: pip install claude-agent-sdk\n(import failed with: ...)"}` | 2 |
| `open` against a non-directory | `{"error": "project_path is not a directory: ..."}` | 1 |
| `query` against unknown handle | `{"error": "session handle not found: ..."}` | 1 |
| Query times out | `{"error": "query timed out after Ns for handle ..."}` | 1 |
| Underlying SDK exception | `{"error": "query failed: <ExcType>: <message>"}` | 1 |
| `close` against unknown handle | `{"session_id": "...", "status": "not_found"}` | **0** (idempotent) |

`close` is intentionally idempotent so retry-on-timeout from a Hermes turn stays safe.

## Extending the skill

To add a new command (for example, `fork <handle>` to clone a session via `fork_session=True`):

1. Add a `cmd_<name>(args)` function returning `int` (exit code).
2. Add a subparser inside `build_parser()` with `set_defaults(func=cmd_<name>)`.
3. Reuse `_load_store()`, `_save_store()`, `_reap()`, and `_run_query()` rather than reaching into the SDK directly.
4. Always emit JSON to stdout via `_print_json(...)`. Errors go through `_die(...)` to stderr.

For richer SDK features, the relevant `ClaudeAgentOptions` fields are already documented in the SDK source: `permission_mode`, `max_turns`, `max_budget_usd`, `allowed_tools`, `system_prompt`, `hooks`, `agents`. The dispatcher currently passes only `cwd` and optionally `resume`; adding more is a matter of plumbing CLI flags into `_run_query`.

## Testing

End-to-end is the only meaningful test for this skill (mocking `claude-agent-sdk` would only test the dispatcher's argparse layer). The verification block in `SKILL.md` doubles as a smoke test. The full PR test transcript lives in the PR description.
