# Architecture Notes

Maintainer reference for the optional `claude-code-sdk` Hermes skill. The
skill's name describes its purpose; its Python dependency is Anthropic's
official `claude-agent-sdk` package.

## Process Model

Every Hermes `terminal` call starts a fresh process. The dispatcher therefore
creates and disconnects one `ClaudeSDKClient` per query and persists only the
data needed to resume later. No daemon or Claude subprocess remains between
calls.

The skill manages two IDs:

- The **handle** is a 12-character token returned by `open` and used by the
  dispatcher commands.
- The **Claude session ID** comes from `ResultMessage.session_id` and is passed
  to `ClaudeAgentOptions.resume` on later queries.

`open` does not call the SDK. The first query uses only `cwd`; after that
query, the returned Claude session ID is saved. Subsequent queries use both
`cwd` and `resume`.

## Profile-Aware State

State lives at:

```text
<HERMES_HOME>/skill-state/claude-code-sdk/
```

This is deliberately outside the installed skill directory. Updating or
reinstalling an optional skill replaces that directory, but must not discard
session records or their cost audit log.

The standalone script mirrors Hermes's home-directory contract because it can
run from a Python environment that does not import the Hermes package:

1. An explicit `HERMES_HOME` wins.
2. Linux and macOS fall back to `~/.hermes`.
3. Windows falls back to `%LOCALAPPDATA%\hermes`, then
   `~/AppData/Local/hermes` when `LOCALAPPDATA` is unavailable.

`sessions.json` is written through a temporary file and atomic rename.
`cost.log` is append-only and contains timestamp, handle, and SDK-reported
query cost separated by tabs.

## Locking

Two lock scopes protect different invariants:

- `.sessions.lock` serializes short load-modify-save operations. Queries using
  different handles release this lock while the SDK is running, so they can
  proceed concurrently.
- A hashed lock file in `.session-locks/` is held for the entire query. Its
  acquisition is non-blocking, so an overlapping query or close on the same
  handle fails immediately with a busy error.

Linux and macOS use `fcntl.flock`; Windows uses `msvcrt.locking`. The handle is
hashed before becoming a filename even though current handles are generated
hex strings.

The query updates `last_activity` before releasing the store lock and calling
the SDK. This prevents the normal idle reaper from removing a record while a
bounded query is active.

## Limits and Costs

The dispatcher passes user-supplied `--max-turns` and `--max-budget-usd`
values directly into `ClaudeAgentOptions`. The local `--timeout` wraps the
async query in `asyncio.wait_for`; it is separate from the SDK's turn and
budget limits.

When `ResultMessage.total_cost_usd` is available, the value is added to the
session record and appended to `cost.log`. Missing cost is represented as
`null` and is never converted to zero.

## Idle Reaper

Each store-oriented command removes handles idle for more than one hour. This
is local bookkeeping only: there is no live process to kill, and the reaper
does not delete Claude's underlying session. Cost-log rows remain for audit.

## Failure Contract

Commands print success payloads as JSON on stdout. Errors are JSON on stderr
and use a nonzero exit code. Notable cases are:

| Condition | Behavior |
|---|---|
| Package missing | `doctor` or `query` exits 2 with the install command. |
| Invalid project | `open` exits 1. |
| Unknown handle | `query` exits 1. |
| Same handle already active | `query` or `close` exits 1 with a busy error. |
| SDK `ResultMessage.is_error` | `query` exits 1 with the SDK result and API status. |
| Query timeout or SDK exception | `query` exits 1 and keeps the handle. |
| Unknown handle on `close` | Returns `not_found` with exit 0 (idempotent). |

## Tests

`tests/skills/test_claude_code_sdk_skill.py` imports the dispatcher as a
module and tests behavior without network access. It covers profile path
selection, cross-process-style lock contention, SDK option forwarding,
session resume bookkeeping, parser validation, and the `doctor` contract.

Run it through the repository test wrapper:

```bash
scripts/run_tests.sh tests/skills/test_claude_code_sdk_skill.py -q
```

An authenticated smoke test can then exercise two separate query processes to
prove resume compatibility with the currently installed `claude-agent-sdk`.
