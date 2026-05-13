# Quinn Ops MCP Snapshot/Diff Design

Status: design contract for repo-side implementation; not live-installed.

## Goal

Add a privacy-safe operational memory layer to `quinn_ops` so Quinn can ask what changed since the last check: gateway restarts, platform state shifts, cron changes, repo changes, runtime file changes, and error-count movement.

## Non-goals

- No config edits.
- No service restarts.
- No platform history reads.
- No transcript/session content reads.
- No raw log storage beyond already-sanitized/count-only overview fields.
- No private credential-like values in the snapshot.

## Boundary clarification

Current `quinn_ops` is operationally read-only. Snapshot/diff adds one narrow write: the server may write its own sanitized state file under Hermes home. This is not an operational action against Hermes, Discord, Telegram, git, systemd, config, or logs.

Allowed write path:

- `$HERMES_HOME/mcp/quinn_ops_state/overview_snapshot.json`

Optional temp/lock files under the same directory are allowed.

Everything else remains read-only.

## Tool additions

Add these tools to `TOOL_FUNCTIONS`:

1. `get_snapshot_status()`
   - Metadata only.
   - Returns whether the snapshot file exists, mtime, size, schema version, snapshot timestamp, and selected baseline identity fields.

2. `get_overview_diff(update_baseline: bool = False)`
   - Collects fresh `get_overview()` data.
   - Loads previous sanitized snapshot if present.
   - Returns structured diff.
   - If `update_baseline` is true, writes the fresh sanitized snapshot after computing the diff.
   - Default must be `False` so casual reads do not mutate state.

3. `save_overview_snapshot()`
   - Collects fresh `get_overview()` data and stores it as the new baseline.
   - Returns metadata and a brief summary of what was stored.

Do not implement a delete/clear tool in v1.

## Snapshot file schema

Use JSON:

- `schema_version`: 1
- `created_by`: `quinn_ops`
- `updated_at_utc`: UTC ISO timestamp
- `overview`: sanitized `get_overview().data`
- `overview_errors`: sanitized `get_overview().errors`
- `overview_warnings`: sanitized `get_overview().warnings`

Store data after normal `sanitize()` has run.

## Diff response schema

`get_overview_diff()` should return the standard response envelope. `data` should include:

- `has_previous`: bool
- `baseline_timestamp_utc`: string or null
- `current_timestamp_utc`: string
- `updated_baseline`: bool
- `summary.changed_count`: int
- `summary.severity`: `info`, `warning`, or `critical`
- `summary.headlines`: compact list of human-readable changes
- `changes`: dictionary grouped by area: `gateway`, `platforms`, `mcp`, `cron`, `sessions`, `repo`, `recent_errors`, `runtime_files`, `toolsets`, `version`

Each change item should include:

- `path`
- `type`: `changed`, `added`, `removed`, `increased`, or `decreased`
- `before`
- `after`
- `severity`

## Diff rules v1

Compare stable metadata only.

Ignore:

- top-level `timestamp_utc`
- session exact `updated_at` movement unless count/type distribution changes
- runtime file exact mtime unless existence or size changes
- log `last_seen` movement unless counts changed

Track gateway:

- `gateway.systemd_active`
- `gateway.pid`
- `gateway.systemd_properties.ActiveState`
- `gateway.systemd_properties.SubState`
- `gateway.systemd_properties.UnitFileState`

Gateway severity:

- inactive/not running: critical
- PID changed while still active/running: info
- enabled to disabled: warning or critical

Track platforms per platform:

- `configured`
- `connected`
- `status`

Platform severity:

- configured true to false: warning
- connected true to false: critical
- unknown/null to configured: info
- configured to unknown: warning

Track MCP:

- configured server list
- `mcp_list.active_like`
- whether `quinn_ops` remains configured/enabled

MCP severity:

- `quinn_ops` removed or not enabled: critical
- other server count changes: info or warning

Track cron:

- `cron.total`
- `cron.active`
- job IDs/names if present

Track sessions:

- `sessions.count`
- `file_count`
- `json_file_count`
- counts by `platform` and `chat_type` derived from `recent_metadata`

Do not store or diff session contents.

Track repo:

- `repo.head`
- `repo.branch`
- `repo.describe`
- `repo.status_short.dirty_count`
- file path additions/removals only

Repo severity:

- head changed: info
- dirty count increased: info
- branch changed: warning

Track recent errors:

- total count
- per-log total
- per-log category counts

Recent error severity:

- count increased: warning
- traceback/exception increased: warning, maybe critical for large jump
- count decreased: info

Do not store or diff snippets in v1.

Track runtime files:

- exists
- size_bytes

Ignore mtime unless file appears/disappears or size changes.

Track toolsets:

- enabled/disabled line set if easy
- summary active_like/total

## Implementation notes

- Use atomic write: create parent directory, write temp file, chmod `0600`, then replace.
- On snapshot read failure, return warning and continue as first snapshot.
- On write failure, return `ok=false` with structured error.
- Ensure all outputs go through existing `response()` / `sanitize()`.
- Keep functions importable without MCP SDK.
- Do not add dependencies.
- Keep the existing path style unless migrating the whole script to profile-aware helpers.

## Tests required

Add tests in `tests/test_quinn_ops_mcp.py` for:

1. Snapshot status when file is missing.
2. Saving snapshot creates parent directory and file with schema version and sanitized data.
3. `get_overview_diff(update_baseline=False)` does not write/update baseline.
4. First diff with no baseline returns `has_previous=false` and helpful warning/headline.
5. Gateway PID change returns info.
6. Gateway active-to-inactive returns critical.
7. Platform configured true-to-false returns warning.
8. Platform connected true-to-false returns critical.
9. Cron total/active changes are detected.
10. Repo dirty_count and file-list changes are detected.
11. Recent error count/category increase is detected without snippets.
12. Private-looking strings are absent from snapshot file and diff output.

Run:

```bash
python3 -m py_compile scripts/mcp/quinn_ops_server.py tests/test_quinn_ops_mcp.py
venv/bin/python -m pytest tests/test_quinn_ops_mcp.py -q
```

## Live promotion requirements

After repo-side tests pass, Quinn should review before live promotion. Live promotion requires owner approval because it changes a live MCP server and restarts the gateway. After promotion, verify with `hermes mcp test quinn_ops`, native MCP calls, and update `/home/quinn/docs/quinn-hermes-server.md`.
