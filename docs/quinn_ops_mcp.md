# Quinn Ops MCP

`quinn_ops` is a local MCP server that gives Quinn structured operational awareness of the Hermes installation. It is intentionally **eyes only** for Hermes operations: no service restarts, no Discord/channel history access, no approval tools, and no live config mutation.

The snapshot/diff extension adds one narrow self-state write so Quinn can compare the current sanitized overview against a previous sanitized baseline:

- `$HERMES_HOME/mcp/quinn_ops_state/overview_snapshot.json`

No other write scope is part of v1.

Version 1 is public-safe/read-only by default. MCP tools may become available across configured Hermes platforms after the gateway is restarted, so outputs must not expose private logs, raw config, session transcripts, auth material, process command lines, or message content.

## Files

- Server: `scripts/mcp/quinn_ops_server.py`
- Tests: `tests/test_quinn_ops_mcp.py`
- Snapshot/diff design: `docs/quinn_ops_snapshot_diff_design.md`

For live use, copy the server to:

```bash
mkdir -p /home/quinn/.hermes/mcp
cp /home/quinn/.hermes/hermes-agent/scripts/mcp/quinn_ops_server.py /home/quinn/.hermes/mcp/quinn_ops_server.py
```

## Config Snippet

Do not add this automatically during development. Add it manually when ready:

```yaml
mcp_servers:
  quinn_ops:
    command: "/home/quinn/.hermes/hermes-agent/venv/bin/python"
    args:
      - "/home/quinn/.hermes/mcp/quinn_ops_server.py"
    timeout: 60
    connect_timeout: 30
    sampling:
      enabled: false
```

Use the Hermes venv Python here if the MCP SDK is installed into the Hermes runtime venv. A bare `python` may resolve to a different interpreter and fail to import `mcp`.

Adding or removing MCP servers requires restarting Hermes/gateway. Promote repo changes live only after review, tests, backup of the previous live server, copy to `/home/quinn/.hermes/mcp/quinn_ops_server.py`, gateway restart, and MCP verification. Do not edit live config unless the task explicitly requires it.

## Verification

```bash
python /home/quinn/.hermes/mcp/quinn_ops_server.py
hermes mcp test quinn_ops
hermes mcp list
```

If the Python MCP SDK is not installed, direct stdio startup fails with a structured missing-dependency error. The collector functions and tests still import without the SDK.

## Tools

- `get_overview()`
- `get_gateway_status()`
- `get_platform_status()`
- `get_mcp_status()`
- `get_toolsets_status()`
- `get_cron_status()`
- `get_sessions_summary()`
- `get_recent_errors(limit=50)`
- `get_config_summary()`
- `get_repo_status()`
- `get_runtime_files_status()`
- `get_snapshot_status()`
- `get_overview_diff(update_baseline=False)`
- `save_overview_snapshot()`
- `healthcheck()`

`get_platform_status()` includes a passive delivery probe for each known platform. It distinguishes `delivery_capable`, `not_configured`, `not_connected`, `configured_delivery_unknown`, and `unknown` using only local `hermes status --all` output. It sets `history_read=false` and `delivery_attempted=false`; it must not read platform/channel history or send test messages.

`get_overview_diff()` includes a privacy-safe `summary.error_delta` object for recent log metadata: total before/after/delta, per-source category deltas, new categories, repeated categories, and last-seen timestamp movement. It must not include snippets or raw log lines.

## Security Boundaries

- Read-only for Hermes operations.
- The only allowed write is the sanitized Quinn Ops baseline at `$HERMES_HOME/mcp/quinn_ops_state/overview_snapshot.json`.
- `get_overview_diff(update_baseline=False)` does not mutate state.
- `save_overview_snapshot()` and `get_overview_diff(update_baseline=True)` write only sanitized overview metadata using an atomic replace and mode `0600` where possible.
- No service restarts or update commands.
- No platform/channel history reads and no test message sends from passive platform probes.
- No action or approval tools.
- No Discord/channel history access.
- Sessions are counted and summarized as metadata only; transcripts are not read.
- Config and logs are recursively redacted.
- Snapshots store sanitized overview data only. They must not contain raw logs, snippets, session transcripts, platform history, config secrets, auth values, or private message content.
- Raw log snippets are disabled by default. `get_recent_errors(..., include_snippets=True)` only returns short sanitized snippets when `QUINN_OPS_ALLOW_LOG_SNIPPETS=1` is explicitly set. Snapshot/diff error deltas remain metadata-only even if snippets are present in source payloads.
- Secrets, tokens, auth contents, headers, and environment values are never returned.
- Subprocess calls use argv lists, `shell=False`, and timeouts.
- Gateway status is built from whitelisted `systemctl --user show` fields only: active state, substate, load state, enabled state, PID, manager, and short booleans.

## Snapshot/Diff Extension

The snapshot/diff extension is documented in `docs/quinn_ops_snapshot_diff_design.md`. It keeps operational actions read-only, but allows one narrow self-state write under `$HERMES_HOME/mcp/quinn_ops_state/` so Quinn can compare the current sanitized overview against a previous sanitized baseline. Do not promote extension changes live without review, owner approval or equivalent active task authorization, gateway restart, and verification.
