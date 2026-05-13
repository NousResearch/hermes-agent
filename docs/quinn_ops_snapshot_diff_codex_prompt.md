# Codex Prompt: Quinn Ops MCP Snapshot/Diff

Repo: `/home/quinn/.hermes/hermes-agent`

Implement todo `qops-snapshot-diff` for Quinn Ops MCP.

Read first:

- `docs/quinn_ops_snapshot_diff_design.md`
- `docs/quinn_ops_mcp.md`
- `scripts/mcp/quinn_ops_server.py`
- `tests/test_quinn_ops_mcp.py`

Files to edit:

- `scripts/mcp/quinn_ops_server.py`
- `tests/test_quinn_ops_mcp.py`
- `docs/quinn_ops_mcp.md` if tool list/security notes need updating
- `docs/quinn_ops_snapshot_diff_design.md` only if implementation intentionally diverges from the design

Do not:

- live install
- copy to `/home/quinn/.hermes/mcp`
- edit live config
- restart gateway
- install packages
- read platform history
- read session transcripts
- store raw logs or private values

Goal:

Add snapshot/diff support to the existing `quinn_ops` MCP server. It should store a sanitized previous overview and report structured changes against a fresh overview.

Add these MCP tools:

1. `get_snapshot_status()`
2. `get_overview_diff(update_baseline: bool = False)`
3. `save_overview_snapshot()`

Allowed self-state write path:

- `$HERMES_HOME/mcp/quinn_ops_state/overview_snapshot.json`

This is the only new write scope. Use atomic write, parent dir creation, and mode `0600` where possible.

Important behavior:

- `get_overview_diff(update_baseline=False)` must not mutate state.
- First diff with no baseline should return `has_previous=false`, zero/empty diff groups, and a helpful headline/warning.
- `save_overview_snapshot()` creates/replaces the baseline using sanitized `get_overview()` data.
- `get_overview_diff(update_baseline=True)` computes diff first, then writes the current sanitized overview as the new baseline.
- All outputs must go through existing `response()` / `sanitize()`.
- Keep functions importable without MCP SDK.
- Do not add dependencies.

Diff stable metadata only. Ignore volatile timestamps except where the design says otherwise.

Minimum required comparisons:

- gateway active/substate/unit state/PID
- platform configured/connected/status per platform
- MCP configured servers and active-like count
- cron total/active/jobs
- sessions count/file counts/type distribution only, no content
- repo head/branch/describe/dirty_count/file path set
- recent error total and per-category counts, no snippets
- runtime file exists/size
- toolset summary counts and simple line-set changes if practical

Tests required:

- snapshot missing status
- save snapshot creates sanitized schema-versioned file
- diff with `update_baseline=False` does not write
- first diff/no baseline behavior
- gateway PID info change
- gateway active-to-inactive critical change
- platform configured true-to-false warning
- platform connected true-to-false critical
- cron count changes
- repo dirty/file-list changes
- recent error count/category increase without snippets
- private-looking strings absent from snapshot file and diff output

Run:

```bash
python3 -m py_compile scripts/mcp/quinn_ops_server.py tests/test_quinn_ops_mcp.py
venv/bin/python -m pytest tests/test_quinn_ops_mcp.py -q
```

Final response to Quinn should include:

- concise implementation summary
- exact tests run and result count
- confirmation that no live install/copy/config/restart/package install happened
- any divergence from the design contract
