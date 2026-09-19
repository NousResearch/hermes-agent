# run-receipts

Tamper-evident, hash-chained evidence records for every Hermes agent run — the
same receipt pattern `hermes update` uses (`~/.hermes/logs/update_receipts/`),
applied to the agent loop itself.

## What it records

Each turn (`run_conversation` invocation) finalizes one JSON line into
`<HERMES_HOME>/receipts/runs.ndjson`:

| field | contents |
|---|---|
| `run_id` / `session_id` / `task_id` / `turn_id` | run identity |
| `platform` / `model` / `provider` | where and what ran |
| `started_at` / `ended_at` / `duration_s` | timing |
| `outcome` / `exit_reason` | `completed`, `failed`, or `interrupted` + the turn's exit reason |
| `tool_calls[]` | per call: id, name, `args_sha256`, `result_sha256`, duration |
| `api_calls[]` | per call: id, model, provider, duration, ok, error type/status |
| `counts` | tool calls, API calls, errors |
| `prev_sha256` / `sha256` | chain link + self-hash |

**Privacy:** arguments, results, and error messages are stored only as sha256
digests — a receipt proves a call happened and lets you compare two runs'
inputs, but contains no content. Chain order is guaranteed by a lockfile
(`runs.ndjson.lock`) so concurrent processes (gateway multiplex, cron
children, CLI) cannot fork the chain.

## Usage

```bash
hermes receipts            # latest receipt
hermes receipts stats      # totals + outcome breakdown
hermes receipts verify     # walk the chain; exit 1 on first break
```

In-session: `/receipts [latest|stats|verify]`.

## Hooks used

`on_session_start`, `post_tool_call`, `post_api_request`, `api_request_error`,
`on_session_end`, `on_session_finalize` — all observer-only. Nothing on the
token path; the single file append happens at turn finalize.
