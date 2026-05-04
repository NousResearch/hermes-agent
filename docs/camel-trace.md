# CaMeL Trace

CaMeL Trace is the observability layer for the Hermes CaMeL runtime.

It records, per turn:

- the trusted operator request
- the active runtime mode (`off`, `monitor`, `enforce`)
- untrusted sources present in context
- suspicious instruction flags seen in those sources
- sensitive tool decisions
- response-layer hijack decisions

Trace files are written to:

```text
~/.hermes/camel_traces/camel_trace_<session_id>.json
```

## Why It Exists

CaMeL Guard already blocks or sanitizes unsafe behavior in `enforce` mode.

CaMeL Trace answers the operator questions that come immediately after that:

- what untrusted source caused the violation
- which capability was requested
- whether the runtime would have blocked it or only flagged it
- what markers were matched in a poisoned response
- how `off`, `monitor`, and `enforce` differ in practice

## CLI

Show the latest trace:

```bash
hermes camel trace
```

Show a specific trace:

```bash
hermes camel trace --session 20260319_033638_f74571
```

Render as markdown:

```bash
hermes camel trace --session 20260319_033638_f74571 --format markdown
```

Write to a file:

```bash
hermes camel trace --session 20260319_033638_f74571 --format markdown --output /tmp/camel-trace.md
```

Rerun the bundled benchmark:

```bash
hermes camel benchmark --write-doc
```

## Runtime Semantics

### `off`

- legacy Hermes runtime
- no CaMeL enforcement
- trace still records what happened if tracing is enabled

### `monitor`

- observe-only mode
- policy violations are recorded and surfaced
- tool execution and final output continue

### `enforce`

- full CaMeL behavior
- sensitive tool calls are blocked unless authorized by the trusted operator plan
- poisoned answer prefixes are sanitized before the final assistant response is returned

## Trace Structure

Each trace includes:

- `summary`
  - `turn_count`
  - `policy_alert_count`
  - `response_alert_count`
  - `unique_untrusted_sources`
- `turns`
  - `operator_request`
  - `goal_summary`
  - `trusted_context_excerpt`
  - `untrusted_sources`
  - `suspicious_flags`
  - `output_markers`
  - `tool_decisions`
  - `response_decision`

Tool decisions include:

- tool name
- capability
- allowed / blocked decision
- reason
- untrusted sources involved
- truncated tool-argument preview

Response decisions include:

- whether the answer was clean or flagged/sanitized
- matched hidden-output markers
- original preview
- final preview

## Recommended Operator Flow

1. Run a session normally with `--camel-guard monitor` when tuning prompts or integrations.
2. Inspect the resulting trace with `hermes camel trace`.
3. Promote to `--camel-guard enforce` once the policy behavior matches the workflow you want.

That gives a clean progression:

- legacy compatibility with `off`
- observability with `monitor`
- full protection with `enforce`
