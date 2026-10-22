---
name: tool-audit
description: "Report per-tool call counts and error rates for a session."
version: 2.0.0
author: nankingjing + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [audit, diagnostics, sessions, tool-usage, self-monitoring]
    related_skills: [hermes-agent]
---

# Tool Audit Skill

Generate an operator-facing report of one session's tool usage from the
canonical SQLite session store (`state.db`): per-tool call counts, results
correlated back to their calls by `tool_call_id`, orphaned calls, and provable
tool errors. The audit is strictly read-only and reports only what the store
can prove — it does not report latency, because per-call `duration_ms` exists
only inside the in-process `post_tool_call` observer hook and is never
persisted to the session store.

## When to Use

- The user asks "how are my tools performing?" or "audit your tool usage"
- After a long session with many tool calls, to spot failing or runaway tools
- When debugging agent loops, repeated tool failures, or orphaned calls
- Don't use for: latency profiling (that needs a `post_tool_call` observer
  plugin), or batch-runner / RL trajectories (not stored in `state.db`)

## Prerequisites

- A readable `state.db` under the active Hermes home. The bundled script
  resolves it profile-aware: `hermes_constants.get_hermes_home()` when
  importable, else the `HERMES_HOME` env var, else the platform default.
- Python 3 available to the `terminal` tool. The script is stdlib-only.

## How to Run

Run the bundled helper with the `terminal` tool (paths relative to this
skill directory):

```
python scripts/tool_audit.py                     # audit $HERMES_SESSION_ID, else the latest session
python scripts/tool_audit.py --session <id>      # audit a specific session (full id or unique prefix)
python scripts/tool_audit.py --json              # machine-readable output
python scripts/tool_audit.py --db /path/state.db # explicit store (e.g. another profile's)
```

## Quick Reference

| Column | Meaning | Source of truth |
|--------|---------|-----------------|
| calls | `tool_calls` entries on assistant rows | `messages.tool_calls` JSON |
| results | tool-role rows matched to a call by `tool_call_id` | `messages.tool_call_id` |
| orphaned | calls with no matching result row | correlation gap |
| errors | matched results whose payload is a JSON object with a truthy `error` key | same rule as the runtime observer (`model_tools._tool_result_observer_fields`) |

Uncorrelated tool results (a result row whose `tool_call_id` matches no
surviving call, e.g. after compaction) are counted separately, never
attributed to a tool.

## Procedure

1. Resolve the target session. Default is `$HERMES_SESSION_ID` (set by the
   running agent), falling back to the most recently started session. Pass
   `--session` when the user names one. Done when the script prints the
   session header (id, source, started, message count).
2. Run the audit via `terminal` as shown above. Done when the per-tool table
   and totals print with exit code 0.
3. Interpret the numbers — only claims the table supports:
   - High `errors` on one tool → quote 1-2 of its error payloads (rerun with
     `--json` and read the `sample_errors` field) and suggest a concrete fix.
   - `orphaned` calls → usually an interrupted turn or a compaction boundary;
     say so instead of calling it a failure.
   - A tool dominating `calls` → look for loops or missed batching.
4. Deliver a short report to the operator: session header, the table, then
   at most three recommendations, each tied to a number in the table. Done
   when every recommendation cites a metric the script actually printed.

## Pitfalls

- Do not invent success rates or latency. The store has no per-call duration;
  anything beyond calls / results / orphaned / errors is fabrication.
- `errors` is a lower bound: only JSON-object results with a truthy `error`
  key are provable. Plain-text failure prose is not counted — say "provable
  errors", not "all errors".
- Legacy `~/.hermes/**/*.jsonl` transcripts are dead: no longer written or
  read. Never grep them; only `state.db` is canonical.
- The script opens the store with SQLite `mode=ro` — never "fix" it to a
  writable connection, and do not run schema queries against a live gateway
  store outside the script.
- Session id prefixes must be unique; the script exits non-zero on ambiguity
  instead of guessing.

## Verification

- [ ] Script exited 0 and printed the session header for the intended session
- [ ] Every number in your report appears in the script output
- [ ] No latency, duration, or "success rate" claims anywhere in the report
- [ ] Store untouched: no `state.db-wal` growth from the audit (read-only URI)
