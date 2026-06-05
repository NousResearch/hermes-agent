# Self-Improvement Telemetry Helpers

This directory contains lightweight, GitHub-ready helpers for recording Hermes
self-improvement telemetry without copying raw transcripts, prompts, tool
outputs, or secrets into logs.

## `log_session_telemetry.py`

Reads Hermes `state.db` and appends one compact task-run record to
`~/.hermes/ops/self-improvement-log/task_runs.jsonl` by default.

```bash
python scripts/self_improvement/log_session_telemetry.py
```

Useful flags:

```bash
# Inspect the payload without writing JSONL
python scripts/self_improvement/log_session_telemetry.py --dry-run

# Record a specific session
python scripts/self_improvement/log_session_telemetry.py --session-id 20260605_182249_c2fb0e

# Allow selecting the newest open/current session when --session-id is omitted
python scripts/self_improvement/log_session_telemetry.py --include-open

# Re-append a session already present in task_runs.jsonl
python scripts/self_improvement/log_session_telemetry.py --force

# Also write a compact workflow event for the self-improvement event watcher
python scripts/self_improvement/log_session_telemetry.py --append-event
```

Default behavior intentionally chooses the newest **ended** session, not the
newest open session. That avoids accidentally attributing the live review
conversation to the task that just completed.

## Output files

- `task_runs.jsonl` — task/session telemetry records.
- `events.jsonl` — optional compact workflow-improvement events when
  `--append-event` is passed.

Each task-run entry stores structured metrics only:

- session id/source/model/timing/end reason;
- message/tool/API counts;
- input/output/reasoning/cache token counts;
- cost fields when Hermes has them;
- role/tool count summaries;
- compact `role:tool:chars` labels for the largest context contributors.

No message content is copied into telemetry output.
