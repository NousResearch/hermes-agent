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

## `audit_memory_context.py`

Audits recalled `<memory-context>` / Mnemosyne-context text for likely
non-durable raw fragments. The helper is deliberately non-destructive: it writes
a JSONL report and can print suggested invalidation commands, but it does not
modify memory itself.

```bash
# Read a captured memory-context text file and append an audit report
python scripts/self_improvement/audit_memory_context.py --input /tmp/memory-context.txt

# Print suggested invalidation calls for entries with memory ids
python scripts/self_improvement/audit_memory_context.py --input /tmp/memory-context.txt --commands --no-write

# Pipe context directly from stdin
python scripts/self_improvement/audit_memory_context.py --input -
```

Candidate reasons include raw `[USER]` fragments, standalone command fragments
such as “proceed”, one-off task prompts, background-process notification
fragments, and low-importance raw conversation entries. Stable distilled rules
and preferences are intentionally ignored even when they mention those words.

## `summarize.py`

Summarizes `task_runs.jsonl` and `memory_context_audit.jsonl` into a compact
review payload. It reports aggregate tokens/tool/API counts, the latest task's
largest context contributors, memory-context candidate counts, and review flags
such as `duplicate_skill_view`, `repeated_cronjob_list`, and
`memory_context_noise`.

```bash
python scripts/self_improvement/summarize.py
```

The summary is safe to paste into review notes because candidate previews include
reason codes, content lengths, and memory ids only — not raw memory content.

## Output files

- `task_runs.jsonl` — task/session telemetry records.
- `events.jsonl` — optional compact workflow-improvement events when
  `--append-event` is passed.
- `memory_context_audit.jsonl` — non-destructive memory-context audit reports.

Each task-run entry stores structured metrics only:

- session id/source/model/timing/end reason;
- message/tool/API counts;
- input/output/reasoning/cache token counts;
- cost fields when Hermes has them;
- role/tool count summaries;
- compact `role:tool:chars` labels for the largest context contributors.

No message content is copied into telemetry output.
