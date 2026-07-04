# Hermes Context Governor

## Purpose

Hermes carries long tool-heavy workflows (CFIPros, marcusgoll.com, GitHub PR operations, release automation, staging/production deploys, Obsidian KB updates, admin email notifications). Current risk: stale GitHub, CI, deploy, file, and documentation state accumulates across turns.

The Context Governor enforces:
- **Retain only recent whole tool call/response pairs** (default: last 5–8)
- **Summarize evicted tool interactions** into a compact task-state ledger
- **Preserve task-level situational awareness** via structured summaries
- **Verify completion by independent read-back** of target state (GitHub, CI, deploy, Obsidian, SES email, app routes)

Research basis: arXiv:2606.10209v1 — last 5 tool pairs + summarization beats full-context retention (91.6% vs 71.0% completion, 553K vs 1.48M tokens, 5.79h vs 14.56h).

---

## Configuration

```yaml
# In config.yaml under `context_governor:`
context_governor:
  enabled: true
  raw_tool_window: 5              # Last N tool call/response pairs to keep raw
  summary_window: 3               # Number of evicted interactions to summarize
  max_state_summary_words: 700    # Word ceiling for task-state summary
  max_raw CI/deploy log lines: 200   # Max lines from CI/deploy logs in context
  verification_required: true     # Independent read-back before completion
```

---

## Context Construction Per Turn

```
SYSTEM / ROLE RULES
PROJECT MEMORY SUMMARY
TASK STATE LEDGER
RECENT RAW TOOL CALLS (last N)
CURRENT OBJECTIVE
NEXT REQUIRED VERIFICATION
```

### Task-State Ledger Schema

```yaml
task_state:
  run_id: "<uuid>"
  repo: "marcusgoll/cfipros"
  objective: "Set up admin email notifications using Amazon SES"
  current_branch: "<branch>"
  last_verified_commit: "<sha>"
  known_constraints:
    - "Amazon SES is the email provider"
    - "Do not expose secrets"
    - "Do not deploy production without explicit approval"
  completed_actions:
    - "Located existing email transport code"
    - "Identified admin notification trigger points"
  open_questions:
    - "Exact admin recipient source not yet verified"
  blockers: []
  verification_evidence:
    tests_run: []
    files_read: []
    external_state_checked: []
  next_action: "Inspect existing notification/email modules before editing"
```

---

## Tool-Output Reducers

| Tool Output | Keep Raw? | Reduce Into |
|-------------|-----------|-------------|
| GitHub PR list | No | PR number, title, branch, draft, mergeability, checks, labels |
| CI logs | Rarely | Failing job, failing command, first error, changed files implicated |
| Deploy logs | Partly | Environment, version, commit SHA, health check, rollback marker |
| Obsidian search | No | Note path, heading, relevant excerpt, last modified |
| Browser screenshots | Partly | URL, viewport, visible issues, measured layout defects |
| SES/email test | Partly | Provider, recipient class, send result, message ID if safe |
| git diff | Partly | Files changed, intent, risky hunks, generated files excluded |

---

## Semantic Batch Summarization

When raw tool calls are evicted from the window, they are summarized into `ContextSummary` objects:

- Goal: from the current task-state ledger.
- Actions completed: patches, writes, executed code, searches, shell commands.
- Files/resources touched: unique paths from read/write/patch calls.
- Verified current state: code/test results that exited cleanly.

Older summaries are retained up to `summary_window * 2` entries; the last `summary_window` are injected into the prompt.

## User-facing Commands

- `/context`, `/ledger`, `/ctx` — show current governor state.
- `/context clear` — clear raw calls and summaries, keep ledger.
- `/context reset` — wipe ledger, raw calls, and summaries.

## Telemetry

In-memory metrics (no external dependency):

| Metric | Kind |
|--------|------|
| `context_governor_load_hits_total` / `context_governor_load_misses_total` | counter |
| `context_governor_persist_total` / `context_governor_persist_failures_total` | counter |
| `context_governor_reset_total` | counter |
| `context_governor_tool_calls_recorded_total` | counter |
| `context_governor_command_invocations_total` | counter |
| `context_governor_prompt_chars` | gauge |

By default, a telemetry snapshot is logged at `INFO` level on session end. To export to Prometheus text format, call:

```python
from agent.context_governor import get_context_governor
get_context_governor().telemetry.write_prometheus_file(Path("/var/lib/node_exporter/textfile/context_governor.prom"))
```

Or register a custom observer:

```python
get_context_governor().telemetry.set_observer(lambda name, value, kind: print(name, value, kind))
```

---

When pruning older tool calls, insert:

```
Summary of previous tool calls:
- Repo inspected: marcusgoll/cfipros.
- Branch: <branch>. Base: main at <sha>.
- Goal: admin email notifications using Amazon SES.
- Relevant files inspected:
  - <path>: existing email transport code.
  - <path>: admin/user settings.
- Decisions made:
  - Do not introduce a second email provider.
  - Reuse existing Amazon SES abstraction unless broken.
- Changes already made: None yet.
- Current verified state:
  - Working tree: clean/dirty.
  - Tests run: <commands and results>.
  - CI/deploy status: <status if checked>.
- Remaining work:
  - Implement admin notification trigger.
  - Add tests.
  - Verify no secrets are logged.
```

---

## Independent Read-Back Verification Rules

| Task Type | Completion Verifier |
|-----------|---------------------|
| PR merged | `gh pr view`, branch state, checks, merge commit |
| Staging deploy | Deployment status, app health endpoint, smoke test |
| Production release | Tag, release notes, deploy health, rollback marker |
| Obsidian doc update | Actual file exists, links resolve, no duplicate source-of-truth |
| Admin email notification | Test event triggers email path, SES mock/local test passes, no secret leak |
| UI change | Screenshot or Playwright check at desktop/mobile |
| Cleanup/refactor | Lint, typecheck, tests, build, route smoke test |

**No task is complete until Hermes independently verifies the target state.** Agent self-report is not evidence.

---

## Anti-Stale-State Rules

1. Older tool outputs are **stale unless re-read**.
2. Never act on old PR, CI, deploy, or route status without fresh verification.
3. Summaries may guide, but raw state must be rechecked before consequential actions.
4. Final completion requires independent read-back.

---

## Context Summary Schema (for agents)

After every tool-heavy batch, produce:

```yaml
context_summary:
  objective: ""
  verified_current_state: []
  actions_completed: []
  files_or_resources_touched: []
  decisions_made: []
  stale_or_discarded_assumptions: []
  blockers: []
  risks: []
  next_required_verification: ""
```

---

## Implementation Notes

The Context Governor sits between the conversation history and the model call.
It uses the existing `trajectory_compressor.py` and `context_compressor.py` infrastructure.

- **Semantic batch summarization:** evicted tool calls are summarized into `ContextSummary` objects (goal, actions completed, files touched, verified state) before injection.
- **Telemetry:** `TelemetryCollector` tracks load/persist/reset/tool-call/command/prompt-char metrics. Register an observer via `governor.telemetry.set_observer(callback)` to export to Prometheus/PostHog/logs.
- **Slash commands:** `/context`, `/ledger`, `/ctx` show state; `/context clear` and `/context reset` clear state.
- **Persistence:** Task-state ledger and raw-tool windows are saved to the session DB (`sessions.context_governor_state`) when a session ends, and restored on session start. The ledger survives new Hermes sessions (new processes, restarts, `/resume`) but is intentionally reset by `/new` or `/reset`.
- No new external services. No secrets, credentials, or production hooks.
- Works with existing `SOUL.md`, `AGENTS.md`, `MEMORY.md`, `USER.md` authority chain.

---

## Benchmarking

Run the same Hermes task twice:

| Run | Context Policy |
|-----|----------------|
| A | Current full-context style |
| B | Last 5–8 tool calls + task-state summary + read-back verifier |

Measure: completion rate, stale assumptions, repeated actions, tool calls, token usage, time to useful PR, verifier failures.

### Measured Baseline

Synthetic benchmark (`scripts/benchmark_context_governor.py`, 100 large `read_file` calls):

| Mode | Prompt chars | Reduction |
|------|-------------|-----------|
| Full context | 6,035 | 1.0× |
| Context Governor | 967 | 6.2× |

Real-file workflow A/B (`scripts/benchmark_context_governor_real_projects.py`):

| Workflow | Full context | Governor | Reduction |
|----------|-------------|----------|-----------|
| CFIPros AKTR OCR test improvement | 3,972 | 1,522 | 2.6× |
| CFIPros Sentry webhook auth investigation | 3,916 | 1,527 | 2.6× |
| Homelab Obsidian Docker restart loop | 2,029 | 1,591 | 1.3× |

Do not claim improvement until Hermes beats its current baseline on real CFIPros tasks.