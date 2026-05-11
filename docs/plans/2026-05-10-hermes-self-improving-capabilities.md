# Hermes Self-Improving Capabilities Implementation Plan

> **For Hermes:** Use subagent-driven-development skill where useful; keep edits tightly scoped and verify locally.

**Goal:** Make the capabilities listed/hinted in Graeme's X posts (self-measuring autonomy, recovery, research/intake, Dreamer/proposal loops, multi-agent build/QA, receipts) present, operator-visible, and smoke-tested in this local Hermes setup.

**Architecture:** Treat the post as a capability checklist, not a marketing target. Prefer local-first primitives: SQLite/session state, structured JSONL logs, inbox/outbox JSON queues, eval/failure stores, CLI slash commands, cron jobs, and reports. Do not add unsafe unattended shell autonomy; keep repair/build loops gated by existing approvals and review flows.

**Tech Stack:** Hermes Python repo, `hermes_state.SessionDB`, `hermes_cli` commands, existing cron scheduler, pytest, local `~/.hermes` artifacts.

---

## Extracted checklist

1. Receipts/scorecards: structured logs, performance reports, scorecards, dashboard visibility.
2. Behavioral eval/canary seed: eval cases, run storage, recent/show/report commands.
3. Failure/recovery: failure classification, aggregation, stalled/loop detection, repair routing artifacts.
4. Proposal/intake queue: local inbox/outbox primitives, acceptance/rejection, package persistence.
5. Research/intelligence pipeline: browser/web/docs/community/transcript/market/build-log sources with verification gates.
6. Multi-agent auto-build: main/planner, coder, QA/reviewer, phased execution, repair loop, E2E verification.
7. Dreamer/subconscious steering: advisory nudges, novelty preservation, prompt guidance, observe-only canaries.
8. Watchdogs/event wakeups: cron watchdogs, scheduler health, delivery failure visibility.
9. Cost/rate/usage awareness: usage insights and quota/status visibility.
10. Operator trust: transparent proposals, no destructive auto-apply, residual gaps explicitly reported.

---

## Task 1: Stabilize local telemetry primitives

**Objective:** Make report/self-review/ops modules import and write local artifacts safely.

**Files:**
- Modify: `utils.py`
- Verify: `tests/hermes_cli/test_local_artifacts.py`, `tests/test_ops.py`

**Steps:**
1. Add `atomic_text_write()` beside `atomic_json_write()`.
2. Run targeted tests and verify import failures are gone.

## Task 2: Add task records to `SessionDB`

**Objective:** Support receipts/scorecards/benchmarks/dashboard queries through a first-class `tasks` table.

**Files:**
- Modify: `hermes_state.py`
- Verify: `tests/test_ops.py`, `tests/test_benchmark_record.py`, dashboard tests.

**Steps:**
1. Add tasks schema and indexes to `SCHEMA_SQL`.
2. Add `SessionDB.create_task()`, `get_task()`, `update_task()`.
3. Cascade/delete task rows in session delete/prune paths.
4. Add `pinned_at` session column expected by recent chat/dashboard tests.
5. Run targeted tests.

## Task 3: Wire eval/failure slash commands

**Objective:** Make behavioral evals and failure analysis reachable from the CLI slash-command path.

**Files:**
- Modify: `hermes_cli/commands.py`
- Modify: `cli.py`
- Verify: `tests/hermes_cli/test_eval_command.py`, `tests/hermes_cli/test_failures_command.py`

**Steps:**
1. Add `/eval` and `/failures` command definitions.
2. Implement `_handle_eval_command()` for help/list/run/recent/show.
3. Implement `_handle_failures_command()` for help/recent/top/show.
4. Add dispatch in `process_command()`.
5. Run targeted tests.

## Task 4: Add test support for approval logging

**Objective:** Keep approval/risk telemetry tests isolated and deterministic.

**Files:**
- Modify: `tools/approval.py`
- Verify: `tests/test_approval_logging.py`

**Steps:**
1. Add a test-only-safe `reset_state()` helper that clears in-memory approval state.
2. Run test and confirm no cross-test leakage.

## Task 5: Run smoke/end-to-end checks

**Objective:** Prove the local setup has working operator paths.

**Commands:**
- Targeted pytest slice covering new/prototype capabilities.
- `hermes status`, `hermes tools list`, `hermes cron list --all`, `hermes mcp list`.
- CLI smoke: eval list/run, ops summary, failures recent/top, report/self-review artifacts.
- Dashboard HTTP smoke if practical.

## Task 6: Residual-gaps report

**Objective:** Summarize what is active/verified, present-but-prototype, configured-untested, blocked, or missing.

**Output:** Short Telegram-friendly wrap-up with exact tests and remaining safe next steps.

---

## Continuation checkpoint — 2026-05-10

### Completed in this pass

- Stabilized eval executor against current `AIAgent` signature.
- Added top-level `hermes ops` command and CLI/operator paths for evals/failures.
- Added packaging coverage for new top-level telemetry modules.
- Removed wildcard CORS from local dashboard JSON responses.
- Added dashboard cross-origin POST rejection for local destructive endpoints.
- Added ops schema reconciliation so `hermes ops` can safely touch an existing `state.db` before the tasks table exists.
- Fixed `hermes ops --help` so it delegates to the real ops subcommand help.
- Created/verified local skills:
  - `project-agent-rules`
  - `prompt-quality-review`
  - `senior-pm-prd-planning`
  - `personal-research-engine`
  - `supervisor-specialist-critic-workflow`
  - `printing-press-pilot`
- Created local workspace starters:
  - `~/Desktop/Hermes Files/projects/personal-research-engine/`
  - `~/Desktop/Hermes Files/projects/printing-press-pilot/`

### Verification

- Targeted pytest slice: `148 passed in 3.65s`.
- Live eval smoke: `hermes ops eval run file-create-and-read` → `1/1 passed`.
- CLI smoke:
  - `hermes ops eval list`
  - `hermes ops failures top 5`
  - `hermes ops summary --json`

### Remaining important cleanup before merge

- Wire structured telemetry deeper into real agent/model/tool execution paths where receipts are still sparse.
- Decide whether unrelated untracked WooCommerce/RadoClones files belong in a separate commit; do not include them in this capability commit by default.
- De-duplicate `/eval` and `/failures` handling between `cli.py` and `hermes_cli.ops` if this grows further.
- Consider stronger dashboard auth/CSRF tokens if dashboard destructive APIs remain enabled beyond localhost.
