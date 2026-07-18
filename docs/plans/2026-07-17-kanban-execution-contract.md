# Kanban Execution Contract Implementation Plan

> **For Hermes:** implement task-by-task with TDD; use deterministic tooling for inspection/benchmarks and one independent review only after the postimage is frozen.

**Goal:** Reduce Kanban token waste from oversized cards, repeated context, redundant reviews, iteration exhaustion, and blind retries without weakening safety gates, tools, model capability, or evidence quality.

**Architecture:** Add a deterministic **Kanban Execution Contract (KEC)** around the existing task/run schema. A pure analyzer classifies scope before dispatch; oversized work can be routed to existing triage/decomposition. Compact context remains reversible through a full-context escape hatch. Exact-postimage reviews use existing idempotency storage through a first-class `review_key`. Kanban workers receive one advisory, cache-stable checkpoint warning and budget-exhausted runs persist one bounded, explicitly unverified continuation handoff; an opt-in `triage_once` policy replaces blind same-card retries with decomposition.

**Tech Stack:** Python 3.11+, stdlib dataclasses/regex, SQLite-backed Kanban, argparse, pytest.

**Compatibility and safety defaults:**

- Lifecycle and context behavior remain compatible by default: `granularity_guard: warn`, `context_profile: full`, and `budget_exhaustion_policy: retry`.
- `kanban.budget_warning_ratio: 0.75` adds only an advisory API-tail marker and idempotent audit event; it does not alter lifecycle, the cached system prompt, or persisted conversation history.
- Operators can opt into `triage`, `compact`, `block`, or `triage_once`; `allow` remains an explicit per-card broad-scope override.
- Compact context never deletes evidence. `kanban context --full` and
  `kanban_show(full_context=true)` expose the historical bounded profile; an
  exact full budget handoff is recovered with its hash-bound `CONTEXT_REF`
  command (`context <task> --run-id <run> --field partial_summary_full`).
- Review deduplication is exact and explicit (`review_key`), not inferred from prose.
- No automatic approval, review PASS, controller decision, deployment, or external action is introduced.

---

### Task 1: Establish deterministic scope assessment

**Objective:** Classify a card as `ok`, `caution`, or `split` using structural signals rather than an LLM call.

**Files:**
- Create: `hermes_cli/kanban_budget.py`
- Create: `tests/hermes_cli/test_kanban_budget.py`

**Steps:**
1. Write RED tests for a narrow bugfix, a broad Nexo-style lifecycle card, Spanish action verbs, many acceptance criteria, and deterministic JSON output.
2. Run `python -m pytest -q tests/hermes_cli/test_kanban_budget.py`; expect import/test failures.
3. Implement `TaskBudgetAssessment` and `assess_task(title, body, max_turns=60)` with explicit action-family/reason counters and no network/model dependency.
4. Re-run the focused tests; expect PASS.
5. Commit `feat(kanban): add deterministic task budget assessment`.

### Task 2: Apply the granularity guard at creation

**Objective:** Persist every assessment and optionally route `split` cards to existing Triage before any worker spends tokens.

**Files:**
- Modify: `hermes_cli/kanban_db.py:create_task`
- Modify: `hermes_cli/config.py:DEFAULT_CONFIG["kanban"]`
- Modify: `hermes_cli/kanban.py:create parser/handler`
- Modify: `tests/hermes_cli/test_kanban_db.py`
- Modify: `tests/hermes_cli/test_kanban_cli.py` or nearest create-command tests

**Steps:**
1. Write RED tests proving `warn` records `granularity_assessed` without changing status, `triage` parks an oversized task in Triage, and `allow` preserves an intentional broad card.
2. Run focused tests and confirm RED.
3. Add `granularity_policy` to `create_task`; resolve `off|warn|triage|allow` fail-closed to `warn` on invalid config.
4. Append a bounded `granularity_assessed` event with verdict, score, reasons, action families, estimated context tokens, and suggested shards.
5. Add CLI `--allow-broad`; keep default compatibility.
6. Re-run focused tests; expect PASS.
7. Commit `feat(kanban): preflight oversized tasks before dispatch`.

### Task 3: Add exact-postimage review deduplication

**Objective:** Prevent duplicate reviews of the same immutable postimage and claim contract without suppressing deliberate new review rounds.

**Files:**
- Modify: `hermes_cli/kanban_db.py:create_task`
- Modify: `hermes_cli/kanban.py`
- Modify: `tools/kanban_tools.py`
- Modify: `plugins/kanban/dashboard/plugin_api.py`
- Modify: relevant CLI/tool/API tests

**Steps:**
1. Write RED tests: identical `review_key` returns the existing non-archived task; a different hash/round creates a new task; conflicting explicit idempotency/review keys are rejected.
2. Run focused tests; confirm RED.
3. Validate the postimage and claims SHA-256 values, then map `review_key` onto the reserved, versioned `idempotency_key` namespace (`review:v1:<postimage>:<claims>[:round-N]`), avoiding a schema migration and preventing generic keys from reserving review identities.
4. Expose `--review-key`, model-tool `review_key`, and dashboard API `review_key`.
5. Re-run tests; expect PASS.
6. Commit `feat(kanban): deduplicate exact-postimage reviews`.

### Task 4: Add a reversible compact worker-context profile

**Objective:** Bound retry/parent/comment context while retaining a full-context escape hatch.

**Files:**
- Modify: `hermes_cli/kanban_db.py:build_worker_context`
- Modify: `hermes_cli/config.py`
- Modify: `hermes_cli/kanban.py:context`
- Modify: `tools/kanban_tools.py:kanban_show`
- Modify: `tests/hermes_cli/test_kanban_db.py`
- Modify: `tests/tools/test_kanban_tools.py`

**Steps:**
1. Write RED tests with large run summaries, comments, and >8 parents. Assert compact mode keeps bounded title/body/latest handoff and `CONTEXT_REF`, lists omitted parent IDs, and is materially shorter; assert the explicit run-field reader recovers the full handoff exactly.
2. Run focused tests; confirm RED.
3. Add context-limit profiles: current values for `full`; bounded attempts/comments/parents/field sizes plus an aggregate 48 KiB UTF-8 ceiling for `compact`.
4. Add explicit omission markers that distinguish bounded full-profile retrieval from exact hash-bound handoff recovery.
5. Add `kanban context --full`, `kanban_show(full_context=true)`, and the read-only exact field surface `kanban context <task> --run-id <run> --field partial_summary_full`.
6. Re-run tests; expect PASS.
7. Commit `feat(kanban): add reversible compact worker context`.

### Task 5: Preserve a compact continuation handoff at exhaustion

**Objective:** Ensure a budget-exhausted worker never forces the next worker to rediscover completed work.

**Files:**
- Modify: `agent/chat_completion_helpers.py:handle_max_iterations`
- Modify: `agent/turn_finalizer.py:finalize_turn`
- Modify: `hermes_cli/kanban_db.py:_record_task_failure`
- Create: `tests/agent/test_iteration_limit_handoff_prompt.py`
- Modify: `tests/agent/test_turn_finalizer_iteration_limit_exit.py`
- Create: `tests/hermes_cli/test_kanban_execution_contract.py`

**Steps:**
1. Write RED tests proving one bounded `[partial_unverified]` run summary is stored, no duplicate comment is injected, and retry context sees the handoff exactly once.
2. Run focused tests; confirm RED.
3. For Kanban workers, request a compact fixed-section continuation summary (`completed`, `changed`, `tests`, `remaining`, `resume`, `invariants`) on the existing toolless summary call.
4. Add optional `partial_summary` plumbing to `_record_task_failure`; persist one structured 4 KiB UTF-8 summary with all required sections and retain the hash-bound full source once in run metadata.
5. Forward the summary from `turn_finalizer`; retain existing circuit-breaker semantics.
6. Re-run tests; expect PASS.
7. Commit `fix(kanban): preserve bounded exhaustion handoffs`.

### Task 6: Add opt-in checkpoint pressure and triage-once recovery

**Objective:** Warn before the hard wall and replace blind same-card retries with one decomposition opportunity.

**Files:**
- Modify: `agent/iteration_budget.py`
- Modify: `agent/conversation_loop.py`
- Modify: `agent/turn_finalizer.py`
- Modify: `hermes_cli/kanban_db.py`
- Modify: `hermes_cli/config.py`
- Create: `tests/agent/test_iteration_budget_checkpoint.py`
- Modify: `tests/run_agent/test_run_agent.py`
- Modify: Kanban DB/finalizer tests

**Steps:**
1. Write RED tests for non-Kanban isolation, one ephemeral Kanban checkpoint message at the configured ratio, no persistent transcript mutation, and no premature “stop now” language.
2. Write RED tests for `retry` compatibility, first `triage_once` exhaustion routing to Triage, and second exhaustion staying blocked.
3. Run focused tests; confirm RED.
4. Implement the pure pressure-message policy and attach it only to the latest previously unseen tool-result tail in the per-call API copy, preserving a byte-stable prefix thereafter.
5. Add `budget_exhaustion_policy: retry|block|triage_once`; fail invalid values to `retry`.
6. Add a bounded `budget_triaged` event and one-shot guard.
7. Re-run tests; expect PASS.
8. Commit `feat(kanban): checkpoint and decompose budget-exhausted work`.

### Task 7: Make decomposition budget-aware

**Objective:** Ensure Triage creates atomic implementation → closure → review → controller graphs and reuses the saved handoff.

**Files:**
- Modify: `hermes_cli/kanban_decompose.py`
- Modify: `tests/hermes_cli/test_kanban_decompose.py`

**Steps:**
1. Write RED tests that the decomposer request contains latest partial handoff, scope assessment, immutable-review guidance, review-key guidance, and “full matrix only at closure”.
2. Run focused tests; confirm RED.
3. Update prompt/template construction with bounded handoff and assessment data.
4. Add lifecycle/sharding constraints without increasing decomposer call count.
5. Re-run tests; expect PASS.
6. Commit `feat(kanban): make decomposition execution-budget aware`.

### Task 8: Add operator visibility and documentation

**Objective:** Make the method measurable through existing task events, explicit CLI/API knobs, documentation, and a reproducible benchmark.

**Files:**
- Modify: `hermes_cli/kanban.py`
- Modify: `website/docs/user-guide/features/kanban.md`
- Add/modify CLI tests
- Add: `scripts/benchmark_kanban_context.py`
- Add: `docs/benchmarks/kanban-context-synthetic-v1.json`

**Steps:**
1. Surface the deterministic assessment through the existing `granularity_assessed` event on every created task.
2. Document safe defaults, opt-in settings, review keys, full-context escape hatch, and lifecycle examples.
3. Add a deterministic synthetic full-versus-compact context benchmark with transparent token approximation.
4. Run docs/config/schema tests.

### Task 9: Verify and benchmark

**Objective:** Prove correctness and quantify the improvement without claiming unmeasured model-cost savings.

**Steps:**
1. Run all focused tests for agent finalization, Kanban DB, CLI, tools, dashboard API, decomposer, config, and gateway watcher.
2. Run `git diff --check`, formatter/linter required by repository guidance, and compile the changed modules.
3. Run a synthetic benchmark comparing current/full versus optimized/compact contexts:
   - context chars and 4-char token estimate;
   - compact-size and full-recovery contracts;
   - byte-identical regeneration of the JSON artifact.
   Retry, review-deduplication, and granularity-policy behavior remain executable test contracts rather than synthetic cost claims.
4. Run one independent code/security review on the frozen diff; remediate findings once, then rerun focused tests.
5. Persist the deterministic benchmark JSON under `docs/benchmarks/`; keep exact test and review commands in the pull request receipt.
6. Do not activate the patch in the live gateway while unrelated workers are running. Install/restart only after a clean drain and explicit runtime verification.
