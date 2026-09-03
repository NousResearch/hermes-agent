# Task Queue — HTR

**Last updated:** 2026-08-24 (Task 30 multi-project registry + isolation — local v1 implemented; Task 28 still not started and not approved)

---

## Active Task

**None** — Task 30 multi-project registry + isolation local v1 is implemented. **Task 28 not started and not approved.** Task 31 is not started and not approved.

---

## Completed (local)

### Task 30 — Multi-project registry + isolation (v1)

**Status:** ✅ Local v1 implemented on `task29-local-merge-g` (parent `3b43f6dfc`). Not an upstream checkpoint.
**Depends on:** existing `runs_root` / `{HERMES_HOME}/runs` path contract (Task 1+); does **not** depend on Task 28 or Task 31.

**Delivered:**

- `htr/project_registry.py` — register / get / list / metadata update; fail-closed identity and path conflicts; atomic exclusive create + flock; isolated `{HERMES_HOME}/.htr/project_registry/`
- `htr/paths.py` — registry path helpers above per-project runs trees
- `htr/ids.py` — `prj_` project identity
- `hermes htr project {register,show,list,update}` plus optional `--project-id` on `observe` / `plan`
- `tests/htr/test_project_registry.py`

**Identity / isolation:**

- Registry location: `{HERMES_HOME}/.htr/project_registry/` (schema `htr.project_registry.record.v1`, `schema_version` 1). Independent of cwd and of any project's `runs_root`.
- Identity is `project_id` (`prj_YYYYMMDD_hex`) + canonical absolute runs-root (symlink-resolved, `normcase` uniqueness). `display_name` and cwd are never identity.
- Immutable identity fields: `project_id`, `runs_root`, `created_at`. `update` may change only `display_name` and `status` (`active` | `archived`). Archived is not delete: records remain readable and listable with `include_archived`.
- Duplicate `project_id` + same canonical path is idempotent (no rewrite).
- Duplicate `project_id` + different path → identity conflict. Same/overlapping path + different id → path conflict. Overlap uses path semantics (`Path.relative_to`), not string prefix.
- An HTR project is a runs-root namespace **inside one Hermes profile**. It is not a Hermes profile and does not cross `HERMES_HOME` / profile boundaries.
- Unregistered single-project `{HERMES_HOME}/runs` workflow is unchanged. No auto-register, no migration, no deletion, no `runs_root` relocate.

**Explicitly not implemented:** Task 31 case history / learning; Task 28 retry/repair; unattended invoke; relocating a project's runs_root; deleting projects; cross-profile orchestration.

**Upstream / authority:** Local v1 on `task29-local-merge-g` is **not** official `upstream/main` completion and **not** an Architect checkpoint. `TASK29_UPSTREAM_COMPLETION=NOT_COMPLETE` is unchanged.

---

## Completed (checkpointed)

### Task 27 — Recovery/Successor Run creation (Path R1 v1 — creation-only)

**Status:** ✅ Path-R1 creation-only v1 checkpoint approved and complete (parent `c1f4fdd8`)
**Depends on:** Task 26C (Path-A v1 — closed); Task 26B.1 baseline stabilization (closed)

**Delivered (13-file manifest):**

- `htr/recovery_runs.py` — Path R1 only; six-file control store at `.control/recovery_runs/`; `recovery_origin.json` linkage; attempt-before-creation; creates **one approved linked Successor Run**; no Task 23 marker during bootstrap
- `htr/io.py` — `reserve_run_root_exclusive`, `bootstrap_reserved_run_workspace` (no `@run_mutation_boundary`)
- `tests/htr/test_recovery_runs.py` (NEW); `tests/htr/test_io.py` (+reserve/bootstrap tests)

**Verification (`--file-retries 0`):** focused recovery + io **209 passed**; explicit 31-file HTR manifest **2252 passed**; **0 failed**; **0 skipped**; **0 FLAKY**; **0 retries**

**Authority boundaries:**

- The **original finalized source Run remains permanently immutable** — no reopen, unlock, edit, or rewrite
- Successor creation **does not authorize successor execution**, completion, closure, or automatic follow-up
- **No** retry, repair, invoke, artifact copy, external side effect, automatic execution, marker disposition, or outcome rewrite authority
- All **six Task 27 outcome non-permission booleans remain `false`**
- **`forward_fix` remains outside Task 27 v1**
- **Path R1 remains the only approved Task 27 eligibility path**

**Explicitly not implemented:** Task 28; push; successor execution/retry/repair/invoke; forward_fix scope; CLI

---

## Completed (checkpointed)

### Task 26B.1 — Concurrent observation publication stabilization

**Status:** ✅ Checkpoint approved and complete (parent `3147ef90`)
**Depends on:** Task 26C (Path-A v1 — closed)

**Delivered:**

- `htr/reconciliation_cases.py` — flock-guarded exact replay for concurrent identical observation creators; `_read_optional_record_if_published`; `_exact_replay_published_record_under_lock`
- `tests/htr/test_reconciliation_cases.py` — parametrized subprocess scenarios (2/4/8 workers); conflicting-intent case

**Verification (`--file-retries 0`):** `test_reconciliation_cases.py` **180 passed**; explicit 30-file HTR manifest **2055 passed**; **0 failed**; **0 skipped**; **0 FLAKY**; **0 retries**

**Explicitly not implemented:** Task 27/28 changes; docs in this narrow checkpoint

---

## Completed (checkpointed)

### Task 26C — Approved marker disposition (Path A only)

**Status:** ✅ Path-A v1 checkpoint approved and complete (parent `a40ec2d0`)
**Depends on:** Task 26B (checkpoint approved and complete)

**Delivered:**

- `htr/marker_disposition.py` — `create_marker_disposition_request`, `issue_marker_disposition_approval`, `revoke_marker_disposition_approval`, `claim_marker_disposition_approval`, `execute_approved_marker_disposition`, `load_marker_disposition_bundle`, `reconcile_marker_disposition_outcome`
- Control store: `{runs_root}/.control/marker_dispositions/{disposition_id}/` with immutable records (`request.json`, `issue.json`, `revoke.json`, `claim.json`, `attempt.json`, `outcome.json`)
- **Path A only** — marker disposition available only through the approved high-risk protocol; requires Task 26B decision `case_closed_deferred_to_protocol` with `marker_disposition_review`
- Coordination via `fcntl.flock(LOCK_EX)` on pinned `.execution_locks` directory fd; 15-minute max approval lifetime; ten outcome classes; all outcome non-permission booleans remain `false`
- `htr/execution_lock.py` — marker-directory coordination flock + lock-order contract + `disposition_unlink_marker`
- `tests/htr/test_marker_disposition.py` (NEW); `tests/htr/test_execution_lock.py` (coordination tests)

**Verification (isolated candidate, `HERMES_TEST_FILE_RETRIES=0`):** focused marker-disposition + execution-lock **226 passed**; full explicit 30-file HTR manifest **2051 passed**; **0 failed**; **0 skipped**; **0 FLAKY**; **0 retries**

**Authority boundaries:** No generic unlock, stale takeover, force, retry, repair, invoke, Recovery Run creation, or outcome-rewrite authority. Successful marker disposition does **not** grant ordinary Run mutation authority. Finalized-run immutability remains enforced.

**Explicitly not implemented:** Path B (deferred); Task 27 Recovery/Successor; Task 28 bounded repair; retry; repair; invoke; CLI

---

## Completed (checkpointed)

### Task 26B — Durable reconciliation cases

**Status:** ✅ Checkpoint approved and complete (parent `8de2b29b`)
**Depends on:** Task 26A (read-only inspection — closed)

**Delivered:**

- `htr/reconciliation_cases.py` — `generate_reconciliation_case_id`, `open_reconciliation_case`, `record_reconciliation_observation`, `record_reconciliation_decision`, `load_reconciliation_case`
- Control store: `{runs_root}/.control/reconciliation/{case_id}/` with immutable O_EXCL records (`open.json`, `observation.json`, `decision.json`)
- 26B-local inspection projection builder + policy-derived decision classes; **decisions grant reconciliation posture only**; six boolean non-permission invariants (all `false`)
- `tests/htr/test_reconciliation_cases.py` — comprehensive matrix (golden projection, idempotency, policy, concurrency, durability, boundaries)

**Explicitly not implemented:** Task 26C marker disposition (not started; not approved); retry; repair; invoke; Recovery/Successor Runs; outcome rewrite; CLI

---

## Completed (checkpointed)

### Task 26A — Read-only execution reconciliation inspection

**Status:** ✅ Checkpoint approved — read-only reconciliation inspection complete (closed; parent Task 25 `c6a9e305`)
**Depends on:** Task 25 `c6a9e305`

**Delivered:**

- `htr/reconciliation_inspection.py` — `inspect_run_completion_reconciliation` (read-only; `complete_run_manually` pilot only)
- Independent axes: approval control, marker, lifecycle evidence; derived `overall_classification`
- `safe_to_retry=false` and `marker_disposition_allowed=false` always
- Semantic inspection digest `htr.reconciliation.inspection.digest.v1` (presentation `observed_at` excluded)
- `tests/htr/test_reconciliation_inspection.py` — **34 tests**

**Explicitly not implemented:** Task 26B durable reconciliation cases; Task 26C marker disposition; retry; repair; Recovery/Successor Runs; CLI

---

## Completed (checkpointed)

### Task 25 — Human-gated single-API invoke pilot

**Status:** ✅ Checkpointed (`c6a9e30542ac2a37bbe83e1f55b0b0f85e443e9b`; parent Task 24.1 `40f4d016`)
**Depends on:** Task 24.1 `40f4d016`

**Delivered:**

- `htr/invoke_run_completion.py` — `invoke_approved_run_completion` pilot bound to **`complete_run_manually` only**
- One continuous `_approval_use_session`: approval claim, lifecycle invoke, mandatory post-observe verification, outcome v2 (`consumed` | `ambiguous`)
- Outcome v2 binds reason and diagnostic evidence; `safe_to_retry=false`; non-null external `project_repository_checkpoint` fail-closed
- `consumed` requires complete verification; `ambiguous` is fail-stop and non-retryable
- No CLI, generic invoke router, retry, reconciliation, marker recovery, or Recovery/Successor Runs

**Tests (formal Git-only isolated archive, pre-commit, zero retries):** full HTR manifest **1623 passed** (27 files); **0 failed**; **0 skipped**

**Explicitly not implemented:** Task 26 reconciliation; general/unattended/multi-API lifecycle invocation

---

## Completed

### Task 24.1 — Execution-Lock Contention Test Harness Repair

**Status:** ✅ Checkpointed (test-only; parent Task 24 `af4868054b0a61fa0511241d58411d16780daa6b`)
**Production diff:** empty — no production modules changed
**Depends on:** Task 24 `af4868054b0a61fa0511241d58411d16780daa6b`

**Context:** Task 24 production checkpoint (`af4868054`) passed pre-commit formal verification with file-retry masking. Its first strict no-retry post-commit archive run exposed a **pre-existing synchronization defect** in `test_concurrent_bootstrap_succeeds` (timing-based release allowed sequential re-acquisition). Parent-versus-child diagnosis on `af4868054` vs `c89f1161` proved equivalent behavior and **no production regression**; held-marker safety (`test_subprocess_o_excl_race_exactly_one_winner`) passed 50/50 on both commits.

**Delivered:**

- `tests/htr/test_execution_lock.py` — `test_concurrent_bootstrap_succeeds` now uses a `release_gate` so the winning worker retains marker ownership until all challengers report (same pattern as `test_subprocess_o_excl_race_exactly_one_winner`)

**Explicitly not implemented:** lifecycle invoke (Task 25), production changes, Task 25 work

---

### Task 24 — Authoritative Approval Control

**Status:** ✅ Checkpointed (fifth Phase 2 **implementation**; commit `af4868054b0a61fa0511241d58411d16780daa6b`; parent Task 23 `c89f1161968931e329f64acb350b166ec564c174`)
**Tests (formal Git-only isolated archive, pre-commit with file-retry):** full HTR manifest **1487 passed** (26 files: Task 23 **1400** + approval-control **87**); **0 failed**; **0 skipped**
**Post-commit note:** first strict no-retry archive run exposed pre-existing flake in `test_concurrent_bootstrap_succeeds` — repaired in Task 24.1 (test-only); production unchanged
**Depends on:** Task 23 `c89f1161968931e329f64acb350b166ec564c174`

**Delivered:**

- `htr/approval_control.py` — authoritative approval SoT at `{runs_root}/.control/approvals/{approval_id}/` with immutable `issue.json`, optional `revoke.json`, singleton `claim.json`, singleton `outcome.json`
- Read APIs: `get_approval`, `list_approvals`, `validate_approval` (advisory only)
- Write APIs under internal `_approval_control_barrier`: `issue_approval`, `revoke_approval`, `claim_approval`, `record_use_outcome`
- Separate `htr.approval.digest.v1` projection; mandatory `expires_at` (max 24h); explicit `event_id` for event-appending APIs
- `{run_root}/approvals.jsonl` documented as inert legacy bootstrap — never read/written by Task 24
- `htr/execution_lock.py` — shared `_acquire_outer_run_marker` helper only; Task 23 `run_write_barrier` seal semantics unchanged
- `htr/paths.py`, `htr/state.py`, `htr/__init__.py` — control-plane paths and approval error types
- `tests/htr/test_approval_control.py` — approval-control hardening matrix (**87 tests**)

**Explicitly not implemented:** lifecycle invoke (Task 25), ambiguous reconciliation (Task 26), Recovery/Successor Runs (Task 27)

---

### Task 23 — Durable Run Write Barrier

**Status:** ✅ Checkpointed (fourth Phase 2 **implementation**; parent Task 22 `896961d0cfbd5a5cce97fc44ad88bf23ec0619eb`)
**Tests (candidate Git-only workspace):** focused execution-lock **37 passed**; finalization **59 passed**; finalization + Task 19/21 **175 passed**; full tracked `tests/htr/` **1400 passed** (25 files)
**Depends on:** Task 22 `896961d0cfbd5a5cce97fc44ad88bf23ec0619eb`

**Delivered:**

- `htr/execution_lock.py` — run-scoped durable write marker (`{runs_root}/.execution_locks/{run_id}.marker`); O_EXCL acquisition; `@run_mutation_boundary` / `run_write_barrier`; `begin_run_write()`; closure-append guard; ownership-checked release
- `htr/events.py`, `htr/io.py`, `htr/contracts.py`, `htr/artifacts.py` — all 25 public/run-aware mutators wired through the barrier
- `tests/htr/test_execution_lock.py` — runtime write-path matrix (25/25); subprocess crash/race/fork tests; path/release tests; literal zero-write proofs
- `tests/htr/test_finalization.py` — literal project zero-write snapshots for finalized/untrusted/replay rejection

**Contract (Task 23):**

- Run-scoped durable write barrier for all 25 committed public/run-aware mutators on supported POSIX/Linux local filesystem
- Read-only preliminary seal classification may only produce terminal read-only outcomes or route toward write intent; preflight never authorizes a write
- Literal zero-filesystem-write paths: exact final-closure replay; preliminary finalized rejection; preliminary suspicious/untrusted closure rejection — no bootstrap, `.execution_locks`, markers, events, or mtime changes
- Write path: read-only preliminary classification → bootstrap → O_EXCL marker → durability → authoritative revalidation → `run_write_started` before first possible Run write → mutation → ownership-checked marker removal + directory fsync
- Existing marker always `occupied_unknown`; no automatic stale cleanup, takeover, force, unlock, skip, env bypass, or public release API
- Same-thread/same-Run nested calls reuse outer marker; other threads/processes not reentrant; cross-key nesting rejected
- Failure before `run_write_started`: no Run write claimed; owned marker cleaned when possible; cleanup uncertainty fails closed
- Failure after `run_write_started`: marker preserved; `mutation_may_have_committed = true`; `safe_to_retry = false`
- First final closure: closure JSON → private final-closure event append while holding active write context; `_append_run_event_internal` requires active ownership, PID/thread/key/token match, positive nested depth, `run_write_started`, and narrow closure-append context
- Observe and plan remain lock-free, read-only, unchanged
- Does **not** claim: database transactionality; atomic multi-file commit; rollback; ambiguous-outcome reconciliation; safe automatic marker recovery; distributed locking; protection against deliberate same-user out-of-band tampering

**Explicitly not implemented:** Task 24 approval, Task 25 invoke, Task 26 reconciliation/marker-residue handling, Task 27 Recovery/Successor Run, Phase 2 lifecycle invocation.

### Task 22 — Immutable Finalized-Run Enforcement

**Status:** ✅ Checkpointed (third Phase 2 **implementation**; parent Task 21 `798bc1ea98b6af8904c9750102c7bfe3917cdfe0`)
**Tests (candidate Git-only workspace):** focused Task 22 **56 passed**; finalization + Task 19/21 **135 passed**; full tracked `tests/htr/` **1360 passed** (24 files)
**Depends on:** Task 21 `798bc1ea98b6af8904c9750102c7bfe3917cdfe0`

**Delivered:**

- `htr/finalization.py` — focused read-only seal evaluator (`not_finalized`, `finalized_valid`, `closure_present_untrusted`, `indeterminate`); `assert_run_mutation_allowed()`; closure event/record matcher; path containment
- `htr/state.py` — `RunFinalizedError`, `RunSealBlockedError` with stable error codes
- Guards on all 25 public/run-aware mutation entry points (workspace, task/attempt, artifacts, events, eleven Phase 1 run-chain APIs)
- `htr/events.py` — JSON-before-event first closure; private `_append_run_event_internal` (validated first-closure path only); public append rejects `run_final_closure_recorded`; exact `record_run_final_closure` replay is sole zero-write replay exception
- `tests/htr/test_finalization.py` — 25/25 mutation callables individually runtime-tested; untrusted-state matrix; guard-order proofs; import smoke

**Contract (Task 22):**

- Valid final closure permanently seals the original run against all normal committed HTR mutation APIs
- Read-only observation (`hermes htr observe`) and read-only planning (`hermes htr plan`) remain allowed
- Valid closure requires trusted JSON/event correspondence and valid preceding frozen chain
- Closure-present-but-untrusted and indeterminate states fail closed (`RunSealBlockedError`); no automatic repair or event-to-JSON reconstruction
- No force, unlock, env-var, low-level helper, or ordinary reopening bypass
- Exact `record_run_final_closure` replay (matching event ID + semantics) returns existing record with zero writes; all other normal mutation/idempotent replay blocked after finalization
- Generic filesystem primitives (`atomic_write_json`, `append_jsonl`, `ensure_dir`) and deliberate manual edits are **not** claimed protected
- Cross-process TOCTOU between seal check and write remains (Task 23 scope)
- Recovery/Successor Run protocol remains Task 27+; Phase 2 lifecycle invoke remains disabled

**Explicitly not implemented:** Task 23 lock/lease, Task 24 approval, Task 25 invoke, Recovery/Successor Run, self-healing, bypass mechanisms.

### Task 21 — Derived Action Plan Generation (read-only)

**Status:** ✅ Checkpointed (second Phase 2 **implementation**; parent Task 20 `2fa580b5f8b5d26657af2af5641724515e114c76`)
**Tests (candidate Git-only workspace):** focused Task 21 **60 passed**; full tracked `tests/htr/` **1304 passed** (23 files)
**Depends on:** Task 20 `2fa580b5f8b5d26657af2af5641724515e114c76`

**Delivered:**

- `htr/action_plan.py` — Hybrid D derived planner on Task 19 snapshots; semantic observation projection + digests; eleven-action frozen Phase 1 catalog; Policy C planning states; no lifecycle import
- `hermes htr plan <run_id>` — JSON stdout; `--summary` stderr; exit 0/1/2; `--runs-root` supplies canonical `project_dir` binding where required
- `tests/htr/test_action_plan.py`, extended `test_phase2_read_only_boundary.py` — contract, digest, Policy C, path-binding, idempotency, runtime tree-hash proofs

**Contract preserved:**

- Strictly read-only — no invoke, append, SoT write, lock, approval, recovery, subprocess, or network
- `proposable` = semantically complete planning proposal only — **not** executable or currently authorized; event identity may remain unbound until invoke
- Policy C at planning layer: finalized original-run mutation → `blocked_finalized`; explicit remediation intent → `recovery_protocol_required`
- Committed `project_dir` = HTR runs-storage root (same role as observer `base_dir` / `--runs-root`); not project repository or run workspace

**Explicitly not implemented:** Task 22 seal, Task 23 lock, Task 24 approval, Task 25 invoke, Recovery/Successor Run (Task 27+), execution authorization.

### Task 20 — Immutable Finalization and Safe Automation Control Boundary

**Status:** ✅ Architecture checkpoint (docs only; Policy C accepted; parent Task 19 `57a1ed651`)
**Tests:** n/a (documentation only)
**Depends on:** Task 19 `57a1ed651d622b3af82939d970b9c7f235ea1764`

**Delivered (documentation only — no runtime code):**

- Accepted **Policy C:** immutable **finalized-run seal** (future Task 22) + **Recovery/Successor Run** (future Task 27) — no in-place reopen/unlock of original runs
- Resolved Task 18 §11 decisions; corrected stale Phase 2 status in `09_PHASE2_RUNTIME_BOUNDARY.md`
- Write-path gate: **no Phase 2 lifecycle write/invoke before Task 22**
- Accepted task sequence Tasks 21–31; Task 21 next (read-only action plan)

**Explicitly not implemented:** finalized-run enforcement, approval storage, lock/lease, invoke, recovery protocol, bypass/unlock mechanisms.

**Historical compatibility:** Task 17.1 semantics preserved — Phase 1 closure was chain-terminal only; Policy C is future Phase 2 enforcement, not retroactive code change.

### Task 19 — Read-Only Runtime Observability (Phase 2 first implementation)

**Status:** ✅ Checkpointed `57a1ed651d622b3af82939d970b9c7f235ea1764` (first Phase 2 **implementation**; builds on Task 18.5 `04b11bc4d`)
**Tests (candidate Git-only workspace):** focused Task 19 **25 passed**; full tracked `tests/htr/` **1246 passed** (22 files)
**Depends on:** Task 18.5 `04b11bc4df883ee1039c0d10fab1ede7b2fc0e7e`

**Scope:** Strictly read-only single-run observation and integrity reporting — foundation for later reliable, traceable, recoverable, human-gated automation; **not** a manual-only or permanent read-only architecture.

**Delivered:**

- `htr/observe.py` — deterministic machine-readable snapshot, Phase 1 chain visibility, task/attempt summaries, integrity findings
- `hermes htr observe <run_id>` — JSON-only stdout; `--summary` on stderr; exit 0/1/2 fail-closed contract
- Read-only boundary tests (AST + runtime tree-hash proofs)

**Explicitly excluded:** artifact observation, transition replay, repair/auto-heal, run listing, snapshot persistence, hard-lock enforcement, new lifecycle schemas/records/events; **no** edits to `htr/events.py` / `htr/schemas.py`.

**Frozen semantics preserved (Task 17.1 historical):** final closure terminal for Phase 1 manual chain; post-closure activity advisory; **current APIs do not yet enforce Policy C immutable seal** (Task 22).

### Task 18.5 — Reconcile Phase 1 Tracked Baseline

**Status:** ✅ Checkpointed `04b11bc4df883ee1039c0d10fab1ede7b2fc0e7e` (additive only; parent `f7e291ff7`)
**Tests (candidate Git-only workspace):** `tests/htr/` — **1221 passed** (20 files: 8 foundation + 12 Phase 1 workflow)
**Depends on:** Task 18 `f7e291ff7`

**Problem:** Phase 1 workflow semantics were closed and tested locally, but Git reproducibility was broken from the first tracked HTR commit: five foundation modules and eight foundation tests were never checkpointed.

**Changes (byte-for-byte admission; no semantic edits):**

- Production: `htr/paths.py`, `htr/ids.py`, `htr/io.py`, `htr/state.py`, `htr/artifacts.py`
- Tests: `tests/htr/test_paths.py`, `test_ids.py`, `test_io.py`, `test_state.py`, `test_artifacts.py`, `test_contracts.py`, `test_events.py`, `test_schemas.py`

**Explicitly excluded (deferred):** `htr/audit.py`, `tests/htr/test_verification.py`, `tests/htr/test_run_completion.py`, all Task 19 paths.

**Frozen / unchanged:** Phase 1 lifecycle, 11-record chain, `htr/contracts.py`, `htr/events.py`, `htr/schemas.py`, frozen workflow tests. Prior checkpoints not rewritten.

**Note:** Semantic closure predated Git reproducibility; Task 18.5 restores Git-only reproducibility without redesign.

### Task 18 — Phase 2 Runtime Boundary Planning (docs only)

**Status:** ✅ Complete (checkpointed `f7e291ff7`)
**Tests:** n/a (docs only; no code/schema/events changes)
**Depends on:** Phase 1 closed at Task 17.1 `8fea4daa0`

Changes:

- Added `docs/runtime_project/09_PHASE2_RUNTIME_BOUNDARY.md`
- Updated `03_PHASE_PLAN.md` to reflect Phase 1 actual freeze + Phase 2 = runtime boundary planning
- Deferred former “Domain Reliability” content to Phase 3
- Status cleanup: Task 17.1 checkpointed; Phase 1 implementation/post-review hardening closed; Phase 2 planning started; Phase 2 implementation not started
- **No implementation** — no runtime, scheduler, queue, database, delegate_task, browser automation
- **No** new lifecycle record/event types; **no** edits to `htr/events.py` / `htr/schemas.py`

### Task 17.1 — Clarify Phase 1 Terminal Semantics and Guard Idempotent SoT

**Status:** ✅ Accepted (checkpointed `8fea4daa0`)
**Tests:** `uv run --extra dev pytest tests/htr/ -v` — **1273 passed**
**Builds on:** Task 17 checkpoint `939e8b606de09532006887c637684cf8baa49d40`

Changes:

- Docs: final closure is terminal for the Phase 1 **manual run-record chain** only
- Docs: `record_run_final_closure` preserves `run_manifest` / `task_status` / `attempt_status` snapshots
- Docs: Phase 1 does **not** install a global hard lock on later task/attempt APIs; operators treat `run_final_closure_record.json` as the boundary; Phase 2 may add a hard lock later
- `htr/events.py`: idempotent replay of manual run-record APIs requires JSON SoT file; event-present / JSON-missing → `InvalidTransition` (no silent heal)
- Tests: rename overclaiming “terminal” wording; add event-present / JSON-missing regression tests
- No new record/event types; no Phase 1 chain change; no global post-closure hard lock
- Closes Phase 1 implementation / post-review hardening

### Task 17 — Phase 1 Boundary / End-to-End Manual Workflow Freeze

**Status:** ✅ Accepted (checkpointed `939e8b606`)
**Tests:** `uv run --extra dev pytest tests/htr/ -v` — **1271 passed** at Phase 1 final verification

Changes:

- Phase 1 boundary constants: `PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN`, `PHASE1_TERMINAL_RECORD_TYPE`, `PHASE1_TERMINAL_EVENT_TYPE`, `PHASE1_BOUNDARY_STATUS`
- `PHASE1_BOUNDARY_STATUS` is a constant/documentation marker only — **not** a lifecycle event
- End-to-end manual workflow regression test through final closure
- Boundary regression tests: no new record/event type, no boundary record file, AST import guards
- Phase 1 terminal record: `run_final_closure_record`; terminal event: `run_final_closure_recorded`
- 11-record manual chain frozen; JSON records are source-of-truth; event log is audit-only
- Final closure is terminal for the manual run-record chain (not a global task/attempt hard lock)
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO; no automation in Phase 1

### Task 16 — Run Final Closure Record

**Status:** ✅ Accepted (checkpointed `1650b9e73`)
**Tests:** `uv run --extra dev pytest tests/htr/ -v`

Changes:

- `run_final_closure_record` contract + schema validation
- `make_run_final_closure_record`, `run_final_closure_fingerprint`
- `validate_run_final_closure_sources_correspond`, `compute_run_final_closure_status`
- `record_run_final_closure()` — manual final closure after full Phase 1 workflow chain
- Fingerprints must match all 10 prior run-level records
- `closure_items` must correspond to post-verification execution verification items (or be global/manual)
- Writes `run_final_closure_record.json`, appends `run_final_closure_recorded` event
- **Terminal for Phase 1 manual run-record chain** — no new followup loop, no automatic validation/test execution, no prior record mutation by this API
- Does not install a global hard lock on later task/attempt APIs (Phase 1)
- Final closure statuses: `closed_verified`, `closed_rejected`, `closed_needs_more_work`, `closed_no_action`
- No artifact/result/verification_result/docs/test-output inspection

### Task 15 — Manual Post-Verification Execution Verification Recording

**Status:** ✅ Accepted (checkpointed `5011ad44c`)
**Tests:** `uv run --extra dev pytest tests/htr/ -v`

Changes:

- `run_post_verification_execution_verification_record` contract + schema validation
- `make_run_post_verification_execution_verification_record`, `run_post_verification_execution_verification_fingerprint`
- `validate_post_verification_execution_verification_items_correspond`, `compute_post_verification_execution_verification_status`
- `record_post_verification_execution_verification()` — manual verification recording after post-verification execution result exists
- Fingerprints must match on-disk result + verification + post-verification follow-up plan + post-verification execution request + post-verification execution result records
- `verification_items` must correspond to post-verification execution result items (or be global/manual)
- Writes `run_post_verification_execution_verification_record.json`, appends `run_post_verification_execution_verification_recorded` event
- **Recording only** — no automatic verification, no test execution, no prior record mutation, no task/attempt creation
- Empty post-verification execution result normally produces `empty` verification; completed/failed/partial result may produce `verified`/`rejected`/`needs_changes` verification
- No artifact/result/verification_result/docs/test-output inspection

### Task 14 — Manual Post-Verification Execution Result Recording

**Status:** ✅ Accepted (checkpointed)
**Tests:** `uv run --extra dev pytest tests/htr/ -v`

Changes:

- `run_post_verification_execution_result_record` contract + schema validation
- `make_run_post_verification_execution_result_record`, `run_post_verification_execution_result_fingerprint`
- `validate_post_verification_execution_result_items_correspond`
- `record_post_verification_execution_result()` — manual result recording after post-verification execution request exists
- Fingerprints must match on-disk result + verification + post-verification follow-up plan + post-verification execution request records
- `result_items` must correspond to post-verification execution request items (or be global/manual)
- Writes `run_post_verification_execution_result_record.json`, appends `run_post_verification_execution_result_recorded` event
- **Recording only** — no execution, no prior record mutation, no task/attempt creation
- Empty post-verification execution request normally produces `empty` result; requested request may produce `completed`/`failed`/`partial` result
- No artifact/result/verification_result/docs inspection

### Task 13 — Manual Post-Verification Execution Request Planning

**Status:** ✅ Accepted (checkpointed)
**Tests:** `uv run --extra dev pytest tests/htr/ -v`

Changes:

- `run_post_verification_execution_request_record` contract + schema validation
- `make_run_post_verification_execution_request_record`, `run_post_verification_execution_request_fingerprint`
- `derive_post_verification_execution_request_items`, `validate_post_verification_execution_request_items_correspond`
- `request_post_verification_execution()` — planning after post-verification follow-up plan exists
- Fingerprints must match on-disk result + verification + post-verification follow-up plan records
- `request_items` must correspond to post-verification follow-up plan items (or be global/manual)
- Writes `run_post_verification_execution_request_record.json`, appends `run_post_verification_execution_requested` event
- **Planning only** — no execution, no prior record mutation, no task/attempt creation
- Empty post-verification follow-up plan normally produces `empty` request; planned follow-up items may produce `requested` items
- No artifact/result/verification_result inspection

### Task 12 — Verification-Driven Follow-up Planning

**Status:** ✅ Accepted (checkpointed `16d81a65f`)
**Tests:** `uv run --extra dev pytest tests/htr/ -v`

Changes:

- `run_post_verification_followup_plan_record` contract + schema validation
- `make_run_post_verification_followup_plan_record`, `run_post_verification_followup_plan_fingerprint`
- `derive_post_verification_followup_items`, `validate_post_verification_followup_items_correspond`
- `plan_post_verification_followup()` — planning after execution verification record exists
- Fingerprints must match on-disk result + verification records
- `followup_items` must correspond to execution items + item verifications
- Writes `run_post_verification_followup_plan_record.json`, appends `run_post_verification_followup_planned` event
- **Planning only** — no execution, no prior record mutation, no task/attempt creation
- Accepted verification normally produces `empty` plan; rejected/needs_changes produce `planned` items
- No artifact/result/verification_result inspection

### Task 11 / Task 10 / Task 0–9

**Status:** ✅ Accepted (checkpointed)

---

## Next Task (Implementer)

**Task 25 — Human-gated single-API invoke pilot.** Required before any Phase 2 lifecycle invoke path is enabled (Task 22 seal ✅; Task 23 write barrier ✅; Task 24 approval control ✅).

**All lifecycle invoke remains disabled** until Task 25 is implemented.

See `09_PHASE2_RUNTIME_BOUNDARY.md` for Tasks 25–31.

---

## Task 24 — Authoritative Approval Control (checkpointed)

**Status:** ✅ Checkpointed

**Delivered:**

- `htr/approval_control.py` — authoritative approval SoT at `{runs_root}/.control/approvals/{approval_id}/` with immutable `issue.json`, optional `revoke.json`, singleton `claim.json`, singleton `outcome.json`
- Read APIs: `get_approval`, `list_approvals`, `validate_approval` (advisory only)
- Write APIs under internal `_approval_control_barrier`: `issue_approval`, `revoke_approval`, `claim_approval`, `record_use_outcome`
- Separate `htr.approval.digest.v1` projection; mandatory `expires_at` (max 24h); explicit `event_id` for event-appending APIs
- `{run_root}/approvals.jsonl` documented as inert legacy bootstrap — never read/written by Task 24
- `htr/execution_lock.py` — shared `_acquire_outer_run_marker` helper only; Task 23 seal semantics unchanged
- Internal `_approval_use_session` hook for Task 25 continuous marker reuse (invoke not implemented)
- `tests/htr/test_approval_control.py` — approval-control hardening matrix (**87 tests**)

**Tests (formal Git-only isolated archive):** **1487 passed** (26 files); **0 failed**; **0 skipped**

**Explicitly not implemented:** lifecycle invoke (Task 25), ambiguous reconciliation (Task 26), Recovery/Successor Runs (Task 27)

---

## Previous next task (superseded)

**Task 24 — Authoritative Approval Control.** Required before Phase 2 human-gated lifecycle invoke (Task 22 seal ✅; Task 23 write barrier ✅).

**Blocked until Task 24:** human-gated lifecycle invoke (Task 25).

See `09_PHASE2_RUNTIME_BOUNDARY.md` for Tasks 24–31.

---

## Backlog

See `03_PHASE_PLAN.md` and `09_PHASE2_RUNTIME_BOUNDARY.md`.
