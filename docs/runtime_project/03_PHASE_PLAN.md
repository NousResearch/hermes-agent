# Phase Plan — HTR

**Baseline:** Architecture Baseline v1.0
**Date:** 2026-07-18
**Updated:** 2026-08-24 (Task 30 multi-project registry + isolation local v1 implemented; Task 28 not started and not approved)

---

## Phase 0: Baseline & Reconnaissance

**Status:** Complete (documentation + recon)

Scope:

- Project control files under `docs/runtime_project/`
- Repository architecture reconnaissance
- Existing component integration map
- ADR decision records
- Test entry confirmation
- First `08_CONTEXT_SUMMARY.md`

---

## Phase 1: Manual Trusted Run-Record Chain (actual)

**Status:** Closed — Task 17 (`939e8b606`) + Task 17.1 (`8fea4daa0`)
Phase 1 implementation and post-review hardening are complete. No further Phase 1 lifecycle work.

Phase 1 as delivered is the **manual 11-record run-level workflow** ending at `run_final_closure_record` (JSON SoT + audit events). It is **not** the earlier aspirational “Trusted Task Loop” list below.

**Frozen deliverables (summary):**

- Run/task/attempt IDs, workspace paths, status machines, events
- Manual run-record chain through final closure
- Idempotent SoT guards (event-present / JSON-missing fails closed)
- No Runtime / delegate_task / scheduler / queue / database / HEAL automation in Phase 1

**Earlier aspirational Phase 1 items (deferred, not unlocked by Phase 1 freeze):**

- Minimal `delegate_task` envelope integration
- HEAL creates new attempt and re-verifies
- Full automated trusted task loop E2E

**Explicitly out of scope for Phase 1:** Dashboard, external DB/queue, full rollback, business idempotency, multi-layer delegation, autonomous Task Graph edits, global post-closure hard lock.

---

## Phase 2: Runtime Integration Boundary

**Status:** **Implementation in progress** (Task 19 observe ✅; Task 21 action plan ✅; Task 22 immutable seal ✅; Task 23 write barrier ✅); architecture checkpoint Task 20 (Policy C) — see `09_PHASE2_RUNTIME_BOUNDARY.md`

Phase 2 defines controlled runtime integration and safe-automation progression.

**Policy C (Task 20 — architecture; Task 22 enforcement ✅):**

1. **Immutable finalization:** valid `run_final_closure_record` → original run sealed against all normal committed HTR mutation APIs (Task 22 ✅).
2. **Recovery/Successor Run:** remediation via linked successor run; original never reopened/unlocked/edited (**Task 27 Path-R1 creation-only v1 checkpoint approved and complete** — creates one approved linked Successor Run only; successor creation does not authorize successor execution).

**Write-path gate:** Task 22 seal ✅; Task 23 write barrier ✅; Task 24 approval control ✅; Task 25 human-gated invoke pilot ✅ (`c6a9e305`). **Task 26A** read-only reconciliation inspection ✅ (closed). **Task 26B** durable reconciliation case control records ✅ (checkpoint approved and complete). **Task 26C** approved marker disposition (**Path-A v1 checkpoint approved and complete**). Retry, repair, invoke, Recovery Run creation, and outcome rewrite **remain prohibited** outside Path A marker disposition.

**Accepted progression:** observe (✅) → action plan (✅ Task 21) → immutable seal (✅ Task 22) → write barrier (✅ Task 23) → approval (✅ Task 24) → human-gated invoke (✅ Task 25 `c6a9e305`) → read-only reconciliation inspection (**Task 26A — closed**) → durable reconciliation cases (**Task 26B — checkpoint approved and complete**) → marker disposition protocol (**Task 26C — Path-A v1 checkpoint approved and complete**) → concurrent observation stabilization (**Task 26B.1 — closed**) → Recovery/Successor creation-only protocol (**Task 27 — Path-R1 v1 checkpoint approved and complete**) → bounded repair (**Task 28 — not started and not approved**) → …

**Historical note:** Phase 1 (Task 17.1) documented chain-terminal closure without global API hard lock; Policy C is forward-looking Phase 2 enforcement.

---

## Phase 3: Domain Reliability (deferred)

Formerly labeled “Phase 2” in baseline notes. Deferred until after Phase 2 runtime boundary acceptance:

- Business idempotency
- Business resource locks (e.g. `ebay_item_id`)
- Full Side-effect Ledger
- Rollback / Compensation
- Simulation / Replay
- Partial success
- Complex dependency graphs

---

## Phase 4: Operations (deferred)

- Dashboard
- Search and audit queries
- HEAL success metrics
- Runtime metrics
- Review queue
- Cost and duration statistics

---

## Current Task Queue Pointer

See `05_TASK_QUEUE.md`. Phase 2 planning document: `09_PHASE2_RUNTIME_BOUNDARY.md`.
