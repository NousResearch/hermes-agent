# Phase 2 — Runtime Integration Boundary

**Status:** Phase 2 **implementation in progress** (Task 19–27 checkpointed; **Task 27 Path-R1 creation-only v1 checkpoint approved and complete**); **Task 28 not started and not approved**. **Updated:** 2026-07-30
**Date:** 2026-07-19 (planning); **updated** 2026-07-21 (Tasks 19–23)
**Depends on:** Phase 1 closed — Task 17.1 `8fea4daa0`; baseline Git-reproducible at Task 18.5 `04b11bc4d`
**Audience:** Architect + Cursor implementer

This document defines what Phase 2 **may** and **may not** do. Task 20 accepts **Policy C** (immutable finalized-run seal + successor-based recovery). **Task 22 implements the finalized-run seal.** **Task 27 Path-R1 creation-only v1 checkpoint approved and complete** — creates one approved linked Successor Run via `.control/recovery_runs/` (parent `c1f4fdd8`); the original finalized source Run remains permanently immutable; successor creation does not authorize successor execution; all six outcome non-permission booleans remain false; forward_fix remains outside v1; Path R1 remains the only approved eligibility path. **Task 28 bounded repair not started and not approved.**

---

## 0. Phase 1 baseline (closed — historical)

Phase 1 delivered a manual 11-record run-level chain ending at `run_final_closure_record`.
Closed at Task 17.1 `8fea4daa0`.

- JSON run-records are source-of-truth (SoT); event log is audit-only.
- Final closure is terminal for the **manual run-record chain** (Task 17.1 **implemented behavior**).
- `record_run_final_closure` does not mutate `run_manifest` / `task_status` / `attempt_status`.
- Task 17.1 did **not** claim or implement a global hard lock; task/attempt APIs remain callable in code today.
- Idempotent replay with matching audit event but missing JSON SoT **fails closed** (`InvalidTransition`).

**Task 20 does not rewrite Phase 1 history.** Task 22 enforces Policy C at canonical mutation boundaries.

---

## 1. Policy C — Immutable finalization + Recovery/Successor Run

**Accepted at Task 20; finalized-run seal enforced at Task 22.**

### 1.1 Immutable finalization (finalized-run seal)

Once a run has a valid `run_final_closure_record`, the **original run** is permanently sealed against **all normal HTR mutation**, including:

- manual, CLI, Cursor, and runtime automation callers;
- run-level lifecycle mutations; task/attempt mutations; artifact-manifest mutations;
- workspace bootstrap/mutation targeting an already-finalized run;
- direct lifecycle-event append or JSON SoT writes on that run.

**Read-only observation** (`hermes htr observe`) and **read-only planning** (`hermes htr plan`) remain allowed.

Enforcement lives at **canonical shared mutation boundaries** (25 public/run-aware mutation APIs). Applies to runs finalized before or after enforcement ships, when the closure record is valid.

**Implemented at Task 22** (`htr/finalization.py` + guards). Valid closure requires trusted JSON/event correspondence and valid preceding frozen chain. Closure-present-but-untrusted and indeterminate states fail closed with no automatic repair. Exact `record_run_final_closure` replay (matching event ID + semantics) is the sole read-only replay exception — literal zero-filesystem-write (Task 23 verified). Generic filesystem primitives and deliberate manual edits are not claimed protected. Single-machine writer TOCTOU addressed at Task 23 for compliant HTR mutation paths on supported POSIX/Linux local filesystem.

### 1.2 Recovery/Successor Run (not in-place recovery)

If a finalized run later requires remediation, the **original run is not reopened, unlocked, edited, or rewritten**. An explicitly approved future process may create a separate **Recovery/Successor Run** linked to the original.

```
final closure
→ original run becomes immutable (Task 22 ✅)
→ read-only observation remains available
→ a problem may produce a recovery proposal (future)
→ explicit high-risk approval required (future)
→ separate linked Recovery/Successor Run may be created (**Task 27 Path-R1 creation-only v1 ✅** — one approved linked Successor Run; creation does not authorize successor execution)
→ diagnosis, remediation, verification, closure in the successor
→ original run unchanged as historical evidence
```

**Do not describe recovery as:** reopening, unlocking, editing final closure, rolling back closure, or resuming mutation on the original run.

**Implemented at Task 27 (Path-R1 creation-only v1).** Creates one approved linked Successor Run only. The original finalized source Run remains permanently immutable. Successor creation does not authorize successor execution, retry, repair, invoke, artifact copy, external side effect, automatic execution, completion, closure, marker disposition, or outcome rewrite. All six Task 27 outcome non-permission booleans remain false. forward_fix remains outside v1. Path R1 remains the only approved eligibility path.

### 1.3 Prohibited bypasses (normal operations)

No ordinary mutation API may gain: `force=True`, `unlock=True`, env-var override, direct SoT/event editing, deleting/renaming closure records, temporary suppression of closure checks, or lower-level helper bypass.

Exceptional legal/security/data-governance correction of a finalized original run requires a **separate Architect-approved exceptional-data-governance design** — not normal recovery.

---

## 2. Write-path gate

**No general Phase 2 lifecycle invoke path is enabled outside the Task 25 pilot API.** Task 22 immutable finalized-run enforcement ✅. Task 23 durable run write barrier ✅. Task 24 authoritative approval control ✅. Task 25 human-gated invoke pilot ✅ (checkpoint pending).

| Work | Status |
|------|--------|
| Read-only observability (Task 19) | ✅ Done |
| Derived action plan (Task 21) | ✅ Done (read-only) |
| Immutable finalized-run seal (Task 22) | ✅ Done |
| Durable run write barrier (Task 23) | ✅ Done |
| Approval persistence (Task 24) | ✅ Done (control plane only; invoke disabled) |
| Human-gated lifecycle invoke (Task 25) | ✅ Implemented — `complete_run_manually` pilot only; ready for checkpoint |
| Bounded repair / unattended automation | ❌ No |
| Recovery/Successor Run creation | ✅ Done (Task 27 Path-R1 creation-only v1 — checkpoint approved and complete) |

Bounded self-healing of finalized-run problems requires the Recovery/Successor Run protocol; **never in-place repair of the original run**.

---

## 3. Integrity, plans, approvals

| Topic | Stance |
|-------|--------|
| **Fail closed** | Required default (Task 19 observe + future invoke) |
| **Silent auto-heal** | Forbidden |
| **Derived action plan** | Task 21 ✅: library/stdout JSON; non-authoritative; not persisted; Hybrid D; eleven-action catalog; digests bind machine-readable state/risk/confidence/prerequisites/idempotency; `proposable` ≠ executable |
| **Observation digest** | Canonical **semantic projection** of snapshot (exclude `observed_at`, presentation-only fields); deterministic JSON |
| **Confidence** | Deterministic classes: `high` / `medium` / `low` / `indeterminate` + reason codes; integrity errors → non-actionable |
| **Persisted approval** | Task 24 ✅: `{runs_root}/.control/approvals/{approval_id}/` immutable issue/revoke/claim/outcome; `{run_root}/approvals.jsonl` inert legacy only |
| **Repair proposal (initial)** | Derived stdout/library JSON only; finalized-run recovery proposal format deferred to Task 27 |

---

## 4. Runtime read/write permissions (current + future)

| Resource | Read | Write (today) | Write (future) |
|----------|------|---------------|----------------|
| Phase 1 JSON SoT | Yes (Task 19) | Manual APIs only | Via Task 25+ invoke after Task 23 |
| Event log | Yes | Lifecycle APIs only | Same |
| Finalized original run | Yes | **Task 22: sealed** | **Task 22: sealed** |
| Recovery/Successor Run | N/A | N/A | Task 27 ✅ (Path-R1 creation-only v1 — one linked Successor Run; no successor execution authority) |

Runtime must not append events or write SoT directly. Writes only through allowlisted canonical lifecycle APIs after Task 22 seal + Task 23 write barrier + Task 24 approval + Task 25 invoke gates pass.

---

## 5. Execution lock, verification, ambiguous outcomes

**Task 23 durable run write barrier (implemented):** run-scoped marker at `{runs_root}/.execution_locks/{run_id}.marker`; O_EXCL acquisition; durable initialization; authoritative revalidation after acquisition; `run_write_started` before first possible Run write; ownership-checked marker removal + directory fsync on success. Read-only preliminary classification may only produce terminal read-only outcomes or route toward write intent — preflight never authorizes a write. Literal zero-filesystem-write paths: exact final-closure replay; preliminary finalized rejection; preliminary untrusted/suspicious closure rejection — no bootstrap, `.execution_locks`, markers, events, or mtime changes. Existing marker always `occupied_unknown`; no automatic stale cleanup, takeover, force, unlock, skip, env bypass, or public release API. Same-thread/same-Run nested calls reuse outer marker; other threads/processes not reentrant; cross-key nesting rejected. Failure before `run_write_started`: no Run write claimed; owned marker cleaned when possible. Failure after `run_write_started`: marker preserved; `mutation_may_have_committed = true`; `safe_to_retry = false`. First final closure: closure JSON → private final-closure event append under active write context. Guarantees compliant single-machine HTR writer coordination on documented POSIX/Linux local-filesystem contract. Does **not** claim transactionality, atomic multi-file commit, rollback, ambiguous-outcome reconciliation, safe automatic marker recovery, distributed locking, or protection against deliberate same-user out-of-band tampering. Deferred: Task 26 ambiguous-outcome reconciliation and marker-residue handling; Task 27 Recovery/Successor Run protocol.

**Task 24 authoritative approval control (checkpointed):** project-scoped SoT at `{runs_root}/.control/approvals/{approval_id}/` with immutable O_EXCL records (`issue.json`, optional `revoke.json`, singleton `claim.json`, singleton `outcome.json`); separate `htr.approval.digest.v1`; mandatory expiry (max 24h); explicit `event_id` for event-appending APIs; read validation advisory only; dedicated `_approval_control_barrier` reusing Task 23 marker without lifecycle seal bypass; internal `_approval_use_session` for Task 25 continuous marker; no lifecycle invoke; no writes to run-tree `approvals.jsonl`; no Recovery/retry/repair/marker reconciliation.

**First human-gated invoke (Task 25 — checkpointed `c6a9e305`):** pre-observe, plan + approval validation, lock (Task 23 ✅), claim inside one continuous approval-use session, **single** allowlisted API (`complete_run_manually`), **mandatory post-observe verification**, outcome v2 (`consumed` | `ambiguous`), fail-stop (no blind retry). Verification cannot be deferred. No retry, repair, marker recovery, or Recovery/Successor Runs.

**Task 26A read-only reconciliation inspection (checkpoint approved — complete):** `inspect_run_completion_reconciliation` for the Task 25 `complete_run_manually` pilot only. Inspects approval/control evidence, read-only marker metadata, and lifecycle JSON/event/manifest correspondence; returns derived `RunCompletionReconciliationInspection` with independent axes and `overall_classification`. Always `safe_to_retry=false` and `marker_disposition_allowed=false`. Literal read-only — no marker bootstrap/acquire/disposition, no invoke/retry/repair. Semantic digest `htr.reconciliation.inspection.digest.v1`.

**Task 26B durable reconciliation cases (checkpoint approved and complete):** Control store at `{runs_root}/.control/reconciliation/{case_id}/` with immutable O_EXCL records (`open.json`, `observation.json`, `decision.json`). Public APIs: `generate_reconciliation_case_id`, `open_reconciliation_case`, `record_reconciliation_observation`, `record_reconciliation_decision`, `load_reconciliation_case`. Observation persists proven Task 26A inspection projection; decision-time revalidation with drift detection; policy-derived decision classes. **Decisions grant reconciliation posture only**; all six non-permission booleans remain **`false`**. Does **not** acquire execution marker, call `begin_run_write`, invoke, repair, create Recovery Runs, or rewrite outcomes.

**Task 26C approved marker disposition (Path A — checkpoint approved and complete):** Control store at `{runs_root}/.control/marker_dispositions/{disposition_id}/` with immutable records (`request.json`, `issue.json`, `revoke.json`, `claim.json`, `attempt.json`, `outcome.json`). Public APIs: `create_marker_disposition_request`, `issue_marker_disposition_approval`, `revoke_marker_disposition_approval`, `claim_marker_disposition_approval`, `execute_approved_marker_disposition`, `load_marker_disposition_bundle`, `reconcile_marker_disposition_outcome`. **Path A only** — requires Task 26B decision `case_closed_deferred_to_protocol` with `marker_disposition_review`. Coordination via `fcntl.flock(LOCK_EX)` on pinned `.execution_locks` directory fd; 15-minute max approval lifetime; ten outcome classes; all permission booleans remain **`false`**. Marker removal **only** via approved execution under coordination flock. Does **not** mutate Task 26B reconciliation records. Retry, repair, invoke, and outcome rewrite **remain prohibited** outside Path A.

**Task 26B.1 concurrent observation stabilization (checkpoint approved and complete):** Hardens `record_reconciliation_observation` concurrent identical-intent paths via flock-guarded exact replay; parametrized subprocess evidence (2/4/8 workers).

**Task 27 approved Successor Run creation (Path-R1 creation-only v1 — checkpoint approved and complete):** Control store at `{runs_root}/.control/recovery_runs/{recovery_request_id}/` with immutable records; `{successor}/recovery_origin.json` linkage. **Path R1 only** — creates **one approved linked Successor Run**; original finalized source Run **permanently immutable**; successor creation **does not authorize successor execution**; all six outcome non-permission booleans remain **`false`**; **`forward_fix` outside v1**; attempt-before-creation; exclusive successor reservation/bootstrap without Task 23 marker. Does **not** grant retry, repair, invoke, artifact copy, external side effect, automatic execution, completion, closure, marker disposition, or outcome rewrite. **Task 28 not started and not approved.**

**Task 30 multi-project registry + isolation (local v1):** Registry SoT at `{HERMES_HOME}/.htr/project_registry/projects/{project_id}/record.json` (schema `htr.project_registry.record.v1`, `schema_version` 1). Identity is `prj_YYYYMMDD_hex` + canonical absolute runs-root; `project_id`, `runs_root`, and `created_at` are immutable; display name and cwd are not identity. Exclusive create + flock covering check+write; list/get/lookup take the same lock; idempotent re-register of the same id+path; identity/path/overlap conflicts fail closed (path semantics, not string prefix). Archived ≠ delete. HTR project ≠ Hermes profile. Unregistered `{HERMES_HOME}/runs` single-project workflow is unchanged. Does **not** implement Task 31 learning, Task 28 repair, unattended invoke, runs-root relocation, or project deletion. Local v1 is **not** official `upstream/main` completion and **not** an Architect checkpoint. `TASK29_UPSTREAM_COMPLETION=NOT_COMPLETE`.

Ambiguous outcomes include: not started; completed and verified; failed before mutation; may-have-completed (lost ack); SoT/event disagree; post-write verification failed; escalation required.

**No general true rollback** in baseline. Successor-based recovery is forward recovery, not rollback.

---

## 6. Self-healing boundary

No self-healing approved yet. Prerequisites: finding taxonomy, repair allowlist, plan digest, immutable seal, lock, approval, budgets, circuit breaker, post-repair verify, Recovery/Successor protocol for finalized-run problems. **Never** auto-reconstruct missing JSON SoT from events alone.

---

## 7. Artifact / link inspection

**Deferred** (Task 29). When introduced: advisory only; must not auto-advance lifecycle state.

---

## 8. Event and schema policy (near-term)

Tasks 21–26: **no new lifecycle record or event types**. Reuse canonical APIs where semantics match. Derived plans, approvals, execution receipts, recovery lineage — **not** disguised as existing lifecycle events.

Recovery/Successor creation types introduced at **Task 27 Path-R1 v1** via control store + `recovery_origin.json` — future extensions require **separate Architect schema task** each time.

---

## 9. Explicit non-goals (Phase 2)

Unchanged from Task 18 planning: no daemon, scheduler, queue, HTR SQLite lifecycle DB, browser automation, silent heal, unattended pipeline, direct raw JSONL/SoT writes by runtime, automatic delegate_task/HEAL loops, changes to frozen 11-record chain.

Additionally: no in-place finalized-run recovery; no ordinary unlock/bypass; no ad hoc recovery-run format outside approved Task 27 Path-R1 v1.

---

## 10. Accepted safe-automation progression

```
read-only observability          ← Task 19 ✅
→ derived action planning        ← Task 21 ✅
→ immutable finalized-run enforcement ← Task 22 ✅
→ durable run write barrier          ← Task 23 ✅
→ authoritative scoped approval      ← Task 24 ✅
→ human-gated single-API invoke  ← Task 25 ✅ (`c6a9e305`)
→ read-only reconciliation inspection ← Task 26A ✅ (closed)
→ durable reconciliation cases ← Task 26B ✅ (checkpoint approved and complete)
→ marker disposition ← Task 26C ✅ (Path-A v1 checkpoint approved and complete)
→ Recovery/Successor Run protocol ← Task 27 ✅ (Path-R1 creation-only v1 checkpoint approved and complete)
→ bounded retry and repair       ← Task 28 (not started and not approved)
→ selective unattended automation
→ multi-project orchestration    ← Task 30 (local v1; not upstream / not Architect checkpoint)
→ controlled learning            ← Task 31
```

Human approval is selective (high-risk, low-confidence, recovery-run creation, repair, escalation) — not the default operating model.

---

## 11. Task 18 §11 decisions (resolved at Task 20)

| # | Question | Resolution (Policy C) |
|---|----------|------------------------|
| 1 | Hard lock before write/invoke? | **Task 22 immutable seal ✅; Task 23 write barrier ✅; Task 24 approval + Task 25 invoke required before lifecycle invoke.** Recovery is successor-based; original never reopened via normal path. |
| 2 | Read-only MVP or invoke? | **Task 19 read-only MVP complete.** Invoke deferred until Task 24+25 prerequisites. |
| 3 | Artifact inspection in MVP? | **Deferred** (Task 29). |
| 4 | Repair proposal form? | **Derived library/stdout JSON** (Task 21); no lifecycle record/event; persistence deferred; finalized-run recovery proposal = Task 27. |
| 5 | New runtime event type? | **Not for Tasks 21–26.** Future approval/recovery/lineage types need separate schema tasks. |

P2-T0 (boundary acceptance) is **passed** for Task 19. Do not reopen “P2-T0 human checkpoint” as a new implementation task.

---

## 12. Phase 2 task map (Task 20–31)

| Task | Name | Status |
|------|------|--------|
| 19 | Read-only observability | ✅ `57a1ed651` |
| 20 | Immutable finalization + safe automation boundary | ✅ Docs checkpoint (Policy C) |
| 21 | Derived action plan (read-only) | ✅ Task 21 |
| 22 | Immutable finalized-run enforcement | ✅ Task 22 |
| 23 | Durable run write barrier | ✅ Task 23 |
| 24 | Approval control schema + API | ✅ Task 24 |
| 25 | Human-gated single-API invoke pilot | ✅ `c6a9e305` |
| 26A | Read-only reconciliation inspection | ✅ Closed (checkpoint approved) |
| 26B | Durable reconciliation cases | ✅ Checkpoint approved and complete |
| 26C | Marker disposition protocol | ✅ Path-A v1 checkpoint approved and complete |
| 26 | Execution reconciliation (umbrella) | ✅ Complete for approved v1 scope (26A/26B/26C Path A done; Path B deferred) |
| 27 | Recovery/Successor Run creation (Path R1 v1) | ✅ Path-R1 creation-only v1 checkpoint approved and complete |
| 28 | Bounded retry/repair framework | Not started |
| 29 | Advisory artifact/link inspection | ✅ local Phase I (MERGE_G `3b43f6dfc`; upstream not complete) |
| 30 | Multi-project registry + isolation | ✅ local v1 (`task29-local-merge-g`; upstream not complete) |
| 31 | Case history + controlled learning | |

---

## 13. Confirmation

- Task 19 checkpointed at `57a1ed651d622b3af82939d970b9c7f235ea1764`.
- Phase 2 **implementation has started** (read-only foundation + immutable seal).
- Task 20 records Policy C; Task 22 **implements finalized-run seal**; Task 23 **implements durable run write barrier**.
- Recovery/Successor creation **implemented** at Task 27 Path-R1 v1 (creation-only; no successor execution authority).
- Task 24 **checkpointed** — authoritative approval control delivered.
- Task 25 **checkpointed** (`c6a9e305`) — narrow `complete_run_manually` pilot only.
- Task 26A **closed** (checkpoint approved) — read-only reconciliation inspection complete.
- Task 26B **checkpoint approved and complete** — durable reconciliation case control records; decisions grant reconciliation posture only; all six non-permission booleans remain `false`.
- Task 26A, 26B, 26B.1, and 26C **closed**; **Task 26 complete for approved v1 scope** (Path B deferred).
- **Task 27 Path-R1 creation-only v1 checkpoint approved and complete** — creates one approved linked Successor Run; original finalized source Run permanently immutable; successor creation does not authorize successor execution; all six outcome non-permission booleans remain false; forward_fix outside v1; Path R1 only eligibility path.
- **Task 28 not started and not approved.**
- **Task 30 multi-project registry + isolation local v1 implemented** — `{HERMES_HOME}/.htr/project_registry` schema v1; unregistered single-project runs root unchanged; HTR project ≠ Hermes profile; not official `upstream/main`; not an Architect checkpoint.
- **Task 29 remains `TASK29_UPSTREAM_COMPLETION=NOT_COMPLETE`.** Local Phase I artifact inspection does not complete upstream Task 29.
- **Task 31 not started and not approved.**
- No general Phase 2 lifecycle invoke path is enabled outside the Task 25 pilot API.
- Phase 1 frozen chain and Task 17.1 historical semantics preserved in §0.
