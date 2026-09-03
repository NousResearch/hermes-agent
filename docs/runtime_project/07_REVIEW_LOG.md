# Review Log — HTR

---

## 2026-07-30 — Task 27: Approved Successor Run creation (Path R1 v1 — creation-only)

**Implementer:** Cursor (isolated candidate tree `/home/unaliu/task27-candidate/`; parent `c1f4fdd8`)
**Scope:** `htr/recovery_runs.py` (NEW), `htr/io.py`, `htr/paths.py`, `htr/ids.py`, `htr/state.py`, `htr/__init__.py`, `tests/htr/test_recovery_runs.py` (NEW), `tests/htr/test_io.py`, runtime docs (5 files)
**Production Runtime modified:** Yes — Path R1 recovery/successor creation control API + exclusive successor reservation/bootstrap (no `@run_mutation_boundary` on reservation path)
**Source Run mutation:** No — original finalized source Run remains permanently immutable
**Task 26A:** Closed
**Task 26B:** Checkpoint approved and complete
**Task 26B.1:** Checkpoint approved and complete
**Task 26C:** Path-A v1 checkpoint approved and complete
**Task 27 checkpoint:** **Path-R1 creation-only v1 approved and complete**
**Task 28:** Not started and not approved
**Entire Task 26:** Complete for approved v1 scope (Path B deferred)
**Verification:** focused recovery + io **209 passed**; explicit 31-file HTR manifest **2252 passed**; 0 failed; 0 skipped; 0 FLAKY; 0 retries (`--file-retries 0`)

### Contract

- Public APIs: `generate_recovery_request_id`, `create_recovery_run_request`, `issue_recovery_run_approval`, `revoke_recovery_run_approval`, `claim_recovery_run_approval`, `execute_approved_recovery_run_creation`, `load_recovery_run_bundle`, `reconcile_recovery_run_outcome`
- Storage: `{runs_root}/.control/recovery_runs/{recovery_request_id}/` with immutable O_EXCL records; `{successor}/recovery_origin.json` linkage
- **Path R1 only** — creates **one approved linked Successor Run**; **Path R1 remains the only approved eligibility path**
- **`forward_fix` remains outside Task 27 v1**
- Successor creation **does not authorize successor execution**, completion, closure, retry, repair, invoke, artifact copy, external side effect, automatic execution, marker disposition, or outcome rewrite
- All **six Task 27 outcome non-permission booleans remain `false`**
- Attempt-before-creation; exclusive successor directory reservation; no Task 23 marker during bootstrap
- Forbidden production modules (`finalization`, `reconciliation_*`, `marker_disposition`, `approval_control`, `invoke_run_completion`, `events`, `execution_lock`) unchanged vs parent

### Explicitly not implemented

- Task 28 bounded repair
- Successor execution / retry / repair / invoke / closure
- forward_fix scope
- push; CLI

---

## 2026-07-28 — Task 26C: Approved marker disposition (Path A — checkpoint approved and complete)

**Implementer:** Cursor (isolated candidate tree `/home/unaliu/task26c-candidate/`; parent `a40ec2d0`)
**Scope:** `htr/marker_disposition.py` (NEW), `htr/execution_lock.py`, `htr/paths.py`, `htr/ids.py`, `htr/state.py`, `htr/__init__.py`, `tests/htr/test_marker_disposition.py` (NEW), `tests/htr/test_execution_lock.py`, runtime docs (5 files)
**Production Runtime modified:** Yes — Path A marker disposition control API + coordination flock on `.execution_locks` directory fd
**Marker mutation:** Yes — **only** via approved `execute_approved_marker_disposition` under coordination flock (Path A high-risk protocol)
**Task 26A:** Closed
**Task 26B:** Checkpoint approved and complete
**Task 26C checkpoint:** **Path-A v1 approved and complete**
**Task 27/28:** Task 27 Path R1 **implementation candidate complete** (not checkpointed); Task 28 not started
**Entire Task 26:** Complete for approved v1 scope (Path B deferred)
**Verification:** focused marker-disposition + execution-lock **226 passed**; full explicit 30-file HTR manifest **2051 passed**; 0 failed; 0 skipped; 0 FLAKY; 0 retries (`HERMES_TEST_FILE_RETRIES=0`)

### Contract

- Public APIs: `create_marker_disposition_request`, `issue_marker_disposition_approval`, `revoke_marker_disposition_approval`, `claim_marker_disposition_approval`, `execute_approved_marker_disposition`, `load_marker_disposition_bundle`, `reconcile_marker_disposition_outcome`
- Storage: `{runs_root}/.control/marker_dispositions/{disposition_id}/` with immutable O_EXCL records
- **Path A only** — marker disposition available only through the approved high-risk protocol
- 15-minute max approval lifetime; ten outcome classes; all outcome non-permission booleans remain `false`
- Coordination: `fcntl.flock(LOCK_EX)` on pinned `.execution_locks` directory fd with documented lock-order contract
- Task 26B reconciliation records are read-only inputs; not mutated by disposition
- **No** generic unlock, stale takeover, force, retry, repair, invoke, Recovery Run creation, or outcome-rewrite authority
- Successful marker disposition does **not** grant ordinary Run mutation authority; finalized-run immutability remains enforced

### Explicitly not implemented

- Path B marker disposition (deferred and unimplemented)
- Retry, repair, invoke, Recovery/Successor Runs, outcome rewrite, CLI
- Task 27/28

---

## 2026-07-25 — Task 26B: Durable reconciliation cases (checkpoint approved)

**Implementer:** Cursor (isolated candidate tree; parent `8de2b29b`)
**Scope:** `htr/reconciliation_cases.py` (NEW), `htr/paths.py`, `htr/ids.py`, `htr/state.py`, `htr/__init__.py`, `tests/htr/test_reconciliation_cases.py` (NEW), runtime docs
**Production Runtime modified:** Yes — reconciliation case control API only (no lifecycle/marker/approval mutation)
**Marker mutation:** No — 26B does not acquire execution marker or call `begin_run_write`
**Task 26A:** Closed (checkpoint approved)
**Task 26C:** Not started and not approved
**Task 27/28:** Task 27 Path R1 **implementation candidate complete** (not checkpointed); Task 28 not started
**Entire Task 26:** Not complete
**Verification:** `tests/htr/test_reconciliation_cases.py` — **176 passed**; full tracked HTR suite **1862 passed** (29 files; baseline 1686 + 176); 0 failed; 0 FLAKY; 0 retries

### Contract

- Public APIs: `generate_reconciliation_case_id`, `open_reconciliation_case`, `record_reconciliation_observation`, `record_reconciliation_decision`, `load_reconciliation_case`
- Storage: `{runs_root}/.control/reconciliation/{case_id}/` with immutable O_EXCL records
- Observation persists proven Task 26A inspection projection; decision-time revalidation with drift detection
- Policy-derived decision classes; consumed outcome never `completion_verified_by_reconciliation`
- **Decisions grant reconciliation posture only**; all six non-permission booleans remain `false`
- Write metadata: `ReconciliationWriteMetadata`; durability failures raise `ReconciliationDurabilityError`

### Explicitly not implemented

- Marker disposition protocol (Task 26C — not started; not approved)
- Retry, repair, invoke, Recovery/Successor Runs, outcome rewrite, CLI

---

## 2026-07-23 — Task 26B: Durable reconciliation cases (isolated candidate implementation — superseded by checkpoint entry above)

**Implementer:** Cursor (isolated candidate tree `/home/unaliu/task26b-candidate/`; parent `8de2b29b`)
**Scope:** `htr/reconciliation_cases.py` (NEW), `htr/paths.py`, `htr/ids.py`, `htr/state.py`, `htr/__init__.py`, `tests/htr/test_reconciliation_cases.py` (NEW), runtime docs
**Production Runtime modified:** Yes — reconciliation case control API only (no lifecycle/marker/approval mutation)
**Marker mutation:** No — 26B does not acquire execution marker or call `begin_run_write`
**Task 26C:** Not started
**Verification:** `tests/htr/test_reconciliation_cases.py` — **28 passed**; full tracked HTR suite **1714 passed** (29 files; baseline 1686 + 28)

### Contract

- Public APIs: `generate_reconciliation_case_id`, `open_reconciliation_case`, `record_reconciliation_observation`, `record_reconciliation_decision`, `load_reconciliation_case`
- Storage: `{runs_root}/.control/reconciliation/{case_id}/` with immutable O_EXCL records
- Observation persists proven Task 26A inspection projection; decision-time revalidation with drift detection
- Policy-derived decision classes; consumed outcome never `completion_verified_by_reconciliation`
- Write metadata: `ReconciliationWriteMetadata`; durability failures raise `ReconciliationDurabilityError`

### Explicitly not implemented

- Marker disposition protocol (Task 26C)
- Retry, repair, Recovery/Successor Runs, CLI

---

## 2026-07-22 — Task 26A: Read-only execution reconciliation inspection (checkpoint approved)

**Implementer:** Cursor (isolated candidate tree; parent Task 25 `c6a9e305`)
**Scope:** `htr/reconciliation_inspection.py`, `htr/state.py`, `htr/__init__.py`, `tests/htr/test_reconciliation_inspection.py`, runtime docs
**Production Runtime modified:** Yes — read-only inspection API only
**Marker mutation:** No — strictly read-only; no bootstrap/acquire/disposition
**Task 26B/26C:** Task 26B checkpoint approved and complete; Task 26C not started and not approved
**Verification:** `tests/htr/test_reconciliation_inspection.py` — **34 passed**; full tracked HTR suite **1657 passed** (28 files; baseline 1623 + 34)

### Contract

- Public API: `inspect_run_completion_reconciliation(approval_id, *, base_dir=None)`
- Pilot scope: `complete_run_manually` approvals only
- Always `safe_to_retry=false`, `marker_disposition_allowed=false`
- Three independent evidence axes + derived `overall_classification`
- Semantic digest: `htr.reconciliation.inspection.digest.v1` (excludes `observed_at`)

### Explicitly not implemented

- Durable reconciliation cases (Task 26B)
- Marker disposition protocol (Task 26C)
- Retry, repair, invoke, Recovery/Successor Runs, CLI

---

## 2026-07-18 — Task 3: Task Card + Result Contract + Artifact Manifest

**Implementer:** Cursor
**Scope:** `htr/contracts.py`, `htr/artifacts.py`, `htr/schemas.py`, `htr/events.py`, `htr/__init__.py`, tests, docs
**Production Runtime modified:** No
**DECO / HEAL integrated:** No
**Verification pipeline:** No
**delegate_task modified:** No
**SQLite introduced:** No

### Changes

**A. Task Card (`htr/contracts.py`)**

- `make_task_card`, `write_task_card`, `read_task_card`
- Path: `tasks/<task_id>/task_card.json` (atomic write)
- Does not mutate task status or create attempts

**B. Attempt Result**

- `make_attempt_result`, `result_fingerprint`
- `submit_attempt_result` in `htr/events.py`
- Writes `output/result.json`, appends `attempt_result_submitted` event, status → `result_submitted` only
- Result idempotency keyed on `result_fingerprint` in event payload (retry-safe after success)

**C. Artifact Manifest (`htr/artifacts.py`)**

- `read/write_artifact_manifest`, `add_artifact`, `list_artifacts`
- `ArtifactConflict` on path+kind mismatch
- Idempotent duplicate path+kind+metadata/checksum/size
- No lifecycle events on add

**D. Checksum**

- `compute_sha256(path)` — streaming via existing `sha256_file`

**E. Schemas**

- Added `task_card`, `attempt_result`, `artifact_entry`; enhanced `artifact_manifest` validation

### Task 3 self-review (final acceptance pass)

Checklist A–L verified. Fixes applied:
- Removed `result_path` from replay core identity
- Early `result_fingerprint` computation in `submit_attempt_result`
- Added 15+ checklist tests (replay no-op, actor conflict, identity mismatch, exports, schemas)

### Verification

```bash
cd /home/unaliu/.hermes/hermes-agent
source venv/bin/activate
python3 -m pytest tests/htr/ -v
# 143 passed in 1.08s
```

### Known limitations (accepted, non-blocking)

- No verification execution or pass/fail decision
- No HEAL execution or runtime integration
- No event replay; JSON snapshot remains operational read source
- No concurrent writer locks
- Artifact manifest is metadata only
- `htr` not yet in `pyproject.toml`

---

## 2026-07-18 — Task 2.1: State/Event API Idempotency Ordering Fix

**Implementer:** Cursor
**Scope:** `htr/events.py`, `tests/htr/test_events.py`, docs
**Production Runtime modified:** No
**DECO / HEAL:** No
**delegate_task modified:** No
**SQLite:** No

### Problem

- `apply_task_transition` / `apply_attempt_transition` ran `_resolve_idempotent_event` before transition validation, allowing duplicate `event_id` to bypass invalid current-state transitions.
- `_semantic_fingerprint` omitted `previous_status`, risking false idempotent match between e.g. `created->running` and `blocked->running`.

### Fix

**A. Transition ordering (task + attempt)**

1. Read current status snapshot
2. Compute `previous_status`
3. `assert_valid_*_transition(previous_status, new_status)` — **before** idempotency
4. Build candidate event
5. `_resolve_idempotent_event`
6. Append event
7. Atomic write status snapshot

**B. `_semantic_fingerprint`**

Now includes: `event_type`, `run_id`, `task_id`, `attempt_id`, `previous_status`, `new_status`, `actor`, `payload` (excludes `created_at`).

**C. `register_attempt`**

Confirmed order: candidate → idempotent resolve → return existing if match → `AttemptAlreadyRegistered` only for different `event_id` → bootstrap → append → update attempts.

### Verification

```bash
cd /home/unaliu/.hermes/hermes-agent
source venv/bin/activate
python3 -m pytest tests/htr/ -v
# 86 passed in 0.65s
```

### New tests

- Duplicate `event_id` + currently invalid transition → `InvalidTransition` (task + attempt)
- Same `event_id` + different `previous_status` → `EventConflict`
- `register_attempt` same `event_id` retry → idempotent return after first success

---

## 2026-07-18 — Task 2: Task/Attempt State Machine + Event Log API

**Implementer:** Cursor
**Scope:** `htr/state.py`, `htr/events.py`, `htr/schemas.py`, `htr/__init__.py`, `tests/htr/test_state.py`, `tests/htr/test_events.py`, docs
**Production Runtime modified:** No
**DECO / HEAL integrated:** No
**delegate_task modified:** No
**SQLite introduced:** No
**Verification pipeline:** No (transitions only)
**Runtime controller:** No

### Changes

**A. `htr/state.py`**

- TaskStatus / AttemptStatus string constants
- Legal transition tables per Owner spec
- `is_valid_*` / `assert_valid_*` transition helpers
- Terminal status helpers
- Exceptions: `HTRStateError`, `InvalidTransition`, `EventConflict`, `AttemptAlreadyRegistered`, `EventValidationError`

**B. `htr/events.py`**

- Event envelope: `make_event`, `append_task_event`, `read_task_events`, `event_exists`
- Lifecycle APIs: `apply_task_transition`, `register_attempt`, `apply_attempt_transition`
- Order: append event → atomic write status snapshot
- Idempotency: same `event_id` + matching semantic fingerprint → return existing
- Semantic fingerprint excludes `previous_status` and `created_at` (retry-safe)
- `register_attempt`: calls `create_attempt_workspace`, appends event, updates `task_status.attempts`
- Same `attempt_id` + different `event_id` → `AttemptAlreadyRegistered`

**C. `htr/schemas.py`**

- Added `event` schema validation (lightweight, no pydantic)

**D. Tests**

- `test_state.py`: 42 tests (legal/illegal transitions, terminal helpers)
- `test_events.py`: 13 tests (round trip, idempotency, lifecycle, field preservation)

### Verification

```bash
cd /home/unaliu/.hermes/hermes-agent
source venv/bin/activate
python3 -m pytest tests/htr/ -v
# 82 passed in 0.55s
```

### Known limitations

- No event replay / snapshot rebuild
- No concurrent write locking
- `htr` not yet in `pyproject.toml`
- Verification / HEAL are state values only

### Open items (for next tasks)

- Task 2-pre: `pyproject.toml` packaging
- Verification pipeline execution
- Runtime controller hooks
- Tool audit binding
- DECO/HEAL bridges
- C-03 Runtime guard

---

## 2026-07-18 — Task 1.1: HTR Core Foundation Hardening

**Implementer:** Cursor
**Scope:** `htr/io.py`, `tests/htr/test_io.py`, docs only
**Production Runtime modified:** No
**DECO / HEAL integrated:** No
**State machine implemented:** No

### Changes

**A. `atomic_write_json` hardened**

- Unique temp via `tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp")`
- UTF-8 JSON write, flush, `os.fsync` on temp fd
- `os.replace` into target
- Best-effort parent directory fsync
- Temp file cleanup on exception
- Removed fixed `{name}.tmp` pattern

**B. Workspace creation idempotency**

- `_init_json_if_missing()` — write JSON only when file absent
- `_touch_jsonl()` — create empty file only when absent (never truncate)
- `create_run_workspace` — does not overwrite existing `run_manifest.json`
- `create_task_workspace` — does not overwrite existing `task_status.json`
- `create_attempt_workspace` — does not overwrite `attempt_status.json` / `artifact_manifest.json`
- Repeated calls preserve `created_at`, status, attempts, JSONL content

**C. Reserved paths documented in docstrings**

- `task_card.yaml` — not created by bootstrap
- `output/result.json` — not created by bootstrap

### Verification

```bash
cd /home/unaliu/.hermes/hermes-agent
source venv/bin/activate
python3 -m pytest tests/htr/ -v
# 27 passed in 0.29s
```

### Open items (unchanged, for Task 2)

- Attempt registration → state/event API, not create_*
- `htr` packaging in pyproject.toml → Task 2-pre or Task 2
- C-03 enforcement → deferred

---

## 2026-07-18 — Owner correction: external component locations (post Task 0)

**Source:** Owner
**Scope:** Documentation alignment only

### Corrected paths

| Component | Path |
|-----------|------|
| DECO policy | `~/hermes-data/hooks/policy_engine.py` + `policy.yaml` |
| HEAL | `~/hermes-data/hooks/heal_overseer.py`, `heal_diagnose.py`, `heal_evolve.py` |
| Side-effect collector | `~/hermes-data/hooks/side_effect_collector.py` |
| L0 Task Card (SOUL) | `~/.hermes/SOUL.md` |

### Conflict reclassification

- **C-01 / C-02:** False alarms (repo boundary — components external to `hermes-agent`)
- **Real gaps:** C-04, C-07; C-05 greenfield → addressed by Task 1
- **ADR-007 clarified:** HTR lifecycle file-only; existing Hermes SQLite untouched
- **C-03:** Deferred — enforce via `max_spawn_depth=1` or equivalent in later task
- **C-08:** Writer confirmed; bridge deferred

Updated: `02_ARCHITECTURE_DECISIONS.md`, `08_CONTEXT_SUMMARY.md`

---

## 2026-07-18 — Task 1: HTR Core Foundation

**Implementer:** Cursor
**Scope:** New `htr/` + `tests/htr/` only
**Production Runtime modified:** No
**Risk:** Low

### Files added

| Path | Purpose |
|------|---------|
| `htr/ids.py` | 10 prefixed ID generators + validate/parse |
| `htr/paths.py` | `~/.hermes/runs/` path contract + traversal guard |
| `htr/io.py` | Atomic JSON/JSONL IO, sha256, workspace bootstrap |
| `htr/schemas.py` | run/task/attempt/manifest validation |
| `htr/__init__.py` | Public exports |
| `tests/htr/test_*.py` | 22 unit tests (all use `tmp_path`) |

### Verification

```bash
cd /home/unaliu/.hermes/hermes-agent
source venv/bin/activate
python3 -m pytest tests/htr/ -v
# 22 passed in 0.24s
```

### Acceptance checklist (Task 1)

- [x] pytest `tests/htr/` all pass
- [x] Tests use `tmp_path` only
- [x] ID format + uniqueness validated
- [x] Path traversal rejected
- [x] Atomic write round-trip
- [x] Full run/task/attempt workspace tree created

### Risks / notes

- `htr/` not yet listed in `pyproject.toml` `[tool.setuptools.packages.find]` — imports work via repo root on `sys.path` (same as tests). Packaging entry can be added in a later task if needed.
- Default runs root uses `hermes_constants.get_hermes_home() / "runs"` when available; tests override with `base_dir`.
- C-05 (runs workspace) foundation delivered; state machine / events not in scope.

### Conflict status (unchanged policy)

| Conflict | Status after Task 1 |
|----------|---------------------|
| C-03 nested delegation | Policy: `max_spawn_depth=1` for HTR (not enforced in code yet) |
| C-04 self-reported results | Deferred — Phase 1 later task |
| C-07 signed audit | Deferred — Phase 1 later task |

---

## 2026-07-18 — Task 0: Baseline landing + repository reconnaissance

**Implementer:** Cursor
**Scope:** Documentation only (`docs/runtime_project/*`)
**Production Runtime modified:** No
**Tests run:** None (docs-only task; test entry confirmed but not executed)

### Deliverables

| File | Action |
|------|--------|
| `docs/runtime_project/00_PROJECT_BRIEF.md` | Created |
| `docs/runtime_project/01_ARCHITECTURE_BASELINE.md` | Created |
| `docs/runtime_project/02_ARCHITECTURE_DECISIONS.md` | Created |
| `docs/runtime_project/03_PHASE_PLAN.md` | Created |
| `docs/runtime_project/04_CURSOR_RULES.md` | Created |
| `docs/runtime_project/05_TASK_QUEUE.md` | Created |
| `docs/runtime_project/06_ACCEPTANCE_CHECKLIST.md` | Created |
| `docs/runtime_project/07_REVIEW_LOG.md` | Created |
| `docs/runtime_project/08_CONTEXT_SUMMARY.md` | Created |

### Repository reconnaissance report (full)

#### A. Repository topology

| Repository | Path | Version / Notes |
|------------|------|-----------------|
| **hermes-agent (primary)** | `/home/unaliu/.hermes/hermes-agent` | v0.18.2, git install, origin `NousResearch/hermes-agent`, HEAD `d59b79fa` |
| **Hermes runtime home** | `/home/unaliu/.hermes` | Sessions, profiles, config, runtime artifacts |
| **ebay_swarm (domain)** | `/home/unaliu/ebay_swarm` | eBay pipeline + overseer/heal prototypes |
| **Windows mirror** | `C:\Users\Unaliu\.workbuddy\hermes\ebay_swarm_code` | Partial copy of swarm code (not primary truth) |

#### B. Integration point map

##### 1. `delegate_task` entry

| Path | Current responsibility | HTR mapping |
|------|------------------------|-------------|
| `tools/delegate_tool.py` | Spawns child agents; `role=leaf|orchestrator`; blocks tools for children; returns summary array to parent | **Primary hook** for Orchestrator–Worker protocol envelope |
| `tools/async_delegation.py` | Background `delegate_task(background=true)` pool | Phase 1 background dispatch candidate |
| `tools/process_registry.py` | Tracks delegate fan-out lifecycle | Attempt lifecycle reference |
| `gateway/run.py`, `gateway/session_context.py` | Gateway integration for delegation events | Event persistence integration point |

**Behavior notes:**

- Leaf (`role='leaf'`) cannot call `delegate_task` (aligned with ADR-011 for leaf).
- Orchestrator children **can** nested-delegate when `delegation.max_spawn_depth >= 2` and `orchestrator_enabled=true` (**conflicts** with baseline "only Main Agent orchestrates").
- Subagent results are **self-reported summaries**, not verified artifacts.

##### 2. Tool runtime / tool call entry

| Path | Current responsibility | HTR mapping |
|------|------------------------|-------------|
| `agent/tool_executor.py` | Sequential/concurrent tool dispatch | Inject run/task/attempt context here |
| `tools/registry.py` | Tool registration and schema | Audit binding at registration/invoke boundary |
| `run_agent.py` | Agent loop wrapper | Top-level orchestration entry |
| `agent/conversation_loop.py` | Main turn loop; uses `task_id` for VM/file isolation | Distinct from HTR Task/Attempt IDs |
| `agent/tool_dispatch_helpers.py` | Batching, result message shaping | Evidence capture hook |
| `tools/terminal_tool.py`, `tools/file_tools.py` | Side-effecting tools; container `task_id` scoping | Tool evidence sources |

##### 3. Audit log / tool audit

| Path | Current responsibility | HTR mapping |
|------|------------------------|-------------|
| `gateway/session.py` | Session store (SQLite primary + legacy JSON); request dumps | Session-level audit, not attempt-level signed audit |
| `agent/trajectory.py` | Optional ShareGPT trajectory JSONL | Training/debug, not HTR contract audit |
| `agent/verification_evidence.py` | SQLite ledger of command verification evidence | Partial overlap with Evidence Verification (coding-focused) |
| `~/.hermes/sessions/*.json` | Persisted session transcripts | Historical tool call records |
| `~/.hermes/side_effects.json` | Runtime side-effect log (104KB+) | **Side-effect ledger data exists; writer code not found in repos** |

**Gap:** No "Signed Tool Audit" module binding `tool_call_id` to `attempt_id` with checksum/immutability as baseline requires.

##### 4. DECO policy / gate / approval

| Path | Current responsibility | HTR mapping |
|------|------------------------|-------------|
| `tools/approval.py` | Dangerous command detection + human/async approval | **Closest to DECO L0/L3** |
| `agent/tool_guardrails.py` | Per-turn loop detection (warn/hard-stop) | **Closest to DECO L2/L4 risk gate (partial)** |
| `agent/file_safety.py` | File mutation safety | Policy adjunct |
| Profile skills (docs only) | e.g. `code-review-gate`, `execution-pregate` under `~/.hermes/profiles/liuqiong/skills/devops/` | Conceptual DECO docs, not runtime module |

**Critical gap:** No code module named DECO with L0–L5 planes. ADR-010 assumes reuse — **must be resolved by Architect** (implement DECO vs map existing gates vs external package).

##### 5. Hermes HEAL (overseer / diagnose / evolve)

| Path | Current responsibility | HTR mapping |
|------|------------------------|-------------|
| **Not found in hermes-agent core** | — | Baseline HEAL is greenfield in core |
| `ebay_swarm/docs/overseer/overseer_agent.py` | Domain loop: detect → fix → verify → red-light stop | Domain overseer prototype, not generic HEAL |
| `ebay_swarm/docs/overseer/heal_submit_fails.py` | Submit failure healing | Domain heal action |
| `ebay_swarm/docs/overseer/oh_heal_dispatcher.py` | HEAL dispatch helper | Domain-specific |
| `~/.hermes/profiles/*/skills/devops/self-healing-system/` | Skill documentation | Guidance only |
| `agent/curator.py`, `agent/error_classifier.py` | Agent self-improvement / error handling | Different semantics from HEAL cycle |

**Gap:** No generic `overseer → diagnose → evolve → new attempt` pipeline in hermes-agent.

##### 6. Side-effect collector / ledger

| Path | Status |
|------|--------|
| `~/.hermes/side_effects.json` | **Exists** (active log with tool entries e.g. `write_file`) |
| Source writer in hermes-agent / ebay_swarm | **Not found** in recon grep |
| `tui_gateway/server.py` | `_mirror_slash_side_effects` — slash command mirroring only |

**Uncertainty:** Side-effect ledger may come from hook, plugin, profile mod, or manual process — needs Architect/Owner confirmation.

##### 7. Verification pipeline (existing)

| Path | Layer | HTR alignment |
|------|-------|---------------|
| `agent/verification_evidence.py` | Command evidence SQLite | Partial Evidence layer (coding) |
| `agent/verification_stop.py` | Blocks completion without fresh evidence | Analogous to gate, not Task/Attempt verifier |
| `agent/verify_hooks.py` | `pre_verify` hook directives | User/plugin policy, not contract verification |

**Gap:** No Contract Verification or Domain Verification framework as baseline defines.

##### 8. Config entry

| Path | Purpose |
|------|---------|
| `~/.hermes/config.yaml` | Primary runtime config (via `hermes_constants.get_config_path()`) |
| `~/.hermes/.env` | Secrets |
| `cli-config.yaml.example` | Example CLI config |
| `hermes_constants.py` | `HERMES_HOME`, profile paths, config resolution |

Delegation knobs: `delegation.max_spawn_depth`, `delegation.orchestrator_enabled`, `delegation.max_concurrent_children` in config.yaml.

##### 9. Logs / run / workspace conventions (current)

| Path | Purpose |
|------|---------|
| `~/.hermes/sessions/` | Session JSON + request dumps |
| `~/.hermes/profiles/{profile}/workspace/` | Profile workspace CWD |
| `~/.hermes/profiles/{profile}/logs/` | Profile logs |
| `~/.hermes/verification_evidence.db` | Verification evidence SQLite |
| `~/.hermes/state.db` (via gateway session store) | Gateway session state |

**Gap:** Baseline `~/.hermes/runs/{run_id}/` tree **does not exist yet** — greenfield for HTR.

##### 10. Test framework

**hermes-agent:**

| Item | Value |
|------|-------|
| Framework | pytest |
| Config | `pyproject.toml` `[tool.pytest.ini_options]`, `testpaths = ["tests"]` |
| Test file count | ~2106 files under `tests/` |
| Runner | `scripts/run_tests.sh` |
| Manual command | `pytest tests/ -v` (from repo root with venv) |
| Markers | `integration`, `real_concurrent_gate`, `real_agent_prewarm` |

**ebay_swarm:**

| Item | Value |
|------|-------|
| Tests | `test_pipeline_profiles.py`, `test_bee_profiles.py` (per SKILL docs) |
| Runner | `python3` direct / pipeline scripts |

**Task 0 test execution:** Not run (documentation-only). Recommended Phase 1 smoke: `scripts/run_tests.sh tests/agent/test_verification_evidence.py -q`

#### C. Architecture conflicts (material)

| ID | Baseline rule | Current code reality | Severity |
|----|---------------|----------------------|----------|
| C-01 | DECO L0–L5 reusable plane | No DECO module found | **Blocker for ADR-010** |
| C-02 | Generic Hermes HEAL cycle | Only domain overseer in ebay_swarm + skills docs | **Blocker for HEAL integration** |
| C-03 | Leaf-only workers; Main orchestrates | `delegate_task` supports nested orchestrator children | **High — ADR-011** |
| C-04 | Subagent result ≠ completion | Parent trusts summary; prompts say "self-reports not verified facts" but no Runtime gate | **High — ADR-002** |
| C-05 | `~/.hermes/runs/{run_id}/` workspace | Not present; uses profile workspace + sessions | **Expected greenfield** |
| C-06 | Phase 1 no new DB | Hermes already uses SQLite for sessions/evidence | **Integration design needed (ADR-007)** |
| C-07 | Signed tool audit + attempt binding | Session/transcript level only | **High — core Phase 1 work** |
| C-08 | Side-effect ledger | Data file exists, collector source unknown | **Medium — uncertain** |

#### D. Phase 1 likely landing zones (recommendation only — not implementing)

1. **New package:** `htr/` or `agent/htr/` — Task Runtime Controller, state machine, event JSONL, workspace IO
2. **Hooks:** `tools/delegate_tool.py` — emit/consume structured task events
3. **Hooks:** `agent/tool_executor.py` — append signed tool audit records scoped to attempt
4. **Hooks:** `tools/approval.py` + `agent/tool_guardrails.py` — DECO adapter facade (pending Architect decision)
5. **New dir:** `~/.hermes/runs/` — attempt workspace root
6. **Tests:** `tests/htr/` — e2e trusted task loop

#### E. Uncertainties requiring Architect review

1. Where does DECO live today (if anywhere off-repo)?
2. What writes `~/.hermes/side_effects.json`?
3. Should HTR live in upstream hermes-agent fork vs local branch vs separate package?
4. How to coexist with Hermes SQLite stores under ADR-007?
5. Is ebay_swarm overseer in scope for generic HEAL or domain plugin only?

### Stop conditions encountered

| Condition | Triggered? | Notes |
|-----------|------------|-------|
| Need modify production Runtime | No (stopped at docs) | Phase 1 will require Runtime hooks |
| Need change baseline | No | Conflicts documented, not silently changed |
| Cannot find delegate_task / tool runtime | **No** — found | |
| Cannot find DECO / HEAL | **Partial** — real generic modules **not found** | Documented as blockers |
| Architecture conflicts | **Yes** | C-01..C-08 documented |
| Test framework unconfirmed | **No** — pytest confirmed | Not executed in Task 0 |

### Cursor self-assessment

Task 0 completed within constraints. Recon based on read-only inspection of WSL paths. DECO/HEAL generic integration points not found — correctly flagged rather than invented.

**Awaiting:** GPT-5.6-Sol review → Task 1 scope + allowed file list.

---

## Task 4 — Manual Verification Record API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 161 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `verification_result` schema with `passed|failed|heal_required` outcomes and check entries
- `make_verification_result()` with None-only defaults (`summary`, `checks`, `metadata`)
- `verification_fingerprint()` — stable JSON with `sort_keys=True`, `separators=(",", ":")`
- `submit_manual_verification()` — `result_submitted → verification_passed|verification_failed|heal_required`
- `manual_verification_submitted` event + replay-only path for terminal verification states
- Minimal state transition update: `result_submitted → heal_required` for manual shortcut outcome

### Non-goals confirmed

- No verification execution, HEAL execution, Runtime/delegate_task integration
- No task_status updates, no task completed, no new attempts
- No SQLite, scheduler, or event replay from log

### Note

`htr/state.py` updated (one transition) — required for `heal_required` outcome from `result_submitted`; outside nominal allowed-file list but necessary for acceptance tests.

**Awaiting:** Architect acceptance before Task 6.

---

## Task 5 — Manual Task Completion API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 194 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `task_completion_record` schema + `make_task_completion_record()` with None-only defaults
- `task_completion_fingerprint()` — stable canonical JSON
- `complete_task_manually()` — requires `verification_passed`, updates `task_status` only
- `manual_task_completed` event + replay-only path for completed tasks
- `_find_task_event_by_id` scoped to task_id for replay lookup

### Non-goals confirmed

- No task execution, verification runner, HEAL, Runtime/delegate_task
- No attempt_status / run_status updates
- No SQLite, scheduler, event replay from log

**Awaiting:** Architect acceptance before Task 6.

---

## Task 6 — Manual Run Completion API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 228 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_completion_record` schema + `make_run_completion_record()` with None-only defaults
- `run_completion_fingerprint()` — stable canonical JSON
- `complete_run_manually()` — requires every listed task already `completed`, updates `run_manifest` only
- `manual_run_completed` event + replay-only path for completed runs
- Run-level event helpers: `make_run_event`, `append_run_event`, `_find_run_event_by_id`
- Run status constants + `assert_valid_run_transition()` in `state.py`
- Event schema: `task_id` optional (run-level events omit it)

### Non-goals confirmed

- No task execution, verification runner, HEAL execution, Runtime/delegate_task
- No task_status / attempt_status updates
- No automatic task discovery or completion
- No SQLite, scheduler, event replay from log

**Awaiting:** Architect acceptance before Task 7.

---

## Task 7 — Manual Run Review API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 265 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_review_record` schema + `make_run_review_record()` with None-only defaults
- `run_review_fingerprint()` — stable canonical JSON
- `review_run_manually()` — requires completed run + existing `run_completion_record.json`
- `manual_run_reviewed` event + replay-only when review record exists
- Decision constants: `accepted`, `rejected`, `needs_followup`
- Does not update `run_manifest`, `task_status`, or `attempt_status`

### Non-goals confirmed

- No task execution, verification runner, HEAL execution, Runtime/delegate_task
- No artifact/result/verification content inspection
- No automatic task discovery or completion
- No SQLite, scheduler, event replay from log

**Awaiting:** Architect acceptance before Task 8.

---

## Task 8 — Review-Gated Follow-up Planning API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 331 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_followup_plan_record` schema + `make_run_followup_plan_record()`
- `run_followup_plan_fingerprint()` — stable canonical JSON
- `plan_run_followup()` — review-gated planning after completion + review records exist
- `manual_run_followup_planned` event + replay-only when follow-up plan record exists
- Plan status constants: `open`, `cancelled`
- `planner` may be human, assistant, tool, or mixed process
- `followup_items` are planning notes only — not tasks

### Design principle

Automate safe bookkeeping (schema, fingerprint, idempotency, replay, audit events).
Do not automate execution, scheduling, delegation, or lifecycle mutation.

### Non-goals confirmed

- No task/attempt creation from follow-up items
- No Runtime/delegate_task/DECO/HEAL/scheduler/queue/database
- No artifact/result/verification content inspection
- No run_manifest/task_status/attempt_status updates

**Awaiting:** Architect acceptance before Task 9.

---

## Task 9 — Review-Gated Execution Request API (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** full HTR suite (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_execution_request_record` schema + `make_run_execution_request_record()`
- `run_execution_request_fingerprint()` — stable canonical JSON
- `request_run_execution()` — review-gated execution request after completion + review + follow-up plan records exist
- `run_execution_requested` event + replay-only when execution request record exists
- Request status constants: `pending`, `cancelled`
- `execution_items` are approved future actions — not performed actions
- `requester` may be human, assistant, tool, or mixed process

### Design principle

Automate safe bookkeeping (schema, fingerprint, idempotency, replay, audit events).
Execution requests prepare controlled automation; they do not execute work.

### Non-goals confirmed

- No actual execution, Runtime/delegate_task/DECO/HEAL/scheduler/queue/database
- No task/attempt creation from execution items
- No artifact/result/verification content inspection
- No run_manifest/task_status/attempt_status updates
- Task 10 not started

**Awaiting:** Architect acceptance before Task 10.

---

## Task 10 — Controlled One-Shot Execution Adapter (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 488 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_execution_result_record` schema + `make_run_execution_result_record()`
- `run_execution_result_fingerprint()` — stable canonical JSON
- `process_execution_items()` — controlled per-item processing without external side effects
- `execute_run_execution_request()` — one-shot adapter after full review chain + pending execution request
- `run_execution_completed` event + replay-only when result record exists
- Result status constants: `completed`, `partial`, `failed`
- Item status constants: `completed`, `skipped`, `failed`, `unsupported`

### Execution behavior

- Manually triggered only; no scheduler, queue, or daemon
- Loads approved `run_execution_request_record.json` from disk
- `command` dict is data, not executable instructions
- No Runtime/delegate_task/subprocess/HTTP/browser/docs mutation

### Non-goals confirmed

- No task/attempt creation or lifecycle mutation
- No artifact/result/verification content inspection
- No HEAL/DECO/scheduler/queue/database integration
- Task 11 not started

**Awaiting:** Architect acceptance before Task 11.

---

## Task 11 — Manual Verification Gate for Execution Results (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Complete — awaiting Architect acceptance
**Tests:** 559 passed (`python3 -m pytest tests/htr/ -v`)

### Delivered

- `run_execution_verification_record` schema + `make_run_execution_verification_record()`
- `run_execution_verification_fingerprint()` — stable canonical JSON
- `verify_run_execution_result()` — manual verification after execution result exists
- Events: `run_execution_verified`, `run_execution_rejected`, `run_execution_needs_changes`
- Item-level verification decisions with run-level consistency rules
- `item_verifications` must correspond to execution result `item_results`

### Design principle

Verification is human/reviewer-provided, not automatically inferred.
Records reviewer decision as source-of-truth JSON; event log is audit-only.

### Non-goals confirmed

- No automatic verification inference or execution
- No prior execution record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- No task/attempt lifecycle mutation
- Task 12 not started

**Awaiting:** Architect acceptance before Task 12.

---

## Task 12 — Verification-Driven Follow-up Planning (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `16d81a65f`)
**Tests:** full HTR suite (`uv run --extra dev pytest tests/htr/ -v`)

### Delivered

- `run_post_verification_followup_plan_record` schema + factory + fingerprint
- `derive_post_verification_followup_items()` — deterministic derivation from verification items
- `plan_post_verification_followup()` — after full execution verification chain
- `run_post_verification_followup_planned` event + replay-only semantics
- Plan statuses: `planned`, `empty`
- Post-verification follow-up kinds for rejected/needs_changes/not_reviewed/manual actions

### Design principle

Planning based only on execution result + execution verification JSON records.
No artifact/result/verification_result inspection; no automatic execution.

### Non-goals confirmed

- No execution, rerun, repair, task/attempt creation, or lifecycle mutation
- No prior record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- Task 13 not started at Task 12 delivery time (subsequently checkpointed in chain)

---

## Task 13 — Manual Post-Verification Execution Request Planning (2026-07-18)

**Implementer:** Cursor
**Status:** ✅ Accepted — checkpointed
**Tests:** full HTR suite (`uv run --extra dev pytest tests/htr/ -v`) — **768 passed**

### Delivered

- `run_post_verification_execution_request_record` schema + factory + fingerprint
- `derive_post_verification_execution_request_items()` — deterministic derivation from follow-up plan items
- `request_post_verification_execution()` — after full post-verification follow-up plan chain
- `run_post_verification_execution_requested` event + replay-only semantics
- Request statuses: `requested`, `empty`
- Post-verification execution request kinds mapped from follow-up kinds

### Design principle

Request planning based only on execution result + execution verification + post-verification follow-up plan JSON records.
No artifact/result/verification_result inspection; no automatic execution.

### Non-goals confirmed

- No execution, rerun, repair, task/attempt creation, or lifecycle mutation
- No prior record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- Task 14 not started

**Awaiting:** Architect scope assignment for Task 14.

---

## Task 14 — Manual Post-Verification Execution Result Recording (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `ea8dbda40`)
**Tests:** full HTR suite — **913 passed** at checkpoint

### Delivered

- `run_post_verification_execution_result_record` schema + factory + fingerprint
- `validate_post_verification_execution_result_items_correspond()` — correspondence to post-verification execution request items
- `record_post_verification_execution_result()` — after full post-verification execution request chain
- `run_post_verification_execution_result_recorded` event + replay-only semantics
- Result statuses: `completed`, `failed`, `partial`, `empty`
- Result item statuses: `completed`, `failed`, `skipped`, `not_applicable`

### Design principle

Result recording based only on execution result + execution verification + post-verification follow-up plan + post-verification execution request JSON records.
No artifact/result/verification_result/docs inspection; no automatic execution or rerun.

### Non-goals confirmed

- No execution, rerun, repair, task/attempt creation, or lifecycle mutation
- No prior record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- Task 15 not started

**Awaiting:** Architect acceptance before checkpoint / Task 15.

---

## Task 15 — Manual Post-Verification Execution Verification Recording (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `5011ad44c`)
**Tests:** full HTR suite — **1078 passed** at Task 16 pre-checkpoint baseline

### Delivered

- `run_post_verification_execution_verification_record` schema + factory + fingerprint
- `validate_post_verification_execution_verification_items_correspond()` — correspondence to post-verification execution result items
- `compute_post_verification_execution_verification_status()` — deterministic status from explicit item decisions
- `record_post_verification_execution_verification()` — after full post-verification execution result chain
- `run_post_verification_execution_verification_recorded` event + replay-only semantics
- Verification statuses: `verified`, `rejected`, `needs_changes`, `empty`
- Item decisions: `verified`, `rejected`, `needs_changes`, `not_applicable`

### Design principle

Verification recording based only on execution result + execution verification + post-verification follow-up plan + post-verification execution request + post-verification execution result JSON records.
No artifact/result/verification_result/docs/test-output inspection; no automatic verification or test execution.

### Non-goals confirmed

- No automatic verification, test execution, rerun, repair, task/attempt creation, or lifecycle mutation
- No prior record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- Task 16 not started

**Awaiting:** Architect acceptance before checkpoint / Task 16.

---

## Task 16 — Run Final Closure Record (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `1650b9e73`)
**Tests:** full HTR suite — **1250 passed** at Task 17 pre-checkpoint baseline

### Delivered

- `run_final_closure_record` schema + factory + fingerprint
- `validate_run_final_closure_sources_correspond()` — correspondence to post-verification execution verification items
- `compute_run_final_closure_status()` — deterministic status from explicit closure item decisions
- `record_run_final_closure()` — after full workflow chain through post-verification execution verification
- `run_final_closure_recorded` event + replay-only semantics
- Final closure statuses: `closed_verified`, `closed_rejected`, `closed_needs_more_work`, `closed_no_action`
- Item decisions: `accepted`, `rejected`, `needs_more_work`, `no_action`

### Design principle

Final closure based only on the full manual workflow chain through post-verification execution verification JSON records.
No artifact/result/verification_result/docs/test-output inspection; no automatic validation; terminal for Phase 1.

### Non-goals confirmed

- No automatic final closure, verification, test execution, rerun, repair, task/attempt creation, or lifecycle mutation
- No new followup loop
- No prior record mutation
- No Runtime/delegate_task/scheduler/queue/database/HEAL/DECO
- Task 17 not started

**Awaiting:** Architect acceptance before checkpoint / Task 17.

---

## Task 17 — Phase 1 Boundary / End-to-End Manual Workflow Freeze (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `939e8b606`)
**Tests:** full HTR suite (`uv run --extra dev pytest tests/htr/ -v`) — **1271 passed** at checkpoint

### Delivered

- Phase 1 boundary constants in `contracts.py` + exports from `__init__.py`
- `tests/htr/test_phase1_manual_workflow_boundary.py` — E2E manual workflow, boundary regression, AST guards
- Documentation freeze for Phase 1 terminal chain and principles
- **No new lifecycle record type, event type, or lifecycle behavior changes**

### Design principle

Task 17 locks the 11-record Phase 1 manual source-of-truth chain and adds regression protection.
`PHASE1_BOUNDARY_STATUS` is constant-only; no boundary lifecycle event or record is created.

### Non-goals confirmed

- No new record/event type, no `phase1_boundary_record.json`, no boundary lifecycle API
- No automation, Runtime/delegate_task/scheduler/queue/database/HEAL/DECO integration
- No Phase 2 implementation; Task 18 not started

**Phase 2:** not started.

---

## Task 17.1 — Clarify Phase 1 Terminal Semantics and Guard Idempotent SoT (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Accepted (checkpointed `8fea4daa0`)
**Tests:** full HTR suite (`uv run --extra dev pytest tests/htr/ -v`) — **1273 passed**
**Builds on:** Task 17 checkpoint `939e8b606` (Phase 1 final verification **1271 passed**)

### Delivered

- Docs clarify: final closure is terminal for the Phase 1 manual run-record chain; `record_run_final_closure` preserves run/task/attempt snapshots; Phase 1 does not install a global post-closure hard lock; operators treat `run_final_closure_record.json` as the boundary; Phase 2 may add a hard lock later
- Idempotent SoT guard on manual run-record APIs: matching audit event + missing JSON → `InvalidTransition` (no silent heal)
- Boundary test renamed to avoid overclaiming a global lifecycle lock
- Regression tests for event-present / JSON-missing on final closure and `plan_run_followup`

### Design principle

Phase 1 freezes the 11-record chain semantics without expanding automation or adding a global hard lock.

### Non-goals confirmed

- No new lifecycle record/event types; no Phase 1 chain change
- No global post-closure task/attempt hard lock
- No Runtime/delegate_task/scheduler/queue/database/browser/HTTP/subprocess automation

**Phase 1 closed** at this checkpoint. Phase 2 planning follows as Task 18.

---

## Task 18 — Phase 2 Runtime Boundary Planning (2026-07-19)

**Implementer:** Cursor
**Status:** ✅ Complete — planning only (awaiting Architect acceptance — not checkpointed)
**Tests:** n/a (docs only)
**Depends on:** Phase 1 closed at Task 17.1 `8fea4daa0`

### Delivered

- `docs/runtime_project/09_PHASE2_RUNTIME_BOUNDARY.md` — may/may-not rules for runtime integration
- `03_PHASE_PLAN.md` updated: Phase 1 closed; Phase 2 = runtime boundary planning; Domain Reliability deferred to Phase 3
- Task queue + context summary status cleanup (Task 17.1 checkpointed; Phase 2 planning started; implementation not started)

### Planning decisions (provisional)

- Closure remains chain-terminal by default; whole-run hard lock optional/gated (separate go/no-go)
- Runtime MVP read-oriented; no direct SoT writes; no direct event append
- Later writes only via approved lifecycle APIs + human checkpoint (if enabled)
- Integrity fail-closed; no silent heal; manual repair proposals deferred/open
- Artifact/link inspection must not auto-advance lifecycle state
- Human checkpoint required for any state-changing / write / hard-lock adoption

### Non-goals confirmed

- No runtime/daemon/scheduler/queue/database/browser/silent-heal/unattended pipeline implementation
- No automatic delegate_task/HEAL loops; no changes to the frozen 11-record chain
- No new lifecycle record/event types; no `htr/events.py` / `htr/schemas.py` changes in this task
- Phase 2 **planning** started; Phase 2 **implementation** not started

**Awaiting:** Architect acceptance of open decisions in `09_PHASE2_RUNTIME_BOUNDARY.md` §11 before any Phase 2 code.

---

## Task 18.5 — Reconcile Phase 1 Tracked Baseline (2026-07-20)

**Implementer:** Cursor
**Status:** ✅ Baseline reconciliation checkpoint (additive; parent `f7e291ff7`)
**Tests (candidate Git-only workspace before staging):** import smoke (`htr`, `htr.contracts`, `htr.events`, `htr.artifacts`) OK; foundation tests **143 passed**; Phase 1 workflow tests **1078 passed**; full candidate `tests/htr/` **1221 passed** (20 files)
**Builds on:** Task 18 checkpoint `f7e291ff7`

### Problem

A read-only dependency audit confirmed the tracked HTR baseline was **non-reproducible from Git alone** since the first tracked HTR commit: five Phase 1 foundation modules and eight foundation tests existed locally and were exercised by passing tests, but were omitted from every prior checkpoint.

### Delivered (byte-for-byte; no semantic edits)

**Production:** `htr/paths.py`, `htr/ids.py`, `htr/io.py`, `htr/state.py`, `htr/artifacts.py`

**Tests:** `tests/htr/test_paths.py`, `test_ids.py`, `test_io.py`, `test_state.py`, `test_artifacts.py`, `test_contracts.py`, `test_events.py`, `test_schemas.py`

### Explicit exclusions (deferred)

- `htr/audit.py` — adjacent functionality + development HMAC secret; separate review
- `tests/htr/test_verification.py` — ownership/scope not established by audit
- `tests/htr/test_run_completion.py` — provenance/overlap with `test_completion.py` unresolved
- All Task 19 paths (`htr/observe.py`, CLI wiring, Task 19 tests)

### Design principle

Baseline reconciliation only — restore Git-only reproducibility for the existing Phase 1 implementation. No lifecycle redesign, no new record/event types, no schema changes, no edits to frozen workflow tests or the 11-record chain.

### Non-goals confirmed

- No rewrite of prior checkpoints
- No Task 19 staging or commit
- No admission of excluded untracked modules

**Task 19** remains the first Phase 2 **implementation** task and is out of scope for this checkpoint.

---

## Task 19 — Read-Only Runtime Observability (2026-07-20)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (first Phase 2 **implementation**; builds on Task 18.5 `04b11bc4d`)
**Tests (candidate Git-only workspace before staging):** import smoke OK; focused Task 19 **25 passed**; full tracked `tests/htr/` **1246 passed** (22 files)
**Depends on:** Task 18.5 `04b11bc4df883ee1039c0d10fab1ede7b2fc0e7e`

### Delivered

- `htr/observe.py` — read-only snapshot builder: frozen Phase 1 chain visibility, task/attempt summaries, integrity findings (JSON SoT, events, fingerprints, correspondence)
- `hermes_cli/htr.py`, `hermes_cli/subcommands/htr.py`, `hermes_cli/main.py` — `hermes htr observe <run_id>`; JSON-only stdout; `--summary` on stderr; exit 0 / 1 / 2
- `tests/htr/test_observe.py`, `tests/htr/test_phase2_read_only_boundary.py` — runtime tree-hash read-only proofs + AST mutator guards

### Design principle

Strictly read-only observability foundation for later reliable automation — **not** lifecycle automation, **not** a permanent manual-only direction. Reuses tracked fingerprint/correspondence helpers; no parallel state machine.

### Non-goals confirmed

- No lifecycle writes, event append, JSON SoT writes, repair execution, auto-heal, artifact inspection, run listing, snapshot persistence, hard-lock enforcement
- No new lifecycle schemas, record types, or event types; no edits to `htr/events.py` / `htr/schemas.py`
- Final closure terminal only for Phase 1 manual chain; post-closure activity advisory only
- Artifact observation and transition replay deferred

**Next:** Task 20 Policy C architecture; Task 21 derived action plan (read-only).

---

## Task 20 — Immutable Finalization and Safe Automation Control Boundary (2026-07-20)

**Implementer:** Cursor
**Status:** ✅ Architecture checkpoint (docs only; Policy C accepted; parent Task 19 `57a1ed651`)
**Tests:** n/a (documentation only)
**Depends on:** Task 19 `57a1ed651d622b3af82939d970b9c7f235ea1764`

### Policy C accepted

1. **Immutable finalization (finalized-run seal):** future Task 22 — original run with valid `run_final_closure_record` sealed against all normal HTR mutation; enforcement at canonical shared mutation boundaries; read-only observe allowed.
2. **Recovery/Successor Run:** future Task 27+ — separate linked run for remediation; **never** reopen, unlock, edit, or roll back the original run via normal paths.

### Write-path gate

No Phase 2 lifecycle write or invoke before Task 22. Task 21 (read-only action plan) may proceed first.

### Task 18 §11 resolved

Hard lock → immutable seal + successor recovery; read-only MVP complete; artifact deferred; derived repair/recovery proposals (non-SoT); no new lifecycle types for Tasks 21–26.

### Historical compatibility

Task 17.1 accurately documented implemented behavior (chain-terminal, no global hard lock). Task 20 does **not** claim current APIs enforce Policy C or rewrite Phase 1 code semantics.

### Non-goals confirmed

- No runtime implementation in Task 20
- No bypass/unlock/force mechanisms approved
- No Recovery/Successor schema, approval storage, lock, invoke, or self-healing
- No in-place recovery of finalized original runs

**Next implementation:** Task 21 — Derived Action Plan Generation (read-only only).

---

## Task 21 — Derived Action Plan Generation (2026-07-20)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (second Phase 2 **implementation**; parent Task 20 `2fa580b5`)
**Tests (candidate Git-only workspace):** focused Task 21 **60 passed**; full tracked `tests/htr/` **1304 passed** (23 files)
**Depends on:** Task 20 `2fa580b5f8b5d26657af2af5641724515e114c76`

### Delivered

- `htr/action_plan.py` — Hybrid D planner: structural next-slot hint; explicit `--action` + inputs for proposals; eleven frozen Phase 1 API catalog; observation digest `htr.observe.semantic.v1`; plan digest `htr.action_plan.digest.v1`; eight canonical plan states with deterministic precedence
- `hermes_cli/htr.py`, `hermes_cli/subcommands/htr.py` — `hermes htr plan <run_id>`; JSON stdout; `--summary` stderr; exit 0/1/2
- `tests/htr/test_action_plan.py`, `tests/htr/test_phase2_read_only_boundary.py` — digest vectors, Policy C, `project_dir` path contract, idempotency, runtime no-write proofs

### Read-only boundary confirmed

No lifecycle invocation, event append, JSON SoT write, workspace/task/attempt/artifact mutation, lock/lease, approval persistence, recovery/successor creation, subprocess, or network.

### Policy C (planning layer)

Trustworthy finalized original run + normal mutation request → `blocked_finalized`. Explicit `--remediation-intent` + mutation → `recovery_protocol_required`. Integrity failure precedes recovery classification. Recovery/Successor protocol **not implemented**.

### Path semantics (audited)

Committed Phase 1 `project_dir` = HTR **runs-storage root** (identical path role to observer `base_dir` and CLI `--runs-root`). Run workspace = runs-root + run_id. `project_repository_checkpoint` is separate opaque identity. Plans bind directory via semantic `project_dir_binding` (not unnecessary absolute paths in output).

### Proposable clarification

`proposable` = semantically complete under Task 21 planning contract only. Does **not** authorize execution. Omitted `event_id` → invoke-time allocation prerequisite; exact event identity not bound until supplied. Future approval must bind execution-ready identity (Tasks 22–25), not treat Task 21 digest alone as sufficient when idempotency remains unbound.

### Non-goals confirmed

- No Task 22 seal, Task 23 lock, Task 24 approval, Task 25 invoke, Recovery/Successor Run creation
- No new lifecycle schemas/records/events; no edits to `htr/events.py`, `htr/schemas.py`, `htr/observe.py`

**Next implementation:** Task 22 — Immutable finalized-run enforcement.

---

## Task 22 — Immutable Finalized-Run Enforcement (2026-07-20)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (third Phase 2 **implementation**; parent Task 21 `798bc1ea`)
**Tests (candidate Git-only workspace):** focused Task 22 **56 passed**; finalization + Task 19/21 **135 passed**; full tracked `tests/htr/` **1360 passed** (24 files)
**Depends on:** Task 21 `798bc1ea98b6af8904c9750102c7bfe3917cdfe0`

### Delivered

- `htr/finalization.py` — seal states, read-only closure evaluation, `assert_run_mutation_allowed()`, event/record matcher
- `htr/state.py` — `RunFinalizedError` (`RUN_FINALIZED`), `RunSealBlockedError` (`RUN_SEAL_BLOCKED`)
- Guards on 25 public/run-aware mutation APIs: workspace (3), task/attempt (7), artifacts (2), events (2), run-chain (11)
- First valid closure: JSON before event via private `_append_run_event_internal` (single production call site: `record_run_final_closure`)
- Public `append_run_event` rejects `run_final_closure_recorded`; exact closure replay is sole read-only replay exception
- `tests/htr/test_finalization.py` — individual runtime matrix for all mutators; untrusted-state matrix; guard-order proofs

### Contract enforced

- Valid final closure seals original run against all normal committed HTR mutation; observe/plan remain read-only
- Trusted closure = valid JSON + fingerprint + source correspondence + valid frozen chain + matching final-closure event
- Untrusted/indeterminate closure states fail closed; no repair or event-to-JSON reconstruction
- No force/unlock/env-var/bypass; generic `atomic_write_json` / `append_jsonl` / `ensure_dir` / manual edits not claimed protected
- Cross-process TOCTOU remains explicit limitation (Task 23); Recovery/Successor Run remains Task 27+; Phase 2 invoke disabled

### Non-goals confirmed

- No Task 23 lock/lease, Task 24 approval, Task 25 invoke, Recovery/Successor protocol, self-healing
- No new lifecycle schemas/records/events; no edits to `htr/schemas.py`, `htr/observe.py`, `htr/action_plan.py`

**Next implementation:** Task 23 — Execution lock/lease.

---

## Task 23 — Durable Run Write Barrier (2026-07-21)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (fourth Phase 2 **implementation**; parent Task 22 `896961d0`)
**Tests (candidate Git-only workspace):** focused execution-lock **37 passed**; finalization **59 passed**; finalization + Task 19/21 **175 passed**; full tracked `tests/htr/` **1400 passed** (25 files)
**Depends on:** Task 22 `896961d0cfbd5a5cce97fc44ad88bf23ec0619eb`

### Delivered

- `htr/execution_lock.py` — run-scoped durable write marker at `{runs_root}/.execution_locks/{run_id}.marker`; O_EXCL acquisition; `@run_mutation_boundary`; `run_write_barrier`; `begin_run_write()`; closure-append guard; ownership-checked release with directory fsync
- All 25 public/run-aware mutators wired: workspace (3), task/attempt (7), artifacts (2), events (`append_run_event`, `record_run_final_closure` + run-chain APIs)
- `tests/htr/test_execution_lock.py` — runtime write-path 25/25; subprocess O_EXCL race, crash phases, fork isolation, first-closure interleaving; path alias/isolation/symlink/bootstrap/release tests; literal zero-write
- `tests/htr/test_finalization.py` — literal project zero-write for finalized/untrusted/replay rejection

### Contract enforced

- Read-only preliminary classification allowed only for terminal read-only outcomes or routing toward write intent; preflight never authorizes a write
- Literal zero-filesystem-write: exact final-closure replay; preliminary finalized rejection; preliminary suspicious/untrusted closure rejection — no bootstrap, `.execution_locks`, markers, events, or mtime changes
- Write path: preliminary classification → bootstrap → O_EXCL marker → durability → authoritative revalidation → `run_write_started` before first possible Run write → mutation → verified marker cleanup + directory fsync
- Existing marker always `occupied_unknown`; no stale cleanup, takeover, force, unlock, skip, env bypass, or public release API
- Same-thread/same-Run nested reuse; other threads/processes not reentrant; cross-key nesting rejected
- Before `run_write_started`: no Run write claimed; owned marker cleaned when possible; cleanup uncertainty fails closed
- After `run_write_started`: marker preserved; `mutation_may_have_committed = true`; `safe_to_retry = false`
- First closure: closure JSON → private `_append_run_event_internal` under active write context with narrow closure-append guard
- Observe and plan remain lock-free and read-only
- Single-machine HTR writer coordination on documented POSIX/Linux local-filesystem contract only

### Non-goals confirmed

- No database transactionality; no atomic multi-file commit; no rollback; no ambiguous-outcome reconciliation; no safe automatic marker recovery; no distributed locking; no protection against deliberate same-user out-of-band tampering
- No Task 25 invoke, Task 26 reconciliation, Task 27 Recovery/Successor Run, Phase 2 lifecycle invocation
- No superseded designs: marker-before-every-read-only decision; abstract Unix socket Phase A; expiring leases; automatic stale takeover; advisory observer lock fields; generic `enforce_seal` / `control_write_barrier` seal-bypass switch

**Next implementation:** Task 25 — Human-gated single-API invoke pilot.

---

## Task 24.1 — Execution-Lock Contention Test Harness Repair (2026-07-21)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (test-only harness repair)
**Depends on:** Task 24 production checkpoint `af4868054b0a61fa0511241d58411d16780daa6b`

### Context

Task 24 production checkpoint (`af4868054`) first strict no-retry post-commit archive run failed on `test_concurrent_bootstrap_succeeds`. Read-only parent-versus-child diagnosis proved **equivalent behavior** on Task 23 parent (`c89f1161`) and Task 24 child (`af4868054`); held-marker safety property passed on both commits; **no production regression**.

### Delivered

- `tests/htr/test_execution_lock.py` — `test_concurrent_bootstrap_succeeds` / `_subprocess_concurrent_bootstrap_worker` now hold the winning marker via `release_gate` until all worker outcomes are collected (pattern from `test_subprocess_o_excl_race_exactly_one_winner`)

### Non-goals confirmed

- No production module changes; no approval schema/API changes; no lifecycle invoke (Task 25); Task 25 not started

**Next implementation:** Task 25 — Human-gated single-API invoke pilot.

---

## Task 24 — Authoritative Approval Control (2026-07-21)

**Implementer:** Cursor
**Status:** ✅ Checkpointed (fifth Phase 2 **implementation**; commit `af4868054b0a61fa0511241d58411d16780daa6b`)
**Depends on:** Task 23 `c89f1161968931e329f64acb350b166ec564c174`

### Delivered

- `htr/approval_control.py` — authoritative SoT at `{runs_root}/.control/approvals/{approval_id}/` with immutable `issue.json`, optional `revoke.json`, singleton `claim.json`, singleton `outcome.json`
- Read APIs: `get_approval`, `list_approvals`, `validate_approval` (advisory only)
- Write APIs under internal `_approval_control_barrier` (not a lifecycle seal bypass)
- `htr/execution_lock.py` — shared `_acquire_outer_run_marker` helper only; Task 23 `run_write_barrier` seal semantics unchanged
- `htr/paths.py` — control-plane paths; `{run_root}/approvals.jsonl` documented inert legacy bootstrap
- `htr/state.py` — approval control error types
- `tests/htr/test_approval_control.py` — approval-control hardening matrix (**87 tests; 0 skipped**)

### Contract enforced

- One approval SoT; no mutable index; list derives from scanning `issue.json`
- Immutable O_EXCL records with fsync file + directory; exact replay idempotent; conflicting replay fails closed
- Issue/new claim rejected on `FINALIZED_VALID`; revoke/outcome allowed after finalization; untrusted/indeterminate seal fails closed on issue/claim
- Singleton `claim.json`; different `claim_id` rejected after first claim; claimant must equal issue `executor_id`
- Dedicated control barrier reuses Task 23 marker; serializes control + lifecycle writers per Run; no lifecycle seal bypass for Run SoT
- No lifecycle invoke; no event append; no writes to run-tree `approvals.jsonl`

### Non-goals confirmed

- No lifecycle invocation (Task 25); no ambiguous reconciliation (Task 26); no Recovery/Successor Runs (Task 27)
- No self-healing; no marker reconciliation; no distributed locking; no finalized-run mutation bypass

**Tests (formal Git-only isolated archive, pre-commit with file-retry):** full HTR manifest **1487 passed** (26 files); **0 failed**; **0 skipped** — strict no-retry post-commit exposed pre-existing test defect; repaired in Task 24.1

**Next implementation:** Task 25 — Human-gated single-API invoke pilot.

---

## Task 25 — Human-Gated Single-API Invoke Pilot (ready for checkpoint, 2026-07-22)

**Base:** Task 24.1 `40f4d01638f3d2f3c16c9c8ef451ab1c20fc21f0`

### Delivered

- Public API: `invoke_approved_run_completion(approval_id, *, claim_id, base_dir=None)` → pilot bound API **`complete_run_manually` only**
- One continuous `_approval_use_session`: validate → claim → invoke once → post-observe → mandatory verification → outcome v2 (`consumed` | `ambiguous`)
- Private in-session helpers: `_claim_approval_during_session`, `_record_use_outcome_during_session` — not exported; PID/thread/token/run-key/depth guarded
- Outcome v2 binds reason and diagnostic evidence; boolean `safe_to_retry` validated before enforcing Task 25 `false` contract; non-null `project_repository_checkpoint` fail-closed
- `consumed` requires complete verification; `ambiguous` is fail-stop and non-retryable
- No generic lifecycle router, CLI, retry, reconciliation, marker cleanup/recovery, or Recovery Run

### Non-goals confirmed

- Task 26 reconciliation **not implemented**
- General, unattended, and multi-API lifecycle invocation **remain disabled**
- `tests/htr/test_run_completion.py` and deferred files untouched

### Verification

**Tests (formal Git-only isolated archive, pre-commit, zero retries):** full HTR manifest **1623 passed** (27 files); **0 failed**; **0 skipped**

**Next:** Task 26 — ambiguous outcome reconciliation (not started).
