# Phase 1 Control-Plane Doctrine (Low-Risk Codification)

Date: 2026-03-19  
Status: active (subordinate doctrine module)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Scope: `/home/yeqiuqiu/clawd-architect` docs/templates only (no runtime mutation in this phase)

## Canonical source set (5 PDFs)
1. `memory/inbound_zips/batch_4_2026-03-09/extracted_txt/Anti-context-failure operating model for a local-first multi-agent assistant.txt`
2. `memory/inbound_zips/batch_4_2026-03-09/extracted_txt/Production-Grade Scheduler Governance Model for a Deterministic Web-Capture Runtime.txt`
3. `memory/inbound_zips/batch_1_2026-03-09/extracted_txt/Native Swarm Runtime for OpenClaw_ Production-Ready Local-First Design.txt`
4. `memory/inbound_pdfs/hl_terminal_research_pack_2026-03-03/LLM Strategy for HL Terminal and OpenClaw.txt`
5. `memory/inbound_zips/batch_4_2026-03-09/extracted_txt/Durable Freshness and Coherence Architecture for a Local-First Agent Control Plane.txt`

---

## 1) Operating doctrine (authority + lane discipline)

### 1.1 Authority model (non-negotiable)
- Chat transcript is **execution cache**, not authority.
- Durable ledger/artifacts are authority.
- Every meaningful action must bind to `work_item_id` and produce immutable evidence metadata.

### 1.2 Work Item minimum contract
Each active work item must keep (durably):
- `work_item_id`, `objective`, `definition_of_done`
- `current_status`, ordered `next_actions`
- `dependencies`, `risks_assumptions`
- `evidence_index` (evidence object refs + hashes)
- `decision_log` (decision, rationale, authorizer, timestamp)
- latest `handover_packet_ref`

### 1.3 Lane boundary rules
- Main lane = control plane + operator UX.
- Delegated branch lane = heavy tool execution and drafting.
- Main lane hard limit for raw tool paste: **max 20 lines or 2 KB**.
- Branch must exit with a **Branch Result Capsule** containing:
  - `claims[]` (evidence-backed or explicitly speculative)
  - `final_recommendation` (or explicit no-conclusion)
  - `evidence_refs[]`
  - `open_questions[]`, `risks[]`
  - `work_item_state_delta`

### 1.4 Reset rules
- Soft reset: prune/summarize branch context; same epoch.
- Hard reset: new `epoch_id`, mandatory reset snapshot.
- No post-reset execution until reset snapshot is shown and active work items are rehydrated.

---

## 2) Cron/watchdog redesign doctrine

### 2.1 Scheduler governance invariants
- Single scheduler writer semantics with leader lease + fencing epoch.
- Scheduler run modes must be explicit: `ACTIVE | SAFE_READONLY | DRAINING | PAUSED`.
- Dispatch invariants:
  - never dispatch while `cooldown_until > now`
  - never dispatch login-required work without satisfied login contract
  - always honor `Retry-After` when present
  - every dispatch carries stable idempotency key/run id
  - retries must use exponential backoff + jitter

### 2.2 Minimal watchdog set (keep few and symptom-based)
1. **Context Pressure Watchdog**
2. **Continuity Watchdog**
3. **Evidence Hygiene Watchdog**
4. **Divergence Watchdog**
5. **Scheduler Governance Watchdog** (lease/fencing, cooldown/backoff compliance, stale inputs)

### 2.3 Escalation thresholds (Phase 1 defaults)
#### Context pressure
- Warn: `context_fill_ratio >= 0.60` or predicted exhaustion `< 10m`
- Gate: `context_fill_ratio >= 0.80` → block new branches, force capsules
- Force-reset: `context_fill_ratio >= 0.90` → hard reset after handover refresh

#### Handover continuity
- Warn: any in-progress work item with handover age `> 15m`
- Gate: `> 30m` → tool calls only with immediate evidence + capsule update
- Force-pause: `> 60m` → continuity recovery mode before continuing

#### Scheduler integrity
- BLOCK posture: hash-chain mismatch, epoch regression, repeated CAS conflicts, split-brain suspicion
- Immediate action: switch to `SAFE_READONLY`, restore known-good snapshot, replay decision journal

---

## 3) Subagent orchestration doctrine (role-separated)

### 3.1 Role boundaries
- `PLANNER`: queue/spec planning only (no production mutation)
- `EXECUTOR`: bounded mutation under lock + checkpoint discipline
- `VALIDATOR`: independent verification + verdict; cannot implement fixes directly
- `RESEARCHER`: read-heavy analysis; no queue mutation by default
- `SRE/WATCHDOG`: recovery, reclamation, incident operations
- `LIBRARIAN`: curation/index/handover packaging

### 3.2 Execution invariants
- No branch creation without scoped objective + expected deliverable.
- No branch continuation past threshold breach without capsule output.
- Executor cannot self-close as DONE without validator verdict when validator lane is available.
- Every slice must declare: objective, scope, expected artifact, verification, time budget, kill condition.

### 3.3 Phase 1 control-loop defaults
- `review_at = spawned_at + 25m`
- `stale_at = spawned_at + 45m`
- max silent interval on blocker work: `30m`

---

## 4) Model routing doctrine (bounded LLM)

### 4.1 Hard no-LLM boundary
LLM must not be authority for:
- scheduler admission/gating decisions
- idempotency/dedup/retry policy
- side-effect authorization/dispatch
- coherence tuple validity checks

### 4.2 LLM contract gates (must-pass)
- JSON schema-valid rate: **>= 99.5%**
- Evidence pointer resolution: **100%**
- No-evidence abstain correctness (F1): **>= 0.95**
- Evidence-present non-abstain recall: **>= 0.95**

### 4.3 Routing principle
- Route by work-unit type, not operator importance.
- Deterministic/local-first path preferred; escalate model only on defined failure or complexity triggers.
- See canonical matrix: `docs/ops/model_routing_no_llm_matrix_v1.md`.

---

## 5) Continuity hardening doctrine (coherence-first)

### 5.1 Required coherence tuple (all surfaces + action payloads)
- `world_anchor_id`
- `policy_epoch_id`
- `connector_snapshot_id`
- `valid_until`
- `build_generation_id`

### 5.2 No-false-green rule
- If `now > valid_until`, readiness must be non-green.
- If any readiness-critical connector exceeds hard TTL, state is blocked/unknown.

### 5.3 Drift classes (must be surfaced explicitly)
- Anchor drift
- Policy drift
- Connector-freshness drift

### 5.4 Publish semantics
- Build full bundle in temp generation dir.
- Atomic rename/pointer swap to active generation.
- Surfaces must render from one generation; mismatch = `INCOHERENT`.

---

## 6) Phase 1 adoption checklist
- [ ] Use worker slice spec template for new delegated work.
- [ ] Use evidence closeout schema for each completed slice.
- [ ] Use doctrine object contract/schema for philosophy/judgment and life-lane principle promotion.
- [ ] Apply no-LLM/model-routing checklist before invoking LLM lanes.
- [ ] Include coherence tuple in closeouts touching readiness/continuity state.
- [ ] Treat threshold breaches as control-loop events, not narrative notes.
