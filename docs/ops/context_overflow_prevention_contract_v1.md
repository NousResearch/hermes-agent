# Context Overflow Prevention Runtime Contract (v1)

Date: 2026-03-30  
Status: active (Program A / Slice A3)  
Owner: Architect  
Scope: OpenClaw continuity runtime overflow prevention and fail-closed handling

---

## 1) Purpose

This contract formalizes OpenClaw context overflow prevention as a deterministic runtime policy.

It converts previously fragmented watcher/checkpoint/compaction behavior into an explicit, testable contract for:
- volumetric overload detection,
- semantic boundary-triggered checkpointing,
- trigger-to-action execution,
- fail-closed mutation posture,
- machine-readable overflow runtime state artifacts.

This contract satisfies queue slice **A3** from:
- `reports/openclaw_system_execution_queue_full_buildout_2026-03-30.md`

---

## 2) Authority and dependencies

## 2.1 Canonical authority for A3
This document is the canonical authority for context overflow prevention runtime behavior.

Machine-readable companions:
- `docs/ops/schemas/context_overflow_status.schema.json`
- `docs/ops/templates/context_overflow_status.template.json`

## 2.2 Required upstream contracts (integrated, not replaced)
- `docs/ops/continuity_integration_contract_v1.md` (A1 canonical continuity contract)
- `docs/ops/schemas/handover_bundle.schema.json` (A2 handover normalization)
- `docs/ops/schemas/checkpoint_bundle.schema.json` (A2 checkpoint normalization)

## 2.3 Scope-out
This contract does not redefine:
- orchestrator API and replay contracts (Program B),
- model routing and qualification doctrine (Program C),
- CSI layer contracts (Program D).

---

## 3) Normative terms

- **MUST**: hard requirement.
- **SHOULD**: expected default unless explicitly justified.
- **MAY**: optional behavior.
- **Volumetric trigger**: threshold based on context usage/session transcript growth.
- **Semantic trigger**: boundary event requiring checkpoint/handover regardless of context percent.
- **Overflow status artifact**: machine-readable runtime state object for trigger evaluation and action outcomes.

---

## 4) Canonical A3 artifact bundle

A3 canonical bundle is exactly:
1. `docs/ops/context_overflow_prevention_contract_v1.md` (this contract)
2. `docs/ops/schemas/context_overflow_status.schema.json`
3. `docs/ops/templates/context_overflow_status.template.json`
4. `reports/execution_A3_context_overflow_prevention_2026-03-30.md`

Runtime state surfaces governed by this contract:
- `state/continuity/latest/context_overflow_status_latest.json` (latest pointer)
- `state/continuity/history/context_overflow_status_history.jsonl` (append history; if enabled)
- `state/continuity/latest/handover_latest.json` (A2 normalized handover)
- `state/continuity/latest/current.json` (continuity current)

---

## 5) Trigger model (volumetric + semantic)

## 5.1 Volumetric thresholds

Thresholds are percentage of context utilization unless noted.

| Band | Condition | Meaning | Required posture |
|---|---|---|---|
| V0 normal | `< 0.80` and no bloat/runtime risk | no immediate overload risk | mutation gate unchanged |
| V1 preventive | `>= 0.80` and `< 0.85` | early pressure; preserve margin | caution allowed; preemptive compaction SHOULD run |
| V2 checkpoint_required | `>= 0.85` and `< 0.90` | overflow risk approaching boundary | checkpoint + handover refresh MUST run |
| V3 emergency | `>= 0.90` OR session bloat OR runtime unhealthy | high/critical overload or unsafe runtime | fail-closed until checkpoint/handover/verify outcomes are valid |
| V4 fail_closed | `>= 0.95` with failed overflow actions OR stale/missing required artifacts | continuity reliability compromised | mutation forbidden |

Mandatory baseline thresholds:
- `warning_threshold_pct = 0.80`
- `checkpoint_threshold_pct = 0.85`
- `emergency_threshold_pct = 0.90`
- `buffer_target_pct = 15` (aligns with continuity integration C2 guidance)
- `session_bloat_threshold_mb = 50` (default local watch guard)

## 5.2 Semantic boundaries

The following boundary classes MUST be supported and MAY independently trigger checkpoint flow even under V0/V1:
- `pre_execution`
- `post_mutation`
- `post_scheduling_change`
- `failure_boundary`
- `handover_boundary`

Semantic trigger doctrine:
- boundary events that can materially affect resumption safety MUST force overflow status evaluation,
- if boundary is high-risk (mutation/scheduling/failure), checkpoint + handover refresh MUST execute,
- missing boundary evidence defaults to caution or forbidden depending on risk tier.

---

## 6) Trigger → action matrix (authoritative)

| Trigger class | Detection condition | Required actions | Mutation gate result |
|---|---|---|---|
| `volumetric_warning` | V1 preventive band | capture ground truth, run compaction, emit status artifact | `caution` |
| `volumetric_checkpoint` | V2 checkpoint_required band | write checkpoint, render handover compat, sync latest artifacts, emit status artifact | `caution` until verify refresh |
| `volumetric_emergency` | V3 emergency by context pct | write checkpoint, render handover, sync latest, require verify-then-resume, emit blocker event | `forbidden` until explicit recovery |
| `session_bloat` | session transcript size >= bloat threshold | write checkpoint with bloat trigger, render handover, compaction SHOULD run | `caution` (or `forbidden` if checkpoint/handover fails) |
| `runtime_unhealthy` | ground-truth anomaly (e.g., gateway unhealthy) | write BLOCKER checkpoint, emit blocker event, require verify-then-resume | `forbidden` |
| `semantic_boundary` | any required boundary event | checkpoint/handover MUST run for mutation/scheduling/failure boundaries | `caution` unless verifier passes |
| `checkpoint_write_failure` | checkpoint action returns non-zero/invalid output | emit fail-closed reason, block mutation, preserve evidence refs | `forbidden` |
| `handover_render_failure` | handover compatibility renderer fails | emit fail-closed reason, block mutation, preserve checkpoint evidence | `forbidden` |

Action primitives (canonical IDs):
- `capture_ground_truth`
- `write_checkpoint`
- `render_handover_compat`
- `sync_latest_artifacts`
- `emit_event_router_signal`
- `run_anchor_preserving_compaction`
- `set_mutation_gate_caution`
- `set_mutation_gate_forbidden`
- `require_verify_then_resume`

---

## 7) Runtime overflow status artifact contract

## 7.1 Artifact identity
The overflow runtime artifact MUST conform to:
- schema: `docs/ops/schemas/context_overflow_status.schema.json`
- template: `docs/ops/templates/context_overflow_status.template.json`
- schema version: `clawd.context_overflow_status.v1`

## 7.2 Required object domains
Overflow status artifacts MUST include:
1. **integration contract linkage** (A1 continuity + A2 bundle schema references)
2. **source input map** (archive-mined + canonical + runtime script references)
3. **runtime measurements** (context pct, token counts, bloat, anomalies, snapshot refs)
4. **trigger evaluation** (volumetric band + semantic boundary + triggered classes)
5. **action plan** (deterministic matrix-driven expected actions)
6. **execution results** (per-action outcomes with evidence refs)
7. **gate posture** (`allowed|caution|forbidden`, safe-to-mutate, fail-closed reasons)
8. **generated artifact references** (contract/schema/template/report paths)

## 7.3 Fail-closed semantics
Overflow status evaluation MUST fail closed when:
- required checkpoint/handover actions fail,
- required runtime evidence is stale/missing for emergency paths,
- required verify-then-resume posture cannot be confirmed.

When fail-closed triggers, artifact MUST set:
- `status = fail_closed`
- `gate_posture.mutation_gate_status = forbidden`
- explicit `fail_closed_reasons[]`

---

## 8) Integration with existing runtime components

This contract integrates and formalizes behavior currently spread across:
- `ops/openclaw/context_runtime_local_watch.sh`
- `ops/openclaw/contract_context_runtime_watchdog.sh`
- `ops/openclaw/snapshot_ground_truth.sh`
- `ops/openclaw/continuity/write_checkpoint.sh`
- `ops/openclaw/continuity/render_context_handover_compat.sh`
- `ops/openclaw/continuity/sync_latest_artifacts.sh`

Existing watcher logic remains valid, but MUST be interpreted through this trigger/action contract and emitted as a schema-valid overflow status artifact.

---

## 9) Source input map (fold-in references)

The following are canonical fold-in inputs for A3 and are classified as source inputs unless explicitly marked active authority.

| Source ID | Source artifact | Classification | Folded into |
|---|---|---|---|
| src_pdf_4d1ba541 | Agent Context Overflow Prevention System (`sha256:4d1ba5414fe314e580191a51d137932b7efacfffde38a4edc773c8872ed53b9f`) | archive_mined | trigger thresholds, checkpoint/reset doctrine |
| src_pdf_c34f592e | AI Continuity Architecture Design (`sha256:c34f592e30a08559f043e07a90de3cc81c8d8f52809865bef203805ecc00da45`) | archive_mined | fail-closed continuity + lineage posture |
| src_pdf_f1fc25fc | Local Assistant Ground Truth Integration Plan (`sha256:f1fc25fce940b647ddb182551a947fa0c11f5ebb13dd8475028b919022db0dc0`) | archive_mined | ground-truth gating + runtime health triggers |
| src_rpt_archive_mining_20260330 | `reports/openclaw_archive_mining_ground_truth_continuity_2026-03-30.md` | archive_mined | hybrid trigger policy + doctrine synthesis |
| src_contract_a1 | `docs/ops/continuity_integration_contract_v1.md` | canonical | C2 trigger policy + C4/C5 gate posture integration |
| src_rpt_a2 | `reports/execution_A2_handover_checkpoint_normalization_2026-03-30.md` | canonical_support | normalized handover/checkpoint surfaces |
| src_schema_handover_a2 | `docs/ops/schemas/handover_bundle.schema.json` | canonical | handover artifact requirements |
| src_schema_checkpoint_a2 | `docs/ops/schemas/checkpoint_bundle.schema.json` | canonical | checkpoint artifact requirements |
| src_runtime_watcher | `ops/openclaw/context_runtime_local_watch.sh` | support_runtime | concrete trigger/action implementation baseline |
| src_rpt_xe202_compaction | `reports/xe202_anchor_preserving_summary_compaction_engine_v1_closeout_2026-03-29.md` | support | anchor-preserving compaction action primitive |

---

## 10) Acceptance criteria for A3

A3 is complete when:
1. One active overflow prevention runtime contract exists (this file).
2. Volumetric and semantic trigger thresholds/actions are explicit and testable.
3. A runtime overflow status artifact schema exists under `docs/ops/schemas/`.
4. A companion overflow status template exists under `docs/ops/templates/`.
5. A strict execution report verifies artifact validity and file existence.

---

## 11) Change control

Any change to thresholds, trigger classes, fail-closed semantics, or action IDs MUST:
1. update this contract,
2. update `context_overflow_status.schema.json` and template,
3. maintain backward-compatible migration notes in a dated execution report,
4. preserve explicit source-input classification discipline.
