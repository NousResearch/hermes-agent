# XO-908 Autonomy Journal + Checkpoint Lineage Contract v1

## Scope
- Slice: `XO-908`
- Lane: `XO` (Optional future upgrades)
- Objective: provide a bounded, deterministic journaling + checkpointing layer for long-running autonomous research/improvement loops without introducing unbounded branch search.

## 1) Design contract

A long-running autonomous loop execution **must** emit an append-only journal stream and checkpoints with bounded branching.

### 1.1 Journal envelope
Each journal record MUST include:
- `entry_id` (opaque stable identifier)
- `workflow_id` (stable run boundary)
- `stage_id`
- `stage_depth` (0-based)
- `attempt`
- `status` (`STARTED` | `WAITING` | `COMPLETED` | `PAUSED` | `FAILED` | `SKIPPED`)
- `occurred_at` (ISO-8601 UTC)
- `parent_entry_id` (previous journal row in same run)
- `lineage_token` (stable checkpoint lineage identifier)
- `parent_lineage_token` (previous lineage in the same replay chain)
- `decision_payload` (bounded action/observation summary)

### 1.2 Checkpoint envelope
Each successful stage completion emits a checkpoint packet with:
- `checkpoint_id`
- `lineage_token`
- `stage_id`
- `stage_depth`
- `attempt`
- `snapshot_ref` (artifact reference)
- `observed_inputs_hash`
- `can_resume` (`true`)
- `resume_token_ttl_minutes`

### 1.3 Resume envelope
A resume packet MUST include:
- `checkpoint_id`
- `resume_stage`
- `resume_step`
- `resume_token`
- `safety_notes`
- `expected_next_stage_id`

## 2) Bounded controls

1. **Depth cap:** at most 3 nested stage depths.
2. **Branching cap:** at most 2 active branches per stage.
3. **Attempt cap:** at most 2 attempts per stage before either transition or explicit `FAILED`.
4. **Queue cap:** at most 2 in-flight resumes per workflow.
5. **No silent drops:** only `SKIPPED` freshness-stage work may be dropped under explicit policy.

Violations of these caps must emit a local fail-close packet and halt autonomous escalation.

## 3) Deterministic recovery model

- Resume is **always** from the latest checkpointing parent in the active lineage chain.
- Resume packets must be replayable from recorded `decision_payload` and journal deltas.
- A run that cannot resume from its own `checkpoint_id` is terminal-failed with operator-visible blocker evidence.

## 4) Operator audit surface

- Runtime pack must include:
  1. `autonomy_journal_lineage_pack`
  2. `checkpoint_resume_simulation`
  3. `stage_lineage_graph`
- Pack must include a PASS/FAIL validation packet with checks for bounds, lineage integrity, and deterministic resume linkage.

## 5) Closeout gate for XO-908

- `autonomy_journal_lineage_pack` schema sanity (non-empty journal + valid parent linkage)
- bounded depth/branch attempts checks
- checkpoint lineage graph complete and acyclic
- resume packet selects an existing checkpoint deterministically
- validation packet status = PASS

## 6) References
- `reports/true_expanded_roadmap_four_repo_recovery_queue_addendum_2026-03-29.md`
- `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md`
- `docs/ops/core_roadmap_queue_layer_doctrine_v1.md`
