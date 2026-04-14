# Research Case Pipeline MVP (2026-03-10)

## Purpose
Land a first concrete governed extraction → understanding → implementation substrate without a broad rewrite.

Primary contract unit: **Research Case (RC)** under `memory/research_cases/<case_id>/`.

This slice is intentionally minimal but durable:
- canonical case container (`rc.json`) with explicit lifecycle + promotion metadata,
- append-only ledger and checkpoint/handover artifacts,
- synthesis head pointer (`SYNTHESIS_HEAD.json`) + candidate gatecheck artifacts,
- publishable status surface (`state/continuity/latest/research_case_*.json` + registry).

## Core files per case
- `rc.json` — canonical lifecycle + reliability + promotion metadata.
- `RAW/raw_manifest.json` — registered source evidence (hash/file fingerprint).
- `SYNTH/synths/<synth_id>.{json,md}` — synthesis artifacts.
- `SYNTH/SYNTHESIS_HEAD.json` — single active synthesis pointer.
- `CANDIDATE/candidates/<candidate_id>/{candidate.json,candidate.md,gatecheck.json,promotion_candidate.json,promotion_gate_decision.json,promotion_gate_decisions.jsonl,promotion_publish_note.md,implementation_queue_item.json}` — promotion package + deterministic promotion gate evidence + implementation queue handoff packet (including B2 orchestration metadata).
- `state/continuity/latest/research_case_capacity_orchestration_runtime.json` — deterministic B2 weighted round-robin-with-aging runtime projection (capacity, starvation, selected/runnable queues, alerts).
- `state/continuity/latest/research_case_batch_replay_runtime.json` — deterministic bounded B2 cohort replay runtime projection (selected replay cohort, lifecycle stage outcomes, pass/fail tuning deltas, replay digest).
- `LEDGER/events.jsonl` — append-only lifecycle events.
- `CHECKPOINT/latest.{json,md}` — interrupt-safe resume packet.
- `reports/research_case_<case_id>_handover_latest.md` — concise handover mirror.

## CLI
Entrypoint: `python3 scripts/research_case_pipeline.py <command> ...`

Commands:
1. `init` — create case scaffold and source contract.
2. `record-synth` — write synthesis artifact, update synthesis head, advance state.
3. `promote` — write candidate + promotion candidate packet, run deterministic promotion gate runner, write gatecheck, emit `implementation_queue_item.json` on PASS, and update promotion metadata fail-closed.
4. `orchestrate-capacity` — run deterministic B2 capacity planning (weighted round-robin with aging), enforce node/global concurrency caps, and emit starvation/concurrency alerts.
5. `replay-batch` — run deterministic bounded cohort replay automation for historical/active B2 implementation queue items, simulate full intake→handoff lifecycle replay stages, and emit pass/fail quality deltas for tuning decisions.
6. `checkpoint` — write/refresh checkpoint + handover artifacts.
7. `lint` — enforce key lifecycle/promotion invariants.
8. `status` — view status and optionally publish latest registry surfaces.

## Lifecycle model (MVP)
Primary states (explicit in `rc.json`):
- `captured`, `extracted`, `triaged`, `understanding_partial`, `synthesis_partial`, `synthesis_complete`, `promotion_ready`, `closed`.

Orthogonal fields (explicit, machine-visible):
- `lifecycle.reading_state`, `lifecycle.synthesis_state`, `lifecycle.promotion_state`, `lifecycle.disposition`
- `reliability_flags.understanding_level`, `work_status`, `freshness`, `partial`, `partial_reason_code`

Hard governance invariant in this MVP:
- `promotion_state=promoted` is invalid unless the last gate decision is `approved` and a gatecheck artifact exists.

## Why this matches the Batch 1–4 arc
- **Batch 1–2**: keeps deterministic local artifacts, head pointers, and continuity-compatible latest surfaces.
- **Batch 3**: models understanding as typed, resumable objects instead of implicit chat memory.
- **Batch 4**: makes promotion explicit and gated (`candidate + gatecheck`) rather than silent synthesis drift.

## Example minimal flow
```bash
python3 scripts/research_case_pipeline.py init --case-id rc_example --title "..." --intent "..." --source <path>
python3 scripts/research_case_pipeline.py record-synth --case-id rc_example --synth-id synth_v1 --takeaway "..."
python3 scripts/research_case_pipeline.py promote --case-id rc_example --candidate-id cand_v1 --requirement "..."
python3 scripts/research_case_pipeline.py lint --case-id rc_example --strict --json
python3 scripts/research_case_pipeline.py status --case-id rc_example --publish --json
```

## XR-007 promoted asset checklist (B2 canonical promotion pack)

Archive/runtime assets promoted to canonical B2 operating checklist:

1. **Multi-case capacity orchestration runtime (`orchestrate-capacity`)**
   - Runtime surface: `state/continuity/latest/research_case_capacity_orchestration_runtime.json`
   - Slice evidence: `reports/b2_multi_case_capacity_orchestration_slice_2026-03-28.md`
   - Required verification refs:
     - `tests/test_research_case_capacity_orchestration.py`
     - `tests/test_research_case_promotion_gate_wiring.py`

2. **Batch-level replay automation runtime (`replay-batch`)**
   - Runtime surface: `state/continuity/latest/research_case_batch_replay_runtime.json`
   - Slice evidence: `reports/b2_batch_level_replay_automation_slice_2026-03-28.md`
   - Required verification refs:
     - `tests/test_research_case_batch_replay_automation.py`
     - `tests/test_wave7_contract_templates.py`

Promotion rule (fail-closed): B2 runtime changes are not promotion-complete unless both command families above are backed by passing verification refs and current runtime artifact projection.
