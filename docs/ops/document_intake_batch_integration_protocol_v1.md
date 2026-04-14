# Document Intake Batch Integration Protocol v1

Date: 2026-03-21  
Status: active (Wave 6 operationalization)

## Purpose
Codify the **subagent-first default intake closeout** for new user-sent document/PDF batches.

This protocol makes batch integration auditable and deterministic before canonical roadmap/spec mutations are promoted.

## Required artifacts
- Contract: `docs/ops/document_intake_batch_integration_protocol_v1.md`
- Packet schema: `docs/ops/schemas/document_intake_batch_integration.schema.json`
- Packet template: `docs/ops/templates/document_intake_batch_integration.template.json`
- Runtime gate: `scripts/document_intake_batch_integration_gate.py`
- Decision log: `state/continuity/knowledge_ingestion/document_intake_batch_integration_decisions.jsonl`

## Required packet content
1. `synthesis_note` in `reports/`.
2. Hash-anchored `inbound_materials` references.
3. Lane-mapped `recommendations` with explicit `promotion_tier`:
   - `promote_now`
   - `promote_later`
   - `reference_only`
4. `integration_summary.counts` that matches recommendation tier totals.
5. Bounded `promote_now_edit_paths` (minimal canonical edit plan).

## Gate order
1. `schema`
2. `synthesis_note`
3. `inbound_artifacts`
4. `lane_mapping`
5. `promotion_tier_accounting`
6. `minimal_edit_plan`

If any gate fails, all downstream gates are marked `skipped`.

## Fail-closed behavior
- Missing validator/schema blocks.
- Missing or hash-mismatched inbound artifacts block.
- Synthesis note outside `reports/` blocks.
- Recommendations without lane mapping block.
- Tier count mismatches block.
- Oversized promote-now edit plans block.

## Commands
- Direct:
  - `python3 scripts/document_intake_batch_integration_gate.py --packet <packet.json> --json`
- Continuity dispatcher:
  - `bash ops/openclaw/continuity.sh doc-intake-closeout --packet <packet.json> --json`

## Operational SLA note
Every inbound doc/PDF batch intended to influence canonical planning must produce one passing closeout packet before promotion-related edits are considered complete.
