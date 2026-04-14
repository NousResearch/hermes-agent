# Source Material Classification Layer v1

Date: 2026-03-20  
Status: active (bounded deterministic classifier)

## Purpose
Classify source artifacts into a constrained taxonomy so downstream ingestion policy can enforce allow/deny gates without relying on ad-hoc operator judgment.

## Taxonomy (v1)
- `architecture_spec`
- `runbook`
- `policy_doctrine`
- `research_report`
- `runtime_evidence`
- `source_document`
- `unknown` (blocked by default)

## Artifacts
- Contract: `docs/ops/source_material_classification_layer_v1.md`
- Packet schema: `docs/ops/schemas/source_material_classification_packet.schema.json`
- Packet template: `docs/ops/templates/source_material_classification_packet.template.json`
- Runtime: `scripts/source_material_classifier.py`
- Default decision log: `state/continuity/knowledge_ingestion/source_material_classification_decisions.jsonl`

## Deterministic classification flow
1. Schema validation.
2. Material artifact path/hash validation.
3. Rule-based scoring (path + extension + bounded content probe + hints).
4. Policy enforcement (`allow_classes`, `block_classes`, `min_confidence`, `allow_generated`).

Output schema: `clawd.source_material_classification.decision.v1`.

## Fail-closed posture
- Missing validator/schema => block.
- Unresolved or mismatched artifact hash => block.
- `unknown` class or confidence below policy threshold => block.
- Generated-origin material blocks unless explicitly allowed by packet policy.

## Command
- Direct:
  - `python3 scripts/source_material_classifier.py --packet <packet.json> --json`
- Continuity dispatcher:
  - `bash ops/openclaw/continuity.sh material-classify --packet <packet.json> --json`

## Out of scope (v1)
- ML-based semantic classification.
- Ontology expansion beyond the bounded taxonomy.
- Multi-document aggregation or cross-document consensus scoring.
