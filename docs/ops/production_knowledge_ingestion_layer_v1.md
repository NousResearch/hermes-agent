# Production Knowledge Ingestion Layer v1

Date: 2026-03-20  
Status: active (bounded eval+apply runtime)

## Purpose
Enforce a strict ingress contract for promoting curated markdown into production knowledge storage.

The ingestion layer requires two upstream decisions to pass:
1. markdown conversion quality gate decision,
2. source material classification decision.

## Artifacts
- Contract: `docs/ops/production_knowledge_ingestion_layer_v1.md`
- Packet schema: `docs/ops/schemas/production_knowledge_ingestion_packet.schema.json`
- Packet template: `docs/ops/templates/production_knowledge_ingestion_packet.template.json`
- Runtime: `scripts/production_knowledge_ingestion.py`
- Destination profile policy: `docs/ops/document_ingestion_destination_profiles_v1.json`
- Destination profile schema/template:
  - `docs/ops/schemas/document_ingestion_destination_profiles.schema.json`
  - `docs/ops/templates/document_ingestion_destination_profiles.template.json`
- Decision log: `state/continuity/knowledge_ingestion/production_knowledge_ingestion_decisions.jsonl`
- Ingestion ledger: `state/continuity/knowledge_ingestion/production_ingestion_ledger.jsonl`
- Latest snapshot: `state/continuity/latest/production_knowledge_ingestion_latest.json`

## Gate order
1. `schema`
2. `artifact_integrity`
3. `markdown_gate`
4. `classification_gate`
5. `destination_policy`
6. `duplicate_guard`

## Apply behavior (`ingest` / `apply`)
If evaluation passes:
- copy markdown artifact to destination,
- write metadata sidecar (`<destination>.meta.json`),
- append immutable ingestion ledger record,
- refresh latest snapshot.

## Fail-closed posture
- Any upstream decision mismatch/schema mismatch blocks.
- Upstream decision SHA linkage must match packet artifacts.
- Destination must stay under the default allowed root or a declared destination profile root (`destination.profile`) from `docs/ops/document_ingestion_destination_profiles_v1.json`.
- Duplicate ingestion ID or duplicate material+markdown+destination tuple blocks.
- Mutating commands (`ingest`, `apply`, `ingest-multi-host`) are wrapper-only fail-closed:
  - require `OPENCLAW_INTERNAL_MUTATION=1`,
  - require allowlisted `OPENCLAW_INTERNAL_MUTATION_CALLSITE` (default: `continuity.sh:knowledge-ingest`),
  - optional allowlist extension via `OPENCLAW_PRODUCTION_KNOWLEDGE_INGESTION_ALLOWED_CALLSITES`.
- Strict replay exception (idempotent no-op): when destination already exists and overwrite is disabled, evaluation may still pass only if destination SHA equals packet markdown SHA **and** ledger contains an exact prior tuple match (`ingestion_id`, `material_sha`, `markdown_sha`, `destination_path`). In that case apply is a no-op (no destination rewrite, no ledger append).
- Apply failures roll back written destination/metadata files when possible.

## Commands
- Evaluate only:
  - `python3 scripts/production_knowledge_ingestion.py evaluate --packet <packet.json> --json`
- Continuity dispatcher:
  - `bash ops/openclaw/continuity.sh knowledge-ingest evaluate --packet <packet.json> --json`
  - `bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest ingest --packet <packet.json> --json`
  - `bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest apply --packet <packet.json> --json`
  - `bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest ingest-multi-host --packet <packet.json> --fault-fixture tests/fixtures/b3/multi_host_fault_injection_fixture_v1.json --json`

## Slice 24 contract extension (high-volume stress)
Canonical stress-fixture artifacts (used for deterministic volume-test closeout):
- Contract schema: `docs/ops/schemas/b3_high_volume_ingestion_stress_fixture.schema.json`
- Contract template: `docs/ops/templates/b3_high_volume_ingestion_stress_fixture.template.json`
- Canonical fixture: `tests/fixtures/b3/high_volume_ingestion_stress_fixture_v1.json`

Stress-fixture rules:
- `batch_size` defines a deterministic packet burst executed against `ingest` mode.
- Assertions require PASS decisions, applied writes, unique ingestion IDs, and ledger row parity.
- Fixture destinations must remain under `memory/knowledge_ingestion/production/...` to preserve lane boundaries.

Verification entrypoint:
- `tests/test_production_knowledge_ingestion.py::test_production_ingestion_high_volume_stress_fixture_contract`

## Slice 25 contract extension (multi-host fault injection + handoff)
Canonical Slice-25 artifacts:
- Contract schema: `docs/ops/schemas/b3_multi_host_fault_injection_fixture.schema.json`
- Contract template: `docs/ops/templates/b3_multi_host_fault_injection_fixture.template.json`
- Canonical fixture: `tests/fixtures/b3/multi_host_fault_injection_fixture_v1.json`
- Runtime latest artifact: `state/continuity/latest/b3_multi_host_fault_injection_latest.json`

Harness command:
- `bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest ingest-multi-host --packet <packet.json> --fault-fixture tests/fixtures/b3/multi_host_fault_injection_fixture_v1.json --json`

Harness guarantees:
- Deterministic chunk-level host assignment across a declared host set.
- Injected active-host drop events force explicit chunk handoff to healthy hosts.
- Recovery fails closed if all candidate hosts for a chunk are unavailable.
- Final destination bytes must match source markdown SHA-256 exactly.

Verification entrypoint:
- `tests/test_production_knowledge_ingestion.py::test_production_ingestion_multi_host_fault_handoff_harness`

## XR-007 promoted asset checklist (B3 canonical promotion pack)

Archive/runtime asset promoted to canonical B3 checklist:

1. **Multi-host fault injection + deterministic handoff harness (`ingest-multi-host`)**
   - Runtime latest artifact: `state/continuity/latest/b3_multi_host_fault_injection_latest.json`
   - Slice evidence: `reports/b3_document_multi_host_reliability_hardening_slice_2026-03-28.md`
   - Required verification refs:
     - `tests/test_production_knowledge_ingestion.py::test_production_ingestion_multi_host_fault_handoff_harness`
     - `tests/test_wave6_contract_templates.py`

Promotion rule (fail-closed): B3 ingestion runtime changes that touch multi-host behavior must include a passing fixture-contract check and fresh `b3_multi_host_fault_injection_latest.json` evidence.

## Out of scope (v1)
- Real distributed lock manager across external hosts (current lane uses deterministic local harness simulation).
- Auto-routing to multiple destination surfaces.
- Automatic rollback to prior destination content snapshot on overwrite mode.
