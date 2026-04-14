# Document/PDF OS Operating Model v1

Date: 2026-03-21  
Status: active (Wave 6 ownership/SLA operationalization)

## Scope
Operational ownership and SLA model for B3 Document/PDF OS lanes:
- markdown conversion gate,
- source classification,
- production ingestion,
- subagent intake closeout integration packets.

## Ownership model
- **Primary owner:** `knowledge_ingestion_primary`
  - accountable for packet contract quality and ingestion gate reliability.
- **Secondary owner:** `operations_release_owner`
  - accountable for destination profiles, release-safety alignment, and incident handoff.
- **Escalation owner:** `architecture_lane_owner`
  - resolves policy conflicts, taxonomy drift, and cross-lane doctrine changes.

## SLA targets (v1)
- Inbound batch triage start: <= 4 hours during active operating window.
- Intake closeout packet generation: <= 24 hours from triage start.
- Promote-now integration edits (if approved): <= 48 hours from closeout PASS.
- BLOCK decision acknowledgment + owner assignment: <= 2 hours.

## Required run surfaces
- `state/continuity/knowledge_ingestion/markdown_conversion_gate_decisions.jsonl`
- `state/continuity/knowledge_ingestion/source_material_classification_decisions.jsonl`
- `state/continuity/knowledge_ingestion/production_knowledge_ingestion_decisions.jsonl`
- `state/continuity/knowledge_ingestion/production_ingestion_ledger.jsonl`
- `state/continuity/knowledge_ingestion/document_intake_batch_integration_decisions.jsonl`

## Operational cadence
- Daily: review BLOCK reasons and duplicates from ingestion ledger/gates.
- Weekly: destination-profile policy review (`document_ingestion_destination_profiles_v1.json`).
- Weekly: intake closeout packet sample audit (tier accounting + lane mapping quality).
- Monthly: SLA adherence summary and backlog pressure review.

## Escalation triggers
- >3 consecutive BLOCKs from same gate type within 24h.
- ingestion ledger write failures or decision-log append failures.
- destination profile policy conflicts with classification labels.
- multi-host fault-injection harness recovery failure (`multi_host_recovery_failed`) or source/destination SHA mismatch.
- intake closeout packets failing tier-accounting repeatedly.
