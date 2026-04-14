# Promotion Protocol Contract v1 (Lane-Local → Shared Doctrine/Memory/Playbooks)

Date: 2026-03-20  
Status: active (bounded contract v1)  
Scope: promotion flow only (no runtime auto-mutation)

## Purpose
Define a fail-closed protocol for promoting lane-local insights into shared knowledge surfaces:
- doctrine (`docs/ops/*.md`)
- memory (`memory/*.md`)
- playbooks (`memory/skills/**/*.md`, or equivalent approved playbook paths)

This contract governs **promotion**, not raw cross-lane leakage.

## Non-goals (v1)
- No automatic write into canonical surfaces.
- No broad refactor of existing queue/watchdog runtimes.
- No implicit trust of chat summaries without source-linked evidence.

## Core invariants
1. **Local by default**: every insight starts lane-local (`promotion_state=LOCAL_ONLY`).
2. **Provenance required**: no provenance, no promotion.
3. **Confidence required**: no confidence score/method, no promotion.
4. **Source refs required**: no source refs, no promotion.
5. **Review required**: no approved review state, no promotion.
6. **Leakage gate required**: failed leakage check blocks promotion.
7. **Fail closed**: unknown/missing fields or unavailable checks => blocked.

## Promotion packet (required shape)
Use schema: `docs/ops/schemas/promotion_candidate.schema.json`

Required top-level fields:
- `promotion_id`
- `created_at`
- `promotion_state`
- `source_lane`
- `insight`
- `provenance`
- `confidence`
- `source_refs[]`
- `review`
- `target`
- `safety`

Template starter:
- `docs/ops/templates/promotion_candidate.template.json`

## State machine (contract-level)
- `LOCAL_ONLY` → candidate exists only in source lane artifacts.
- `PENDING_REVIEW` → packet is complete enough for review queue.
- `APPROVED` → review gate passed; still not yet published.
- `PROMOTED` → target surface updated with traceable reference.
- `REJECTED` / `BLOCKED` → cannot be promoted as-is.

Allowed transitions:
- `LOCAL_ONLY -> PENDING_REVIEW`
- `PENDING_REVIEW -> APPROVED | REJECTED | BLOCKED`
- `APPROVED -> PROMOTED | BLOCKED`

Any other transition is invalid.

## Gate sequence (must pass in order)
1. **Schema gate**
   - Packet validates against `promotion_candidate.schema.json`.
   - Fail => `promotion_state=BLOCKED`, reason `schema_invalid`.

2. **Provenance gate**
   - All `source_refs[].path` must resolve to real artifacts.
   - `content_hash` must be present for each source ref.
   - Fail => `BLOCKED`, reason `provenance_unresolved`.

3. **Confidence gate**
   - `confidence.score` in `[0,1]` and `confidence.method` non-empty.
   - Minimum score by target:
     - doctrine: `>= 0.80`
     - playbook: `>= 0.70`
     - memory: `>= 0.60`
   - Fail => `BLOCKED`, reason `confidence_below_threshold`.

4. **Review gate**
   - `review.state` must be `approved`.
   - Reviewer role requirements:
     - doctrine: `VALIDATOR` or `LIBRARIAN` (recommended dual sign-off outside v1 schema)
     - playbook: `VALIDATOR` or domain owner role
     - memory: `LIBRARIAN` or `VALIDATOR`
   - Fail => `BLOCKED`, reason `review_not_approved`.

5. **Leakage gate**
   - `safety.leakage_check` must be `pass`.
   - `safety.redaction_applied` must be true when classification is `restricted`.
   - classification `secret` is not promotable in v1.
   - Fail => `BLOCKED`, reason `leakage_risk`.

6. **Publish gate**
   - Target path exists and is in approved surface class (`doctrine|memory|playbook`).
   - Promotion write must include `promotion_id` in the resulting artifact/change note.
   - On success => `PROMOTED`.

## Required traceability on promoted artifacts
Every promoted change must carry:
- `promotion_id`
- source reference IDs (`source_refs[].ref_id`)
- reviewer identity (`review.reviewer_id`) and timestamp
- confidence score used at decision time
- promotion trace manifest (`clawd.promotion_trace_manifest.v1`) linking candidate hash, decision hash, and verified source hashes
- for memory target promotions: shared-memory object ID/path (`clawd.shared_memory_object.v1`)

## Fail-closed behavior summary
- Missing field: block.
- Unknown enum/state: block.
- Gate unavailable/timeouts: block (`gate_unavailable`).
- Review pending: remain `PENDING_REVIEW` (not promotable).
- Any contradiction between packet and target class: block.

## Minimal implementation posture (bounded)
- Produce packets as JSON artifacts using template + schema validation.
- Route packets through human review before any canonical write.
- Keep promotion log append-only (packet history + decision timestamps).
- Queue/runtime auto-mutation remains deferred; contract gates are wired into explicit workflow entrypoints (`research_case_pipeline.py promote`, `continuity.sh promotion-gate`).

## Next-slice recommendations
1. Add dual-signoff policy for doctrine promotions as hard gate.
2. Add replay checker over cross-case promotion gate ledgers.
3. Add multi-queue capacity governance for implementation handoff surge windows.
