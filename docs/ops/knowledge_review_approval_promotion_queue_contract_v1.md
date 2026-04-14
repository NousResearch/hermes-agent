# Knowledge Review / Approval / Promotion Queue Contract v1

Date: 2026-03-20  
Status: legacy compatibility (wave-4/5 bounded helper contract; canonical queue runtime is `docs/ops/knowledge_review_approval_promotion_queue_v1.md`)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`  
Aligned contracts:
- `docs/ops/promotion_protocol_contract_v1.md`
- `docs/ops/doctrine_object_contract_v1.md`
- `docs/ops/controlled_cross_lane_bridge_contract_v1.md`

## 0) Intent and bounded scope

Define a deterministic, fail-closed queue artifact for **knowledge object review, approval, and promotion readiness**.

v1 goals:
- queue knowledge objects with explicit owner/reviewer/approver metadata,
- gate progression by trust tier + evidence requirements,
- enforce explicit state transitions,
- record decisions append-only.

Non-goals (v1):
- no broad ingestion/runtime refactor,
- no auto-write into canonical doctrine/memory/playbook surfaces,
- no bypass of Wave-4 promotion/leakage gates.

## 1) Core invariants

1. **Promotion, not leakage**: queue carries references + hashes, not raw transcript dumps.
2. **Fail closed**: unknown states, missing metadata, unavailable validators, or unsafe paths block.
3. **Wave-4 contracts remain authoritative**:
   - promotion candidate semantics stay in `promotion_protocol_contract_v1.md`.
   - doctrine object semantics stay in `doctrine_object_contract_v1.md`.
4. **Bounded queue**: queue has a fixed max size; full queue blocks enqueue.
5. **Append-only decision ledger**: review/approval/promotion decisions are append-only JSONL rows.

## 2) Queue artifact model (v1)

Queue item schema:
- `docs/ops/schemas/knowledge_review_queue_item.schema.json`

Starter template:
- `docs/ops/templates/knowledge_review_queue_item.template.json`

Legacy compatibility runtime helper:
- `scripts/knowledge_review_queue.py`

Mutation ingress contract (enqueue/transition):
- direct script entry is wrapper-only fail-closed;
- required runtime context: `OPENCLAW_INTERNAL_MUTATION=1` plus allowlisted `OPENCLAW_INTERNAL_MUTATION_CALLSITE`;
- canonical operator path: `bash ops/openclaw/continuity.sh knowledge-queue <enqueue|transition> ... --json`.

Canonical queue runtime path:
- `scripts/knowledge_promotion_queue.py` (contract: `docs/ops/knowledge_review_approval_promotion_queue_v1.md`)

Legacy local runtime artifacts (default):
- queue snapshot: `state/continuity/knowledge_review_queue/queue.json`
- decision ledger: `state/continuity/knowledge_review_queue/decisions.jsonl`

## 3) Required metadata fields

Every queue item must include:
- **Owner metadata**: `ownership.owner_id`, `ownership.owner_role`, `ownership.lane_id`, `ownership.work_item_id`
- **Reviewer metadata**: `review.required`, `review.reviewer_id`, `review.reviewer_role`, `review.state`
- **Approver metadata**: `approval.approval_required`, `approval.approver_id`, `approval.approver_role`, `approval.decision`
- **Trust tier**: `trust.tier`
- **Evidence requirements**: `evidence_requirements.*`

## 4) Trust tiers (v1)

- `t0_untrusted`: collected but not eligible for approval/promotion transitions.
- `t1_provisional`: reviewable, but not eligible for approval/promotion transitions.
- `t2_verified`: eligible for approval and promotion-ready transitions.
- `t3_canonical`: eligible for promotion and canonical publication flow (still gated).

Promotion-related transitions (`APPROVAL_PENDING`, `APPROVED`, `PROMOTION_READY`, `PROMOTED`) require trust tier in `{t2_verified, t3_canonical}`.

## 5) Evidence requirements (v1)

Required checks before promotion-path transitions:
1. minimum source evidence count (`evidence_requirements.min_source_refs`),
2. provenance hash presence when required (`require_provenance_hashes=true`),
3. required gate decision refs (`required_gate_decisions[]`) present in `decision_refs[]`.

Recommended default required gate decisions:
- `promotion_gate:PASS`
- `doctrine_lint:PASS` (for doctrine object flow)

## 6) Queue state model and transitions

Queue states:
- `PENDING_REVIEW`
- `UNDER_REVIEW`
- `CHANGES_REQUESTED`
- `REJECTED`
- `APPROVAL_PENDING`
- `APPROVED`
- `PROMOTION_READY`
- `PROMOTED`
- `BLOCKED`
- `EXPIRED`

Allowed transitions (v1):
- `PENDING_REVIEW -> UNDER_REVIEW | REJECTED | BLOCKED | EXPIRED`
- `UNDER_REVIEW -> CHANGES_REQUESTED | APPROVAL_PENDING | REJECTED | BLOCKED`
- `CHANGES_REQUESTED -> PENDING_REVIEW | REJECTED | BLOCKED`
- `APPROVAL_PENDING -> APPROVED | REJECTED | BLOCKED`
- `APPROVED -> PROMOTION_READY | BLOCKED | EXPIRED`
- `PROMOTION_READY -> PROMOTED | BLOCKED`
- `BLOCKED -> PENDING_REVIEW | EXPIRED`

Any other transition is invalid and must block.

## 7) Role gates (v1)

- Review transitions require reviewer role (`VALIDATOR|LIBRARIAN|RESEARCHER|SRE|PLANNER|EXECUTOR`).
- Approval transitions require approver role (`VALIDATOR|LIBRARIAN`).
- Terminal safety transitions (`BLOCKED|EXPIRED`) require governance-capable actor (`VALIDATOR|LIBRARIAN|SRE`).

## 8) Append-only decision recording

Every enqueue/transition action must append a row with:
- `queue_item_id`
- `action` (`enqueue|transition`)
- actor (`actor_id`, `actor_role`)
- `from_state` / `to_state` (for transitions)
- `decision` (`PASS|BLOCK`)
- `block_reason` (if any)
- evidence evaluation summary
- timestamp

The decision log is append-only; existing rows must not be edited or deleted.

## 9) Fail-closed rejection reasons (canonical v1)

- `schema_invalid`
- `queue_full`
- `duplicate_queue_item_id`
- `knowledge_object_unresolved`
- `knowledge_object_hash_mismatch`
- `knowledge_object_schema_invalid`
- `state_transition_invalid`
- `actor_role_not_allowed`
- `trust_tier_insufficient`
- `evidence_requirements_unsatisfied`
- `promotion_id_missing`
- `unsafe_path`
- `gate_unavailable`

## 10) Out of scope (v1)

- Automatic canonical surface mutation on queue approval.
- Queue scheduler/orchestrator integration.
- Secret-class promotion exceptions.
