# Hermes Memory Bench v0.1

Hermes Memory Bench is a read-only benchmark scaffold for proving that Hermes
Memory Fabric is more than a generic memory wrapper. It makes the evaluation
dimensions explicit, starts with deterministic local fixtures, and emits a
stable JSON report that can later be compared across memory systems.

v0.1 does not call live Memory Fabric tools, write durable memory, write the
Memory Graph, modify OpenClaw configuration, approve allowlists, create
proposals, or create operation-ledger events. The benchmark only reads local
fixtures and writes the requested JSON report.

## Run

```bash
python -m benchmarks.hermes_memory_bench.run --suite smoke --output /tmp/hermes-memory-bench-report.json
```

## Dimensions

- `recall_accuracy`
- `temporal_accuracy`
- `source_provenance_accuracy`
- `governance_write_safety`
- `project_scope_isolation`
- `contradiction_handling`
- `hybrid_retrieval_fusion`
- `bitemporal_fact_graph`
- `contradiction_engine`
- `memory_compiler`
- `memory_blocks`
- `memory_block_review_queue`
- `memory_review_decision_gate`
- `memory_proposal_draft_builder`
- `memory_proposal_governance_gate`
- `memory_governance_submission_packet`
- `memory_human_review_outcome_gate`
- `memory_real_proposal_creation_plan`
- `memory_real_proposal_dry_run`
- `memory_real_proposal_write_lock_gate`
- `memory_human_approval_token_request`
- `memory_human_approval_token_review_gate`
- `memory_human_approval_token_issuance_plan`
- `memory_human_approval_token_issuance_dry_run`
- `latency_ms`

## Hybrid Retrieval Fusion v0.1

Hybrid Retrieval Fusion v0.1 lives in `agent.memory_retrieval_fusion` and
exposes `fuse_memory_retrieval(...)`. It is a deterministic, read-only scorer
for candidate memory records. v0.1 intentionally uses lexical matching instead
of external embeddings so benchmark runs are reproducible and do not call live
services.

Each result includes the original query, selected memories, rejected memories,
per-candidate scores, policy, and an explanation. The scorer always returns
rejected candidates with a reason and component scores; losing records are not
hidden.

Scoring dimensions:

- `semantic_score` — token-overlap similarity between query and candidate text
- `keyword_score` — exact token overlap plus exact query phrase match
- `entity_score` — entity id intersection
- `temporal_score` — current validity and recency, with expired records
  penalized
- `project_scope_score` — project match for scoped queries
- `source_trust_score` — provenance-bearing sources score highest
- `governance_score` — read-only and proposal-governed records score highest;
  unsafe write/config/graph/allowlist indicators are rejected
- `final_score` — deterministic weighted total

## Bi-temporal Fact Graph v0.1

Bi-temporal Fact Graph v0.1 lives in
`agent.memory_bitemporal_fact_graph`. It exposes a deterministic in-memory
fact model for temporal fact reasoning without writing durable memory, the
Memory Graph, config, proposals, or operation-ledger events.

Fact records include domain validity (`valid_from`, `valid_until`) separately
from system timing (`system_created_at`, `system_invalidated_at`). The v0.1
operations normalize facts, select current project-scoped facts, supersede
facts by returning updated copies, detect contradictory subject/predicate/project
claims, and explain fact lineage so superseded historical facts remain visible.

The smoke suite includes `bitemporal_fact_graph`, proving that a newer valid
fact wins while the older historical fact remains explainable through lineage.

## Contradiction Engine v0.1

Contradiction Engine v0.1 lives in
`agent.memory_contradiction_engine`. It exposes deterministic read-only
classification for memory facts and candidates with relation labels:
`supports`, `updates`, `contradicts`, `unrelated`, and `needs_review`.

The engine imports `normalize_fact` from the Bi-temporal Fact Graph module, so
validity windows, system timestamps, source episode ids, provenance, confidence,
and governance metadata are preserved without mutating inputs. Contradictions
are flagged for review or later governed proposal handling; the engine does not
write memory, write the Memory Graph, modify config, approve allowlists, create
proposals, or create operation-ledger events.

The smoke suite includes `contradiction_engine`, proving that a candidate fact
which conflicts with an existing fact returns a review recommendation rather
than being merged directly.

## Memory Compiler v0.1

Memory Compiler v0.1 lives in `agent.memory_compiler`. It compiles normalized
Bi-temporal Fact Graph facts and Contradiction Engine groups into review-only
methodology and procedure candidates. The compiler detects stable repeated
claims, superseded lineage evolution, and contradiction groups without writing
durable memory, writing the Memory Graph, modifying config, approving
allowlists, creating proposals, or creating operation-ledger events.

Compiler output includes the project scope, input and current fact counts,
contradiction group count, extracted patterns, methodology candidate, procedure
block candidate, review recommendation, trace, and read-only policy. Candidates
always use `review_required` status and explicitly state that they have not
been applied.

The smoke suite includes `memory_compiler`, proving that facts can compile into
a review-only procedure candidate while preserving the benchmark schema.

## Memory Blocks v0.1

Memory Blocks v0.1 lives in `agent.memory_blocks`. It converts review-only
compiler output into deterministic memory block candidates that future agents
can inspect without applying them. Supported block types are `human_profile`,
`persona`, `collaboration_style`, `project_context`, `procedural_rules`,
`safety_policy`, `methodology`, and `current_task_state`.

Every candidate includes a deterministic `block_id`, `review_required` status,
source pattern and fact ids, confidence, validity, `proposal_only` mutation
policy, `direct_write_allowed: false`, and explicit read-only policy proving
that it does not write memory, write the Memory Graph, modify config, approve
allowlists, create proposals, or create operation-ledger events.

The smoke suite includes `memory_blocks`, proving that a compiler procedure
candidate becomes a review-only `procedural_rules` block candidate while the
JSON report schema remains stable.

## Memory Block Review Queue v0.1

Memory Block Review Queue v0.1 lives in
`agent.memory_block_review_queue`. It wraps Memory Block candidates in
deterministic `pending_review` queue items so humans or agents can inspect them
later without applying blocks or creating any durable side effects.

Queue items include a deterministic `queue_item_id`, block identity, project
scope, priority, risk level, source pattern and fact ids, block validation,
recommended action, an immutable block snapshot, and explicit read-only policy.
Priority is deterministic: invalid blocks are highest, safety policy blocks are
high risk, methodology and procedural rules are medium risk, persona and
collaboration style blocks are medium risk, and project context and current task
state blocks are low risk.

The smoke suite includes `memory_block_review_queue`, proving that a
`procedural_rules` block candidate becomes a `pending_review` queue item while
the JSON report schema remains stable.

## Memory Review Decision Gate v0.1

Memory Review Decision Gate v0.1 lives in
`agent.memory_review_decision_gate`. It evaluates pending review queue items
into deterministic decision candidates with labels `approve_to_proposal`,
`reject`, `request_more_evidence`, and `defer`.

Decision candidates include queue and block identity, reviewer, rationale, risk,
source ids, queue validation, decision validation, next-step recommendation,
queue item snapshot, and explicit read-only policy. The gate does not apply
decisions, create real proposals, write durable memory, write the Memory Graph,
modify config, or create operation-ledger events.

The smoke suite includes `memory_review_decision_gate`, proving that a sourced
`procedural_rules` queue item becomes an `approve_to_proposal` decision
candidate without creating a real proposal.

## Memory Proposal Draft Builder v0.1

Memory Proposal Draft Builder v0.1 lives in
`agent.memory_proposal_draft_builder`. It converts approved review decision
candidates into deterministic `draft_review_required` proposal draft candidates
for a separate governed proposal flow.

Only `approve_to_proposal` decisions can produce valid drafts.
`request_more_evidence`, `reject`, `defer`, and invalid decision candidates
produce invalid draft candidates with explicit reasons. Drafts preserve the
block payload preview, source pattern ids, source fact ids, source decision
snapshot, validation, next-step recommendation, routing, and read-only policy.
The builder does not create real proposals, persist approvals, write memory,
write the Memory Graph, modify config, or create operation-ledger events.

The smoke suite includes `memory_proposal_draft_builder`, proving that an
approved procedural rules decision candidate becomes a
`draft_review_required` proposal draft candidate without creating a real
proposal.

## Memory Proposal Governance Gate v0.1

Memory Proposal Governance Gate v0.1 lives in
`agent.memory_proposal_governance_gate`. It validates proposal draft candidates
and creates deterministic `governance_review_required` governance submission
candidates for manual governed proposal creation.

Only valid drafts with `proposal_status: draft_review_required` can produce
valid submission candidates. Invalid drafts, missing payload previews, or
missing source evidence produce invalid candidates with explicit reasons. The
gate preserves payload previews, source ids, draft validation, source draft
snapshots, routing, and read-only policy. It does not submit to governance,
create real proposals, apply proposal drafts, persist approvals, write memory,
write the Memory Graph, modify config, or create operation-ledger events.

The smoke suite includes `memory_proposal_governance_gate`, proving that a
valid proposal draft becomes a `governance_review_required` submission
candidate without creating a real proposal or governance submission record.

## Memory Governance Submission Packet v0.1

Memory Governance Submission Packet v0.1 lives in
`agent.memory_governance_submission_packet`. It turns valid governance
submission candidates into deterministic `human_review_packet_required` packet
candidates for manual human review before real proposal creation.

Only valid governance submission candidates can produce valid packets. Invalid
submission candidates, missing payload previews, or missing source evidence
produce invalid packets with explicit reasons. Packets include source identity,
payload preview, evidence summary, deterministic human review checklist, risk
notes, source submission snapshot, validation, recommendation, and read-only
policy. The packet builder does not submit to governance, create proposal
records, persist approvals, convert packets to real proposals, write memory,
write the Memory Graph, modify config, or create operation-ledger events.

The smoke suite includes `memory_governance_submission_packet`, proving that a
valid governance submission candidate becomes a `human_review_packet_required`
packet candidate without creating a real proposal.

## Memory Human Review Outcome Gate v0.1

Memory Human Review Outcome Gate v0.1 lives in
`agent.memory_human_review_outcome_gate`. It turns human-review packet
candidates into deterministic `review_outcome_candidate` artifacts with outcome
labels `approve_real_proposal_creation`, `request_changes`, `reject`, and
`defer`.

Invalid packets produce `reject` outcomes, missing payload previews or source
evidence produce `request_changes`, and valid packets with payload preview plus
source evidence produce `approve_real_proposal_creation`. Explicit supported
outcome overrides are recorded only as read-only candidates. The gate does not
create real proposals, submit to governance, apply proposal drafts, persist
approvals, write memory, write the Memory Graph, modify config, or create
operation-ledger events.

The smoke suite includes `memory_human_review_outcome_gate`, proving that a
valid human-review packet becomes an `approve_real_proposal_creation` outcome
candidate without creating a real proposal.

## Memory Real Proposal Creation Plan v0.1

Memory Real Proposal Creation Plan v0.1 lives in
`agent.memory_real_proposal_creation_plan`. It turns an
`approve_real_proposal_creation` human-review outcome candidate into a
deterministic `manual_creation_plan_required` plan candidate.

Only valid approve outcomes with payload preview and source evidence can produce
valid plans. `request_changes`, `reject`, `defer`, invalid outcomes, missing
payload previews, and missing source evidence produce invalid plans with
explicit reasons. The planner creates plan candidates only; it does not create
real proposals, submit to governance, apply proposal drafts, persist approvals,
write memory, write the Memory Graph, modify config, or create operation-ledger
events.

The smoke suite includes `memory_real_proposal_creation_plan`, proving that a
valid `approve_real_proposal_creation` outcome becomes a
`manual_creation_plan_required` plan candidate without creating a real proposal.

## Memory Real Proposal Dry Run v0.1

Memory Real Proposal Dry Run v0.1 lives in
`agent.memory_real_proposal_dry_run`. It turns a valid
`manual_creation_plan_required` plan candidate into a deterministic
`manual_final_preflight_required` dry-run candidate.

The dry run previews the proposal record, operation-ledger record, deterministic
target paths, payload, source evidence, and final preflight checklist. Invalid
plans, missing payload previews, missing source evidence, or missing creation
steps/preflight checks produce invalid dry runs with explicit reasons. The dry
run creates preview candidates only; it does not create real proposals, write
proposal files, write operation-ledger events, submit to governance, persist
approvals, apply proposals, write memory, write the Memory Graph, or modify
config.

The smoke suite includes `memory_real_proposal_dry_run`, proving that a valid
manual creation plan becomes a `manual_final_preflight_required` dry run without
creating a real proposal or operation event.

## Memory Real Proposal Write Lock Gate v0.1

Memory Real Proposal Write Lock Gate v0.1 lives in
`agent.memory_real_proposal_write_lock_gate`. It turns a valid
`manual_final_preflight_required` dry-run candidate into a deterministic
write-lock gate candidate.

Valid dry runs with preview-only proposal, operation-ledger, and target-path
previews become `eligible_for_human_approval_token`. Invalid dry runs, missing
previews, missing source evidence, or preview artifacts with written/created
flags remain `locked` with explicit reasons. The gate creates write-lock
candidates only; it does not create real proposals, write proposal files, write
operation-ledger events, submit to governance, persist approvals, apply
proposals, write memory, write the Memory Graph, or modify config.

The smoke suite includes `memory_real_proposal_write_lock_gate`, proving that a
valid dry run becomes eligible for a separate human approval token flow without
creating a real proposal or operation event.

### Memory Human Approval Token Request v0.1

Implemented in `agent.memory_human_approval_token_request`. It turns an
`eligible_for_human_approval_token` write-lock gate candidate into a
deterministic `approval_token_review_required` human approval token request
candidate.

Invalid or locked write-lock gates, missing previews, missing source evidence,
or preview artifacts with written/created flags remain `locked` with explicit
reasons. The request candidate never issues real approval tokens, persists
approvals, creates real proposals, writes proposal files, writes
operation-ledger events, submits to governance, writes memory, writes the Memory
Graph, or modifies config.

The smoke suite includes `memory_human_approval_token_request`, proving that an
eligible write-lock gate becomes review-required without issuing a token,
creating a real proposal, or creating an operation event.

### Memory Human Approval Token Review Gate v0.1

Implemented in `agent.memory_human_approval_token_review_gate`. It turns an
`approval_token_review_required` request candidate into a deterministic
`token_review_outcome_candidate` with outcome labels `approve_token_issuance`,
`request_changes`, `reject`, and `defer`.

Invalid or locked request candidates reject. Missing proposal record,
operation-ledger, target-path previews, or source evidence request changes.
Valid review-required requests with intact previews and source evidence become
`approve_token_issuance` candidates. Explicit supported outcome overrides are
recorded only as read-only candidates; the gate never issues approval tokens,
persists approvals, creates real proposals, writes proposal files, writes
operation-ledger events, submits to governance, writes memory, writes the Memory
Graph, or modifies config.

The smoke suite includes `memory_human_approval_token_review_gate`, proving that
a valid approval token request becomes `approve_token_issuance` without issuing
a token, creating a real proposal, or creating an operation event.

### Memory Human Approval Token Issuance Plan v0.1

Implemented in `agent.memory_human_approval_token_issuance_plan`. It turns an
`approve_token_issuance` review outcome candidate into a deterministic
`manual_token_issuance_plan_required` plan candidate.

Only valid `approve_token_issuance` review outcomes with proposal-record,
operation-ledger, target-path previews, and source evidence can create a valid
plan. `request_changes`, `reject`, `defer`, invalid review outcomes, and missing
required previews or source evidence produce locked plan candidates. The plan
never issues approval tokens, persists approvals, creates real proposals, writes
proposal files, writes operation-ledger events, submits to governance, writes
memory, writes the Memory Graph, or modifies config.

The smoke suite includes `memory_human_approval_token_issuance_plan`, proving
that a valid `approve_token_issuance` review outcome becomes
`manual_token_issuance_plan_required` without issuing a token, creating a real
proposal, or creating an operation event.

### Memory Human Approval Token Issuance Dry Run v0.1

Implemented in `agent.memory_human_approval_token_issuance_dry_run`. It turns a
valid `manual_token_issuance_plan_required` plan candidate into a deterministic
`manual_token_issuance_final_preflight_required` dry-run candidate.

The dry run previews the approval token record, approval audit record, token
target paths, inherited proposal and operation-ledger previews, and final token
preflight checklist. It never issues real approval tokens, persists approvals,
creates real proposals, writes proposal files, writes operation-ledger events,
writes token files, writes approval audit records, submits to governance, writes
memory, writes the Memory Graph, or modifies config.

The smoke suite includes `memory_human_approval_token_issuance_dry_run`,
proving that a valid token issuance plan reaches final manual preflight without
issuing a token, persisting approval, creating a real proposal, or creating an
operation event.

## Report Schema

The runner emits JSON with these top-level fields:

- `benchmark_type`
- `generated_at`
- `suite`
- `scores`
- `cases`
- `aggregate`
- `policy`

The `policy` object is part of the benchmark contract and must show:

```json
{
  "read_only": true,
  "would_write_memory": false,
  "would_modify_config": false,
  "would_write_graph": false,
  "does_not_create_operation_events": true
}
```

## Comparison Roadmap

Future versions can add adapters for Mem0, Graphiti, Letta, Cognee, and other
memory systems. Each adapter should answer the same case fixtures and report the
same dimensions, so Hermes can be compared on grounded recall, temporal
preference handling, source provenance, governance safety, project isolation,
contradiction handling, and latency. Live adapters must keep this benchmark's
safety contract: benchmark runs are read-only unless a separate human-approved
test harness explicitly permits isolated writes in a disposable environment.
