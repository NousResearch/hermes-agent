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
