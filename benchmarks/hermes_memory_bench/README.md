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
