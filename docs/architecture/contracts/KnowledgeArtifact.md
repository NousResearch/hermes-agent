# KnowledgeArtifact v2.1 — Architectural Contract

**Frozen.** This contract is depended upon by KnowledgeProvider v1.1 and the Context System.
No structural changes without a major version bump and migration of all dependents.

---

## Why This Subsystem Exists

`KnowledgeArtifact` is an immutable pipeline message that transports normalized knowledge from a `KnowledgeProvider` to the Context System. It is the single unit of content throughout the retrieval and assembly pipeline. It is not a persistence model, cache entry, telemetry record, or runtime state object — it is a message, produced by a provider and consumed by the Context Provider's assembly step.

---

## What It Owns

| ID | Responsibility |
|----|---------------|
| R1 | Carry loaded content from a KnowledgeProvider to the Context System |
| R2 | Report the token cost of the loaded content for budget arithmetic |

---

## What It Refuses to Own

| Non-Responsibility | Belongs To |
|--------------------|-----------|
| Never carries identity fields (artifact_id, retrieval_key, source URI) | Pipeline context — the Context Provider tracks which plan item produced which artifact internally |
| Never carries type classification (artifact_type) | RetrievalPlan — the RetrievalItem.type already identifies the category |
| Never carries section breakdown metadata | Context Provider — ResolvedMetadata (internal) holds the section inventory |
| Never carries cache metadata (created_at, TTL, staleness, cache key) | Cache layer — the cache wraps the provider and manages its own state |
| Never carries telemetry (access_count, load duration, tokens served) | Telemetry subsystem |
| Never carries serialization metadata (schema_version, checksum) | Implementation detail; integrity is the transport layer's responsibility |

---

## Schema

```
KnowledgeArtifact {
    content:        str
    token_count:    int
}
```

### Fields

| Field | Type | Required | Empty Value | Description |
|-------|------|----------|-------------|-------------|
| `content` | `str` | Yes | `""` | The text that was loaded. The complete content as returned by the provider for the given retrieval key and selection. May be empty when no content was found or when the request specified a selection that matched nothing. |
| `token_count` | `int` | Yes | `0` | Estimated token count of `content`, computed per the Hermes Token Estimation Policy v1. `0` when `content` is empty. |

### Validation Rules

| Rule | Description |
|------|-------------|
| V1 | `content` is a string |
| V2 | `token_count >= 0` |
| V3 | If `content` is empty, `token_count` is `0` |

---

## Inputs (Conceptual)

The artifact is produced by a KnowledgeProvider during a `retrieve()` call. It receives no external inputs — it is the output of retrieval.

---

## Outputs (Conceptual)

- **`content`** — The knowledge text. Inserted into the prompt by the Context Provider's assembly step.
- **`token_count`** — The token cost. Used by the Context Provider's assembly step to decide whether the artifact fits within the remaining budget.

---

## Architectural Invariants

| Invariant | Rationale |
|-----------|----------|
| **Immutability after creation** | The artifact crosses a producer/consumer boundary. If either side mutates it, the other side cannot trust its view. Immutability guarantees that the assembly step receives exactly what the provider returned. |
| **Lifetime determined by the runtime** | The contract does not prescribe when the artifact is discarded. The cache layer may hold it and serve it multiple times. The Context Provider may hold it across multiple assembly passes. The only guarantee is that the artifact will not change while it is held. |
| **Token count is provider-computed** | The Context System aggregates estimates from multiple providers. If each provider uses a different estimation strategy, the aggregate is incoherent. The Hermes Token Estimation Policy v1 ensures all providers estimate tokens the same way. This invariant is enforced by the policy, not by the artifact structure. |
| **Section metadata is not carried on the artifact** | The artifact carries what was loaded, not what could have been loaded. The section inventory (which sections exist, their sizes, their order) is loading-time metadata held internally by the Context Provider. The artifact is the post-load result. |

---

## Dependencies

**Depends on:**
- Hermes Token Estimation Policy v1 — defines how `token_count` is computed

**Depended on by:**
- KnowledgeProvider v1.1 — produces `KnowledgeArtifact` from `retrieve()`
- Context Provider (Context System) — consumes `KnowledgeArtifact` in the assembly step

---

## Future Boundaries

| Feature | Rightful Home | Why Not Here |
|---------|--------------|--------------|
| Section-level progressive loading | Context Provider internal metadata (ResolvedMetadata) | The artifact carries the loaded result. The decision about which sections to load is a loading-time concern. |
| Type-aware routing (SKILL before MEMORY) | RetrievalPlan / Context Provider | The RetrievalItem.type already carries the category. Duplicating it on the artifact creates two sources of truth. |
| Caching and deduplication | Cache layer | The cache wraps the provider. The artifact is the cached value, not the cache key. |
| Tracing and observability | Pipeline context (TaskDescriptor.task_id) | The pipeline carries a correlation ID. Individual artifacts are traced by their position in the pipeline. |

---

## Provider-Agnostic Validation

Both `ConfigProvider` and `MemoryProvider` produce the same `KnowledgeArtifact` without special cases.

**ConfigProvider:**
```json
{
    "content": "model:\n  default: deepseek-v4-pro\n  provider: deepseek\n...",
    "token_count": 2400
}
```

**MemoryProvider:**
```json
{
    "content": "User prefers minimal approval friction. approvals.mode=off.",
    "token_count": 15
}
```

**SkillProvider (full skill):**
```json
{
    "content": "<full skill text, 8,200 tokens>",
    "token_count": 8200
}
```

No provider-specific fields. No conditional logic in the Context System based on artifact shape. Every artifact has the same two fields.

---

## Simplification History

| Version | Fields | Changes |
|---------|--------|---------|
| v1.0 | 26 | Included identity, cache, telemetry, tracing, and infrastructure fields |
| v2.0 | 4 + Section (5) | Removed all infrastructure concerns. Added `artifact_type` for routing, `content` for text, `token_count` for budgeting, and `sections` for progressive loading |
| v2.1 | 2 | Removed `artifact_type` (duplicates `RetrievalItem.type` in pipeline context) and `sections` (loading-time metadata belongs in `ResolvedMetadata`, not on the post-load artifact). Two fields remain — the minimum required for the Context System to insert knowledge into a prompt and budget for it. |

---

**Contract version:** 2.1
**Status:** Frozen
**Governed by:** Hermes Governance v1.0
**Supersedes:** KnowledgeArtifact v2.0
**Amendments require:** A full architecture review explicitly referencing this contract by version number.
