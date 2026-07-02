# KnowledgeProvider v1.1 — Architectural Contract

**Frozen.** This contract is depended upon by the Context System.
No structural changes without a major version bump and migration of all dependents.

---

## Why This Subsystem Exists

`KnowledgeProvider` is the interface that every knowledge source implements. It has one responsibility: resolve a retrieval key to existence and metadata, and retrieve content for that key. It is consumed only by the Context System — specifically, by the Context Provider's internal `_resolve_metadata()` and `_load_content()` methods. No other subsystem calls it.

---

## What It Owns

| ID | Responsibility |
|----|---------------|
| R1 | Check whether content exists for a given retrieval key and return its metadata without loading content |
| R2 | Load content for a given retrieval key and return it as a KnowledgeArtifact |
| R3 | Report accurate token counts per the Hermes Token Estimation Policy v1 |
| R4 | Return structured results for any expected operational failure |
| R5 | Allow both methods to be called independently — `retrieve()` is valid without a prior `resolve()`, and `resolve()` is valid without a subsequent `retrieve()` |

---

## What It Refuses to Own

| Non-Responsibility | Belongs To |
|--------------------|-----------|
| Never caches retrieval results | Cache layer — wraps the provider |
| Never manages its own lifecycle (startup, shutdown, health checks) | Provider registry / Context System |
| Never registers itself in the provider registry | Context System's provider registry |
| Never makes routing decisions (which provider handles which key) | RetrievalPlan — routing is specified by the plan |
| Never estimates tokens with a provider-specific strategy | Hermes Token Estimation Policy v1 — one policy for all providers |
| Never interprets `selection` as anything other than an opaque hint | Provider implementation — the contract carries the hint; the provider interprets it (or ignores it) |
| Never performs telemetry or metrics | Telemetry subsystem |
| Never converts programming errors into empty structured results | The provider may raise for unexpected implementation failures |

---

## Behavioral Contract

A `KnowledgeProvider` is stateless with respect to retrieval. Calling `resolve()` twice with the same key returns equivalent results. Calling `retrieve()` after `resolve()` returns the content described by the prior resolution. The provider never caches — the cache layer wraps it.

The provider is **deterministic given stable store state.** If the underlying store has not changed, repeated calls with the same key return identical results.

**`retrieve()` is valid without a preceding `resolve()`.** The two methods are independent entry points. A caller may resolve to check existence, retrieve directly if it already knows it wants the content, or call either alone. The contract imposes no ordering.

**Providers may internally reuse work between calls.** A provider that parses its store on the first `resolve()` and holds the result for subsequent `resolve()` and `retrieve()` calls is fully compliant. The caller sees two independent calls; the provider owns the interior.

---

## Required Methods

### `resolve(key: str) -> ProviderResolveResult`

Check whether content exists for the given retrieval key and return its metadata. Does not load content.

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | The retrieval key. Provider-defined semantics. For ConfigProvider: a section path like `"model"`. For MemoryProvider: a search query like `"user preferences"`. For SkillProvider: a skill name like `"hermes-agent"`. |

**Returns:** `ProviderResolveResult`

```
ProviderResolveResult {
    status:         ResolveStatus
    token_count:    int
    sections:       list[SectionMeta]
}

ResolveStatus = FOUND | NOT_FOUND | ERROR

SectionMeta {
    section_id:     str
    title:          str
    token_count:    int
}
```

| Field | Description |
|-------|-------------|
| `status` | `FOUND` — content exists and is retrievable. `NOT_FOUND` — the key does not correspond to any content. `ERROR` — the store is unreachable or retrieval failed for an expected operational reason. |
| `token_count` | Estimated tokens if the full content were loaded. `0` when `status` is not `FOUND`. Estimated per the Hermes Token Estimation Policy (§9). |
| `sections` | Available sub-sections with individual sizes. Empty list when the content has no section structure, or when `status` is not `FOUND`. |

### `retrieve(key: str, selection: list[str] | None = None) -> KnowledgeArtifact`

Load content for the given retrieval key.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | (required) | The retrieval key. Same semantics as `resolve()`. |
| `selection` | `list[str] \| None` | `None` | Opaque provider-specific retrieval hint. Interpreted only by providers that support selective retrieval. Providers that do not support it ignore the parameter. `None` means "return full content." The hint carries no prescribed structure — it may be a list of section IDs, a list of category names, or any other provider-defined interpretation. |

**Returns:** `KnowledgeArtifact` (v2.1)

```
KnowledgeArtifact {
    content:        str
    token_count:    int
}
```

| Call | Behavior |
|------|----------|
| `retrieve("hermes-agent")` | Full content (no hint) |
| `retrieve("hermes-agent", selection=["instructions"])` | Provider interprets the hint; returns what it can |
| `retrieve("hermes-agent", selection=[])` | Provider interprets the hint; empty list is not equivalent to `None` |

Providers that do not support selective retrieval ignore `selection` and return full content regardless.

---

## Inputs (Conceptual)

- **Retrieval key** — A string identifying the knowledge to retrieve. Provider-defined semantics.
- **Selection hint** — An opaque provider-specific hint for selective retrieval. Always optional.

---

## Outputs (Conceptual)

- **ProviderResolveResult** — Existence status, estimated token count, and available section metadata.
- **KnowledgeArtifact** — The loaded content and its token count.

---

## Architectural Invariants

| # | Invariant | Rationale |
|---|-----------|-----------|
| I1 | `resolve()` returns a valid `ProviderResolveResult` for every input | The Context System must be able to call resolve on any key without crashing the pipeline |
| I2 | `retrieve()` returns a valid `KnowledgeArtifact` for every input | Same guarantee — the Context System calls retrieve on any key and receives a structured result |
| I3 | If `resolve(key).status == FOUND`, then `retrieve(key).content` is non-empty and `retrieve(key).token_count > 0` | Content that exists must produce content when loaded. An empty artifact for a FOUND key is a provider bug. |
| I4 | If `resolve(key).status != FOUND`, then `retrieve(key).content` is empty and `retrieve(key).token_count == 0` | Content that doesn't exist (or can't be reached) must produce an empty artifact. A non-empty artifact for a NOT_FOUND key is inconsistent. |
| I5 | `resolve(key).token_count == retrieve(key).token_count` for the same key when `status == FOUND` and no `selection` is provided | The resolve and retrieve estimates must agree. If resolve says 1200 tokens and retrieve returns 3400, budgeting is unreliable. |
| I6 | Every `SectionMeta.section_id` is unique within a single `ProviderResolveResult` | The Context System references sections by ID. Duplicates create ambiguity. |
| I7 | The sum of all `SectionMeta.token_count` values ≤ `ProviderResolveResult.token_count` | Section sizes are a partition of the total. The sum may be less (e.g., when sections omit non-content structure), but never greater. |
| I8 | Repeated calls to `resolve()` with the same key return results with the same `status`, `token_count`, and `sections` — provided the underlying store has not changed | Determinism. The Context System's planning depends on stable metadata. |
| I9 | Repeated calls to `retrieve()` with the same key and same `selection` return artifacts with identical `content` — provided the underlying store has not changed | Determinism. Identical retrieval requests produce identical artifacts. |

---

## Error Handling

The contract distinguishes two categories of failure.

### Expected operational failures

These are failures inherent to retrieval — the key doesn't exist, the store is down, the content is corrupt. The provider returns a structured result. It never raises.

| Failure | `resolve()` result | `retrieve()` result |
|---------|-------------------|---------------------|
| Key does not exist | `status: NOT_FOUND, token_count: 0, sections: []` | `content: "", token_count: 0` |
| Store unreachable (disk full, database down, network timeout) | `status: ERROR, token_count: 0, sections: []` | `content: "", token_count: 0` |
| Permission denied | `status: ERROR, token_count: 0, sections: []` | `content: "", token_count: 0` |
| Content corrupted or unparseable | `status: ERROR, token_count: 0, sections: []` | `content: "", token_count: 0` |
| Key is syntactically invalid for this provider | `status: NOT_FOUND, token_count: 0, sections: []` | `content: "", token_count: 0` |

### Unexpected implementation failures

These are programming errors — invariant violations, assertion failures, impossible states, bugs in the provider's parser or logic. The provider **may raise.** These should not be caught and converted to empty artifacts, because doing so would mask genuine bugs during development and testing.

| Failure | Behavior |
|---------|----------|
| Assertion failure | May raise `AssertionError` |
| Internal invariant violated | May raise |
| Impossible state reached | May raise |
| Memory corruption | May raise |
| Bug in provider logic | May raise |

The caller (Context Provider) wraps provider calls in a try/except block. Expected failures produce structured results and the pipeline continues. Unexpected failures are logged as provider bugs and the pipeline degrades gracefully — the Context Provider treats an unhandled exception the same as `ERROR`.

---

## Provider Identity

Each provider implementation has a stable `provider_id` string. The Context System uses this to route retrieval plan items to the correct provider. The `provider_id` is assigned by the Context System's provider registry at registration time. The provider itself does not need to know its ID.

```
Context System provider registry:
    "skill_store"     → SkillProvider
    "memory_store"    → MemoryProvider
    "config_store"    → ConfigProvider
    "session_store"   → SessionProvider
```

---

## Token Estimation Policy

All providers must follow the **Hermes Token Estimation Policy v1.**

### Policy

Providers estimate `token_count` using one of:

| Strategy | Description |
|----------|-------------|
| **Primary:** Model-native tokenizer | The tokenizer used by the active reasoning model. Produces exact counts. |
| **Fallback:** Character-based approximation | `token_count = len(content) / chars_per_token` where `chars_per_token` is a documented constant. For English text, `4` is a reasonable default. |

A provider documents which strategy it uses and the value of `chars_per_token` if using the fallback. The strategy must be stable — a provider does not switch strategies between calls.

### Rationale

The Context Planner's budget aggregates estimates from all providers. If SkillProvider estimates at characters/3 and MemoryProvider estimates at characters/5, the aggregate budget is noisy — the planner cannot trust that 10,000 estimated tokens from one provider is comparable to 10,000 from another. A single policy ensures that estimates are coherent across providers.

The policy does not mandate a specific tokenizer implementation. It mandates that all providers use the same estimation approach, so budgets are comparable. When the reasoning model changes and brings a new tokenizer, the policy is updated once and all providers follow.

---

## Compliance Tests

Every `KnowledgeProvider` implementation must pass these tests:

### Core tests

| Test | Description |
|------|-------------|
| T1 | `resolve()` on an existing key returns `FOUND` with `token_count > 0` |
| T2 | `resolve()` on a nonexistent key returns `NOT_FOUND` with `token_count == 0` and `sections == []` |
| T3 | `retrieve()` on an existing key returns non-empty `content` with `token_count > 0` |
| T4 | `retrieve()` on a nonexistent key returns `content == ""` and `token_count == 0` |
| T5 | Neither `resolve()` nor `retrieve()` raises for expected failures — tested with nonexistent, empty, and malformed keys |
| T6 | `resolve(key).token_count == retrieve(key).token_count` when no `selection` is provided |
| T7 | Repeated calls return equivalent results (determinism under stable store state) |

### Section tests

| Test | Description |
|------|-------------|
| T8 | Section IDs within a single `ProviderResolveResult` are unique |
| T9 | Sum of section `token_count` values ≤ total `token_count` |
| T10 | `retrieve(key, selection=["instructions"])` returns content consistent with the hint (for providers that support selective retrieval) |

### Independence tests

| Test | Description |
|------|-------------|
| T11 | `retrieve()` succeeds without a prior `resolve()` call |
| T12 | `resolve()` succeeds without a subsequent `retrieve()` call |

### Error distinction tests

| Test | Description |
|------|-------------|
| T13 | Expected operational failure (nonexistent key) returns structured result — does not raise |
| T14 | Expected operational failure (store unreachable) returns structured result — does not raise |
| T15 | Provider accepts syntactically invalid keys (`""`, `None`, non-string types) and returns structured result — does not raise |
| T16 | Programming error (invariant violation, assertion failure) *may* raise — the test verifies that the Context Provider's try/except wrapper handles this gracefully |

---

## Dependencies

**Depends on:**
- KnowledgeArtifact v2.1 — the output type of `retrieve()`
- Hermes Token Estimation Policy v1 — defines how `token_count` is computed

**Depended on by:**
- Context Provider (Context System) — calls `resolve()` and `retrieve()` as part of `provide()`
- Every provider implementation (ConfigProvider, MemoryProvider, SkillProvider, SessionProvider, etc.)

---

## Future Boundaries

| Feature | Rightful Home | Why Not Here |
|---------|--------------|--------------|
| Caching strategy (TTL, eviction, staleness) | Cache layer | The provider is wrapped by the cache. Caching policy is a cache-level decision. |
| Provider lifecycle (startup, shutdown, health checks) | Provider registry | The registry manages provider instances. The provider contract is the interface, not the lifecycle. |
| Telemetry and metrics | Telemetry subsystem | Metrics are observed by the infrastructure serving the provider, not properties of the provider itself. |
| Structured `selection` parameter (beyond `list[str]`) | KnowledgeProvider v2.0 or provider implementation | The current selection type is a `list[str]`. If a future provider needs a richer selection model (e.g., `{sections: [...], max_examples: 2}`), that is a contract amendment. The contract documents `selection` as an opaque hint precisely to leave this door open without changing the conceptual model. |
| Provider-specific error codes or messages | Provider implementation detail | The contract defines three statuses (FOUND, NOT_FOUND, ERROR). Richer error detail (error codes, messages) is an implementation choice the provider may add internally, but the contract does not require or structure it. |

---

## Implementation Notes

These are clarifications for implementors. They are not part of the frozen contract and may evolve as implementation proceeds.

### section_id conventions

While the contract does not mandate specific `section_id` values, the following conventions are recommended for consistency across providers:

| `section_id` | Convention |
|-------------|-----------|
| `"instructions"` | Primary instructional content |
| `"references"` | Supplementary reference material |
| `"examples"` | Worked examples |
| `"scripts"` | Executable scripts or templates |
| `"preamble"` | Frontmatter or metadata block |
| `"body"` | Default section ID for single-section artifacts |

### JSON serialization

The canonical serialization format for both `ProviderResolveResult` and `KnowledgeArtifact` is JSON with `snake_case` field names. This is an implementation convention — the contract does not prescribe a wire format, but adopting a common format simplifies provider composition.

**ProviderResolveResult:**
```json
{
    "status": "FOUND",
    "token_count": 8200,
    "sections": [
        {"section_id": "instructions", "title": "CLI Reference", "token_count": 1200},
        {"section_id": "references", "title": "Ollama Cloud Config", "token_count": 3800}
    ]
}
```

**KnowledgeArtifact:**
```json
{
    "content": "<full skill text>",
    "token_count": 8200
}
```

---

## Contract

| Property | Value |
|----------|-------|
| **Purpose** | Interface for knowledge sources consumed by the Context System |
| **Methods** | `resolve(key) -> ProviderResolveResult`, `retrieve(key, selection=None) -> KnowledgeArtifact` |
| **Call independence** | `retrieve()` is valid without prior `resolve()`. Providers may internally reuse work between calls. |
| **Error model** | Structured results for expected operational failures. May raise for unexpected implementation failures. |
| **Caching** | Provider never caches. The cache layer wraps the provider. |
| **Determinism** | Deterministic given stable store state |
| **Token estimation** | Hermes Token Estimation Policy v1 — single policy across all providers |
| **Selection** | Opaque provider-specific retrieval hint. Interpreted only by providers that support it. |
| **Version** | 1.1 |
| **Governed by** | Hermes Governance v1.0 |

---

**Contract version:** 1.1
**Status:** Frozen
**Governed by:** Hermes Governance v1.0
**Amendments require:** A full architecture review explicitly referencing this contract by version number.
