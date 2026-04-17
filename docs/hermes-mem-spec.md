# hermes-mem-spec

Design spec for a native Hermes persistent-memory subsystem inspired by claude-mem, but built around Hermes's existing architecture, local-first priorities, and auditability requirements.

---

## Overview

Hermes already has several memory-adjacent systems:

- built-in durable memory via `memory` and `USER PROFILE`
- cross-session transcript recall via `session_search`
- pluggable memory backends via `MemoryProvider` and `MemoryManager`
- context compression via `agent/context_compressor.py`
- persistent session storage in SQLite via `hermes_state.py`

What Hermes does not yet have natively is an operational memory layer that continuously turns session activity into reusable, inspectable observations with stable IDs, cheap progressive recall, and a local viewer for audit/debugging.

This spec proposes that layer under the working name `hermes-mem`.

---

## Goal

Add a native, local-first memory subsystem that:

- captures useful observations from agent work across sessions
- compresses noisy interaction history into reusable memory units
- supports progressive recall with low token cost
- provides stable observation IDs and citations
- exposes a simple local inspection UI
- fits the current Hermes runner without forcing an external SaaS dependency

---

## Why This Fits Hermes

The right move for Hermes is not to clone claude-mem literally.

The right move is to use the primitives Hermes already has and close the missing loop:

1. capture
2. compress
3. index
4. recall
5. audit

Hermes already has the right extension seam:

- `agent/memory_provider.py` defines the lifecycle hooks needed for a real memory engine
- `agent/memory_manager.py` already routes providers and fences memory context
- `run_agent.py` already initializes a configured memory provider and injects provider tool schemas

That means `hermes-mem` should land first as a native provider plugin, not as an invasive rewrite of the core loop.

---

## Current State vs Target State

### Current Hermes memory capabilities

Hermes today can already do all of the following:

- persist compact durable facts about the user and environment
- search and summarize previous sessions with `session_search`
- activate external memory providers such as `supermemory`
- compress long conversations when context gets tight
- observe delegation results through `on_delegation`

### Gaps

Hermes does not yet have a first-class native concept of:

- observation IDs as durable memory units
- chronological memory timelines around a result
- structured session-derived observations separate from raw transcripts
- recall in layers: compact index → nearby context → full detail
- a local web viewer showing what was stored and what was reinjected
- a local-first memory backend purpose-built for Hermes rather than delegated to a third party

---

## Proposed Architecture

### 1. New provider: `plugins/memory/hermes_mem`

Recommended first implementation path:

- add a new native provider named `hermes_mem`
- keep it selectable via `memory.provider`
- let it use the existing `MemoryProvider` lifecycle
- avoid spreading bespoke hooks all over the codebase until the design proves itself
- start from the committed storage draft in `plugins/memory/hermes_mem/schema.sql`

This preserves the current model:

- built-in memory remains separate and always available
- `hermes_mem` handles operational memory and recall
- external providers remain optional alternatives, not blockers

### 2. Data layers

#### Layer A: raw sessions

Keep using existing session persistence from `hermes_state.py`.

This remains the canonical source for full transcripts.

#### Layer B: observations

Add a new local store for structured observations extracted from turns, delegations, compression events, and session-end synthesis.

An observation is the core atomic unit of reusable operational memory.

#### Layer C: summaries

Persist composite summaries by:

- session
- topic
- project/workspace
- time window

This allows cheap recall without reprocessing entire transcripts.

---

## Observation Model

Use `observation` as the primary durable unit.

Suggested minimum schema:

```python
@dataclass
class Observation:
    id: int
    session_id: str
    project: str | None
    kind: str
    title: str
    summary: str
    detail: str
    source_type: str
    source_ref: str | None
    created_at: str
    importance: float
    tags: list[str]
```

Recommended `kind` values:

- `fact`
- `decision`
- `bugfix`
- `investigation`
- `tool_result`
- `file_change`
- `user_preference`
- `delegation_result`
- `session_summary`

Design rule:

- raw tool output is not the memory unit
- the useful conclusion is the memory unit

---

## Capture Pipeline

### Turn-level capture

Use `sync_turn()` to extract compact candidate observations from:

- the user's request
- the assistant's final response
- selected tool-call outcomes

This is for lightweight, low-latency memory creation.

### Session-end synthesis

Use `on_session_end()` to generate:

- denser observations that require broader session context
- one session summary record
- optional topic or project summaries if enough signal exists

### Compression hook integration

Use `on_pre_compress()` to preserve learnings from messages about to be summarized or discarded.

### Delegation capture

Use `on_delegation()` to record task/result pairs from subagents as high-value observations.

That is especially important because delegated work often contains the best problem-solving signal.

---

## Progressive Recall Model

Hermes should adopt a 3-layer recall flow.

### Layer 1: compact search index

Tool: `memory_search`

Purpose:

- return cheap results with IDs and short summaries
- let the agent filter before fetching full detail

Suggested output fields:

- `id`
- `kind`
- `title`
- `summary`
- `score`
- `created_at`

### Layer 2: timeline context

Tool: `memory_timeline`

Purpose:

- show what happened around a memory
- expose nearby events before the model spends tokens on full payloads

Useful for:

- debugging sequences
- architecture decisions
- multi-step bugfixes
- understanding what came before/after a result

### Layer 3: full detail fetch

Tool: `memory_get`

Purpose:

- fetch full details only for selected IDs
- avoid flooding the prompt with large payloads upfront

This follows the same broad idea that makes claude-mem attractive, but adapted to Hermes's provider/tool model.

---

## Automatic Context Injection

`prefetch()` should inject only a small, fenced context block.

Recommended policy:

- include the most relevant 2–5 observations
- include stable user/project facts when clearly relevant
- include observation IDs for citation/debugging
- prefer summaries over detail bodies
- keep the injected block compact and deterministic

Important:

- full observation payloads should remain opt-in via tools
- reinjection should never behave like dumping raw history into the prompt

---

## Storage Strategy

### MVP storage

Required:

- SQLite
- FTS5 full-text index

Optional later:

- embeddings
- hybrid semantic retrieval
- local vector index or pluggable embedding backends

Rationale:

- SQLite/FTS5 matches Hermes's current storage posture
- it keeps the MVP local-first and operationally simple
- it reduces external dependencies and makes debugging easier

---

## Viewer / Audit UI

A local inspection surface is part of the product, not a nice-to-have.

### MVP viewer goals

- recent observation feed
- search UI
- filter by `kind`, session, project, and date
- observation detail page
- session summary view
- visible record of what was reinjected on the last turn

### MVP HTTP surface

Suggested endpoints:

- `GET /health`
- `GET /observations`
- `GET /observations/{id}`
- `GET /search?q=...`
- `GET /sessions/{id}/summary`
- `GET /recall/latest`

The first version does not need rich styling. It needs trust and debuggability.

---

## Relationship To Existing Hermes Features

### Built-in memory

Keep built-in `memory` and `USER PROFILE` focused on compact durable facts.

Do not overload them with operational event history.

### `session_search`

Keep `session_search` as transcript-centric cross-session recall.

`hermes_mem` is complementary:

- `session_search` answers: "what happened in that conversation?"
- `hermes_mem` answers: "what reusable observations matter now?"

Longer-term, `hermes_mem` may internally reuse session data and summarization logic from `session_search`, but the mental model should remain separate.

### External memory providers

`hermes_mem` should not delete the provider abstraction.

Instead it should become the native local-first option in that ecosystem.

---

## Recommended MVP Scope

### In scope

- new provider `plugins/memory/hermes_mem`
- local SQLite/FTS5 observation store
- observation extraction from turns, session end, delegation, and pre-compression
- tools:
  - `memory_search`
  - `memory_timeline`
  - `memory_get`
- compact automatic prefetch injection
- local read-only viewer
- stable observation IDs and citations

### Out of scope for v1

- mandatory embeddings
- cloud sync
- heavy UI polish
- collaborative multi-user knowledge management
- aggressive auto-learning of every tool result
- editing/forget UI beyond basic debug workflows

---

## Design Constraints

1. Local-first by default.
2. Avoid mandatory external SaaS dependencies.
3. Preserve current provider architecture.
4. Prefer compact recall over verbose reinjection.
5. Make every stored unit auditable.
6. Separate durable user/profile memory from operational project memory.

---

## Open Questions

1. Should `hermes_mem` eventually become the default memory provider?
2. Should observation extraction happen on every turn, session end, or both with different density levels?
3. Should the viewer be shipped inside the provider or as a separate memory-inspection tool?
4. Should `session_search` be reused internally for session synthesis, or kept as a parallel system?
5. When should embeddings become worth the complexity increase?

---

## Recommended Implementation Order

1. create the provider skeleton in `plugins/memory/hermes_mem/`
2. build `store.py` and SQLite schema
3. implement `sync_turn`, `prefetch`, `on_session_end`, `on_delegation`, and `on_pre_compress`
4. add `memory_search`, `memory_timeline`, and `memory_get`
5. add tests for storage, extraction, and provider lifecycle
6. add a minimal local viewer

This path delivers the core value quickly without overengineering the first version.
