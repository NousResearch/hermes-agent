# Knowledge-Routing Design

Date: 2026-04-20
Audit ID: F-D1 (design-first; no code until an approach is agreed)

## Context

Hermes currently has **three parallel knowledge systems** that store
different kinds of "things the agent learned" with no orchestration
between them. They were each built for a valid local reason, but
nothing routes a new fact to the right layer, nothing deduplicates
across layers, and the agent has no policy for which layer to query
first. This doc surveys what exists, what the observed couplings are
already telling us, and proposes three concrete routing designs so we
can pick one before writing any code.

The motivation for picking now (rather than next quarter): the
graphify pass over `hermes-agent/` flagged `GraphManager` and
`kb_tool` as two of the ten "god nodes" in the graph, with a real
cross-system edge (`tools/kb_tool.py:566` imports `GraphManager` for
search fallback). The existing code is already whispering the need
for routing; formalizing it now is cheaper than once more call sites
accrete organic couplings.

## Current state

### Three stores, three storage layers, three query surfaces

| System | File | Backing store | Public API |
|---|---|---|---|
| Wiki KB | `tools/kb_tool.py` | `~/.hermes/wiki/*.md` (markdown + YAML frontmatter) | `kb_tool(action="search"/"list"/"read"/"file"/"log"/"lint", ...)` — single dispatcher |
| Context graph | `agent/graph_manager.py` | `~/.hermes/context-graph/kuzu_db/` (Graphiti on Kuzu) | `add_episode`, `add_academic_episode`, `search`, `get_episodes`, `decay_knowledge_graph`, `reinforce_edges` |
| Structured records | `agent/knowledge_manager.py` | SessionDB (SQLite) mirrored to Obsidian vault | `save_note`, `save_person`, `save_project`, `save_decision`, `write_decision_trace`, `sync_episodes` |

### What they each solve

- **Wiki KB** — human-readable reference pages. Long-lived concepts
  the user or agent want to browse later as prose. Free-text regex
  search. No embedding or graph structure; that is intentional — it
  is the layer you read with your eyes, not the agent.
- **Context graph** — entity/relation extraction via LLM, queryable
  semantically. Designed for "what do we know about X" across many
  ingestion episodes. Has decay + reinforcement — the only layer with
  temporal dynamics.
- **Structured records** — typed persistence for decisions,
  people, projects, notes. The Obsidian mirror is a bidirectional
  human editing surface.

### What the coupling graph already shows

- `kb_tool.py:566` imports `GraphManager` inside `_search()` as a
  fallback when the markdown wiki returns no hits. This is the only
  explicit cross-system edge.
- There is no cross-edge between `KnowledgeManager` and either of
  the other two systems.
- `knowledge_manager.sync_episodes()` looks promotion-shaped (its
  docstring talks about syncing episodes) but it writes into the
  Obsidian vault, not into either of the other knowledge stores.

### What it costs to have no router

1. **Double/triple storage of the same fact.** A user says "my
   co-founder's name is Priya" → if the agent uses `save_person`,
   only `KnowledgeManager` knows. If it uses `add_episode`, only
   `GraphManager` knows. If it writes an "entities/priya.md" wiki
   page, only `kb_tool` knows. Usually only one layer gets the write
   because the agent has no policy; the other two remain ignorant.
2. **Search is not even attempted across layers.** The agent has no
   "search everything I know about X" tool — it has three different
   tools, and nothing forces it to ask all three.
3. **Consolidation never happens.** Over a year, the wiki
   accumulates markdown pages, the graph accumulates episodes, and
   SQLite accumulates decision rows — without any promotion or
   deduplication between them.
4. **Entity IDs diverge.** The graph's node UUID for "Priya" has no
   link to the wiki's `entities/priya.md` slug or the SQLite row
   `persons.id=42`.

## Design principles

1. **Decide write-once, not replicate-three-times.** A new fact
   should land in exactly one system as the source of truth; other
   layers can subscribe to that store for their own projections.
2. **Search must be cross-layer.** Whatever the top-level query tool
   is, it must hit all three stores under the hood (in parallel,
   with result fusion) — even if writes stay single-homed.
3. **Respect what each store is optimized for.** Graph for
   relations, wiki for prose, SQLite for typed records. Routing is
   about choosing the right write target, not about making every
   store serve every query.
4. **Promotion, not duplication.** Low-value content in `MEMORY.md`
   should be able to *graduate* into the wiki or the graph when it
   earns value, via a pipeline that deletes from the source. No
   silent copy.
5. **The entity-ID namespace is shared even when the stores are
   not.** Introduce a stable external ID (a hash of normalized
   entity name + type) that every store carries. Cross-store joins
   become possible without migrating data.
6. **Changes are behind a feature flag first.** Routing layer is
   non-trivial; we want a period where the old three-tool surface
   still works and the router is opt-in for experimentation.

## Proposals

The three proposals differ on *where the coordination lives* and
*how invasive the change is*. They are not combinable — we pick one.

### Proposal A — `KnowledgeRouter` facade, stores unchanged

A new `agent/knowledge_router.py` becomes the single write and query
surface for all agent-authored knowledge. The three existing stores
keep their current APIs; the router decides which one to call based
on the content shape.

```
┌───────────────────────────────────────────────────┐
│ agent/knowledge_router.py                         │
│   KnowledgeRouter.add(kind, payload, metadata)    │
│   KnowledgeRouter.search(query, layers=ALL)       │
│   KnowledgeRouter.get(entity_id)                  │
└──────────┬─────────────┬──────────────┬──────────┘
           │             │              │
   ┌───────▼──┐  ┌──────▼────┐  ┌──────▼──────────┐
   │ kb_tool  │  │ GraphMgr  │  │ KnowledgeManager │
   └──────────┘  └───────────┘  └──────────────────┘
```

**Routing policy (`KnowledgeRouter.add`):**
- `kind == "decision"` → `KnowledgeManager.save_decision` +
  write_decision_trace, plus episode into GraphManager.
- `kind == "person" | "project"` → `KnowledgeManager.save_person /
  save_project`, plus episode into GraphManager tagged with the
  structured row's external ID.
- `kind == "concept" | "reference"` with prose body ≥ N chars →
  `kb_tool(action="file")` + episode into GraphManager.
- `kind == "episode"` (conversation turn, tool output summary) →
  `GraphManager.add_episode` only.

**Search policy (`KnowledgeRouter.search`):** scatter-gather across
all three stores with reciprocal-rank fusion
(`GraphManager.reciprocal_rank_fusion` already exists). Single
result list with source-layer tags.

**Pros.** Minimally invasive; existing stores unchanged; one place
for routing logic; easy to flag-gate. Easy to add new layers later.

**Cons.** No promotion pipeline (entries don't migrate between
layers). Still three ID namespaces — the router tags each record
with a shared external ID but doesn't unify storage. Silent writes
to a "wrong" store are still possible if an old call site bypasses
the router.

**Effort.** ~600 LOC router + ~150 LOC tests. 1-2 days of work.

**Flag-gating.** `knowledge.routing.enabled: false` in
`DEFAULT_CONFIG`. Router becomes the canonical path when true;
existing tool calls still work when false.

### Proposal B — `KnowledgeManager` becomes canonical; others project from it

Make `KnowledgeManager` the single write surface. Wiki KB and
context graph become *read-only replicas* maintained by a
subscription process.

```
┌──────────────────────────┐
│ agent/knowledge_manager  │  ← all writes land here (SQLite)
│      (canonical)         │
└────────────┬─────────────┘
             │ change events
      ┌──────┴──────┐
      ▼             ▼
┌────────────┐  ┌────────────┐
│  kb_tool   │  │ GraphMgr   │
│ (replica)  │  │ (replica)  │
└────────────┘  └────────────┘
```

**Mechanism.** `KnowledgeManager` emits events on every `save_*`
call. A background worker (or the existing Obsidian sync loop)
fans the events out:
- For every `save_person/save_project` → `kb_tool(action="file",
  page_type="entity", ...)` to create the wiki page + add a
  GraphManager episode for entity extraction.
- For every `save_decision` → wiki "decision" page + graph episode.
- Notes flow into GraphManager only (no wiki page for every note).

Wiki KB and GraphManager become strictly *derived* stores;
direct `kb_tool(action="file")` calls are deprecated over a release
cycle, and GraphManager's `add_episode` becomes internal-only.

**Pros.** Cleanest. One source of truth eliminates entire classes
of inconsistency. Agent has one obvious write path. Replicas can
be rebuilt from scratch if they drift.

**Cons.** Most invasive of the three. `kb_tool` loses its direct
write surface (which the wiki skill-docs currently encourage).
GraphManager's episode-ingestion latency (~2-10s per LLM entity
extraction) becomes a write-path concern — either the fan-out is
async (eventual consistency, which the agent has to reason about)
or synchronous (long write latencies).

**Effort.** ~1200 LOC (event bus + fan-out workers + deprecation
shims + event replay tool). 1-2 weeks + deprecation period.

**Flag-gating.** Harder — the contract change is in-band.

### Proposal C — Promotion pipeline only; keep three tools independent

Don't build a router. Accept that writes are single-homed and
explicit. Build only the *promotion* pipeline: a periodic job that
moves high-value content between layers based on earned evidence.

```
   MEMORY.md (lowest)
          │
          │   promotion when score ≥ N
          ▼
   Wiki KB (reference prose)
          │
          │   promotion when referenced k times across sessions
          ▼
   Context graph (semantic, long-lived)
          │
          │   structured form detected
          ▼
   Structured records (typed SQLite)
```

**Mechanism.** A new cron job (`hermes cron` compatible) runs nightly:

1. Read `~/.hermes/MEMORY.md`. For each entry scoring ≥ 8 in the
   existing `memory_curator` pipeline, call `kb_tool(action="file",
   page_type="concept")` with the entry content. Remove from
   MEMORY.md.
2. For each wiki page that was read ≥ k times in the last 30 days
   OR whose slug matches an existing graph entity, call
   `GraphManager.add_episode` with the page content.
3. For each graph entity with ≥ m outgoing relations and a
   recognizable type (Person/Project/Decision), auto-create a
   `KnowledgeManager` structured row tagged with the graph entity's
   external ID.

No new router — just a promotion daemon and the existing three
tools keep their current surfaces.

**Pros.** Minimum new abstractions. Keeps the three stores'
current simplicity. Addresses the single biggest symptom (content
never flows to the right long-term home). No flag needed — either
the cron job runs or it doesn't.

**Cons.** Doesn't fix write-time routing (the agent still has to
pick a tool). Doesn't unify search — a query still has to hit
three tools. Doesn't establish entity-ID namespace sharing.

**Effort.** ~400 LOC promotion daemon + ~100 LOC tests + 1 cron
recipe. 3-5 days.

**Flag-gating.** Implicit — the promotion cron job is
user-installable rather than auto-enabled.

## Recommendation

**Proposal A + the cron half of Proposal C.**

- Proposal A gives the agent a single write API, a single cross-
  layer search, and a shared external-ID namespace — which solves
  the three root-cause problems without making any of the existing
  stores canonical.
- Adding C's promotion cron on top of it gives us the
  content-flow-between-layers behavior without needing the
  invasive event-bus machinery of B.
- B is the "right long-term shape" but has too much deprecation
  churn for the current cadence — we can migrate to B *from* A+C
  if we ever decide we want a canonical store. Going straight to B
  would collide with in-flight refactors in the F-L1 series on
  `agent/core.py`.

## Phasing (if A + C-cron is approved)

1. **Week 1 — shared external-ID format and feature flag.**
   Add `knowledge.routing.enabled` to `DEFAULT_CONFIG`. Add the
   `knowledge_external_id(kind, name) -> str` helper (simple hash;
   unit-tested). Emit it from every new write site in the three
   existing stores (append-only to row schemas — no migration).

2. **Week 2 — `KnowledgeRouter` scaffolding.**
   `agent/knowledge_router.py` with `add(kind, payload)` and
   `search(query)` methods. Both delegate to the real stores.
   Behind the flag. Not yet called from the agent.

3. **Week 3 — agent integration.**
   Add a `knowledge` tool whose handler delegates to
   `KnowledgeRouter` when flag is on, falling back to the existing
   `kb_tool`/`memory`/`context_graph` tools when off. Agent system
   prompt gains a directive to prefer the `knowledge` tool.

4. **Week 4 — promotion cron.**
   `hermes cron add` recipe wiring
   `agent/knowledge_promotion.py` (MEMORY → wiki → graph →
   structured). `[SILENT]` prefix by default; output lands under
   `~/.hermes/cron/output/knowledge_promotion/`.

5. **Week 5 — observation window.**
   Real usage for two weeks. Success criteria:
   - No regression in `kb_tool`/`GraphManager`/`KnowledgeManager`
     standalone tests.
   - ≥90% of new agent-authored facts go through the router
     (measured by an audit count in `memory_curator`'s next run).
   - Zero cross-layer double-writes in a representative session
     trace.

## Open questions

1. **Is the wiki KB's markdown canonical or derivable?** If
   derivable, we could consider B later. If canonical (user edits
   files directly), A is the only viable shape.
2. **Does the graph need a write gate?** Currently GraphManager is
   free to add arbitrary episodes. With a router, should direct
   `add_episode` calls be deprecated (as in B) or allowed (as in A)?
3. **How do we reconcile the graph's entity UUIDs with the
   external ID namespace?** Simplest: add `external_id` as a node
   property; leave the UUID alone. Any graph query can join on
   either.
4. **Flag default — off or on?** Off during phasing; flip to on
   after Week 5 observation window clears.
5. **Do we port this to `hermes-companion`?** The Swift companion
   has its own knowledge storage in `~/.hermes/state.db`; the
   router would have to be duplicated or the companion has to call
   the agent gateway for knowledge writes. Out of scope for F-D1.

## Out of scope

- Changing any existing store's storage format (no SQL migration
  or wiki reflow).
- Re-indexing Graphiti's existing Kuzu DB (shared external IDs are
  append-only).
- Touching `hermes-companion`'s knowledge path (independent).
- Writing any code (this doc is gate 1; approval required before
  touching `agent/knowledge_router.py`).
