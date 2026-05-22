# ContextOps Epistemic State Engine Roadmap

> **For Hermes / future agents:** This is a documentation and planning artifact for a standalone ContextOps lane. Do not collapse it into existing `#hermes-main` memory work. Use the roadmap to create durable Kanban/implementation tasks only after the operator approves execution.

**Goal:** Build a cognitive phase continuity layer that restores active thought-lines, unresolved tensions, and epistemic mode across turns/sessions without relying on giant summaries or naive retrieval.

**Architecture:** Start with an AI-first, inspectable file-backed prototype. Let a strong LLM perform routing/extraction while engineering provides raw event logging, state boundaries, context pack limits, provenance, auditability, and contamination controls. Only harden deterministic rules after observing real state transitions.

**Tech stack:** Python, JSONL/YAML file state, pytest, optional Pydantic schemas, later SQLite/Postgres + vector/graph indexes only after MVP proves the cognitive object model.

**Repository boundary decision:** For now, manage ContextOps inside this repo as a separate repo-local module and lane because it is tightly coupled to Hermes conversation/runtime evidence. Design every interface so it can later be extracted into an independent package/repo: keep ContextOps core free of gateway dispatch, Hermes durable-memory mutation, Obsidian/Qdrant/Neo4j writes, and non-ContextOps control-plane assumptions.

---

## Lane contract

### This lane owns

- ContextOps product/design documents.
- Epistemic State Engine object model.
- Cognitive Router / Thread Orchestrator / Context Pack Builder behavior.
- Thread/Tension/Heat semantics.
- Evaluation rubric for cognitive continuity.
- Prototype storage and experiments.

### This lane does not own initially

- Existing Hermes persistent memory plugin replacement.
- Existing `#hermes-main` compaction/ACK control-plane fixes.
- Gateway restart or live production injection.
- Obsidian/Qdrant/Neo4j writes.
- Automatic durable user memory mutation.

### Required return path

For any implementation/review graph created from this roadmap:

```yaml
origin: Devhub/#contextops
return_to: Devhub/#contextops
final_report: GO | BLOCK | NEED_MORE
include:
  - files changed
  - tests run
  - contamination/scope risks
  - next milestone recommendation
```

## MVP boundary

The first MVP is not a full memory platform. It is a working, observable loop that proves whether context packs derived from thread/tension state improve cognitive continuity.

### MVP must do

1. Log raw session events.
2. Maintain active thread/tension state.
3. Route a new user message to candidate threads.
4. Build a minimal context pack with `restore` and `avoid` sections.
5. Extract a state delta after the assistant response.
6. Update working state without auto-promoting durable memory.
7. Provide a human-readable dashboard/status view.
8. Run offline fixture tests for routing, context pack construction, and contamination guards.

### MVP must not do

- No live gateway integration.
- No external paid LLM calls in tests.
- No automatic writes to canonical Hermes memory/user profile.
- No graph DB/vector DB requirement.
- No complex ontology/taxonomy.
- No cross-lane state injection without explicit lane tag and evidence.

## Milestone 0 — Canonical docs and lane anchor

**Objective:** Make the idea durable and discoverable before coding.

**Deliverables:**

- `docs/contextops/epistemic-state-engine.md`
- `docs/plans/2026-05-17-contextops-epistemic-state-engine-roadmap.md`
- `docs/index.md` pointer

**Acceptance criteria:**

- Docs explicitly state `Thread != topic`, `Heat != recency`, `Compaction != summary`.
- Docs include MVP boundary and long-term roadmap.
- Docs mark this as standalone ContextOps lane.
- Markdown fences are balanced.

## Milestone 1 — File-backed cognitive state prototype

**Objective:** Implement an inspectable local substrate with no live integration.

**Suggested paths:**

```text
contextops/
  __init__.py
  models.py
  store.py
  router.py
  context_pack.py
  extractor.py
  heat.py
  cli.py
tests/contextops/
  test_models.py
  test_store.py
  test_router_contracts.py
  test_context_pack.py
  test_state_delta_update.py
  fixtures/
    epistemic_state_engine_seed.yaml
```

**Data files for local prototype:**

```text
.data/contextops/
  events.jsonl
  threads.yaml
  tensions.yaml
  hypotheses.yaml
  context_packs.jsonl
  state_deltas.jsonl
  memory_candidates.jsonl
```

**Tasks:**

1. Create Pydantic or dataclass models for Event, Thread, Tension, ContextPack, StateDelta.
2. Add file store that reads/writes JSONL/YAML atomically.
3. Add deterministic validation rules:
   - thread ID cannot be generic topic labels like `trading`, `tarot`, `memory` without a cognitive-pressure qualifier.
   - every context pack must include `restore` and `avoid`.
   - every state delta must include evidence refs or mark itself as low confidence.
4. Add CLI:
   - `contextops log-event --file ...`
   - `contextops status`
   - `contextops build-pack --message ...`
   - `contextops apply-delta --file ... --dry-run`
5. Add offline tests only.

**Acceptance criteria:**

- `pytest tests/contextops -q` passes offline.
- Example seed state can produce a context pack.
- Status view lists active threads/tensions sorted by heat and explains heat components.
- Applying a delta updates working state but not durable memory.

## Milestone 2 — AI-first router/extractor harness

**Objective:** Use a strong model to observe actual cognitive routing/extraction behavior without hardening rules too early.

**Tasks:**

1. Define prompt contracts for Cognitive Router.
2. Define prompt contracts for State Extractor.
3. Add JSON schema validation for model outputs.
4. Add fixture-driven fake model for tests.
5. Add replay command:

```bash
contextops replay --events fixtures/session.jsonl --state fixtures/state.yaml --dry-run
```

6. Store model output as proposals with confidence and raw refs.

**Acceptance criteria:**

- Invalid model JSON fails closed.
- Router output explains conceptual continuation, not just keywords.
- Extractor output contains deltas, not summaries.
- Dry-run replay writes no canonical state unless `--apply` is explicitly set.

## Milestone 3 — Context pack A/B evaluation

**Objective:** Prove whether ContextOps packets beat giant summaries / recent history for continuity.

**Experiment variants:**

A. Recent raw turns only
B. Giant summary
C. Context pack only
D. Context pack + minimal raw excerpts

**Evaluation rubric:**

- Thought-line restoration.
- Unresolved tension preservation.
- Hidden reference detection.
- Premature closure avoidance.
- Contamination avoidance.
- Correct lane/scope authority.
- Human auditability.

**Tasks:**

1. Create evaluation fixtures from real but sanitized conversation snippets.
2. Build scoring template.
3. Add `contextops eval` command that emits side-by-side packs/outputs.
4. Add human review fields.

**Acceptance criteria:**

- At least 10 fixtures covering memory architecture, hidden coupling, lane contamination, and stale summary risks.
- Human reviewer can mark GO/BLOCK per variant.
- Reports show which context style preserved cognitive pressure best.

## Milestone 4 — ChannelWorkingState integration design

**Objective:** Define how ContextOps interacts with Hermes channel/session state without mutating live memory.

**Tasks:**

1. Map ContextOps state to ChannelWorkingState.
2. Define authority ranking in code/docs.
3. Add read-only adapter from Hermes session events to ContextOps Event.
4. Add dry-run hydration preview:

```bash
contextops hydrate-preview --channel '#contextops' --message '...' --no-dispatch
```

**Acceptance criteria:**

- Adapter is read-only.
- Hydration preview shows selected threads, tensions, context pack, and excluded stale/contaminating candidates.
- No gateway restart, no memory writes, no message dispatch.

## Milestone 5 — Human-reviewed memory candidate queue

**Objective:** Separate working state from durable long-term memory.

**Tasks:**

1. Add MemoryCandidate model.
2. Add promotion criteria fields.
3. Add reviewer decision workflow:
   - proposed
   - accepted
   - rejected
   - needs_more_evidence
4. Add export format compatible with future DocOps/Obsidian review queue.

**Acceptance criteria:**

- No candidate becomes durable memory without explicit review.
- Candidate text is declarative, compact, and non-stale.
- Working-state items like current tensions do not auto-promote.

## Milestone 6 — Deterministic heat and lifecycle rules

**Objective:** Stabilize only rules proven by MVP observation.

**Tasks:**

1. Implement heat component updates:
   - recency
   - recurrence
   - unresolvedness
   - emotional_salience
   - contradiction_density
   - cross_thread_connectivity
   - explicit_reactivation
2. Add lifecycle transitions:
   - active → dormant
   - active → resolved
   - dormant → active via explicit reactivation or strong conceptual continuation
3. Add audit log for heat changes.

**Acceptance criteria:**

- Heat changes are explainable.
- A dormant high-unresolvedness tension can reactivate even after long silence.
- Low-confidence router matches do not silently boost heat.

## Milestone 7 — Thread graph and hidden coupling layer

**Objective:** Represent conceptual bridges between threads without premature ontology.

**Tasks:**

1. Add lightweight edges:

```yaml
edge:
  from_thread: independent_systems_hidden_coupling
  to_thread: memory_retrieval_vs_epistemic_restoration
  relation: "shared anomaly pattern"
  evidence_refs: []
  confidence: 0.0-1.0
```

2. Add hidden-coupling detection proposal output.
3. Add graph status view.
4. Defer graph DB until lightweight graph proves useful.

**Acceptance criteria:**

- Graph links explain why a seemingly random user message belongs to an older thought-line.
- Edges have evidence and confidence.
- No ontology lock-in.

## Milestone 8 — Controlled Hermes integration

**Objective:** Integrate ContextOps as an optional context-engine layer after offline proof.

**Hard stop:** This milestone requires separate explicit operator approval.

**Tasks:**

1. Add feature flag: disabled by default.
2. Add shadow mode: build packs but do not inject.
3. Add compare mode: injected vs non-injected answer preview.
4. Add live opt-in for specific lane/channel only.
5. Add rollback switch.

**Acceptance criteria:**

- Shadow mode produces audit artifacts without changing responses.
- Live mode cannot activate globally by accident.
- Operator can inspect exactly what was injected and why.

## Milestone 9 — Productization

**Objective:** Turn ContextOps into a reusable layer for agents/apps.

**Possible deliverables:**

- Local UI/dashboard for active cognitive state.
- SDK APIs:
  - `route(message)`
  - `build_context_pack(route_result)`
  - `extract_delta(turn)`
  - `apply_delta(delta)`
- Export/import of state packs.
- Team/shared lane support.
- Evaluation benchmark set.
- Optional vector/graph backend adapters.

**Acceptance criteria:**

- ContextOps can run as a library, CLI, or service.
- Backends are replaceable.
- Core semantics remain Thread/Tension/Mode/ContextPack, not generic RAG.

## First safe parallel Kanban wave

Only create these after operator approval. All cards in this wave stay inside the
standalone ContextOps lane; origin and `return_to` are `Devhub/#contextops`.

### Wave graph

Each implementation card has a **reviewer child**. In Hermes Kanban, the
implementation parent completes as `done` with a `GO-for-review` handoff after local
verification; that completion promotes the reviewer child. Downstream implementation
work depends on the relevant **reviewer child** returning `GO`, not merely on the
implementation parent finishing. The fan-in card is the single final-ACK node.

```text
   Card A (docs)            Card B (models/store)
        │                         │
        ▼                         ▼
   Review A  (GO)            Review B  (GO)
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
          Card C (context pack)            Card D (router/extractor)
                  │                                 │
                  ▼                                 ▼
            Review C  (GO)                    Review D  (GO)
                  │                                 │
                  └────────────────┬────────────────┘
                                   ▼
                        Fan-in card — MVP status/report
                                   │
                                   ▼
                     Final ACK → Devhub/#contextops
                          (GO | BLOCK | NEED_MORE)
```

### Dependency semantics

- **Depends-on** means: the upstream card's *reviewer child* has returned `GO`. A
  dependent card must not start while any dependency is still `proposed`,
  `in_progress`, or `BLOCK`/`NEED_MORE`.
- A `BLOCK` or `NEED_MORE` from a reviewer child propagates downstream: dependents
  stay blocked until the upstream pair re-runs and clears to `GO`.
- Cards with no dependency on each other run in parallel (A ∥ B; C ∥ D).
- The fan-in card depends on **all** of Review A/B/C/D returning `GO`
  (fan-in = AND of every reviewer child).

### GO-for-review rule

When an implementation card has a review child, the implementation worker does
**not** claim final acceptance. Instead:

1. Implementation worker finishes scope, runs local verification, and records a
   structured handoff with changed files, tests, decisions, and residual risks.
2. If the review child already exists or is linked, the implementation card completes
   as `done` with a **GO-for-review** summary. This is the dependency-engine signal
   that promotes the reviewer child; it is not final product acceptance.
3. The reviewer child independently returns `GO`, `BLOCK`, or `NEED_MORE`.
4. Downstream implementation cards depend on reviewer `GO` edges. Reviewer
   `BLOCK`/`NEED_MORE` creates a remediation loop and keeps downstream cards from
   starting.
5. Use `review-required` blocking only when no valid review child/path exists; do
   not block a parent that already has a waiting reviewer child.

### Card A — Docs/contract hardening

- Assignee profile: `ccsupervisor`
- Reviewer child: Review A — profile `ccreviewer`
- Scope: refine docs, add diagrams, add acceptance checklist.
- No code beyond docs.
- Depends on: nothing (wave root).

### Card B — Models/store prototype

- Assignee profile: `ccsupervisor`
- Reviewer child: Review B — profile `ccreviewer`
- Scope: models + file store + offline tests.
- Depends on: Milestone 0 docs (Card A reviewer `GO`).

### Card C — Context pack builder prototype

- Assignee profile: `ccsupervisor`
- Reviewer child: Review C — profile `ccreviewer`
- Scope: deterministic builder from seed state + tests.
- Depends on: Card B model definitions (Review B `GO`).

### Card D — Router/extractor prompt contracts

- Assignee profile: `ccsupervisor`
- Reviewer child: Review D — profile `ccreviewer`
- Scope: prompt contract docs + fake model fixtures + schema validation tests.
- Depends on: Card B models (Review B `GO`).

### Fan-in card — MVP status/report

- Assignee profile: `ccreviewer`
- Scope: verify all cards, run tests, evaluate the MVP acceptance checklist in
  `docs/contextops/epistemic-state-engine.md`, and emit the single final ACK.
- Depends on: Review A, Review B, Review C, Review D all `GO`.
- Final report path: `GO | BLOCK | NEED_MORE` to `Devhub/#contextops`, using the
  `final_report` contract under [Required return path](#required-return-path).

## Implementation guardrails

- TDD for code milestones.
- Offline tests before any live model/API use.
- Strong model calls allowed for exploration only, not required for CI.
- No paid/remote calls in automated tests.
- No durable memory writes before reviewed candidate queue exists.
- No gateway/live injection before shadow mode has evidence.
- Every state mutation must be reversible/auditable.
- Every extracted state item must have evidence refs.

## Open questions

1. Should ContextOps live inside Hermes repo initially, or become its own repo/package after MVP?
2. What is the first real evaluation corpus: #contextops only, or sanitized cross-lane samples?
3. Should the first UI be CLI status, markdown report, or small web dashboard?
4. Which model should be the exploratory router/extractor for non-test runs?
5. How much of user epistemic style belongs in ContextOps working state vs durable user memory?

## Recommended next action

Complete Milestone 0, then create the first Kanban wave for Milestones 1-2. Keep it explicitly standalone from existing Hermes-main memory work, with final reports back to `Devhub/#contextops`.
