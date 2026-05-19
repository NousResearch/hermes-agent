# ContextOps: Epistemic State Engine

> **Lane anchor:** This is a standalone ContextOps lane. It is related to prior Hermes/Agent OS memory work, but it is not subordinate to the existing `#hermes-main` memory/compaction track. The priority here is cognitive phase continuity: restoring the user's active thought-state, unresolved tensions, and epistemic stance across turns/sessions without flattening contamination.

## Operator summary

기존 memory/RAG 시스템의 기본 질문은 보통 이렇다.

```text
무엇을 저장했고, 무엇을 다시 검색할 것인가?
```

ContextOps의 질문은 다르다.

```text
현재 사용자는 어떤 사고 국면에 있으며,
어떤 unresolved tension을 이어가고 있고,
다음 응답은 어떤 epistemic mode에서 시작해야 하는가?
```

따라서 이 시스템의 핵심은 `MemoryService`가 아니라 다음에 가깝다.

```text
Cognitive state transition orchestration
```

또는:

```text
Epistemic State Engine
Cognitive Phase Continuity Layer
Thread/Tension Orchestrator
```

## Non-negotiable distinctions

1. **Thread is not topic.**
   - `사주`, `타로`, `트레이딩`, `AI memory`는 topic이다.
   - `독립이어야 하는 시스템이 은밀히 coupling되는 anomaly`, `retrieval과 cognitive continuity의 차이`가 thread다.

2. **Heat is not recency.**
   - 최근 언급은 하나의 signal일 뿐이다.
   - 실제 heat는 unresolvedness, emotional salience, contradiction density, cross-thread connectivity, explicit reactivation으로 결정된다.

3. **Compaction is not summary.**
   - 나쁜 compaction: `사용자는 사주와 유전에 대해 대화했다.`
   - 좋은 compaction: `사용자는 출생 시점 기반 사주 체계와 유전 기반 기질 체계가 독립이어야 하는데 실제 해석 경험에서는 coupling되는 듯 보이는 anomaly를 unresolved 상태로 유지 중이다.`

4. **Memory is not retrieval.**
   - 검색된 fact가 많아도 사고 국면이 복원되지 않으면 continuity가 없다.

5. **Context is not history.**
   - 전체 transcript를 넣는 것이 목표가 아니다.
   - Main LLM이 올바른 cognitive phase에서 시작할 수 있는 최소 상태 packet을 넣는 것이 목표다.

6. **State update is not note-taking.**
   - 응답 후 저장해야 하는 것은 요약이 아니라 다음 응답을 바꿀 cognitive delta다.

7. **Forgetting is less dangerous than flattening contamination.**
   - 잘못된 giant summary, noisy semantic match, 오래된 compaction이 현재 사고장을 오염시키는 것이 단순 망각보다 더 위험하다.

### Distinctions quick reference

These five distinctions are the contract surface of the lane. Any doc, schema, or
prompt that blurs them is a contamination bug, not a style choice.

| Concept | Is NOT | Is |
| --- | --- | --- |
| **Thread** | a topic label (`사주`, `트레이딩`, `AI memory`) | a persistent cognitive line — an unresolved line of thinking that survives topic changes |
| **Heat** | recency of last mention | a composite of unresolvedness, emotional salience, contradiction density, cross-thread connectivity, and explicit reactivation |
| **Compaction** | a shorter summary of what was said | preservation of cognitive pressure — the unresolved core stays unresolved |
| **Context pack** | a transcript or history window | a minimal phase-restoration packet with `restore` and `avoid` fields |
| **StateDelta** | note-taking / after-the-fact minutes | only the cognitive deltas that change the *next* response, with evidence refs |

## Core philosophy

```text
Working Memory = Epistemic State Restoration
```

더 정확히는:

```text
Raw events are evidence.
Threads are cognitive streams.
Tensions are living pressure.
Context packs are phase restoration packets.
Long-term memory is only stabilized identity/project bias.
```

## Authority ranking

ContextOps는 memory contamination을 막기 위해 authority ranking을 명시해야 한다.

충돌 시 우선순위:

1. latest explicit user message
2. active ContextContract / TaskContract
3. safety and hard guardrails
4. authoritative tool/runtime evidence
5. ChannelWorkingState / active lane state
6. recent same-channel turn digests
7. ContextOps context pack / compaction
8. persistent semantic memory / user profile
9. skills and project conventions
10. older session search results
11. inference

장기 메모리나 기존 compaction은 background bias일 뿐, 현재 사용자의 명시적 correction이나 active scope를 덮어쓸 수 없다.

## Key objects

### Event

Raw log. 해석된 상태와 섞지 않는다.

```yaml
event:
  id: uuid
  session_id: string
  lane: contextops | hermes-main | research | trading | other
  channel: string
  role: user | assistant | tool | system
  content_ref: string
  created_at: timestamp
  metadata:
    model: string
    source: discord | telegram | cli | cron | kanban | tool
    parent_event_id: optional string
```

### Thread

Thread는 topic이 아니라 지속되는 사고선이다.

```yaml
thread:
  id: memory_retrieval_vs_epistemic_restoration
  title: "Retrieval과 cognitive continuity의 차이"
  summary: "기억의 핵심이 fact retrieval이 아니라 현재 사고 국면 복원이라는 문제의식"
  status: active | dormant | resolved | archived
  heat: 0.0-1.0
  heat_components:
    recency: 0.0-1.0
    recurrence: 0.0-1.0
    unresolvedness: 0.0-1.0
    emotional_salience: 0.0-1.0
    contradiction_density: 0.0-1.0
    cross_thread_connectivity: 0.0-1.0
    explicit_reactivation: 0.0-1.0
  epistemic_modes:
    - anomaly_investigation
    - anti_premature_closure
  active_tensions:
    - retrieval_vs_state_restoration
  linked_threads:
    - compaction_contamination_problem
  last_touched_at: timestamp
```

### Tension

Tension이 1급 객체다. Thread의 실제 생명력은 tension에서 나온다.

```yaml
tension:
  id: retrieval_vs_state_restoration
  thread_id: memory_retrieval_vs_epistemic_restoration
  claim_a: "기존 memory system은 retrieval을 잘하면 continuity가 생긴다고 본다"
  claim_b: "사용자가 체감하는 continuity는 epistemic phase restoration에 더 가깝다"
  unresolved_core: "정보 검색과 사고 상태 복원은 다른 문제다"
  pressure: 0.0-1.0
  status: unresolved | reframed | resolved | abandoned
  last_delta: "Memory라는 이름 자체가 storage/retrieval 사고를 유도하는 naming contamination으로 확장됨"
```

### Epistemic mode

Tone보다 중요한 응답 자세다.

```yaml
epistemic_mode:
  - anomaly_investigation
  - contradiction_tolerance
  - anti_premature_closure
  - technical_but_exploratory
  - avoid_schema_lockin
```

### Context pack

Context Pack은 transcript가 아니라 cognitive phase restoration packet이다.

```yaml
context_pack:
  current_phase: "technical design without over-specification"
  active_thread:
    id: memory_retrieval_vs_epistemic_restoration
    title: "Epistemic State Engine 설계"
  current_tensions:
    - "문제는 retrieval이 아니라 cognitive state restoration이다"
    - "Memory라는 이름은 storage/retrieval implementation bias를 유도한다"
  user_epistemic_style:
    - anomaly seeking
    - hidden coupling fascination
    - anti-premature closure
  response_mode:
    - technical but exploratory
    - architecture-level
    - preserve unresolved pressure
  restore:
    - "thread는 topic이 아니라 사고선이다"
    - "compaction은 summary가 아니라 cognitive pressure preservation이다"
  avoid:
    - "일반 RAG 설명으로 flattening"
    - "복잡한 ontology/schema 조기 고정"
    - "topic label만으로 routing"
  minimal_raw_excerpts:
    - event_ref: "optional safe pointer"
      reason: "only if exact wording matters"
```

### State delta

State Extractor는 요약기가 아니다. 다음 응답을 바꾸는 delta만 뽑는다.

```yaml
state_delta:
  new_tensions:
    - name: memory_word_contamination
      description: "Memory라는 이름이 retrieval/storage 설계로 엔지니어링 사고를 오염시킬 위험"
  updated_tensions:
    - tension_id: retrieval_vs_state_restoration
      delta: "철학적 구분에서 naming/API/schema decision으로 확장됨"
  new_hypotheses:
    - "초기 구현은 cognitive ethnography에 가까워야 하며 deterministic schema는 관찰 이후에 안정화한다"
  resolved_items: []
  epistemic_mode_shift:
    from:
      - architecture_exploration
    to:
      - naming_precision
      - anti_overengineering
      - prototype_scaffolding
  thread_heat_delta:
    memory_retrieval_vs_epistemic_restoration: 0.12
```

## Core loop

```text
User Message
  ↓
Session Event Logger
  ↓
Cognitive Router
  ↓
Working State Loader
  ↓
Context Pack Builder
  ↓
Main LLM
  ↓
State Extractor
  ↓
Working State Update
  ↓
Long-Term Memory Candidate Queue
```

### Architecture overview

The same loop, annotated with the stores each component reads from (`R`) and writes
to (`W`). Note that the raw event ledger and the working cognitive state never share
a store — that separation is what keeps a bad extraction from corrupting evidence.

```text
                ┌──────────────────────────────────────────────┐
                │              Raw Event Ledger                 │
                │                (events.jsonl)                 │
                └───────▲───────────────────────────────┬───────┘
                        │ W                             │ R
   User Message ──► Session Event Logger                 │
                        │                                │
                        ▼                                │
                  Cognitive Router  ◄─────────────────────┤ R
                        │  (matched threads, lane check)  │
                        ▼                                 │
                Working State Loader  ◄───────────────────┼──┐ R
                        │                                 │  │
                        ▼                                 │  │
               Context Pack Builder ──► context_packs ────┘  │
                        │  (restore + avoid)               W │
                        ▼                                    │
                    Main LLM                                 │
                        │                                    │
                        ▼                                    │
                  State Extractor                            │
                        │  (deltas + evidence refs)           │
                        ▼                                     │
              Working State Update ──► threads/tensions ──────┘ W
                        │
                        ▼
        Long-Term Memory Candidate Queue ──► memory_candidates  W
                        │
                        ▼
              Human review (no auto-promotion)

   Working Cognitive State store: threads.yaml · tensions.yaml · hypotheses.yaml
```

## Component responsibilities

### 1. Session Event Logger

- Store raw evidence.
- Do not merge raw log and interpreted state.
- Preserve provenance: session, channel, lane, source, model, timestamps.
- Event log must remain auditable even if state extraction is wrong.

### 2. Cognitive Router

Given current message + active threads + recent events, determine:

- matched active threads
- whether to create a new thread
- which tensions are relevant
- what context is required
- whether this is an explicit lane switch or hidden continuation

Router must prefer conceptual continuation over keyword matching.

### 3. Working State Loader

Load only selected thread/tension/hypothesis state. Do not load all memory.

### 4. Context Pack Builder

Build a minimal cognitive phase packet. Include both `restore` and `avoid` fields.

### 5. Main LLM

Receives:

- current user message
- context pack
- optional minimal raw excerpts

Does not receive giant raw history by default.

### 6. State Extractor

Extract cognitive deltas after the turn:

- new/updated tensions
- hypothesis changes
- mode shifts
- heat deltas
- resolved/abandoned items

### 7. Working State Update

Apply deltas conservatively:

- add only what changes the next response
- preserve audit trail
- record extractor confidence
- keep raw evidence references

### 8. Long-Term Memory Candidate Queue

Do not auto-promote working state to durable memory. Promotion requires criteria:

```yaml
promotion_criteria:
  repeated_across_sessions: true
  high_heat: true
  affects_future_responses: true
  identity_or_project_relevant: true
  human_reviewed: true
```

## Storage boundaries

### Must stay separate

1. Raw event ledger
2. Working cognitive state
3. Context packs generated for model hydration
4. State deltas extracted after responses
5. Long-term memory candidate proposals
6. Approved durable memory / user profile
7. Vector/graph indexes derived from above

### Initial storage recommendation

Start with files, not DB-first architecture:

```text
contextops/
  events.jsonl
  threads.yaml
  tensions.yaml
  hypotheses.yaml
  context_packs.jsonl
  state_deltas.jsonl
  memory_candidates.jsonl
```

This makes early cognitive ethnography easy to inspect before the system hardens into a schema.

## Safety and contamination rules

1. No giant summary injection unless explicitly requested for audit.
2. No topic-only routing.
3. No automatic long-term memory promotion.
4. No old context overriding latest user correction.
5. No cross-lane residue unless router explains why it is conceptually relevant.
6. Context packs must include `avoid` when contamination risk is high.
7. Every extracted state change must carry evidence refs and confidence.
8. If router confidence is low, prefer a minimal clarification or neutral response over hallucinated continuity.

### Contamination guard examples

Concrete failure/repair pairs. Each guard rule above should be testable against a
case like these.

**Topic-only routing (rule 2).**

```text
BAD  : message mentions "타로" → route to every thread tagged tarot.
GOOD : message mentions "타로" → router checks for an active cognitive line
       (e.g. independent_systems_hidden_coupling); routes only if the conceptual
       continuation holds, otherwise opens a new thread.
```

**Cross-lane residue (rule 5).**

```text
BAD  : a #contextops turn silently inherits a trading-lane tension because a
       vector match scored high.
GOOD : router either drops the trading tension, or includes it with an explicit
       lane tag + evidence ref + a one-line reason for conceptual relevance.
```

**Stale compaction overriding live correction (rule 4).**

```text
BAD  : old compaction says "user wants schema-first design"; user just said
       "schema 조기 고정하지 말자" — pack still restores the schema-first stance.
GOOD : authority ranking puts the latest explicit user message above old
       compaction; pack's `avoid` field carries "복잡한 schema 조기 고정".
```

**Giant summary injection (rule 1).**

```text
BAD  : context pack embeds a 2k-token recap of the whole session.
GOOD  : pack carries phase + tensions + restore/avoid; raw excerpts appear only
       under minimal_raw_excerpts when exact wording matters.
```

## Evaluation rubric

This system should not be evaluated like ordinary RAG.

Useful evaluation axes:

- Does the response restore the correct thought-line?
- Does it preserve unresolved tension instead of prematurely resolving it?
- Does it avoid flattening to topic labels?
- Does it catch hidden/implicit continuation?
- Does it keep unrelated memory out?
- Does it honor current lane/scope over old memory?
- Does the next state delta actually change future responses?
- Can a human audit why a thread was activated?

## MVP acceptance checklist

The MVP is the first observable loop (see roadmap Milestone 1). It is "done enough"
only when every box below can be checked with offline evidence. This checklist is
the GO/BLOCK gate for the fan-in card.

**Core loop**

- [ ] Raw session events are logged to an append-only ledger with provenance
      (session, channel, lane, source, model, timestamps).
- [ ] Raw ledger and working cognitive state are physically separate stores.
- [ ] A new user message routes to candidate threads with a stated reason.
- [ ] A context pack is built from seed state with both `restore` and `avoid`.
- [ ] A state delta is extracted after the response and contains deltas, not a summary.
- [ ] Applying a delta updates working state without auto-promoting durable memory.

**Distinction integrity**

- [ ] Thread IDs reject bare topic labels (`trading`, `tarot`, `memory`) with no
      cognitive-pressure qualifier.
- [ ] Heat is computed from its components, not from recency alone.
- [ ] Compaction output preserves the unresolved core, not just a topic recap.
- [ ] Context pack is a phase packet, not a transcript window.
- [ ] StateDelta entries each carry evidence refs or are marked low-confidence.

**Contamination guards**

- [ ] No giant summary is injected unless explicitly requested for audit.
- [ ] No topic-only routing path exists.
- [ ] Latest explicit user correction outranks old compaction / long-term memory.
- [ ] Cross-lane state only appears with an explicit lane tag + evidence + reason.
- [ ] Low router confidence yields clarification/neutral output, not invented continuity.

**Observability & safety**

- [ ] A human-readable status view lists active threads/tensions sorted by heat
      and explains heat components.
- [ ] Every state mutation is reversible and auditable.
- [ ] Offline fixture tests cover routing, context-pack construction, and
      contamination guards; `pytest tests/contextops -q` passes with no paid/remote calls.
- [ ] No live gateway integration, no durable memory writes, no message dispatch.

## Product stance

Early ContextOps is not mainly a programming exercise. It is partly cognitive ethnography:

```text
Observe how LLMs naturally represent thought-state.
Stabilize only the structures that repeatedly prove useful.
```

Engineering owns continuity, provenance, limits, audit, and side-effect safety.

LLM owns meaning-level judgments: conceptual continuation, hidden coupling, unresolved tension phrasing, and mode inference.
