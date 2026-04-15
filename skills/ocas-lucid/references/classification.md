# Classification Reference

Relevance scoring model, filing taxonomy, and wing/room assignment rules for Lucid's journal classification phase.

## Relevance scoring

Each journal entry receives a numeric relevance score. The score determines filing vs skip.

### Scoring signals (additive)

| Signal | Points | Detection |
|--------|--------|-----------|
| Decision keywords | +3 | Content contains: decided, confirmed, agreed, resolved, committed, approved, rejected |
| Entity density | +2 | 3+ named entities (people, places, organizations, tools) in decision or action fields |
| Novel entities | +3 | Entities not yet present in MemPalace (check via `mempalace_search`) |
| Correction or lesson | +4 | Content contains: mistake, lesson, learned, corrected, wrong, fixed, should have |
| User-directed action | +3 | Journal records an action taken on behalf of or directed by the operator |
| Emotional signal | +2 | Content contains: frustrated, impressed, surprised, disappointed, pleased, grateful |
| Cross-skill reference | +2 | Journal references entities or decisions from a different skill's domain |
| Relationship signal | +3 | Content describes a relationship between people, or between a person and a project/organization |

### Scoring penalties (subtractive)

| Signal | Points | Detection |
|--------|--------|-----------|
| Pure metrics | -3 | Journal contains only numeric metrics with no narrative content in decision field |
| Routine health check | -4 | Source skill is ocas-custodian and event_type is routine health check |
| Duplicate content | -5 | `mempalace_check_duplicate` returns a match above 0.9 similarity |

### Thresholds

- Score >= 5: file (MemPalace drawer, KG, or Elephas Signal depending on content type)
- Score 3-4: skip with recirculation (add to recirculation queue for re-evaluation in 7 days)
- Score <= 2: skip without recirculation

Thresholds are configurable in `config.json` under `classification.file_threshold` and `classification.recirculate_threshold`.

## Filing taxonomy

### MemPalace drawer (verbatim filing)

File as a drawer when the journal contains:
- Session reasoning, tradeoffs, or context that explains *why* a decision was made
- Multi-step workflow narrative (not just the outcome, but the process)
- Conversation content worth retrieving verbatim in future sessions
- Research findings with sources and analysis
- Lessons learned with enough context to be useful later

Drawer content is the full journal entry or a relevant extract. Do not summarize -- MemPalace's value is verbatim retrieval.

### MemPalace KG (relationship filing)

File as a KG triple when the journal contains:
- A factual relationship between entities (person works_on project, tool uses_model X)
- A temporal state change (project completed, role changed, preference updated)
- An entity attribute worth tracking over time

Use `mempalace_kg_add` with:
- `subject`: the primary entity
- `predicate`: the relationship type (use snake_case verbs: works_on, uses, decided, prefers)
- `object`: the related entity or value
- `valid_from`: the journal's timestamp

### Elephas Signal (Chronicle promotion)

Emit a Signal when the journal contains:
- A structured entity meeting the ontology types in spec-ocas-ontology.md (Person, Place, Event, etc.)
- A relationship between ontology-typed entities with confidence >= med
- An entity not yet in Chronicle (check via `elephas.query` if available)

Write the Signal to the `signal` payload field in Lucid's own dream journal entry. A single dream journal may carry multiple Signals. Use the Signal schema from spec-ocas-shared-schemas.md:

```json
{
  "signal_id": "sig_{hash}",
  "timestamp": "ISO 8601",
  "source_skill": "ocas-lucid",
  "source_journal_type": "Action",
  "user_relevance": "user|agent_only",
  "payload": {
    "type": "Person|Place|Event|...",
    "data": {}
  },
  "confidence": "high|med|low",
  "status": "active"
}
```

A single journal may produce both a MemPalace filing and an Elephas Signal if it contains both verbatim context worth preserving and a structured entity worth promoting.

### Skip

Skip when the journal contains:
- Routine operational metrics with no narrative (latency, retry counts, token usage only)
- Health check confirmations with no anomalies
- Content already filed in MemPalace (duplicate check positive)
- Content with insufficient context to be useful in isolation

Log the skip reason in `decisions.jsonl`. If score is 3-4, add to recirculation queue.

## Wing and room assignment

### Wing selection (by source skill domain)

| Source skill domain | Wing |
|---------------------|------|
| Scout, Sift, Look | wing_research |
| Elephas, Weave | wing_knowledge |
| Praxis, Dispatch | wing_operations |
| Voyage, Spot, Sands | wing_logistics |
| Rally | wing_portfolio |
| Taste | wing_preferences |
| Corvus, Thread | wing_patterns |
| Mentor, Forge, Fellow | wing_evolution |
| Vesper | wing_briefings |
| Custodian, Triage, Haiku, Bower | wing_system |
| Lucid (own journals) | Do not file own journals to MemPalace |

### Room selection (by content topic)

Rooms are derived from the journal's content, not its source skill. Use the most specific room that applies:

- Extract the primary topic from the journal's `decision.payload` or `decision.description`
- Check if a room for that topic already exists via `mempalace_list_rooms` for the selected wing
- If a matching room exists, file there
- If no match, create a new room using a concise topic label (e.g., "auth-migration", "graphql-api", "quarterly-review")

Prefer merging into existing rooms over creating near-duplicate rooms. When uncertain, check existing room names via `mempalace_get_taxonomy` before creating.

## Recirculation re-evaluation

When a recirculated journal is re-evaluated:

1. Recompute relevance score using current state (new entities in MemPalace may boost novel entity signal)
2. Check if the journal's key topics appear in journals filed since the original skip (cross-reference boost: +3 if topic appeared in 2+ subsequent filed journals)
3. If new score >= file threshold: file normally, remove from recirculation queue
4. If still below threshold: leave in queue for one more cycle. After 3 consecutive re-evaluations without promotion, archive from queue permanently

## Content extraction from journals

Journals follow the JournalEntry schema. Extract filing content from these fields:

- `decision.description` and `decision.payload`: primary content for classification and filing
- `decision.reasoning_summary`: context for why the decision was made (high value for drawer filing)
- `action.side_effect_intent`: what was done (useful for KG triples)
- `metrics`: skip if this is the only substantive content
- `okr_evaluation`: skip (Mentor's domain)

Do not file the full raw journal JSON as a drawer. Extract the meaningful narrative content and file that.
