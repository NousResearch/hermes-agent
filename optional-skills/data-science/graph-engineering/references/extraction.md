# Knowledge Extraction: Entities, Relations, Events
*(Course lectures 4-7, translated and adapted)*

## Contents
- [Extraction by source type](#extraction-by-source-type)
- [Entity extraction](#entity-extraction)
- [Relation extraction](#relation-extraction)
- [Event extraction](#event-extraction)
- [LLM extraction prompt pattern](#llm-extraction-prompt-pattern)
- [Failure modes](#failure-modes)

## Extraction by source type

(Lecture 4.) Match method to source structure — using NLP on data that is already structured
is the classic beginner waste:

- **Structured (databases, CSVs, APIs):** direct mapping, no NLP. Write a per-source mapping
  from columns → ontology types (the course's D2R idea). Deterministic code, not LLM.
- **Semi-structured (HTML tables, infoboxes, wikis, JSON blobs):** wrappers/parsers per
  layout family; LLM only for the messy cells.
- **Unstructured (text, transcripts, PDFs):** the NER → RE → EE pipeline below.

## Entity extraction

(Lecture 5's method ladder, compressed for the LLM era.)

The course traces: rules/dictionaries → HMM/CRF → BiLSTM-CRF → semi-supervised → transfer →
pretrained (BERT) → LLM. What survives into practice:

1. **Dictionary/rule extraction first** for closed vocabularies you already have (your product
   names, team roster, ticker symbols, ontology enum values). Exact match beats any model —
   free, deterministic, 100% precision.
2. **LLM extraction** for everything open-ended, with the ontology's entity types + definitions
   + examples in the prompt (pattern below).
3. **Always capture:** surface form, canonical form (best guess), type, source pointer
   (doc id + char span or sentence), confidence.

Classical lesson that still applies to LLM output: **nested and discontinuous mentions**
("University of California, Berkeley professor John Smith" contains an ORG inside a PERSON
context) and **type ambiguity** ("Apple") drive most errors — require the model to quote its
evidence sentence, which forces disambiguation from context.

## Relation extraction

(Lecture 6.) Course inventory: template-based → supervised → weakly supervised → distant
supervision → unsupervised open IE → deep/RL methods. Modern distillation:

- Extract relations **only between entities that passed stage 4** — never let relation
  extraction invent new entities. This single constraint kills most compounding errors.
- Constrain output to the ontology's relation list; **validate domain/range in code**
  (an `EMPLOYED_BY` edge from Org → Org is auto-rejected).
- Distant supervision's core insight still matters for evaluation: if two entities co-occur,
  a model will happily assert the relation the prior suggests. Guard: require an evidence
  quote that *asserts* the relation, not just co-occurrence ("Musk discussed Twitter" is not
  `OWNS`).
- Keep un-modeled but repeated relations in a `candidate_relations` side-list — review weekly;
  promote real ones into the ontology rather than forcing them into wrong types.

## Event extraction

(Lecture 7.) Use when the domain is dynamic — news, incidents, transactions, funding rounds.

An event = **trigger** (the word/phrase signaling it) + **typed arguments** (participants,
time, place, values) + **event type** from the ontology. Events are first-class nodes with
edges to their arguments — never flatten a 4-argument event into 6 pairwise edges (you lose
which acquisition happened at which price).

The course's finance case study generalizes: define per-event-type argument schemas
(`Acquisition: {acquirer, target, price, date}`), extract into that schema, reject events
missing the trigger evidence.

**Event-logic graphs (事理图谱):** a distinctive idea from the course worth knowing — graphs
whose nodes are events and edges are causal/temporal/conditional relations between events
("rate hike → bond selloff"). Build one when the user asks "what leads to what," not just
"what is related to what."

## LLM extraction prompt pattern

One pass per stage (entities, then relations, then events), not one mega-prompt:

```
You are extracting knowledge for a graph with this ontology:
<ontology>   (verbatim from the project's ontology file)

From the text below, extract every entity matching the ontology types.
For each: {surface, canonical, type, evidence: "<exact sentence>", confidence: high|med|low}
Rules:
- Only types from the ontology. Unknown-but-recurring concepts → list separately under "candidates".
- Evidence must be a verbatim quote containing the mention.
- Do not merge distinct mentions; deduplication happens later.
<text>
```

Relation pass adds: the recognized-entity list, the relation inventory with domain/range, and
"assert only relations the evidence sentence states directly."

Chunking: overlap chunks by 10-15% so sentence-boundary entities aren't lost; extract per
chunk; fusion (stage 8) reconciles.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Graph full of `Concept`/`Thing` nodes | Extracting without ontology | Stage 3 first; re-extract |
| Same person as 4 nodes | Skipped canonical-form rule | Define rule in ontology; fusion pass |
| Confident wrong relations | Co-occurrence treated as assertion | Evidence-quote requirement + domain/range validation |
| Events flattened to edge soup | No event schema | First-class event nodes with argument schemas |
| Precision collapses at scale | Prompt drift across doc types | Per-source-type prompts; stage 7 gate per source |
