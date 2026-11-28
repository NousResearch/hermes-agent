# Typed Graph Mode (opt-in)

Read this only when the user explicitly asks for graph-shaped work on their wiki: typed
relations ("who ACQUIRED whom"), multi-hop questions ("who worked with X on projects using
Y"), merging duplicate entities ("SEU" and "Southeast University" are one page), or event
timelines. Ordinary ingest, query and lint stay exactly as SKILL.md describes. Nothing here
runs automatically, and a wiki whose `SCHEMA.md` has no `## Relations` section is not in
graph mode.

## Ground rules

- **Opt-in per wiki.** Graph mode is on only after the user approves a `## Relations`
  section in `SCHEMA.md`. Never add it on your own initiative.
- **Additive.** Typed relations live in a new optional `relations:` frontmatter field.
  Existing `[[wikilinks]]`, page bodies, tags and thresholds are untouched; Obsidian keeps
  working.
- **Merges and bulk rewrites need approval.** Propose; the user confirms; then apply. An
  erroneous merge silently fuses two entities' facts and is far worse than a missed one.
- **No retrofit sweeps.** Do not back-fill `relations:` across an existing wiki unless the
  user asks for that scope explicitly (SKILL.md's 10+ pages rule still applies).

## 1. Scope test

Before proposing graph mode, check a graph actually beats the plain wiki. It pays off when
questions are multi-hop, when the same entities recur across many sources, or when the
relationships ARE the data. If the user's questions are single-page lookups, say so and stop.

## 2. Relations schema

Start from 5-10 **competency questions**: the questions the graph must answer. They are the
spec and the test suite. Then propose this section for `SCHEMA.md`:

```markdown
## Relations
<!-- Graph mode. Entity types are page tags from the taxonomy. -->
Competency questions:
- Which engineers contributed to services that had incidents last quarter?

| Relation | From (tag) | To (tag) | Notes |
|---|---|---|---|
| CONTRIBUTED_TO | person | service | |
| AFFECTED | incident | service | attrs: severity |

Canonical names: full official name, English; other forms go in `aliases:`.
Events: incident {service, start, resolved, severity}
```

Rules:
- Precise verb names (`ACQUIRED`, `CITES`, `DEPENDS_ON`), never `RELATED_TO`.
- Every relation has a domain and range expressed as existing taxonomy tags.
- Start minimal (10-30 relations). Walk each competency question through the table; a
  question you cannot path through means a missing relation, not a reason to add ten.
- If one relation keeps needing a qualifier (`role: author | editor`), split it in two.
- Canonical-name rule is written here; merging (step 5) enforces it.
- Recurring relations that do not fit go in a `Candidate relations:` list for the user to
  promote later, never forced into a wrong type.

To draft the table from existing content, read 3-5 representative pages or raw sources and
propose relations **with evidence quotes**; the user prunes. Never auto-accept an induced schema.

## 3. Typed relations on pages

On ingest in graph mode, record each relation the source asserts on the subject's page:

```yaml
relations:
  - rel: ACQUIRED
    to: "[[target-company]]"
    source: raw/articles/acme-news-2026.md
    evidence: "Acme completed its acquisition of Target Co. on 3 March."
    since: 2026-03-03          # optional validity start
    confidence: high
aliases: [Acme Inc., ACME]     # optional, other surface forms of this entity
```

- Extract relations only between pages that exist or meet the Page Thresholds. Relation
  extraction never invents entities.
- **Domain/range check:** the subject page's tags must include the relation's From tag, the
  target's the To tag. A mismatch is rejected, not stored.
- **Evidence must assert the relation.** Co-occurrence is not a relation ("Musk discussed
  Twitter" is not `OWNS`). No evidence quote, no relation.
- Still add the plain `[[wikilink]]` in the body as usual; `relations:` is metadata on top.
- Conflicting facts: keep both with `since`/`source` and follow the Update Policy. Facts
  change; the graph records the change instead of overwriting it.

**Events** (news, incidents, funding rounds): only when the schema lists event types. An
event is its own page (`type: entity`, tag `event`, added to the taxonomy first) with typed arguments in frontmatter
(`args: {acquirer: "[[a]]", target: "[[b]]", price: ..., date: ...}`). Never flatten a
4-argument event into pairwise relations; you lose which deal had which price.

## 4. Quality gate

After the first graph-mode ingest (and per new source type), sample 50 new relations (all of
them if fewer) and check each against its evidence quote: is the entity real and correctly
tagged, does the sentence actually state the relation? Target 90%+ precision. Below that,
fix how you are extracting (prompt, schema definitions), remove the bad relations, and
re-run; recall improves with later passes, bad precision poisons every query.

## 5. Alias merging

Duplicates are the main reason graphs give confidently wrong answers: a path breaks at
"SEU" vs "Southeast University". Run this as an explicit operation, not on every ingest.

1. **Block** with `execute_code`: group pages by same type tag plus a cheap key (shared
   name token, acronym match, normalized name, `aliases:` overlap). Compare only within a
   block.
2. **Match** on layered evidence: names and aliases; compatible attributes (dates,
   domains); **shared neighbours** (two "J. Smith" pages linking to the same three
   coauthors are one person; same name, disjoint neighbours are two people).
   Acronyms: initial-letter rules miss intra-word acronyms ("SEU" for So·uth·east
   University yields "SU"), so use alias lists, subsequence matching or your own judgment.
3. **Propose** a merge list to the user: pair, evidence, confidence. Apply only approved
   pairs.
4. **Apply** each merge: keep the canonical page per the schema rule; move content and
   `relations:` over, union `aliases:` and `sources:`, keep conflicting values side by side
   with sources; record `merged_from: [old-slug]`; archive the other page through the
   Archiving procedure; rewrite inbound `[[old-slug]]` links; log
   `## [YYYY-MM-DD] merge | old-slug → canonical-slug`.

## 6. Graph queries

For multi-hop questions, build a throwaway edge table from frontmatter with `execute_code`
(walk `entities/`, `concepts/`, read each page's `relations:`, emit
`subject|rel|object|source|since` rows). Treat it as derived data: rebuild per query, never
hand-edit, never commit it as a source of truth.

- Link the question's entities to pages (names and `aliases:`).
- Retrieve **paths** between them, 1-2 hops; past 2 hops results turn to noise.
- Answer from the path, citing each hop's page and source: "[[a]] ACQUIRED [[b]]
  (raw/articles/x.md) ..." File substantial answers to `queries/` as usual.
- "What are the big themes" questions: cluster by shared relations and summarize per
  cluster instead of dumping the graph.

## 7. Extra lint checks (graph mode only)

Add to the normal lint report:
- relations whose tags violate the From/To columns
- relations missing `evidence` or `source`, or pointing at missing pages
- relation names not in the schema table
- two pages sharing an alias (merge candidates; report, never auto-merge)

## Pitfalls

- Turning graph mode on for a user who asked a single lookup question.
- Writing `relations:` on pages the user did not ask to touch.
- Auto-merging. Merges are always proposed first.
- Vague relation names; they make every later query ambiguous.
- Storing relations without evidence; you can neither audit nor merge them later.

## Credits

Adapted from [codejunkie99/graph-engineering](https://github.com/codejunkie99/graph-engineering),
itself distilled from Southeast University's graduate Knowledge Graph course
([npubird/KnowledgeGraphCourse](https://github.com/npubird/KnowledgeGraphCourse), Prof. Peng
Wang). Upstream license:

> MIT License
>
> Copyright (c) 2026 codejunkie99
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
> to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
> DEALINGS IN THE SOFTWARE.
