---
name: evidence-reports
description: "Use when writing research reports with checked claims."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Reports, Citations, Evidence, Sources]
    category: research
---

# Evidence-Gated Research Reports

Use this skill to turn multi-source research into a self-contained report that a
reader can verify. The report separates facts, attributed statements, analysis,
and unknowns. It records enough source text for another person to check important
claims without trusting the writer's memory.

The workflow works with ordinary Hermes tools and local files. It does not require
a private service, a particular citation manager, or another unpublished skill.

## When to use it

Use it for:

- research briefs, backgrounders, timelines, and comparison reports;
- current-state research that combines several websites or documents;
- reports with disputed claims, important figures, or a high cost of error;
- research handed to another person or agent for design, implementation, or review.

For a quick answer based on one easily checked source, use normal citations instead.

## Deliverables

Create one project directory. At minimum, produce:

- `README.md`: scope, question, date of research, source-selection rules, and
  limitations;
- `RESEARCH_REPORT.md`: the content authority and final report;
- `sources/`: saved source text or downloaded documents when available;
- `SOURCE_INDEX.md`: source IDs, titles, URLs, retrieval dates, and notes;
- `EVIDENCE_LOG.md`: claim-to-source mapping for material claims.

Add `HANDOFF_*.md` files only when another role or agent needs explicit
instructions. A handoff must name the report as its content authority and must say
that downstream edits may not change claims or citations without review.

Keep all paths relative to the project directory unless a handoff is being passed
to a process that cannot resolve relative paths.

## Procedure

### 1. Define the question and evidence bar

Write the research question before searching. State:

- what the report will and will not cover;
- the relevant time range and geography;
- what counts as a primary, secondary, or community source;
- which claims require direct quotations, a second source, or both;
- what would remain unresolved if evidence is unavailable.

Do not silently turn a broad question into the narrow question that happened to be
easier to search.

### 2. Search broadly, then select deliberately

Search one query per sub-question, time period, or competing interpretation. Use
several source types when the question warrants it: official documentation,
regulators or public institutions, original papers, reputable reporting, project
repositories, and clearly labeled first-person accounts.

With Hermes, start with `web_search` and use `web_extract` for pages that support
load-bearing claims. If a route is unavailable, use one of these fallbacks:

1. open the page in the browser and save or copy the relevant text;
2. fetch a public HTML, PDF, or plain-text URL with `curl` or another available
   command-line client;
3. use an accessible mirror or an official alternate format;
4. record the failed URL and the missing evidence instead of inventing a result.

Search-result snippets are leads, not evidence for claims that require the page
body. Record why a source was selected and note obvious gaps or access failures.

Example source set for a report on Python's release support policy:

- Python Developer's Guide, `https://devguide.python.org/versions/` (primary);
- the relevant Python release page or PEP (primary);
- a reputable technical article only for context or reported user impact.

Example source set for a public-health backgrounder:

- a national public-health agency or WHO page (primary/institutional);
- the cited study or dataset (primary);
- reputable reporting for chronology or public response.

### 3. Register sources as you retrieve them

Give each source a stable ID immediately, such as `S01`, `S02`, and so on. Add an
entry to `SOURCE_INDEX.md` before drafting:

```markdown
| ID | Title | URL | Retrieved | Type | Used for |
|---|---|---|---|---|---|
| S01 | Python versions | https://devguide.python.org/versions/ | 2026-09-22 | official docs | support dates |
```

Use the retrieval date supplied by the environment, not a guessed date. Preserve
the exact URL used. If a URL redirects, record both the requested URL and the final
URL when the tools expose both.

If a citation-ledger tool is available, it may own the ID-to-URL mapping. Treat its
IDs as immutable, register URLs at retrieval time, and generate the final Sources
block from the ledger. The report must still be understandable if that optional
tool is absent.

### 4. Archive and inspect source text

Save the page text, PDF, or relevant excerpt in `sources/` when licensing and
access permit. Use a descriptive, filesystem-safe filename such as
`sources/S01-python-versions.txt`. Keep the original URL and retrieval date in the
file header or `SOURCE_INDEX.md`.

For a long page, archive the complete accessible text before selecting quotes. For
a short page, an excerpt is acceptable when it includes the heading, URL, date, and
the full sentences needed to check the claim. For a scanned document, mark the
text as OCR and inspect important numbers against the page image when possible.

If extraction fails, do not treat an unseen page as read. Mark the source
`unverified-access` and either find an alternate source or narrow the claim.

### 5. Build an evidence log before synthesis

For each material claim, record:

| Claim ID | Claim | Source IDs | Exact supporting text or location | Status |
|---|---|---|---|---|
| C01 | Python 3.9 is in security-fix-only status. | S01 | section “Status of Python versions” | verified |

Use these statuses:

- `verified`: the source directly supports the wording;
- `corroborated`: independent sources support the same claim;
- `attributed`: a source reports the statement, but it is not independently proved;
- `contested`: credible sources disagree;
- `open`: no adequate evidence was found;
- `inferred`: analysis derived from cited facts, not a direct source statement.

For high-stakes claims, copy the exact sentence or a precise page/section/table
location. Never create a quote from memory. If the source is dynamic, record the
access date and the page heading or anchor used.

### 6. Write with clear evidence boundaries

Separate these registers in the report:

- **Documented fact:** “The release page lists 14 March as the release date.”
- **Attributed report:** “The regulator reports 12 incidents.”
- **Analysis:** “This suggests the change affected maintenance work first.”
- **Tradition or allegation:** “The account is repeated in later sources, but no
  contemporaneous record was located.”
- **Unknown:** “The available sources do not establish the subgroup total.”

Put a citation immediately after the sentence or table cell it supports. A source
can support only what it actually says. Do not convert a regional, aggregate, or
modeled figure into a subgroup-specific figure unless the source does so.

When sources disagree, state the disagreement, cite each reading, and explain the
basis for any weighting. Do not silently choose the more convenient number. Do not
present a search snippet, an uncited model recollection, or a plausible inference as
settled fact.

### 7. Apply the evidence gate

Before delivery, check every material sentence and table row:

- Is its provenance visible through a source ID, inline citation, or explicit
  `unverified`/`open` marker?
- Does the source text support the exact scope, date, number, and certainty of the
  wording?
- Are important claims supported by a second independent source when the question
  calls for corroboration?
- Are quotes copied from archived source text rather than retyped?
- Are conflicting accounts labeled rather than merged?
- Are limitations and failed retrievals visible?
- Does `SOURCE_INDEX.md` contain every cited source, with no unused source silently
  presented as support?

A simple manual gate is enough when no ledger script is available. A small local
script may also check that every citation ID in the report appears in
`SOURCE_INDEX.md` and that every evidence-log source ID exists there.

If an optional citation ledger is installed, run its verification command after
this manual check. Render the Sources block from the ledger rather than editing it
by hand, then re-run verification after every report edit.

### 8. Hand off through files

A downstream brief should contain:

```markdown
# Handoff: design

Input authority: ./RESEARCH_REPORT.md
Output: ./DESIGN_SPEC.md

Use the report's claims and source IDs exactly. Do not add, remove, or rewrite
claims. Put presentation choices and unresolved questions in the design spec.
Before reporting done, check that every factual statement in the output points back
to a report section or source ID.
```

Dispatch dependent work only after the preceding artifact exists and passes the
evidence gate. Verify the final artifact from disk, not from a chat summary.

## Minimal report structure

```markdown
# Research Report: <question>

- Scope:
- Research date:
- Evidence boundary:
- Executive finding:

## Findings

### <sub-question>

Claim with an inline source marker. [S01]

## Disputed or limited findings

State competing readings and what each source supports. [S02][S03]

## Open questions

List claims that remain unverified and why.

## Method and limitations

Describe search coverage, access failures, and selection rules.

## Sources

- [S01] Title. URL. Retrieved YYYY-MM-DD.
```

## Common failure modes

- **Registering after drafting:** source IDs and URLs then come from memory. Register
  at retrieval time.
- **Using search snippets as proof:** extract or open the source body first.
- **Hand-editing generated citations:** regenerate the Sources block from the source
  index or optional ledger.
- **Quoting a paraphrase:** copy the exact wording from archived text and record its
  location.
- **Treating one source as universal proof:** match the claim's scope to the source's
  scope and seek independent corroboration when needed.
- **Hiding access failures:** label the source unavailable and reduce the claim.
- **Letting a downstream editor improve the facts:** downstream work may change
  layout or presentation, not the report's claims or source IDs.
- **Leaving stale claims after correction:** edit or remove the old claim and update
  its evidence-log row; do not rely on a later note to cancel it.

## Completion criteria

The report is ready when the project contains the required files, every material
claim has a visible evidence status, every citation resolves to a source entry,
important quotes or locations are preserved, disagreements and gaps are explicit,
and a fresh reader can reproduce the main checks from the project directory.
