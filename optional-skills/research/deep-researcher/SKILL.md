---
name: deep-researcher
description: Track research claims and resume evidence dossiers.
version: 1.0.0
author: Brent Wilkins (BrentWilkins)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, web-search, web-extraction, evidence, citations]
    category: research
    related_skills: [grounded-citations, research-paper-writing, arxiv, blogwatcher]
---

# Deep Researcher Skill

Search discovers candidates; retrieved passages support claims; the ledger controls synthesis.
Never use search snippets or model memory to fill a missing source-backed conclusion.
Use native `web_search`, `web_extract`, `browser_navigate`, `read_file`,
`write_file`, and `patch` tools. Do not type tool calls
into a terminal or substitute direct search-engine scraping. Scripts below are local
bookkeeping helpers, not alternative retrieval backends.

## When to Use

Use for substantive multi-source investigations and comparisons requiring
persistent claims, contradiction analysis, or work across sessions. Simple
lookups, single-claim checks, and synthesis of supplied documents usually need
a smaller workflow such as `grounded-citations`.

## Prerequisites

Python 3.10+ for the standard-library bookkeeping scripts, `terminal` for running
them, and native retrieval tools configured in Hermes. No extra Python packages,
API keys, or research-access plugin are required by this skill itself. Retrieval
services may have their own setup requirements.

## How to Run

Invoke the commands below through `terminal`. Replace SKILL with this skill's
absolute directory and RUN with a new dedicated research directory; quote paths
containing spaces. Use the available Python interpreter (`python3` below).

```sh
python3 SKILL/scripts/research_run.py init RUN --question "Research question" --scope "Audience, timeframe, exclusions" --mode standard
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Topic" --question "Q1: Question"
```

## Quick Reference

- `research_run.py init|checkpoint|status`: initialize, save progress, or resume.
- `evidence_ledger.py add-source|add-claim`: record passages and linked claims.
- `evidence_ledger.py check|brief`: validate references and derive a compact brief.
- `audit_report.py REPORT --ledger LEDGER --no-live`: offline citation audit.
- `research_run.py archive|restore`: verified lossless archival and recovery.

All helpers live under `scripts/`; each command supports `--help`.
Read [run management](references/run-management.md) for lifecycle commands.

## Procedure

### 1. Frame and budget

Establish the decision, audience, scope, timeframe, and exclusions. Ask only when an
unresolved choice materially changes the task; otherwise record assumptions and proceed.
Decompose into focused questions, including material counterarguments or failure modes.
Define what evidence would answer each question and which source types can establish it.

Use a new dedicated run directory. Resolve this skill's directory and the run directory
to absolute paths; commands below use SKILL and RUN as placeholders, not persistent shell
variables. Read [run management](references/run-management.md) when initializing, resuming,
or archiving a run.

Planning defaults (override to match the task and record the effective values):
- Quick: 4 search calls, about 4–6 useful sources.
- Standard: 8 search calls, about 8–15 useful sources.
- Deep: 12 search calls, about 15–25 useful sources.
- Stop after four search/extraction/gap-analysis iterations by default. An explicit zero
  removes the iteration ceiling only; search, context, and storage budgets still apply.
- Reserve at most two additional searches for a specific high-impact gap after extraction.
- Reserve roughly one quarter of available context for synthesis and audit. Extract at
  most 8,000 characters per page initially; read relevant cached sections as needed.
  Keep raw excerpts in context below roughly 35% of the available window and discovery/tool
  overhead near 10%; reduce collection further when the session already has substantial history.
- Use a 64 MiB warning budget for registered working artifacts, configurable at initialization.
  This is a collection guideline, not a downloader quota or an OS disk limit.

Counts are ceilings or planning estimates, never success targets. Stop once material
questions are adequately answered or further collection is unlikely to change the result.
Expose unresolved gaps when the budget is exhausted.
In quick mode, a decisive claim needs a strong primary source or independent corroboration.
In standard/deep modes, seek two independently assessed origins per major claim and label
single-source exceptions. Deep mode also requires a counter-evidence search per major theme.

### 2. Discover and retrieve

1. Search before extraction unless URLs were supplied. Start each evidence gap with one
   concise query. Try a materially broader query after an empty result; stop that gap
   after a second empty result. Stop discovery after four empty responses across the report.
   Before each follow-up, compare its evidence gap and candidate URLs with prior queries;
   a reworded query is not a new evidence need. Stop discovery after two searches add no new
   credible canonical URLs. Synthesize or report the gap; do not silently raise budgets to
   avoid stopping. Record user-requested scope/budget changes in the run state.
2. Keep a compact search log: query, gap, outcome, newly shortlisted URLs. Do not retain
   repeated snippets as evidence. Favor primary evidence for facts and independent work
   for interpretation. Popularity and domain suffix are not quality scores.
   Assess directness, author authority, methodological transparency, independence, recency,
   and incentives/conflicts. Do not compare options on dimensions lacking comparable evidence.
3. Extract at most three promising canonical URLs per batch. Record actual citation identity,
   retrieval date, relevant passages, coverage, and limitations. Never invent metadata.
4. A successful HTTP response containing a cookie wall, challenge, or empty body is a
   retrieval failure. For permitted sites, try an available native browser session or an
   authoritative alternative (canonical page, print/PDF, original study, official API, or
   repository copy). Use at most one browser fallback per failed URL and two consent interactions,
   preferring necessary-only consent, then record persistent blocks. Respect site
   access policies; tool availability does not authorize paid services or new accounts.
5. Abstract-only retrieval cannot establish unreported full-paper methods or results.
   Full-text availability does not establish study quality, and cached but unread sections
   have not been reviewed. For clinical claims, follow secondary leads to underlying
   studies, reviews, guidelines, or regulator documents.
   If the underlying clinical claim cannot be verified, omit it from clinical conclusions
   or explicitly label it unverified. Record retrieval method, timestamp, and article identifiers
   when available. Keep extracted, failed, considered, and cited sources distinct.
6. Deduplicate by source identity as well as URL. Copies, syndicated reports, and articles
   relying on the same study share an origin group even on different domains.

Native retrieval tools own extraction; do not bypass them with terminal
Firecrawl calls, inline Python, or generated extraction scripts. Invoke the
bundled bookkeeping scripts through `terminal`.

### 3. Record evidence and checkpoint

Initialize the run manifest first, then the ledger:

```sh
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Topic" --question "Q1: Question"
```

After each extraction batch, add concise source records and atomic claims using
`evidence_ledger.py add-source` and `add-claim`. See `--help` and the
[worked example](references/worked-example.md) for paired passages and locators.

- Use repeatable `--evidence '{"excerpt":"Exact short passage","locator":"Section/page/paragraph"}'`
  on sources. Locators may include a registered run-relative cache path and line numbers.
  Record a separate excerpt/locator pair for each passage.
- Link claims with `--evidence S1E1` and the corresponding `--source 1` or
  `--counter-source 1`. Evidence IDs remain stable within the ledger.
- Independence starts unverified. Set both `--independence-group` and
  `--independence-note` only after tracing the underlying data or origin. Group IDs are
  assessments, not proof; the script cannot establish semantic independence.
- Keep excerpts short and relevant (usually one to three per source). Prefer source
  locators over entire page copies; retain a single local text copy only when needed for
  audit or offline reproducibility. Do not collect browser profiles, cookies, or credentials.
- Checkpoint completed work, current gaps, the next action, and artifact paths after each
  batch or before context compaction. Register every retained run artifact. Resume from
  the manifest and compact brief, opening only the passages needed for the next decision.

Older ledgers remain readable. Legacy domain-based groups count as unverified until an
explicit assessment and note are recorded. Do not infer independence while migrating.
For corrections, use the native file editor on the ledger and rerun its checks.

### 4. Check gaps and synthesize

Run `evidence_ledger.py check RUN/evidence.json` and
`evidence_ledger.py brief RUN/evidence.json`; save the brief as RUN/evidence-brief.md.
Checks warn about missing corroboration and locators; they do not prove truth.
The brief defaults to at most 24,000 characters. If it reports truncation, read relevant
ledger records before synthesis; use `--max-chars` only when the context budget allows.
Seek counter-evidence for decisive claims and inspect conflicts in definitions, populations,
dates, methods, and incentives. Sources repeating one origin do not supply independent support.
After each batch, check unanswered questions, interested-party-only support, stale evidence,
disputed claims without counter-evidence, and recommendations missing tradeoffs or downsides.
Resolve the most consequential gap first. Formulate the strongest plausible contrary claim
before searching for evidence that would support it.

Draft from the brief and ledger using [the dossier template](references/dossier-template.md),
adapting its length and sections to the user's request. Keep these distinctions visible:
sourced fact, interpretation, analyst inference, recommendation, and unresolved gap.
Place citations next to the exact claims they support. Summarize uncertainty and what would
change the conclusion. Count only successfully retrieved sources actually used in the report.
Each paragraph-end citation must support the whole paragraph; otherwise split the claims.
List excluded and failed candidates separately. Calibrate confidence per finding:
- High: direct evidence with independent corroboration and no serious unresolved contradiction.
- Medium: credible evidence with limitations in independence, coverage, recency, or methods.
- Low: sparse, indirect, disputed, or materially uncertain evidence.

### 5. Audit and finish

Run `audit_report.py RUN/report.md --ledger RUN/evidence.json` (use `--no-live` for
offline structural checking). Review warnings and inspect each major claim's exact scope,
tense, supporting passage, opposing evidence, and independence assessment. A reachable URL
or a clean script result does not establish semantic support.
Fix structural errors, explain unresolved warnings, and remember that a blocked live link alone
does not invalidate a source already retrieved. Revise once for accuracy and once for usefulness.

Check the executive summary against the findings, revise, and mark the run complete only
after the requested report and audit are finished. Then offer or perform lossless archiving
when requested, using [run management](references/run-management.md). Keep report.md readable.
Compression reduces disk storage, not token usage; read selectively rather than inflating
an archive into model context. Never archive an active run or prune unregistered artifacts.

## Pitfalls

- Different hostnames do not establish independent evidence.
- Ledger checks validate structure, not truth or verbatim passage accuracy;
  compare recorded excerpts with retrieved text during the semantic audit.
- Archives are not encrypted backups. Never include credentials or browser state.
- Compression saves disk space, not context; the storage budget is a warning,
  not a hard quota. Missing optional codecs fall back without installing packages.

## Verification

After writing the report, run this offline check through `terminal`:

```sh
python3 SKILL/scripts/audit_report.py RUN/report.md --ledger RUN/evidence.json --no-live
```

Fix structural errors and review every material claim against its supporting
passages; a clean exit does not certify the report's conclusions.

Maintained at [BrentWilkins/research-skills](https://github.com/BrentWilkins/research-skills).
Related design references: [SkillMedev's deep-research skill](https://github.com/SkillMedev/skills/blob/main/skills/deep-research/SKILL.md)
(scope and examples) and [LovStudio's deep-research skill](https://github.com/lovstudio/deep-research-skill/blob/main/SKILL.md)
(evidence locators and run manifests).
