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
`write_file`, and `patch` tools. When `execute_code` is available, batch discovery
and extraction by calling the same native tools from code (`from hermes_tools import
web_search, web_extract`): see [search as code](references/search-as-code.md).
Never substitute direct search-engine scraping, `curl`, or vendor SDK calls. Scripts
below are local bookkeeping helpers, not alternative retrieval backends.

## When to Use

Use for substantive multi-source investigations and comparisons requiring
persistent claims, contradiction analysis, or work across sessions. Simple
lookups, single-claim checks, and synthesis of supplied documents usually need
a smaller workflow such as `grounded-citations`.

## Prerequisites

Python 3.11+ for the standard-library bookkeeping scripts, `terminal` for running
them, and native retrieval tools configured in Hermes. No extra Python packages,
API keys, or research-access plugin are required by this skill itself. Retrieval
services may have their own setup requirements.

## How to Run

Invoke the commands below through `terminal`. Replace SKILL with this skill's
absolute directory and RUN with a new dedicated research directory; quote paths
containing spaces. Use the available Python interpreter (`python3` below).

```sh
python3 SKILL/scripts/research_run.py init RUN --question "Research question" --scope "Audience, timeframe, exclusions" --mode standard --execution direct
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Topic" --question "Q1: Question"
```

## Quick Reference

- `research_run.py init|checkpoint|status`: initialize, save progress, or resume.
- `research_run.py budget|reserve-search|record-search`: persist the effective
  allowance and actual query receipts before native retrieval.
- `evidence_ledger.py add-batch LEDGER PAYLOAD`: validate and commit a retry-safe
  JSON batch; [workflow contracts](references/workflow-contracts.md) gives its schema.
- `evidence_ledger.py add-source|add-claim`: individual additions remain available.
- `evidence_ledger.py check|brief LEDGER`: validate and derive a compact brief.
- `passages.py CACHE --term TEXT --max-chars 4000`: bounded literal character windows.
- `semantic_review.py prepare|show|template|apply|check RUN`: optional audit packet,
  only when the user requests a separate deeper review.
- `audit_report.py REPORT --ledger LEDGER --no-live`: offline citation audit.
- `research_run.py archive|restore`: verified lossless archival and recovery.
- [Search as code](references/search-as-code.md): tested native-tool discovery,
  passage selection, recording, and recovery; no generated bookkeeping loops.

`--question` takes the full question text or its leading ID (`Q1`). The report audit
expects inline numeric markers (`[1]`, `[2-3]`) and a final `## References` (or
`## Citation Links` / `## Bibliography`) section of numbered entries
`1. [Title](https://url)` whose URLs match ledger sources.
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

Initialize with `--execution direct|batched`; use `--search-budget N` for an explicit
allowance. The saved manifest and printed usage govern the plan. Planning defaults:
- Quick: 4 queries one at a time, or 6 in one search-as-code round; about 4–6 useful sources.
- Standard: 8 queries one at a time, or 16 across at most two rounds; about 8–15 useful sources.
- Deep: 12 queries one at a time, or 40 across at most three rounds; about 15–25 useful
  sources. Give independent workers distinct runs; never concurrently write one manifest.
- The larger batched budgets apply only when queries run inside `execute_code`: results
  are filtered in code, so extra queries add coverage without adding raw snippets to context.
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
2. Reserve every query before native retrieval and record its result afterward, using
   the helper protocol in [search as code](references/search-as-code.md). Failed calls
   consume allowance and remain failures, distinct from empty results. The manifest
   retains query, gap, outcome, and candidate URLs. Do not retain
   repeated snippets as evidence. Favor primary evidence for facts and independent work
   for interpretation. Popularity and domain suffix are not quality scores.
   Assess directness, author authority, methodological transparency, independence, recency,
   and incentives/conflicts. Do not compare options on dimensions lacking comparable evidence.
3. Extract at most three promising canonical URLs per batch (five per `web_extract`
   call inside a search-as-code round, with passages selected in code). Record actual citation identity,
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
Firecrawl calls, raw HTTP fetches, or scraping code. Code that calls
`hermes_tools.web_search`/`web_extract` inside `execute_code` uses the native tools
and is the preferred way to batch them. Invoke the bundled bookkeeping scripts
through `terminal`.

### 3. Record evidence and checkpoint

Initialize the run manifest first, then the ledger:

```sh
python3 SKILL/scripts/evidence_ledger.py init RUN/evidence.json --topic "Topic" --question "Q1: Question"
```

After each extraction batch, create a JSON payload with `write_file` and call
`evidence_ledger.py add-batch RUN/evidence.json RUN/batch.json` through `terminal`.
The helper assigns stable IDs and validates the whole staged ledger before saving;
identical retries return the receipt, while changed retries fail. See
[workflow contracts](references/workflow-contracts.md) for fields and examples.
Never add records by hand or generate subprocess loops to guess source IDs.

- Record exact short excerpts with a page/section/paragraph locator and link claims
  to those passage IDs. For flattened PDFs, use `passages.py` on native cached text
  to retain bounded windows with exact character offsets; a first-line slice can
  miss a decisive passage near the end of a huge line.
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

Check the executive summary against the findings. During drafting, verify material
numbers against their source periods, populations, units, and denominators; trace
organization names and shared data origins to the retained passages. Read relevant
footnotes and qualify uncertain inferences beside the affected claim. Correct actual
errors without generating a separate approval record or rereading every source by default.
Then run `research_run.py checkpoint RUN --status complete`. Completion requires
the report, valid nonempty ledger, brief, cited-source summaries, precise linked
passages, resolved search receipts, and citation audit. A clean script exit does
not establish factual accuracy.
Only when the user requests a separate deeper audit, use the optional review packet
in [workflow contracts](references/workflow-contracts.md). It is not a completion
requirement and has no demonstrated speed or quality benefit in the current local tests.
Archive only when requested, following [run management](references/run-management.md).
Archiving repeats completion checks; keep report.md readable.

## Pitfalls

- Different hostnames do not establish independent evidence. Original and
  secondary copies of one dataset keep the same origin group; source type
  records whether a publication is primary or secondary.
- Inspect source footnotes before claiming coverage details are unavailable.
  Say what was not retrieved in this run; do not turn missing evidence into
  claims that a dataset does not exist or information is not public.
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

Then run the completion checkpoint.
A clean exit records structural checks; it does not
certify passage authenticity or the report's conclusions.

Maintained at [BrentWilkins/research-skills](https://github.com/BrentWilkins/research-skills).
Related design references: [SkillMedev's deep-research skill](https://github.com/SkillMedev/skills/blob/main/skills/deep-research/SKILL.md)
(scope and examples) and [LovStudio's deep-research skill](https://github.com/lovstudio/deep-research-skill/blob/main/SKILL.md)
(evidence locators and run manifests).
