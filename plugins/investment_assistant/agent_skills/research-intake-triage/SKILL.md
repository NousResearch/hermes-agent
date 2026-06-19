---
name: research-intake-triage
description: Select filing analyses to read before deep research.
version: 0.1.0
---

# Research Intake Triage

Use this skill after candidate triage and before deep research. It decides
which offline filing-analysis materials should be opened for detailed reading.
It does not generate filing summaries, does not analyze full filings, and does
not produce portfolio weights.

## Inputs

- candidate triage artifact
- research budget allocations and layer audits
- lightweight Futu evidence
- symbol data manifests
- company profiles
- SEC companyfacts key metrics
- filing metadata and filing-summary freshness
- optional user constraints or must-read symbols

If the task needs the local filing data layout, read
`references/offline-filing-data-layout.md`.

## Purpose

Triage decides broad research budget. Research intake decides which existing
offline materials are worth opening now. Keep this stage cheap and narrow:

- Use compact facts first: manifest status, company profile, exact numeric
  metrics, filing dates, and filing-summary freshness.
- Treat `filing_summary.md` and other long narrative artifacts as expensive
  reads.
- Select a smaller `must_read_filing_analysis` set from the deep queue.
- Promote only a few watchlist exceptions when lightweight evidence suggests a
  possible omission.
- Keep deferred and rejected candidates closed unless the user explicitly asks
  otherwise.

## Required Process

1. Preserve the triage evidence boundary.
   - Use the triage artifact as the universe boundary.
   - Do not recover symbols from model memory or discovery output directly.
   - Do not invent missing financial facts.

2. Build a light index for each candidate.
   - Read manifest layer status and warnings.
   - Read company profile fields such as business description, sector,
     industry, ETF/fund flags, market cap, beta, exchange, and trading status.
   - Read SEC companyfacts key metrics and provenance.
   - Read filing metadata dates and forms.
   - Read filing-summary meta status without opening the long markdown unless
     the candidate is selected for reading.

3. Decide filing-read budget by layer.
   - Start from `research_budget_allocations`.
   - Allocate reads to layers where fundamentals or business mix can change the
     conclusion: semiconductors, memory/storage, optical/networking, power,
     cloud infrastructure, software, and recent listings.
   - ETF anchors usually need ETF holdings/overlap data rather than SEC filing
     analysis.
   - Mature mega-cap anchors may need fewer filing reads unless the layer
     tradeoff depends on segment quality, capex, margins, or AI exposure.

4. Select read actions.
   - `must_read_filing_analysis`: open the offline filing analysis before deep
     research.
   - `optional_read_filing_analysis`: useful if budget remains.
   - `profile_metrics_only`: current company profile and exact metrics are
     sufficient for this pass.
   - `promote_from_watchlist_for_reading`: watchlist exception that deserves a
     filing-analysis read.
   - `do_not_read_yet`: defer because evidence is weak, data is stale, or the
     candidate is outside current research budget.

5. Explain skipped reads.
   - Do not silently skip high-priority deep candidates.
   - For each skipped deep candidate, explain whether the reason is ETF
     structure, duplicate exposure, missing/stale materials, lower layer
     priority, or enough light evidence for now.

## Output Artifact

Produce `research_intake_triage` with:

- generated_at
- source_artifacts
- intake_summary
- layer_read_budgets
- must_read_filing_analysis
- optional_read_filing_analysis
- profile_metrics_only
- promote_from_watchlist_for_reading
- do_not_read_yet
- stale_or_missing_materials
- skipped_deep_candidate_audit
- data_gaps
- warnings

For each symbol decision include:

- symbol
- original triage bucket
- layer_keys
- priority
- read_action
- available_light_materials
- missing_or_stale_materials
- reason

## Boundaries

- Do not produce final promote/keep/demote/reject investment judgments.
- Do not assign target weights, price targets, or trade actions.
- Do not open or summarize full filing sections in this stage.
- Do not use narrative filing text as the source for exact financial numbers.
- Do not treat unavailable profile or metrics data as positive evidence.
