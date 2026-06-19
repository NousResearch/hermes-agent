---
name: fundamental-analysis
description: Assess candidates using SEC and financial artifacts.
version: 0.1.0
---

# Fundamental Analysis

Use this skill after a candidate pool exists and fundamental evidence needs to
be interpreted. The Fundamental Agent judges whether company fundamentals
support the theme role. It does not perform technical analysis, portfolio
weighting, or trade planning.

## Inputs

- theme discovery and candidate pool artifacts
- SEC/companyfacts context
- filing metadata and event freshness
- optional earnings release or call-summary artifacts when implemented
- freshness report and data gaps

## Required Process

1. Check freshness before analysis.
   - Inspect whether fundamental context exists for each candidate.
   - If context is missing or cache-stale, request one refresh through the
     deterministic data tool.
   - If the provider has no fresher source data, mark stale_source and continue
     with lower confidence.

2. Separate facts from interpretation.
   - Numeric financial claims must come from structured artifact fields.
   - Narrative summaries can support qualitative interpretation, but cannot be
     the source for exact financial numbers.
   - Missing values remain missing.

3. Assess each candidate.
   - Theme fit: direct, enabling, second-order, or weak exposure.
   - Revenue quality: scale, growth, cyclicality, segment relevance when known.
   - Profitability: margin, operating income, net income, operating leverage.
   - Balance sheet: assets, liabilities, debt burden, resilience.
   - Cash generation: operating cash flow and capex when available.
   - Filing/event risk: stale filings, recent 8-K, earnings uncertainty.
   - Data gaps: fields or sources that block confidence.

4. Assign candidate action.
   - keep: fundamentals support the theme role enough for thesis synthesis.
   - watch: relevant but evidence, valuation, profitability, or freshness is
     incomplete.
   - exclude: evidence contradicts the theme role or quality is unacceptable.
   - needs_refresh: analysis should pause because key data is missing and a
     refresh path exists.

## Output Artifact

Produce `fundamental_analysis` with per-candidate:

- symbol and theme layer
- theme fit
- financial quality
- evidence strength
- cycle position if inferable from artifacts
- key evidence
- red flags
- data gaps
- candidate action
- rationale

Also include global warnings, stale-source summary, missing-source summary, and
artifact ids used.

## Boundaries

- Do not produce target weights.
- Do not recommend trades.
- Do not use technical indicators as fundamental evidence.
- Do not turn unavailable data into positive evidence.
