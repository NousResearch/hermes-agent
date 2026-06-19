---
name: candidate-triage
description: Enrich and triage candidates before deep research.
version: 0.1.0
---

# Candidate Triage

Use this skill after theme discovery. Candidate triage owns the full pre-research
candidate narrowing stage:

1. Plan and collect lightweight Futu evidence for discovered candidates.
2. Use that evidence to allocate downstream research budget across queues.

It does not author final portfolio weights.

## Inputs

- policy and user intent
- theme discovery artifact
- domain tree, coverage requirements, seed symbols, and required symbols
- executed Futu probes and filter audits
- research trace and search queries
- omissions to investigate
- lightweight and deep enrichment artifacts when already present
- available Futu quote, K-line, owner-plate, screener, and optional options
  data tools

## Purpose

Discovery optimizes for not missing investable branches. Enrichment-guided
triage optimizes research budget while data is being collected. Triage is not
the final stock-selection step. It decides which symbols deserve expensive deep
research now, which peers should stay visible, and which names should wait for
better evidence. Keep the goals separate, but let enrichment and triage run as
one staged process:

- Discovery may return 80-120 candidates to avoid missing important subdomains.
- Triage should first choose cheap checks for the broad set, then use those
  checks to allocate a deep-research budget, usually 25-40 candidates.
- Deep enrichment should run on the shortlist, required symbols, and selected
  watchlist exceptions, not on every discovery candidate.
- New enrichment evidence may promote, demote, defer, or reject a candidate.

## Required Process

1. Preserve the theme map.
   - Start from discovery layers and coverage requirements.
   - Do not collapse a layer just because another layer already has strong
     candidates.
   - Keep user-required symbols as constraints, not quality proof.

2. Build a research-budget frame before spending research budget.
   - Identify core layers, important layers, optional layers, and unresolved
     layers.
   - Decide an approximate lightweight-check and deep-research budget per layer.
   - Reserve room for bottlenecks, emerging candidates, and controversial names
     that require validation.
   - Record the final per-layer budget in `research_budget_allocations`.
   - Treat `deep_enrichment_queue` as research spend, not as an investable
     shortlist.

3. Plan lightweight Futu evidence collection.
   - Quote and snapshot should cover every discovered candidate when available:
     price, market cap, valuation fields, turnover, volume, bid/ask spread,
     quote validity, and quote freshness.
   - Daily K-line should cover every discovered candidate when available:
     trend, relative strength, return windows, realized volatility, drawdown,
     and moving-average context.
   - Owner-plate checks are useful when layer mapping or theme classification
     needs validation. ETF composition and overlap are not owner-plate tasks.
   - Options checks are selective. Use them only for liquid anchors or names
     where option availability/IV matters for later strategy design.
   - Do not ask for SEC filings, earnings transcripts, news summaries, current
     holdings, orders, or trade plans during lightweight triage.

4. Label data quality before triage.
   - Mark each candidate as fresh, partial, stale, unavailable, or
     quote_unavailable.
   - Record missing fields and source warnings.
   - Unsupported tickers remain visible only as important omissions, watchlist
     names, or deferred names; do not treat them as fully validated.

5. Run cheap checks before expensive checks.
   - For broad discovery candidates, prefer quote validity, market cap,
     liquidity, price trend, basic profitability, and source/probe support.
   - Use lightweight evidence to form a `deep_research_queue`.
   - Do not send every discovered candidate through SEC, earnings transcripts,
     long filings, or expensive event summarization.

6. Use evidence, not bare familiarity.
   - Prefer candidates supported by multiple probes, research sources, or layer
     audits.
   - Prefer candidates that represent distinct economic exposure, not only the
     largest familiar tickers.
   - Do not discard a plausible discovered candidate solely because a stronger
     same-layer candidate exists. Move it to watchlist unless there is a hard
     exclusion reason.
   - Scores can rank attention, but they do not author the investment thesis.
     Preserve weaker-scoring candidates when they cover a required layer or have
     important optionality.

7. Apply staged triage criteria.
   Consider these signals when available:
   - required symbol or must-consider flag
   - layer importance and coverage scarcity
   - probe count and quality of supporting probes
   - research source count and freshness
   - Futu symbol validity and quote availability
   - market cap and liquidity sanity
   - theme purity and directness of revenue exposure
   - duplicate share class or overlapping ETF/individual exposure
   - ADR, OTC, recent listing, or unsupported security type risk
   - whether the candidate needs SEC/fundamental validation before inclusion

8. Separate research-budget buckets from portfolio conclusions.
   - Use `deep_research_queue` for candidates that should receive deeper
     SEC/fundamental/event/technical/options work now.
   - Use `watchlist` for plausible candidates that should remain visible but do
     not need immediate full SEC/fundamental processing.
   - Use `defer` for candidates that need another data source or user direction
     before research spend.
   - Use `reject` only for hard reasons such as invalid symbol, clear theme
     mismatch, duplicate share class, unsupported security type, explicit user
     exclusion, or failed basic tradability/size gate.

9. Re-triage after deep research returns.
   - Promote candidates whose fundamentals or events confirm the thesis.
   - Demote candidates whose evidence is weak, stale, contradicted, or only
     price-momentum driven.
   - Keep unresolved but plausible names in watchlist instead of hiding them.

10. Audit omissions and retained watchlist names.
   - For every important layer, explain why coverage is sufficient or what is
     still missing.
   - For every high-salience omitted candidate, give a hard reason.
   - For candidates like storage, optical networking, power, advanced packaging,
     or other bottleneck branches, prefer watchlist over rejection when the
     thesis is plausible but current evidence is incomplete.

## Output Artifact

Produce `candidate_triage` with:

- research budget summary
- per-layer research budget allocations
- lightweight Futu check plan
- market snapshot, technical, liquidity, valuation, owner-plate, and optional
  options evidence summaries
- data quality and risk flags
- deep research queue
- watchlist candidates
- deferred candidates
- rejected candidates with hard reasons
- per-layer coverage summary
- triage criteria used
- candidates requiring SEC/fundamental analysis
- candidates requiring only market/technical/liquidity checks
- post-research promotions and demotions, when applicable
- data gaps and warnings

## Boundaries

- Do not generate final portfolio maps.
- Do not assign target weights or cash allocation.
- Do not create trade plans, orders, price targets, or options strategies.
- Do not read current holdings.
- Do not invent missing Futu, market, SEC, or financial facts.
- Do not remove discovery evidence; preserve source ids and trace ids for
  auditability.
