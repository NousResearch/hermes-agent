---
name: deep-research
description: Read filing summaries and rank candidates.
version: 0.1.0
---

# Deep Research

Use this skill after `research-intake-triage` has selected which offline
filing analyses to open. This stage reads the selected `filing_summary.md`
files plus compact structured metrics, then produces source-grounded candidate
research cards and layer-level peer tradeoffs.

## Inputs

- research-intake-triage artifact
- candidate triage artifact when provided
- `filing_summary.md`
- `filing_summary.meta.json`
- `sec_companyfacts.json`
- `fmp_company_profile.json`
- `filing_metadata.json`
- optional lightweight market/Futu facts when provided

## Purpose

Deep research turns routed reading materials into a research dataset for the
future portfolio architect. It should answer:

- Is the theme exposure real or indirect?
- What do filings say about demand, margin, capex, customer concentration,
  backlog, product cycle, and risks?
- Which same-layer peers look stronger, weaker, or complementary?
- Should a candidate be a portfolio candidate, a satellite/watch candidate, or
  deferred/rejected before portfolio construction?

## Required Process

1. Respect the evidence boundary.
   - Use only provided artifacts.
   - Do not invent market facts, financial numbers, customer names, or news.
   - Exact numbers must come from structured numeric artifacts such as
     `sec_companyfacts.json` or provider profile fields, not prose.

2. Read selected filings as qualitative evidence.
   - Use `filing_summary.md` for management commentary, business mix, demand
     signals, margin/cost signals, AI/data-center relevance, risk factors, and
     data-quality notes.
   - Preserve source labels when useful.
   - Surface truncation or partial-summary warnings.

3. Compare within layers.
   - Do not rank all symbols in one flat list.
   - For each layer, explain selected, watch, deferred, and rejected peers.
   - Same-layer tradeoffs are more important than generic quality language.

4. Keep portfolio construction separate.
   - Do not assign target weights.
   - Do not produce target prices, buy/sell/hold recommendations, orders, or
     trade plans.
   - You may classify candidates as `core_candidate`, `satellite_candidate`,
     `watchlist`, `defer`, or `reject` for the next architect stage.

## Output Expectations

For each researched candidate include:

- symbol
- layer_keys
- intake_action
- theme_exposure
- business_quality
- fundamental_snapshot
- filing_takeaways
- key_risks
- peer_positioning
- candidate_decision
- confidence
- evidence_refs
- data_gaps

For each layer include:

- layer_key
- selected_symbols
- watchlist_symbols
- deferred_symbols
- rejected_symbols
- peer_tradeoff_summary
- unresolved_questions

## Boundaries

- No portfolio weights.
- No trade actions.
- No price targets.
- No unstated facts.
- If evidence is weak, say so and lower confidence.
