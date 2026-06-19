---
name: portfolio-architect
description: Build portfolio maps from triaged evidence.
version: 0.1.0
---

# Portfolio Architect

Use this skill after `deep-research` has produced candidate research cards. The
architect turns a researched candidate set into target portfolio maps. It does
not discover new candidates, reread filings, or create orders.

## Evidence Boundary

Use only the supplied artifacts:

- user objective and portfolio constraints
- deep-research report
- candidate triage artifact
- optional Futu lightweight/enrichment data supplied with the package

Do not use theme discovery output directly. Discovery is an upstream production
step; candidate triage defines the upstream universe, and deep-research defines
the researched construction candidates.

Do not use model memory for facts about companies, recent events, prices,
financial numbers, or ETF composition. If a fact is not in the supplied
artifacts, mark it as a data gap or open question.

Treat the deep-research report as the primary evidence layer:

- `candidate_cards` are the main per-symbol evidence surface.
- `layer_conclusions` are the main same-layer peer-tradeoff surface.
- `candidate_decision` is a research-stage label, not a command.
- `high_conviction_candidate` and `core_candidate` are natural construction
  inputs.
- `satellite_candidate` can be used to express style differences or complete a
  layer.
- `watchlist`, `defer`, and `reject` should not receive target weights unless
  the map explicitly explains the exceptional reason using supplied evidence.
- Do not reinterpret raw filing summaries if the deep-research report already
  summarized them.

## Required Output Behavior

Produce two or three distinct target portfolio maps. Each map must include:

- a post-research selection step before assigning weights
- selection summary explaining the narrowing logic
- peer tradeoffs for important overlapping or substitutable candidates
- every selected candidate must include role, why_selected, and evidence_refs
- positioning and best-fit investor preference
- total theme sleeve weight and cash weight
- explicit sleeves with sleeve weights and holding symbols
- individual holdings with target weights, roles, rationale, and evidence refs
- map-level weight rationale explaining holding count, sleeve weights, high-beta
  sizing, selected-but-unheld candidates, and risk-budget tradeoffs
- key risks and missing exposure
- omission audit for important triaged candidates not selected

Weights are target portfolio-map weights only. Do not produce:

- buy, sell, hold, trim, add, or reduce instructions
- price targets
- trigger prices
- simulated orders
- options strategies
- current-holding adjustments

## Portfolio Construction Guidance

Start from the deep-research candidate cards, layer conclusions, and candidate
decisions. Use candidate triage only as the upstream universe and layer/source
context. Preserve distinct economic exposures where the research supports them.
Do not collapse the map into a generic large-cap technology template when the
research contains validated bottleneck branches such as memory/storage,
optical/networking, power/cooling, semiconductor equipment, cloud/platforms, or
software/security.

Before weights, group candidates that represent overlapping or substitutable
exposure. Decide which researched candidates enter the construction universe,
which remain watchlist, and which are deferred. Explain the tradeoff using
supplied deep-research evidence. This is a process requirement, not a hardcoded
ticker rule.

Peer tradeoffs are not layer inventories. For any layer where some high-priority
candidates are selected and some are not, record comparable symbols, selected
symbols, non-selected symbols, and the reason for the choice.

Treat required symbols as constraints, not proof of quality. They should appear
unless the policy or supplied evidence makes them impossible, in which case the
failure must be explicit.

Use deep-research `evidence_refs` in holdings and omission audits. If exact
financial numbers are needed, only use numbers already present in the supplied
deep-research report or structured context. Do not extract new authoritative
financial numbers from prose summaries.

## Omission Audit

For each researched candidate with `candidate_decision` equal to
`high_conviction_candidate` or `core_candidate` that is not selected in a map,
explain whether it was omitted because of:

- overlap with selected holdings
- weaker deep-research evidence
- valuation, profitability, or quality concern from supplied data
- momentum or technical concern from supplied Futu data when present
- data quality gap
- sleeve or single-name weight budget
- risk or cyclicality
- scope mismatch

If a same-layer candidate is selected instead, name the substitute holding.
