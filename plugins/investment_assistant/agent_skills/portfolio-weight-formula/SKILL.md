---
name: portfolio-weight-formula
description: Score portfolio weights for formula allocation.
version: 0.1.0
---

# Portfolio Weight Formula

Use this skill when the system needs auditable scores that a deterministic
formula can convert into target weights.

## Purpose

This stage does not directly assign portfolio weights. It scores sleeves and
candidates using only supplied artifacts. Deterministic code then converts the
scores into weights.

## Inputs

- user objective or revision intent
- portfolio structure without target weights, either from a draft map or from
  deep-research candidate/layer artifacts
- candidate research cards
- policy constraints
- portfolio style selected by the user or HITL
- evidence refs

## Scoring Requirements

Before scoring, read the selected portfolio style:

- `balanced`: broad coverage, moderate dispersion.
- `conviction`: stronger concentration in the best-evidenced sleeves and names.
- `bottleneck_barbell`: concentrate around infrastructure/supply-chain
  bottlenecks plus a few core anchors when evidence supports that style.
- `concentrated_growth`: compact high-growth map; peripheral diversification
  should receive very low scores unless it is central to the thesis.

The style is not a recommendation by itself. It controls score separation. The
agent still decides which sleeves and candidates deserve high scores from the
supplied artifacts.

Score each sleeve and candidate using comparable 0-1 or bounded factors.
Explain each score with evidence refs.

Candidate scores should cover:

- role importance
- theme fit
- evidence strength
- business quality
- growth quality
- market or technical signal when supplied
- valuation adjustment when supplied
- liquidity
- risk penalty
- overlap penalty
- why not higher
- why not lower

Sleeve scores should cover:

- importance
- opportunity
- evidence strength
- risk penalty
- overlap penalty
- why not higher
- why not lower

## Calculation design

The exact deterministic weight calculation is documented in `docs/portfolio_weight_formula.md`. In short:

1. The AI agent outputs bounded sleeve/candidate scores only, not final target weights.
2. Code computes raw sleeve/candidate scores from multiplicative factor formulas.
3. Code applies the selected portfolio-style exponents to control concentration.
4. Code normalizes sleeve weights to the theme `sleeve_weight`, then normalizes candidate weights within each sleeve.
5. Code applies sleeve/candidate floors, max caps, candidate-capacity caps, and `single_name_limit`.
6. Code rounds with largest-remainder rounding to the configured precision.
7. Code emits `formula` plus run-specific `calculation_steps` in `PortfolioFormulaAllocationReport` so reviewers can audit how every target weight was produced.

## Boundaries

- Do not output final target weights.
- Do not infer hidden reference weights. If no draft map is supplied, treat the
  supplied sleeves and candidates as a fresh initial map candidate universe.
- Do not create orders, price targets, or trade actions.
- Do not use outside market facts or model memory.
- If evidence is missing, lower confidence and mark the data gap.
