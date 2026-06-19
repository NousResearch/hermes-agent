# Portfolio Weight Formula Calculation Design

## Goal

The portfolio-weight-formula stage makes portfolio-map weights auditable without letting deterministic code author the investment thesis.

- The AI scoring agent reads only supplied workflow artifacts and outputs bounded sleeve/candidate scores plus rationale.
- Deterministic code converts those scores into target weights.
- The emitted allocation artifact records both the formulas and the concrete calculation steps used for the run.

This preserves the investment-assistant contract: AI owns the investment judgment; deterministic code owns arithmetic, caps, rounding, validation, and auditability.

## Inputs

The deterministic allocator reads:

- `policy.sleeve_weight`: total investable weight assigned to the theme map.
- `policy.cash_weight`: reserved cash weight, carried through for reporting.
- `policy.single_name_limit`: maximum portfolio weight for any single candidate unless a stricter candidate cap is supplied.
- `policy.portfolio_style`: one of `balanced`, `conviction`, `bottleneck_barbell`, or `concentrated_growth`.
- `policy.portfolio_style_profile.sleeve_score_exponent`: concentration exponent for sleeve scores.
- `policy.portfolio_style_profile.candidate_score_exponent`: concentration exponent for candidate scores.
- AI-authored `SleeveFormulaScore` rows.
- AI-authored `CandidateFormulaScore` rows.
- Optional AI-authored `min_weight` / `max_weight` constraints on sleeves and candidates.
- `precision`: rounding unit, default `0.001`.

## Score formulas

The AI agent supplies bounded factors. Code computes raw scores by multiplication.

Sleeve raw score:

```text
sleeve_raw_score = importance_score
                 * opportunity_score
                 * evidence_strength
                 * (1 - risk_penalty)
                 * (1 - overlap_penalty)
```

Candidate raw score:

```text
candidate_raw_score = role_importance
                    * theme_fit
                    * evidence_strength
                    * business_quality
                    * growth_quality
                    * market_signal
                    * valuation_adjustment
                    * liquidity_score
                    * (1 - risk_penalty)
                    * (1 - overlap_penalty)
```

All terms are clamped positive during multiplication with a tiny floor (`1e-6`) so a single zero does not create division failures. The Pydantic schemas bound the factor ranges.

## Style adjustment

Portfolio style controls concentration by exponentiating raw scores.

```text
sleeve_style_adjusted_score = sleeve_raw_score ** sleeve_score_exponent
candidate_style_adjusted_score = candidate_raw_score ** candidate_score_exponent
```

Current style profiles:

- `balanced`: sleeve `1.0`, candidate `1.0`.
- `conviction`: sleeve `1.35`, candidate `1.45`.
- `bottleneck_barbell`: sleeve `1.55`, candidate `1.6`.
- `concentrated_growth`: sleeve `1.75`, candidate `1.85`.

Higher exponents increase dispersion: high-scoring sleeves/candidates receive more weight and lower-scoring ones compress faster.

## Sleeve normalization

Sleeve weights are normalized across sleeves to consume `policy.sleeve_weight`.

Base formula:

```text
sleeve_weight = sleeve_weight_total
              * sleeve_style_adjusted_score
              / sum(sleeve_style_adjusted_score across sleeves)
```

Before rounding, `_normalize_with_caps()` applies:

1. Sleeve floors from `SleeveFormulaScore.min_weight`.
2. Sleeve caps from `SleeveFormulaScore.max_weight`.
3. Aggregate candidate-capacity caps: a sleeve cannot receive more weight than its candidates can absorb under candidate caps and `single_name_limit`.
4. Proportional redistribution of remaining weight among uncapped sleeves.

If floors exceed the available total, floors are scaled down proportionally instead of over-allocating.

## Candidate normalization

Candidate weights are normalized within each sleeve budget.

Base formula:

```text
candidate_weight = sleeve_weight
                 * candidate_style_adjusted_score
                 / sum(candidate_style_adjusted_score in same sleeve)
```

Before rounding, `_normalize_with_caps()` applies:

1. Candidate floors from `CandidateFormulaScore.min_weight`.
2. Candidate caps from `CandidateFormulaScore.max_weight`.
3. `single_name_limit` as the default cap for every candidate.
4. Proportional redistribution of remaining sleeve budget among uncapped candidates.

## Rounding

`_round_weights_exact()` uses largest-remainder rounding to `precision`:

1. Convert each unrounded weight into units: `weight / precision`.
2. Floor each unit count.
3. Compute residual units needed to match the requested total.
4. Give residual units to entries with the largest fractional remainders.
5. Convert units back to weights.

This keeps rounded sleeve totals and rounded candidate totals equal to `sleeve_weight` within validation tolerance.

## Validation

After allocation, `_validate_allocation_report()` verifies:

- Candidate weights sum to `report.sleeve_weight` within tolerance.
- Sleeve weights sum to `report.sleeve_weight` within tolerance.
- No candidate exceeds `single_name_limit` plus tolerance.

Scoring validation separately verifies that:

- Every expected sleeve has exactly one score.
- Every expected candidate has exactly one score.
- Candidate scores reference known sleeve keys.
- Each score includes rationale and `why_not_higher` / `why_not_lower` explanations.
- Candidate scores include evidence refs.

## Artifact fields

`PortfolioFormulaAllocationReport.formula` contains human-readable formulas:

- `sleeve_raw_score`
- `candidate_raw_score`
- `sleeve_style_adjusted_score`
- `candidate_style_adjusted_score`
- `sleeve_normalization`
- `candidate_normalization`
- `caps`
- `rounding`

`PortfolioFormulaAllocationReport.calculation_steps` contains run-specific audit data:

- `allocation_pipeline`: `score → style_adjust → normalize_with_floors_and_caps → round_largest_remainder → validate`.
- `policy_inputs`: actual policy values and style exponents used.
- `sleeves`: per-sleeve style-adjusted score and final target weight.
- `candidates_by_sleeve`: per-candidate style-adjusted score, normalized within-sleeve score, target weight, and factor terms.

## Example

If one sleeve has a `0.12` budget and two candidates score in a 3:1 ratio, with no caps binding:

```text
US.MU raw/style-adjusted score   = 1.0
US.SNDK raw/style-adjusted score = 0.333333
sum                              = 1.333333
US.MU target_weight              = 0.12 * 1.0 / 1.333333 = 0.09
US.SNDK target_weight            = 0.12 * 0.333333 / 1.333333 = 0.03
```

The unit test `test_portfolio_weight_formula_allocates_from_ai_scores_deterministically` covers this exact calculation and asserts that formula/calculation-step metadata is emitted.
