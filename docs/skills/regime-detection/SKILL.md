# Regime Detection

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the framework for market regime identification and regime-conditional signal evaluation. Multiple research digests and the DEM's own backtest results show regime is the dominant confound. This skill bridges the gap between the implicit regime awareness already in the system (bear/neutral/bull performance split) and an explicit detection and response framework.

---

## Why Regime Matters for This System

### DEM Backtest Evidence (True PIT, Spec 050)

| Regime | Alpha (pp/mo) | Hit Rate | Character |
| --- | --- | --- | --- |
| Bear | +3.37 | 75% | Strong alpha engine |
| Neutral | +6.23 | 93% | Strongest alpha engine |
| Bull | -0.37 | 50% | Bounded underperformance |
| Pooled | +2.34 | 69% | Bear/neutral carry the average |

**Critical insight:** The DEM is a bear/neutral alpha engine. In strong bull markets, expect flat to slightly negative relative performance. Any signal evaluation or portfolio decision that ignores regime is incomplete.

### Research Digest Evidence (May 2026)

| Source | Finding | Implication |
| --- | --- | --- |
| Multiple 2025-2026 studies | LLM-based signals are strongly regime-dependent | Regime detection is a structural requirement |
| Wasserstein HMM | Sharpe 2.18 vs 1.59 baseline; max DD -5.43% vs -14.62% SPX | Regime-aware allocation materially reduces drawdowns |
| Regime-aware agentic framework (Springer 2026) | +0.373 Sharpe improvement net of transaction costs | Regime conditioning adds economic value |
| LLM-Enhanced Black-Litterman (arXiv 2504.14345) | Different LLMs exhibit distinct "investment styles"; success depends on regime alignment | Model routing by regime is a portfolio construction decision |

---

## Current Regime Handling (Implicit)

### In the Backtest

The True PIT backtest segments results by bear/neutral/bull regime post-hoc. This is descriptive analysis -- it tells us *how* the system performed in each regime but does not *predict* regime in advance.

### In the Ranker (Alt 10)

The selector-only comparator (Spec 094) showed regime confounding: selector wins 4/6 in clean window (+0.020) but 0/5 in regime window (-0.025). Pooled: 4/11 (ranker-override slightly better, dominated by regime confounding). This is the strongest evidence that regime is an active confound in production.

### In the Daily Pipeline

`backtest/regime.py` (6.7KB) implements regime classification for backtest analysis. It is NOT wired into the production scoring or construction pipeline.

---

## Regime Classification Methods (Reference)

### Hidden Markov Models (HMM)

- Wasserstein HMM achieves Sharpe 2.18, max DD -5.43%
- 2-3 state models (bull/bear or bull/neutral/bear)
- Trained on return distributions, volatility, and cross-asset correlations
- Latency: 1-5 day detection lag typical

### Ensemble ML

- Combine multiple regime indicators (VIX level, yield curve slope, credit spreads, momentum breadth)
- Random forest or gradient boosting for regime classification
- Can incorporate macro indicators (inflation expectations, PMI, employment)

### XBI / IBB as Biotech Regime Proxies

| ETF | Use | Current Level (May 18) |
| --- | --- | --- |
| XBI | SPDR S&P Biotech ETF (equal-weight small/mid cap) | $130.69 (-2.98% week of May 12) |
| IBB | iShares Biotech ETF (cap-weighted) | $166.83 (-1.83% week of May 12) |

XBI is the more relevant proxy for the DEM's universe (small/mid cap biotech). XBI drawdown from 52-week high, VIX level, and XBI relative strength vs SPY are candidate regime indicators.

---

## Regime Interaction with DEM Signals

### coinvest_score_z

Pre-cohort (clean window): mean IC = -0.051, hit rate 11.1%
Post-cohort (contaminated): mean IC = -0.008, hit rate 60.0%

The April 2026 selloff drove pre-cohort negativity. This is likely a regime effect -- coinvest signals may perform differently in broad biotech drawdowns vs recoveries.

### inst_delta_z

Zeroed in selector at IC = -0.097 over 36 dates. The negative IC may be partially regime-driven (QoQ institutional changes lag market moves). Reinstatement conditions should include regime-conditional IC analysis.

### financial_score (Negative Weight)

The stress-upside thesis (negative ranker weight = prefer financially stressed names) is inherently regime-sensitive:
- In bear markets: stressed names may face existential financing risk
- In bull markets: stressed names have maximum optionality
- The negative weight persists across both bull (NW-t = -3.42) and bear (-3.38) regimes, but the MAGNITUDE of alpha contribution may differ

---

## Design Candidates (Post-Freeze)

### Tier 4 Evaluation Path

Regime detection is a Tier 4 governance question -- it would change how signals are interpreted, not what signals are computed. Any implementation requires:

1. Design memo with clear hypothesis
2. Historical validation (does regime conditioning improve PIT backtest results?)
3. Forward shadow arm (add a regime-conditioned arm to coinvest_shadow_tracker)
4. Operator approval before any production integration

### Possible Implementations

| Approach | Complexity | Value | Risk |
| --- | --- | --- | --- |
| Regime label in daily output (descriptive only) | LOW | Improves operator context | None (no model change) |
| Regime-conditional IC reporting | LOW | Better signal evaluation | None (diagnostic only) |
| Regime-conditional position sizing | MEDIUM | Reduce bull-market drag | Construction change, needs Checklist v2 |
| Regime-aware selector weighting | HIGH | Potentially large | Overfitting risk, needs extensive validation |

### Recommended Sequence

1. Add regime label to daily pipeline output (descriptive, no model change)
2. Segment all IC and performance reporting by regime
3. Accumulate 6+ months of regime-labeled forward data
4. Evaluate whether regime conditioning would have improved decisions
5. If evidence supports, design a regime-conditional construction change
6. Checklist v2 battery on regime-conditioned variant

---

## External Benchmarks

### ODIN Confidence Tiers (Catalyst-Level Regime)

ODIN assigns binary confidence tiers that correlate with post-approval stock performance. This is a form of catalyst-level regime detection (market receptivity to positive catalysts varies by broader regime).

### FDA Real-Time Clinical Trial Initiative

If clinical trials become continuous rather than phase-gated, the binary catalyst model evolves toward continuous information release. This is a structural regime change, not a cyclical one.

---

## Key Constraints

1. Do NOT implement regime-conditional scoring during the architecture freeze
2. Regime detection is descriptive analysis until proven predictive through forward evidence
3. Any regime-aware construction change requires full Checklist v2 battery
4. The dead lane for "options surface-shape as regime indicator" (50-month negative IC) should be respected -- regime detection via options is a closed lane for this universe