# Backtest Framework

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the methodology for constructing valid, PIT-safe backtests in the biotech screener context. This skill covers *how* to build a backtest that produces trustworthy evidence, complementing ic-evaluation (which covers *how to interpret* backtest results) and selector-ranker (which reports *what* the backtests found).

## Preconditions

- All backtests MUST use an explicit `as_of_date` parameter. Never use `datetime.now()`.
- PIT cutoff: `source_date <= as_of_date - 1` (standard) or `source_date < as_of_date - 2` (strict mode).
- All scoring arithmetic uses `Decimal`. Statistical analysis (IC, bootstrap, Spearman) may use float/numpy/scipy.
- Forward returns must be computed from prices AFTER the as_of_date, never from contemporaneous data.

---

## Backtest Construction Rules

### Rule 1: Score Field Identity

The backtest MUST explicitly declare which score field it measures. The output header must state the field name and confirm it matches the production sort key.

**Lesson from Spec 095/100:** `run_rank_ic_backtest.py` measured `composite_score` IC while claiming to measure production performance. `composite_rank` correlates only 0.25 with `actionable_rank`. Top-30 overlap was 23%. This conflation invalidated all prior ranker IC claims.

**Prevention:** Every backtest output file must include:
```json
{
  "score_field_measured": "final_score",
  "production_sort_key": "final_score",
  "field_match_verified": true
}
```

### Rule 2: Point-in-Time Safety

| Violation | Detection Method | Consequence |
| --- | --- | --- |
| Future data in features | `source_date > as_of_date` check | Reject unconditionally |
| Future data in returns | Return window starts before as_of_date | Reject unconditionally |
| Stale data masquerading as current | Gate 2 staleness thresholds (see biotech-validation) | Apply staleness penalties |
| Survivorship bias | Universe defined at as_of_date, not at evaluation date | Exclude companies that delisted after as_of_date |

### Rule 3: Survivorship Bias Avoidance

The backtest universe at each snapshot date must include ALL tickers that were eligible at that date, including those that subsequently delisted, were acquired, or went bankrupt.

**Deprecated evidence (from ic-evaluation):** All survivorship-only benchmark numbers (+93.7pp, +110.5pp, etc.) are invalid and must not be cited.

### Rule 4: Transaction Cost Modeling

The True PIT backtest uses net-of-cost returns. Cost model components:

| Component | Source File | Key Parameters |
| --- | --- | --- |
| Spread cost | `backtest/cost_model.py` | Bid-ask spread estimate by market cap tier |
| Market impact | `backtest/cost_model.py` | Volume-based impact model |
| Turnover | Measured from portfolio changes | Monthly rebalance frequency |

**Production evidence (net-of-cost):** +2.34pp/mo, t = 2.57, 69% hit rate, 67 monthly periods (Jun 2020 - Apr 2026).

### Rule 5: Regime Conditioning

Backtests MUST report results segmented by market regime:

| Regime | DEM Performance | Hit Rate |
| --- | --- | --- |
| Bear | +3.37pp/mo | 75% |
| Neutral | +6.23pp/mo | 93% |
| Bull | -0.37pp/mo | 50% |

**Critical insight:** The system is a bear/neutral alpha engine. Expect bounded underperformance in strong bull markets. Any backtest that reports only pooled results without regime segmentation is incomplete.

### Rule 6: Contamination Tagging

After adding new managers to the institutional signal pipeline, a contamination window opens (typically 20 trading days). IC measurements during this window must be:
- Flagged as `contaminated: true`
- Reported separately from clean-window IC
- Excluded from promotion-grade evidence

### Rule 7: Serial Correlation Awareness

Daily snapshots with 5-day forward return windows create heavy serial correlation. 14 snap dates yields ~37 effective observations, not 14 independent observations. IC t-stats from overlapping windows are indicative only, not promotion-grade.

---

## Backtest Codebase Map

| Component | File | Size | Purpose |
| --- | --- | --- | --- |
| IC measurement | `backtest/ic_measurement.py` | 70KB | Spearman IC, cohort segmentation |
| Cost model | `backtest/cost_model.py` | 7KB | Transaction cost estimation |
| Portfolio metrics | `backtest/portfolio_metrics.py` | 23KB | Return, Sharpe, hit rate |
| Factor attribution | `backtest/factor_attribution.py` | 23KB | Factor decomposition |
| Regime analysis | `backtest/regime.py` | 7KB | Bear/neutral/bull classification |
| Fama-MacBeth | `backtest/fmb.py` | 12KB | Cross-sectional regression |
| Walk-forward OOS | `backtest/walkforward_oos_m3.py` | 25KB | Out-of-sample validation |
| Stability attribution | `backtest/stability_attribution.py` | 16KB | Signal robustness across slices |
| Returns provider | `backtest/returns_provider.py` | 10KB | Forward return computation |
| Data readiness | `backtest/data_readiness.py` | 17KB | Input quality verification |
| Panel builder | `backtest/panel_builder_m1.py` | 16KB | Snapshot panel construction |
| Sanity metrics | `backtest/sanity_metrics.py` | 22KB | Output validation |

---

## Evidence Hierarchy (Cross-Reference: ic-evaluation)

| Rank | Source | Strength | Backtest Type |
| --- | --- | --- | --- |
| 1 | Checklist v2 rerun (2026-04-04) | STRONGEST (signals) | Full battery (FM + bootstrap + FDR + LOSO + year stability) |
| 2 | True PIT backtest (Spec 050) | STRONGEST (portfolio) | Net-of-cost, regime-segmented |
| 3 | Pairwise feature audit (2026-04-04) | SUPPORTING | Within-cohort feature analysis |
| 4 | Forward shadow | MONITORING | True out-of-sample daily accumulation |
| 5 | Old PIT benchmark (Spec 048) | SUPERSEDED | Pre-Checklist-v2 methodology |

---

## What Makes a Bad Backtest (Anti-Patterns)

1. **Score field conflation** (Spec 095): Measuring composite_score but claiming production performance
2. **Survivorship bias**: Only including companies that survived the full backtest period
3. **Pooled-only reporting**: Not segmenting by regime, hiding bear/bull asymmetry
4. **Gross-of-cost claims**: Ignoring transaction costs in a high-turnover strategy
5. **Contaminated IC**: Including post-manager-addition windows in clean IC calculations
6. **Overlapping-window t-stats**: Treating serial-correlated daily observations as independent
7. **Backfill masquerading as forward**: Historical data recomputed with current logic (pseudo-PIT) cited as forward evidence

---

## Promotion Path

A signal or construction change must pass ALL of the following before entering production:

1. Checklist v2 battery: 5/5 gates (FM, bootstrap, FDR, LOSO, year stability)
2. True PIT backtest: positive net-of-cost alpha across regimes
3. Forward shadow: 30+ trading days of true out-of-sample evidence
4. Operator approval: explicit written approval from Darren
5. Architecture freeze compliance: no changes during freeze windows