# Performance Attribution

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the framework for measuring, attributing, and validating investment performance. Draws from Darren's institutional benchmarking career ($14B+ AUM, Barclays Agg, custom composites) and the biotech screener's True PIT methodology.

---

## Operator Context

- **Darren Schulz, CFA, CAIA** -- Career-long institutional benchmarking (Barclays US Aggregate Bond Index, custom composites), return attribution, and risk-adjusted performance evaluation.
- As Deputy CIO / Interim CIO at NDRIO, oversaw performance measurement across 8 pension funds, insurance pool, and sovereign wealth fund.
- CFA credential covers quantitative methods, performance measurement, and attribution.
- Background informs the evidence hierarchy and the distinction between backtest artifacts and true forward evidence.

---

## Return Calculation Methods

### Time-Weighted Return (TWR)

Standard for evaluating manager skill (removes impact of cash flow timing).

```
TWR = product(1 + r_i) - 1 for each sub-period i
```

**Use when:** Evaluating manager/model performance independent of cash flow timing. This is the standard for institutional performance reporting (GIPS-compliant).

### Money-Weighted Return (MWR / IRR)

Reflects actual investor experience including cash flow timing.

```
Sum of PV(cash_flows, IRR) = 0
```

**Use when:** Evaluating the actual return experienced by the investor, including the impact of contribution/withdrawal timing. Relevant for PE pacing (J-curve), SFO liquidity modeling.

### Holding Period Return

```
HPR = (ending_value - beginning_value + income) / beginning_value
```

**Use when:** Simple single-period return calculation. Building block for TWR.

---

## Attribution Methods

### Brinson Attribution (Sector/Selection Decomposition)

Decomposes total return into:

| Component | Formula | What It Measures |
| --- | --- | --- |
| Allocation effect | Sum[(w_p - w_b) * r_b] | Value from sector/allocation tilts |
| Selection effect | Sum[w_b * (r_p - r_b)] | Value from stock/position selection |
| Interaction effect | Sum[(w_p - w_b) * (r_p - r_b)] | Cross-term between allocation and selection |
| Total active return | Allocation + Selection + Interaction | Total alpha vs benchmark |

**DEM context:** The biotech screener is primarily a selection engine (coinvest signals select *which* names). Allocation effects are minimal in the EW Top-30 construction. Almost all active return comes from selection.

### Factor Attribution

Decomposes returns into systematic factor exposures:

| Factor | Proxy | DEM Relevance |
| --- | --- | --- |
| Market (beta) | XBI or IBB | Biotech sector beta |
| Size | Market cap quintiles | Small-cap biotech tilt |
| Momentum | Prior 12-1 month return | Implicit in coinvest signal |
| Quality | Financial health score | Negative weight = anti-quality tilt |
| Catalyst | Catalyst proximity | catalyst_decay_w exposure |

**Source files:** `backtest/factor_attribution.py` (22.5KB), `backtest/stability_attribution.py` (16KB)

---

## Benchmark Selection

### For the Biotech Screener

| Benchmark | Use Case | Limitation |
| --- | --- | --- |
| XBI (SPDR S&P Biotech ETF) | Equal-weight small/mid biotech | Best match for DEM universe construction |
| IBB (iShares Biotech ETF) | Cap-weighted biotech | Dominated by mega-cap (Amgen, Gilead, etc.) |
| S&P 500 | Broad market | Not directly comparable, but shows diversification value |
| Russell 2000 | Small cap | Captures size factor but not biotech-specific risk |

**Primary benchmark:** XBI (equal-weight biotech, most comparable to EW Top-30 construction).

### For the SFO Liquidity Model

| Component | Benchmark |
| --- | --- |
| Public equity | MSCI ACWI or custom composite |
| Public bond | Bloomberg US Agg (formerly Barclays Agg) |
| PE | Cambridge Associates PE Index (vintage-year matched) |
| Real estate | NCREIF ODCE (core real estate) |
| Total portfolio | Custom policy benchmark (weighted blend of component benchmarks) |

---

## Risk-Adjusted Metrics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Sharpe ratio | (R_p - R_f) / sigma_p | Excess return per unit total risk |
| Sortino ratio | (R_p - R_f) / sigma_downside | Excess return per unit downside risk |
| Information ratio | (R_p - R_b) / tracking_error | Active return per unit active risk |
| Calmar ratio | Annualized return / max drawdown | Return per unit worst drawdown |
| Hit rate | % of periods with positive active return | Consistency measure |

### DEM Production Metrics (True PIT Backtest)

| Metric | Value | Period |
| --- | --- | --- |
| Monthly alpha (net-of-cost) | +2.34pp | Jun 2020 - Apr 2026 |
| t-statistic | 2.57 | 67 monthly periods |
| Hit rate | 69% | Monthly |
| Bear alpha | +3.37pp (75% hit) | Bear regime periods |
| Neutral alpha | +6.23pp (93% hit) | Neutral regime periods |
| Bull alpha | -0.37pp (50% hit) | Bull regime periods |

---

## Validation Framework

### What Makes Valid Performance Evidence

| Criterion | Required | DEM Status |
| --- | --- | --- |
| Point-in-time safe | No lookahead bias | Yes (PIT enforcement) |
| Survivorship-free | Includes delisted companies | Yes (universe at as_of_date) |
| Net of transaction costs | Realistic cost model | Yes (backtest/cost_model.py) |
| Regime-segmented | Not just pooled results | Yes (bear/neutral/bull split) |
| Statistically significant | t-stat > 2.0 | Yes (t = 2.57) |
| Economically meaningful | Alpha > transaction costs | Yes (+2.34pp/mo >> costs) |
| Replicable | Same inputs = same output | Yes (deterministic) |

### Red Flags in Performance Claims

1. **Gross-of-cost only:** Ignoring transaction costs in high-turnover strategies
2. **Survivorship bias:** Only including companies that survived the full period
3. **Look-ahead bias:** Using information not available at decision time
4. **Cherry-picked periods:** Reporting only favorable time windows
5. **Pooled-only reporting:** Hiding regime-dependent performance
6. **Wrong score field:** Measuring performance of a different signal than production (Spec 095)

---

## Post-Mortem Tracking (Biotech Earnings)

The Biotech Earnings Post-Mortem Tracker routine classifies multi-day price action after results:

| Classification | Code | Definition |
| --- | --- | --- |
| Sustained Move | SM | D0 direction continues through D+2 |
| Accelerating | ACC | D0 direction AND magnitude increases |
| Mean Reversion | MR | D0 direction reverses by D+2 |
| Consolidation | CON | D0 move holds but doesn't extend |

### Cumulative Stats (Weeks 1-3, |D0| >= 5%)

n=27: SM=11 (41%), ACC=10 (37%), MR=2 (7%), CON=4 (15%), SM+ACC=21 (78%).

**Key finding:** MR rate historically low at 7% -- markets are confirming moves far more than fading them in the current regime.

---

## Source Files

| Component | File |
| --- | --- |
| Portfolio metrics | `backtest/portfolio_metrics.py` (23KB) |
| Factor attribution | `backtest/factor_attribution.py` (23KB) |
| Stability attribution | `backtest/stability_attribution.py` (16KB) |
| IC measurement | `backtest/ic_measurement.py` (70KB) |
| Cost model | `backtest/cost_model.py` (7KB) |
| Regime analysis | `backtest/regime.py` (7KB) |
| Sanity metrics | `backtest/sanity_metrics.py` (22KB) |