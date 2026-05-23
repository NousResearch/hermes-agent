# Options & Derivatives

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the framework for options analysis and derivatives usage relevant to Wake Robin's investment activities. Covers both the general analytical framework (IV, skew, term structure) and the domain-specific lessons from the biotech screener about where options signals add value and where they demonstrably do not.

---

## Operator Context

- **Darren Schulz, CFA, CAIA** -- LinkedIn headline includes "Options, Analytics"
- Self-forwards Natenberg and Sinclair content to work email as reference material
- Published commentary on CDX vs SPX options pricing and credit-equity market integration
- Experience with treasury futures for interest rate exposure management and synthetic equity exposure (S&P 500)
- Tracks options/volatility trading concepts actively via LinkedIn engagement

---

## General Options Framework

### Implied Volatility (IV) Analysis

| Metric | What It Measures | Use Case |
| --- | --- | --- |
| IV rank | Current IV vs 52-week range | Relative cheapness/richness |
| IV percentile | % of days IV was below current level | Historical context |
| IV term structure | IV across expirations | Event premium identification |
| IV skew | IV difference between OTM puts and calls | Risk sentiment |
| Realized vol vs implied | Historical vol minus IV | Over/underpricing detection |

### Key References

| Author | Work | Key Concept |
| --- | --- | --- |
| Sheldon Natenberg | *Option Volatility and Pricing* | Comprehensive options valuation |
| Euan Sinclair | *Volatility Trading* | Practical vol trading strategies, edge estimation |
| Nassim Taleb | *Dynamic Hedging* | Non-linear risk management |

### Binary Event Pricing (Biotech-Specific)

Biotech options pricing around catalysts (PDUFA dates, Phase 3 readouts) is dominated by binary outcomes:

```
Expected move = IV * sqrt(T) * stock_price * (1/sqrt(252))
```

For binary events, the market prices an implied probability:
```
implied_prob = straddle_price / stock_price
```

**Key characteristic:** Biotech binary events create extreme IV spikes (often 100-300% annualized) that collapse to near-zero after the event. This makes standard mean-reversion vol strategies unreliable.

---

## DEM Dead Lanes: What Didn't Work

### Options Surface-Shape as Ranker (DEAD)

- **Status:** DEAD LANE -- do not reopen without fundamentally new data
- **Evidence:** 50-month IC negative across ALL horizons
- **Root cause:** In small-cap biotech, vol surface shape is dominated by binary catalyst outcomes, not continuous information flow. The surface reflects event timing and binary payoff structure, not alpha-predictive information.
- **Lesson:** Options-derived signals work in markets where information is incorporated gradually (large cap equity). They fail in markets where price moves are binary and event-driven.

### Options-as-Alpha (Spec 053, CLOSED)

- **Status:** CLOSED -- 37 signals tested, ALL fail
- **Evidence:** Exhaustive signal search across IV rank, IV percentile, skew, term structure, put/call ratios, and composite signals. Zero survived even basic IC screening.
- **Root cause:** The options market for small-cap biotech is too thin and binary-event-driven for continuous alpha extraction. Bid-ask spreads are wide, volume is low, and the signal-to-noise ratio is dominated by catalyst timing effects already captured by `catalyst_decay_w`.
- **Lesson:** Options data in thin markets is a reflection of known catalyst timing, not independent information. When the underlying alpha source is already captured by the catalyst pipeline, options add noise, not signal.

---

## Where Options/Derivatives DO Add Value

### Treasury Futures for Duration Management

- Darren's fixed income experience at TMRS and NDRIO used treasury futures for interest rate exposure management
- Synthetic duration adjustment without selling physical bonds
- Cost-efficient tactical tilts around rate expectations

### Synthetic Equity Exposure

- S&P 500 index options/futures for tactical equity exposure management
- Used in institutional contexts for cash equitization (deploying cash inflows quickly)
- Lower transaction costs than physical equity trading for short-term exposure

### Credit Derivatives (CDX)

- Darren published commentary on CDX vs SPX options pricing and credit-equity market integration
- Credit default swap indices (CDX IG, CDX HY) as credit risk indicators
- CDX-SPX correlation as a regime indicator (credit-equity convergence/divergence)

### ODIN External Benchmark

ODIN's 51-feature clinical trial prediction model includes "options market implied approval probability" as a feature category:
- Extracts implied binary event probability from options pricing around PDUFA dates
- Uses it as one input among 51 features (not standalone)
- AUC 0.9363 on 2,210 historical FDA events
- **DEM implication:** Options-implied probability may work as a *calibration input* for catalyst_quality (external benchmark), even though it failed as a standalone *alpha signal* in Spec 053. The distinction is: standalone alpha vs. calibration input for an existing signal.

---

## Portfolio-Level Derivatives Usage

### Hedging Framework

| Strategy | When to Use | Cost Profile |
| --- | --- | --- |
| Protective puts (XBI/IBB) | Concentrated biotech exposure | Premium = insurance cost |
| Collar (sell calls, buy puts) | Willing to cap upside for downside protection | Net premium near zero |
| Put spread | Limited budget, defined risk tolerance | Lower premium, capped protection |
| Variance swap | Systematic vol exposure management | Institutional-only |

### Position-Level Options

| Strategy | Use Case |
| --- | --- |
| Covered calls on held names | Generate income on low-catalyst positions |
| Protective puts pre-catalyst | Risk management around binary events |
| Straddle/strangle (research only) | Implied vs realized vol comparison |

---

## Key Constraints

1. Options-as-alpha is a DEAD LANE for the DEM biotech universe (Spec 053). Do not reopen without fundamentally new data or a different universe.
2. Options surface-shape as ranker is DEAD (50-month negative IC). Same constraint.
3. Options-derived data may have value as CALIBRATION INPUTS (external benchmark for catalyst_quality) through the T5 promotion path, but this requires separate evaluation.
4. For portfolio-level hedging, XBI/IBB options are more liquid and appropriate than individual name options in the small-cap biotech universe.
5. Natenberg/Sinclair frameworks apply to markets with continuous information flow; biotech binary events require modified frameworks.