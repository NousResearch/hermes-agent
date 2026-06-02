---
name: dossier-generation
triggers:
  - IC memo
  - investment dossier
  - portfolio report
  - weekly IC presentation
  - score breakdown
  - pipeline output report
  - top holdings report
  - dossier structure
description: "Generate biotech investment dossiers and IC memos."
---

# Dossier Generation Skill

## Purpose

Generate investment dossiers and IC (Investment Committee) memos from pipeline output. This skill defines the structure, cadence, and content requirements for transforming raw screening results into actionable investment documents.

## Preconditions

- Pipeline run has completed successfully (all audit stages OK).
- Governance metadata is present in screening output.
- All scores referenced in the dossier are from the same `as_of_date`.
- No future data has leaked into the scoring (PIT-verified).

---

## Weekly Cadence

| Day | Activity | Owner |
|-----|---------|-------|
| Tuesday | Machine processing: full pipeline run, data collection, scoring | Automated |
| Wednesday | Human scoring: manual review of flagged tickers, override decisions | Analyst |
| Thursday | IC presentation: ranked portfolio, position sizing, risk overlays | Committee |

---

## Dossier Structure

### Section 1: Run Summary

```
Run ID:           <deterministic hash>
As-of Date:       <YYYY-MM-DD>
PIT Cutoff:       <as_of_date - 1>
Score Version:    <version>
Schema Version:   <version>
Parameters Hash:  sha256:<hash>
```

### Section 2: Universe Overview

| Metric | Value |
|--------|-------|
| Total tickers screened | N |
| Active (passed gates) | N |
| Excluded (SEV3) | N |
| Shell companies filtered | N |
| Below liquidity gate | N |
| Below market cap gate | N |

### Section 3: Signal Coverage Dashboard

| Signal | Coverage | Threshold | Status |
|--------|---------|-----------|--------|
| Financial scores | X/N (Y%) | 80% | OK/DEGRADED |
| Clinical scores | X/N (Y%) | 80% | OK/DEGRADED |
| Catalyst events | X/N (Y%) | 10% | OK/DEGRADED |
| Market data | X/N (Y%) | 0% | OPTIONAL |
| Smart money (13F) | X/N (Y%) | 0% | OPTIONAL |
| Short interest | X/N (Y%) | 0% | OPTIONAL |
| PoS scores | X/N (Y%) | 0% | OPTIONAL |

### Section 4: Regime Context

| Indicator | Value | Regime Signal |
|----------|-------|--------------|
| VIX level | X.X | Calm/Normal/Elevated/High/Extreme |
| XBI vs SPY (30d) | X.X% | Outperform/Underperform |
| Yield curve (10Y-2Y) | X bps | Normal/Inverted |
| HY credit spread | X bps | Normal/Elevated/Crisis |
| Classified regime | X | BULL/BEAR/VOLATILITY/etc. |

### Section 5: Top Holdings (Ranked Portfolio)

For each ticker in the top N (default 20), present:

```
Rank: #X
Ticker: XXXX
Composite Score: XX.XX / 100
Position Size: X.X%

Score Breakdown:
  Clinical:   XX.XX (weight: XX%)  [Phase: X, Indication: X]
  Financial:  XX.XX (weight: XX%)  [Runway: Xmo, Severity: X]
  Catalyst:   XX.XX (weight: XX%)  [Events: X, Proximity: X days]
  PoS:        XX.XX (weight: XX%)  [LOA: X.XXX, Confidence: X.XX]
  Short Int:  XX.XX (weight: XX%)  [SI: X%, DTC: X, Signal: X]

Enhancement Flags:
  - E1 Regime Gating:    [applied/not applied]
  - E2 Existential Flaw: [applied/not applied] [cap: XX]
  - E3 Confidence Gate:  [components gated: X]
  - E4 Score Ceiling:    [ceiling: XX, reason: X]
  - E5 Asymmetric:       [transform applied: X]
  - E6 Contradictions:   [conflicts: X]

Risk Flags:
  - Dilution Risk: [NO_RISK/LOW/MEDIUM/HIGH] (score: X.XX)
  - Liquidity Risk: [flags]
  - Crowding Risk: [LOW/MEDIUM/HIGH]
  - Staleness: [PASS/WARN/SOFT_GATE]

Smart Money:
  - Managers holding: X/33 elite
  - Coordinated activity: [ADD/EXIT/NONE]
  - Crowding: [YES/NO] (threshold: 6+)
```

### Section 6: Exclusions Report

| Ticker | Exclusion Reason | Gate |
|--------|-----------------|------|
| XXXX | Runway < 6 months | SEV3 (financial) |
| XXXX | ADV < $500K | Liquidity hard gate |

### Section 7: Score Distribution

| Component | Mean | Median | Std Dev | Min | Max |
|----------|------|--------|---------|-----|-----|
| Composite | X.X | X.X | X.X | X.X | X.X |
| Clinical | X.X | X.X | X.X | X.X | X.X |
| Financial | X.X | X.X | X.X | X.X | X.X |
| Catalyst | X.X | X.X | X.X | X.X | X.X |

### Section 8: Week-over-Week Changes

| Change Type | Count | Details |
|------------|-------|---------|
| New to top 20 | N | [tickers] |
| Dropped from top 20 | N | [tickers] |
| Rank change > 10 | N | [tickers with direction] |
| New catalyst events | N | [tickers with event types] |

### Section 9: Position Sizing Summary

| Metric | Value |
|--------|-------|
| Total positions | N (max: 60) |
| Max single position | X.X% (limit: 10%) |
| Min single position | X.X% (limit: 0.5%) |
| HHI (concentration) | X.XXXX |

---

## Composite Weight Sets

### V3 Enhanced (all signals available)

| Component | Weight |
|----------|--------|
| Clinical | 26% |
| Financial | 24% |
| Catalyst | 16% |
| PoS | 14% |
| Momentum | 9% |
| Short Interest | 6% |
| Valuation | 5% |

### V3 Default (no enhancement data)

| Component | Weight |
|----------|--------|
| Clinical | 40% |
| Financial | 35% |
| Catalyst | 25% |

### V3 Partial (some enhancements)

| Component | Weight |
|----------|--------|
| Clinical | 33% |
| Financial | 28% |
| Catalyst | 18% |
| Momentum | 9% |
| Short Interest | 7% |
| Valuation | 5% |

### Baker-Style Fundamental-Concentrated

| Component | Weight | Rationale |
|----------|--------|-----------|
| Clinical | 35% | Core thesis (biology quality) |
| PoS | 18% | Core thesis (probability) |
| Financial | 22% | Survivability |
| Valuation | 15% | Mispricing |
| Catalyst | 7% | Timing only |
| Momentum | 2% | Overlay |
| Short Interest | 1% | Risk context |

---

## Enhancement Flags Reference (E1-E6)

### E1: Hard Regime Gating

| Regime | Momentum Cap | Financial Penalty |
|--------|-------------|-------------------|
| BEAR | 30% of deviation | 1.25x |
| BULL | 100% (full) | 0.85x |
| NEUTRAL | No gating | - |

### E2: Existential Flaw Escalation

- Runway < 9 months AND early-stage (phase_1, phase_2): cap score at 65

### E3: Confidence-Weighted Aggregation

Per-component confidence multipliers. Components below 0.40 confidence are soft-gated.

### E4: Dynamic Score Ceilings

| Condition | Ceiling |
|----------|---------|
| Phase 3 stage | <= 85 |
| No catalyst events | <= 60 |
| Commercial stage | <= 90 |

### E5: Convex Downside, Concave Upside

Reflects loss aversion appropriate for biotech risk.

### E6: Contradiction Detector

| Conflict Pair | Resolution |
|--------------|------------|
| Strong momentum + low liquidity | Cap momentum contribution |
| Cheap valuation + financing pressure | Flag, reduce conviction |

---

## Expected Return Model

```
Score -> Rank -> Percentile (Blom plotting position)
Percentile -> Z-score (Acklam inverse normal)
Z-score -> Expected excess return = z * lambda * (12 / holding_period_months)
```

- **DEFAULT_LAMBDA_ANNUAL**: 0.08 (8% per 1-sigma per year)
- **Model ID**: zscore_linear_lambda v1.0.0

---

## Output Format Requirements

1. All monetary values in USD with commas (e.g., $1,234,567)
2. All scores to 2 decimal places
3. All percentages to 1 decimal place
4. All dates in ISO 8601
5. Governance block attached to every output document
6. Content hash (SHA256) of the dossier included in audit log

---

## Source Files

| Component | File |
|----------|------|
| Pipeline Orchestrator | `run_screen.py` |
| Composite Scoring | `module_5_composite_v3.py` |
| Regime Engine | `regime_engine.py` |
| Smart Money | `manager_momentum_v1.py` |
| Expected Returns | `common/score_to_er.py` |
| Audit Log | `governance/audit_log.py` |
| Position Sizing | `module_5_composite_with_defensive.py` |
