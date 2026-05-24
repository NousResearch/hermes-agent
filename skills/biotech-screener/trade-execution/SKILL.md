# Trade Execution

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the bridge from model output (ranked list) to portfolio action (trades). The selector-ranker skill covers *what* to buy; this skill covers *how* to execute it -- trade plan generation, position sizing, shadow portfolio mechanics, and broker integration.

---

## Pipeline Steps (Daily Production)

The trade execution steps are the final stages of the 13-step daily pipeline:

| Step | Name | Output |
| --- | --- | --- |
| 9 | Shadow portfolio | Shadow portfolio state file |
| 10 | Trade plan | Trade recommendations |
| 11 | Portfolio report | Performance and holdings summary |

These run after scoring (steps 1-6), manifest/promotion (step 6), drift report (step 7), and action packet (step 8).

---

## Position Sizing (Decision Engine L3)

The L3 sizing layer applies two adjustments based on `ev_severity_score` (from runway severity v1.1):

### Dilution Haircut

```
dilution_haircut = 0.35 * ev_severity_score
```

Reduces position size for companies with high financing pressure. Range: 0% (healthy) to 35% (maximum financial stress).

### Size Multiplier

```
size_multiplier = max(0.40, 1.0 - 0.60 * ev_severity_score)
```

Floor at 40% of base position size. A company with maximum `ev_severity_score` = 1.0 gets 40% of the base weight.

### Derived Field Contracts

Both fields must satisfy these formulas for all non-null rows (tolerance 1e-6):
```
dilution_haircut == 0.35 * ev_severity_score
size_multiplier == max(0.40, 1.0 - 0.60 * ev_severity_score)
```

`check_severity_formulas()` QA validation runs every snapshot (Spec 101, commits eaa4ea87 + cba4ee0f).

---

## Construction: EW Top-30

### Rules

- Equal-weight across top 30 names ranked by `final_score`
- K=30 validated by PIT sweep (stable K=25-35 plateau, net-of-cost peak)
- Rank-weighting does NOT help: RW-EW delta = -0.09pp, t = -0.95
- Base position weight = 1/30 = 3.33% before sizing adjustments

### Position Weight After Sizing

```
position_weight = (1/30) * size_multiplier * (1 - dilution_haircut)
```

Weights are then renormalized to sum to 100%.

---

## Shadow Portfolio

### Purpose

The shadow portfolio tracks hypothetical performance of the model's recommendations without live capital at risk. It serves as:
- Forward evidence accumulation (true out-of-sample)
- Validation against the True PIT backtest
- Benchmark for live portfolio performance comparison

### Mechanics

- Updated daily after the production pipeline completes
- Uses same construction rules as live portfolio (EW Top-30, sizing adjustments)
- Tracks entry/exit dates, holding period returns, and attribution
- Does NOT include transaction costs (those are modeled separately in backtest/cost_model.py)

### Shadow Tracker (coinvest_shadow_tracker v2)

7 arms tracking different signal combinations and construction variants:
- Wired into `run_daily.py`
- Accumulates true out-of-sample data daily
- Evaluate after 30+ trading days
- DO NOT backfill from historical

---

## Trade Plan Generation

### Inputs

1. Current shadow portfolio holdings
2. New `rankings.csv` from today's pipeline run
3. Position sizing adjustments (dilution haircut, size multiplier)

### Outputs

1. **Buys:** Names entering the top-30 that are not currently held
2. **Sells:** Names exiting the top-30 that are currently held
3. **Size adjustments:** Positions that remain but whose sizing changed
4. **No-trade zone:** Names within the top-30 that require no action

### Turnover Considerations

- Monthly rebalance frequency assumed in cost modeling
- High turnover signals potential construction damage (see dead lane: fixed sleeve budgets, +153.6pp drag)
- Jaccard similarity between consecutive top-30 lists is a health metric

---

## Broker Integration

### Robinhood

- **Routine:** Robinhood Activity Tracker (incoming_email trigger)
- **Functionality:** Processes incoming Robinhood trade emails, logs activity (executions, options, proxy votes, dividends, transfers)
- **Output:** Robinhood Activity Log document
- **Limitations:** Read-only email parsing; does not execute trades programmatically

### Alpaca

- **Status:** API integration available (per user profile technical infrastructure)
- **Use case:** Potential programmatic trade execution
- **Current state:** Not wired into the daily pipeline

---

## Key Constraints

1. Shadow portfolio is the primary execution context; live trading is not automated
2. Position sizing is deterministic given `ev_severity_score`
3. Equal-weight construction is the production standard; rank-weighting is a dead lane
4. Transaction cost modeling is in the backtest layer, not the execution layer
5. All trade plans are advisory outputs -- execution requires operator action