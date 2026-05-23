# Decision Audit Trail

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 4 of 7

## Purpose

Annotate key design decisions, parameter choices, and closed research lanes with their rationale, the evidence base at the time, and the conditions under which the decision should be revisited.

## Decision Entry Schema

```
decision_id: D-YYYY-NNN
date: YYYY-MM-DD
component: [skill or module name]
decision: One-line summary
rationale: Why this choice was made
evidence: What data/analysis supported the decision
alternatives_considered: What else was evaluated and why rejected
revisit_conditions: When this decision should be re-examined
related_specs: [spec numbers]
status: ACTIVE | SUPERSEDED | UNDER_REVIEW
```

---

## Catalog

### D-2026-001 | Gate 3 Penny Stock Threshold ($5.00)

- **Component:** biotech-validation (Gate 3)
- **Decision:** Exclude all tickers below $5.00 from the rankable universe
- **Rationale:** Not documented in any current skill. The W7 note explains the relationship between the $5.00 gate and the $2.00 financial-health penalty, but not why $5.00 was chosen.
- **Evidence:** MISSING. Presumably based on institutional liquidity constraints and SEC penny stock definitions.
- **Alternatives considered:** UNKNOWN.
- **Revisit conditions:** If the biotech universe shifts significantly in price distribution (e.g., many clinical-stage companies trading $3-5 after market correction), this gate could exclude otherwise legitimate candidates.
- **Status:** ACTIVE (needs rationale backfill)

### D-2026-002 | Construction Size K=30

- **Component:** selector-ranker
- **Decision:** Equal-weight top 30 names by final_score
- **Rationale:** Validated by PIT sweep (stable K=25-35 plateau, net-of-cost peak)
- **Evidence:** PIT sweep results exist but are not linked from the skill.
- **Alternatives considered:** K=20 (too concentrated, higher turnover), K=40 (dilution of conviction signal). RW-EW delta = -0.09pp showed rank-weighting does NOT help.
- **Revisit conditions:** If the rankable universe shrinks significantly (e.g., from coinvest-only selector reducing eligible names), K=30 may be too large. If universe expands, K=30 may be too small.
- **Status:** ACTIVE

### D-2026-003 | inst_delta_z Zeroing Threshold

- **Component:** institutional-signal, selector-ranker
- **Decision:** Zero inst_delta_z weight in selector (v1.14.0, 2026-05-04)
- **Rationale:** Mean IC = -0.097 over 36 dates, confirmed across two independent measurement frames
- **Evidence:** Two-frame IC confirmation. Negative IC means the signal was pointing the wrong direction.
- **Alternatives considered:** Reduce weight (e.g., 35% to 15%) instead of zeroing. Rejected because negative IC means any positive weight is actively harmful.
- **Revisit conditions:** Reinstatement requires IC recovery evidence documented in governance log. Signal remains active in ranker (NW-t = +3.32) where it operates within a different scope.
- **Status:** ACTIVE

### D-2026-004 | Contamination Window Duration (20 Trading Days)

- **Component:** institutional-signal
- **Decision:** After adding new managers, IC measurements during a 20-trading-day window are flagged as contaminated
- **Rationale:** Not explicitly documented. Presumably reflects time needed for score distribution to stabilize after manager addition.
- **Evidence:** MISSING. Was this calibrated empirically?
- **Alternatives considered:** UNKNOWN.
- **Revisit conditions:** If manager additions cause longer-duration score instability, or if clearance trigger (>=34/48 managers filed) interacts with contamination windows unexpectedly.
- **Status:** ACTIVE (needs rationale backfill)

### D-2026-005 | financial_score Negative Ranker Weight (Stress-Upside Thesis)

- **Component:** selector-ranker
- **Decision:** financial_score has weight -0.0533 in the production ranker, penalizing financially safe names
- **Rationale:** Intentional stress-upside thesis (Spec 074, reconfirmed Spec 093). Within the coinvest-selected universe, financially safe names are less catalytic.
- **Evidence:** Six-diagnostic audit confirmed TRUE PENALTY in both bull (NW-t = -3.42) and bear (-3.38) regimes. Persists across cohorts and regimes, ruling out artifact.
- **Alternatives considered:** Zeroing financial_score in ranker (would remove tilt). Using absolute value (would lose directional signal).
- **Revisit conditions:** If coinvest universe shifts toward earlier-stage cash-burning companies where financial health IS the binding constraint. Also revisit if bear-market drawdowns exceed historical bounds.
- **Status:** ACTIVE

### D-2026-006 | Dead Lane Closures (11 Research Lanes)

- **Component:** selector-ranker
- **Decision:** 11 research lanes closed/dead
- **Rationale and key learnings:**
  - *Options surface-shape (DEAD):* 50-month IC negative all horizons. Vol dominated by binary catalyst outcomes, not information flow.
  - *Options-as-alpha (Spec 053, CLOSED):* 37 signals tested, ALL fail. Options market for small-cap biotech too thin and binary-event-driven for continuous alpha.
  - *Static execution features (Spec 054, CLOSED):* All noise/destructive. Trade execution characteristics don't predict forward returns.
  - *Clinical composites as ranker (Spec 055, CLOSED):* Negative across ALL slices. Clinical stage doesn't predict relative performance within coinvest-selected universe (coinvest already incorporates clinical conviction).
  - *Fixed sleeve budgets (RETIRED):* Primary construction damage (+153.6pp drag). Forcing sector/stage allocation constraints destroyed the most value of any single design choice.
- **Revisit conditions:** Only with fundamentally new data (new options data source, or structural change to biotech market microstructure). Do not reopen on hope.
- **Status:** ACTIVE (all lanes remain closed)

### D-2026-007 | DEM Tier 6 Decision: Local LLMs Deferred

- **Component:** openclaw-agent-optimize
- **Decision:** Do not deploy local LLMs for agent inference at this time
- **Rationale:** Cost-performance analysis: local LLM breakeven vs frontier APIs is 3-6 months on DGX Spark; against cheap open-weight cloud APIs ($0.30/1M tokens), local never breaks even within hardware life.
- **Evidence:** Benchmark data in `local-llm-cost-vs-performance-agents-2026` (ai-projects). Qwen 2.5 Coder 32B scored 9.3/10 on tool calling but requires 24GB+ VRAM (current hardware: 16GB).
- **Alternatives considered:** DGX Spark purchase ($3K), cloud GPU rental, hybrid local+API routing
- **Revisit conditions:** If API costs increase significantly, if hardware costs drop, or if a breakthrough model fits in 16GB VRAM with comparable quality
- **Status:** ACTIVE

---

## Usage Rules

1. When a parameter or threshold is questioned, check this catalog first.
2. If no entry exists, flag the decision as needing rationale backfill (mark `evidence: MISSING`).
3. When conditions change (market regime shift, universe expansion, new data source), scan revisit_conditions for affected decisions.
4. SUPERSEDED decisions should retain their full history -- do not delete, only change status.