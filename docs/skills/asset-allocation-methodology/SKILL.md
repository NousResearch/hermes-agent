# Asset Allocation Methodology

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Bridge the gap between the three SFO-specific skills (pe-pacing, sfo-liquidity-architecture, spending-liquidity) and the broader strategic allocation methodology. Those skills cover specific layers; this skill covers how the layers fit together and how key allocation decisions (CMAs, optimizer selection, rebalancing, scenarios) are made.

---

## Architecture Context

**Repo:** `Warrenpoobear/asset-allocation`
**Current phase:** 23 (PE real-data commitment input layer)
**Authoritative docs:** MODEL_DOCUMENTATION.md (567KB), PROJECT_SCOPE.md (17KB), SPEC.md (19KB)
**Tests:** 386 passing
**Stack:** Python 3.12, pydantic v2, numpy, pandas, pyarrow

### Layer Integration

The seven-layer SFO stack (from sfo-liquidity-architecture) forms the core. This skill focuses on how the Allocation/Policy layer (3.7) draws from all other layers:

```
3.1 Entity layer (not yet built)
3.2 Account/Position layer (partial)
3.3 Cash-flow layer (not yet built)
3.4 PE Pacing layer (shipped) -----> feeds PE call/distribution projections
3.5 RE + OpCo layer (not yet built)
3.6 Liquidity layer (partial) -----> enforces illiquidity overlay
3.7 Allocation/Policy layer (shipped) <-- THIS SKILL'S FOCUS
```

---

## Capital Market Assumptions (CMAs)

**Config:** `configs/cma.yaml`

### Baseline CMAs (Immutable)

| Bucket | Expected Return | Volatility | Liquidity |
| --- | --- | --- | --- |
| cash | Low | 0.005 | Liquid |
| public_bond | Moderate | 0.04 | Liquid |
| public_equity | Higher | 0.16 | Liquid |
| pe_buyout | Highest | 0.20 | Illiquid |

**Rule:** CMA baseline is immutable. All analysis starts from this baseline. Changes are expressed as scenario perturbations, not baseline modifications.

### CMA Construction (Institutional Context)

From Darren's career experience and the research digests:

| Method | Description | Use Case |
| --- | --- | --- |
| Building blocks | Risk-free rate + risk premia by asset class | Baseline construction |
| Historical analysis | Long-run return/vol/correlation from historical data | Calibration check |
| Equilibrium models | CAPM-implied returns from market cap weights | Reverse optimization |
| Forward-looking | Current yields, spreads, earnings yields, P/E normalization | Tactical adjustment |
| Consultant CMAs | Callan, J.P. Morgan, Lombard Odier annual publications | External benchmark |

**Darren's experience:** Worked with Callan on capital markets assumptions for NDRIO ($14B+). Conducted asset allocation reviews incorporating capital markets projections, risk/return optimization, and rebalancing frameworks.

### Scenario Generation

**Config:** `configs/scenarios.yaml`

Scenarios are perturbations to the CMA baseline. They do NOT modify the baseline itself.

| Scenario Type | What Changes | Use |
| --- | --- | --- |
| Return shock | Expected returns shifted +/- | Stress testing |
| Volatility shock | Volatilities scaled | Tail risk analysis |
| Correlation shock | Correlation matrix perturbed | Diversification stress |
| Regime-specific | Multiple parameters shifted simultaneously | Realistic stress scenarios |

---

## Allocation Engines

### Engine ABC

**File:** `src/aa_model/allocation/base.py`

All allocation engines implement a common interface, enabling swap-and-compare analysis.

| Engine | File | Status | Character |
| --- | --- | --- | --- |
| Stub | `stub.py` | Production reference | Config-driven fixed weights from `public_allocation.yaml` |
| Riskfolio | `riskfolio_adapter.py` | Shipped (Phase 3a) | Mean-variance optimization family |
| cvxportfolio | `cvxportfolio_adapter.py` | Shipped (Phase 3b) | Cost-aware, transaction-cost-sensitive |
| Liquidity overlay | `liquidity_overlay.py` | Shipped (Phase 8) | Illiquidity enforcement |

### Engine Selection Criteria

| Criterion | Stub | Riskfolio | cvxportfolio |
| --- | --- | --- | --- |
| Simplicity | Best | Moderate | Complex |
| Transaction cost awareness | None | Limited | Strong |
| Illiquidity handling | Manual | Via overlay | Via overlay |
| Rebalancing optimization | None | Static | Dynamic |
| Estimation error sensitivity | None (fixed weights) | HIGH (Markowitz curse) | Moderate (regularized) |

### Liquidity Overlay (Phase 8)

The liquidity overlay is NOT an allocation engine -- it is a post-optimizer constraint layer:
- PE buckets are structurally illiquid and DO NOT participate in rebalancing
- Liquid NAV residual (public_equity, public_bond, cash) absorbs all rebalancing
- This is the first place the model structurally honors the four-line principle

---

## Rebalancing Framework

### Band Calibration

Rebalancing bands define the tolerance around target weights before triggering trades.

| Consideration | Impact |
| --- | --- |
| Narrower bands | More frequent rebalancing, higher transaction costs, tighter tracking |
| Wider bands | Less frequent rebalancing, lower costs, more drift |
| Asset class volatility | Higher-vol assets need wider bands to avoid excessive trading |
| Tax implications | Taxable accounts favor wider bands (defer capital gains) |

### Rebalancing Constraints

- Rebalance flows are zero-sum per quarter on the ledger
- PE buckets are excluded from rebalancing (illiquidity overlay)
- Reserve floor (18 months of spending) is enforced BEFORE rebalancing
- All rebalancing happens on the quarterly ledger spine

---

## Spending Policy Integration

The spending rate denominator is the critical bridge between allocation and liquidity:

| Spending Base | What's Included | Appropriate When |
| --- | --- | --- |
| Total NAV | Everything (including PE, development RE, OpCo) | Legacy models, simple entities |
| Liquid NAV | Only liquid buckets | Conservative, multi-entity SFO |
| Distributable income | Income-producing assets only | Most realistic for this SFO |

**Why this matters:** Total NAV materially overstates spendable resources for Wake Robin's multi-entity structure. The spending base denominator is a policy decision, not a calculation -- it must be explicitly configured.

---

## Institutional Allocation Expertise (Operator Context)

Darren's allocation background provides the domain lens for this model:

- **North Dakota Legacy Fund:** Built from inception (100% cash) to diversified multi-asset allocation. Drew parallels to Norway's Government Pension Fund and Alaska Permanent Fund.
- **NDRIO pension/insurance:** 8 separate pension funds + insurance pool, ~$14B combined. Asset allocation reviews with Callan.
- **Fixed income redesign:** Redesigned fixed income allocation to integrate short-duration and unconstrained mandates for rising-rate environment.
- **Alternatives:** PE, private real estate, infrastructure, timber, commodities, inflation-linked assets.
- **Spending policy:** Developed ~2-3% spending policies and rebalancing frameworks for state investment boards.

---

## Research Digest Integration (May 2026)

### AI-Enhanced Allocation (from research digests)

| Approach | Source | Finding |
| --- | --- | --- |
| Self-Driving Portfolio | arXiv 2604.02279 | ~50 agents generating CMAs, 20+ construction methods, meta-agent learning |
| LLM-Enhanced Black-Litterman | arXiv 2504.14345 | LLMs as systematic view generators; different LLMs = different "styles" |
| CNN-LSTM + LLM hybrid | Springer 2025 | 1.623 Sharpe, LLM sentiment alone adds +28% Sharpe improvement |
| Regime-aware allocation | Multiple 2025-2026 | +0.37-1.0 Sharpe improvement through regime conditioning |

### Family Office Allocation Gap

65% of family offices prioritize AI investments, but 50%+ have zero allocation to growth equity/VC and 79% lack infrastructure exposure -- the very markets where AI innovation occurs. This structural misalignment is an opportunity for Wake Robin's allocation framework.

---

## Key Constraints

1. MODEL_DOCUMENTATION.md (567KB) is doc-as-spec for the asset-allocation repo
2. Every behavior change requires a matching MODEL_DOCUMENTATION.md update
3. Phase gates are real -- design-lock commit before implementation
4. CMA baseline is immutable; scenarios are perturbations
5. Quarterly ledger is the spine; all flows land on it
6. Cash-flow worksheet alignment across 4 dimensions (timing, flow, source, reconciliation)