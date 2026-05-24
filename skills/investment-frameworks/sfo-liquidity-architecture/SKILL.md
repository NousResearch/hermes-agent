---
name: sfo-liquidity-architecture
triggers:
  - SFO architecture
  - Wake Robin liquidity
  - four-line principle
  - quarterly ledger
  - allocation engine
  - asset allocation model
  - ledger spine
  - PE pacing architecture
  - liquidity overlay
  - capital market assumptions
  - cash-flow workbook
description: >
  Architecture reference for the Wake Robin Liquidity Architecture — a
  deterministic, multi-engine modeling stack for a Gen3-Gen5 single-family
  office. Covers the four-line principle (NAV is not liquidity), seven-layer
  architecture, quarterly ledger schema and invariants, allocation engines
  (stub/Riskfolio/cvxportfolio/liquidity overlay), spending rules, PE pacing
  integration, CMA baseline, and governance rules. NOT a generic asset-allocation
  framework — every decision is downstream of the four-line principle.
---

# SFO Liquidity Architecture Skill

## Purpose

Reference for the Wake Robin Liquidity Architecture — a deterministic, multi-engine modeling stack for a Gen3-Gen5 single-family office. Every modeling decision is downstream of the four-line principle.

---

## The Four-Line Principle (Load-Bearing)

```
NAV is not liquidity.
Appraisal value is not spending capacity.
Development / land value is not distributable income.
OpCo value is not automatically portfolio liquidity.
```

This principle governs every phase of work.

---

## Architecture Overview

**Repo**: `Warrenpoobear/asset-allocation`
**Current Phase**: 23 (PE real-data commitment input layer)
**Stack**: Python 3.12, pydantic v2, numpy, pandas, pyarrow

### Seven Layers (dependency order)

| Layer | Status | Description |
|-------|--------|-------------|
| 3.1 Entity | Not yet built | Multi-entity SFO chart (LLCs, trusts, individuals) |
| 3.2 Account/Position | Partial | Per-account holdings, asset-class taxonomy |
| 3.3 Cash-flow | Not yet built | Entity-by-entity quarterly forecast |
| 3.4 PE Pacing | Shipped (Phases 1,7,8) | Commitment, call, distribution, NAV projection |
| 3.5 RE + OpCo | Not yet built | Stabilized RE, development RE, operating companies |
| 3.6 Liquidity | Partial (Phase 8) | Illiquidity overlay; full tier system not built |
| 3.7 Allocation/Policy | Shipped (Phases 1-11) | Weights, bands, rebalance, spending, scenarios |

---

## Quarterly Ledger (The Spine)

**File**: `src/aa_model/integration/ledger.py`

Every module produces or consumes rows on the quarterly ledger. It is the central object.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| quarter | Period[Q] | e.g. 2026Q2 |
| bucket | str | public_equity, public_bond, cash, pe_buyout |
| flow_type | str | Canonical ordering below |
| amount_usd | float | Signed dollar impact |
| nav_start_usd | float | Bucket NAV before this flow |
| nav_end_usd | float | Bucket NAV after this flow |
| source | str | Producing module name |
| run_id | str | Manifest run ID |

### Canonical Intra-Quarter Flow Ordering

1. `inflow` — external contributions
2. `return` — mark-to-market on liquid buckets
3. `pe_call` — capital deployed into PE
4. `pe_distribution` — capital returned from PE
5. `pe_nav_mark` — PE NAV growth + yield
6. `spend` — withdrawals
7. `rebalance` — intra-portfolio transfer (sums to zero)

### Invariants

- Per-row: `nav_end_usd == nav_start_usd + amount_usd`
- Chain: within each (run_id, bucket) chain, start = prior end
- Rebalance is zero-sum per quarter
- Total NAV conservation: moves only via market P&L and external cash
- No NaN in amount, nav_start, nav_end
- Determinism: identical inputs produce byte-identical ledger.parquet

---

## Allocation Engines

**ABC**: `src/aa_model/allocation/base.py`

| Engine | File | Status |
|--------|------|--------|
| Stub | `stub.py` | Production reference (config-driven weights) |
| Riskfolio | `riskfolio_adapter.py` | Shipped (Phase 3a) |
| cvxportfolio | `cvxportfolio_adapter.py` | Shipped (Phase 3b, cost-aware) |
| Liquidity overlay | `liquidity_overlay.py` | Shipped (Phase 8) |

### Liquidity Overlay (Phase 8)

Liquid NAV residual rebalances; PE buckets do not. This is the first place the model structurally honors the four-line principle.

---

## Spending Rules

**ABC**: `src/aa_model/spending/base.py`

| Rule | Description |
|------|-------------|
| flat_real | Fixed real spending, inflation-adjusted |
| smoothing | Weighted average of prior spend and current NAV-implied |
| Owl (Guyton-Klinger) | Guardrail rules with prosperity/capital preservation triggers |

### Spending Base (Phase 12/12.5)

Configurable denominator for spending rate calculation. Critical: total NAV materially overstates spendable resources for this household.

### Liquidity Coverage

**File**: `src/aa_model/liquidity/coverage.py`

Coverage ratios, reserve floor (18 months default), shortfall frequency.

---

## PE Pacing

### Takahashi-Alexander Model

**File**: `src/aa_model/pe/ta_model.py` — deterministic PE cash-flow projection.

Default parameters: lifetime_years=12, commitment_period_years=4, rate_of_contribution=[0.25, 0.30, 0.25, 0.20], bow=2.5, growth_pct=0.13.

### STAIRS Adapter

**File**: `src/aa_model/pe/stairs_adapter.py` (Phase 7) — market-state-coupled PE pacing.

### Call Obligation & Reconciliation (Phases 19-21)

- `call_obligation.py` — PE capital call bridge (workbook-wins default)
- `call_reconciliation.py` — Workbook vs model reconciliation
- `reconciliation_gates.py` — Configurable gates (advisory / warning / requires_override / hard_fail)

---

## Cash-Flow Worksheet Alignment (Standing Constraint)

The model must stay aligned with `Cashflow Modeling v7.xlsx`. Four dimensions:

1. **Timing** — quarters, fiscal periods, lookahead windows match
2. **Flow** — spending, distributions, calls map to worksheet lines
3. **Source** — every obligation carries provenance (explicit_config / cashflow_workbook / pe_pacing_model / investment_summary / synthetic_fixture)
4. **Reconciliation** — reports show where model totals reconcile to worksheet

**Boundary rules**: Read the worksheet. Normalize it. Reconcile to it. Do NOT mutate it.

---

## Capital Market Assumptions

**File**: `configs/cma.yaml` — CMA baseline is immutable. Scenarios are perturbations.

| Bucket | Vol (annual) | Liquidity |
|--------|-------------|-----------|
| cash | 0.005 | liquid |
| public_bond | 0.04 | liquid |
| public_equity | 0.16 | liquid |
| pe_buyout | 0.20 | illiquid |

---

## Configuration

| Key | Default |
|-----|---------|
| governance.size_usd | 100,000,000 |
| solver.preferred | clarabel (fallback: scs, osqp) |
| liquidity.floor_months | 18 |
| pe.sleeve_target_pct | 0.25 |
| rebalance.frequency | quarterly |

---

## Governance Rules (Do Not Violate)

1. Ledger is sole state spine — no sidecars, no hidden state
2. CMA baseline immutable; scenarios are perturbations
3. No implementation before design lock (docs commit first)
4. MODEL_DOCUMENTATION.md updated for any behavior change
5. Identical inputs produce byte-identical ledger.parquet
6. No overwriting run directories
7. PROJECT_SCOPE.md is authoritative for reference architecture

---

## Source Files

| Component | File |
|----------|------|
| Quarterly Ledger | `src/aa_model/integration/ledger.py` |
| Orchestrator | `src/aa_model/integration/orchestrator.py` |
| Allocation (stub) | `src/aa_model/allocation/stub.py` |
| Riskfolio Adapter | `src/aa_model/allocation/riskfolio_adapter.py` |
| cvxportfolio Adapter | `src/aa_model/allocation/cvxportfolio_adapter.py` |
| Liquidity Overlay | `src/aa_model/allocation/liquidity_overlay.py` |
| Spending Rules | `src/aa_model/spending/rules.py` |
| Owl Adapter | `src/aa_model/spending/owl_adapter.py` |
| Spending Base | `src/aa_model/spending/spending_base.py` |
| Liquidity Coverage | `src/aa_model/liquidity/coverage.py` |
| Manager Terms | `src/aa_model/liquidity/manager_terms_diagnostics.py` |
| TA Model | `src/aa_model/pe/ta_model.py` |
| Run Script | `scripts/run_sfo_study.py` |
