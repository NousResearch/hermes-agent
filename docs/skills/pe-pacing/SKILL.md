# PE Pacing Skill

## Purpose

Reference for the private equity pacing model within the Wake Robin Liquidity Architecture. Covers the Takahashi-Alexander deterministic model, STAIRS market-coupled adapter, capital call obligations, and workbook reconciliation with configurable gates.

---



## Operator Context

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **PE expertise**: CAIA-credentialed. Career-long experience across private equity, private real estate, infrastructure, timber, and commodities. Managed PE sleeve allocations within $14B+ multi-asset portfolios (NDRIO). Evaluated GP terms, commitment pacing, and J-curve management for state pension and sovereign wealth fund portfolios. Led manager searches and due diligence across alternatives.
- **Reconciliation authority**: The operator is the sole authority for reconciliation gate overrides (requires_override level) and hard_fail investigations. All STAIRS adapter design decisions and PE sleeve policy changes require explicit operator approval.



## Takahashi-Alexander Model

**File**: `src/aa_model/pe/ta_model.py`

Deterministic PE cash-flow projection used as the canonical PE engine.

### Default Parameters \(pinned in `configs/pe_pacing.yaml`\)

| Parameter | Default | Description |
| --- | --- | --- |
| lifetime\_years | 12 | Total fund life |
| commitment\_period\_years | 4 | Capital call window |
| rate\_of\_contribution | \[0.25, 0.30, 0.25, 0.20\] | Year-by-year call schedule \(sums to 1.0\) |
| bow | 2.5 | Distribution curve shape |
| yield\_pct | 0.0 | Current yield |
| growth\_pct | 0.13 | NAV growth rate |

### Golden CSV Regression Test

`tests/golden/ta_single_fund.csv` generated from a single fund: $100M commitment, 2024Q1 vintage, default params, 48 quarters. Regression test asserts byte-equality.

### Ledger Integration

PE flows on the quarterly ledger follow canonical ordering:

1. `pe_call` \- capital deployed \(negative amount\)
2. `pe_distribution` \- capital returned \(positive amount\)
3. `pe_nav_mark` \- NAV growth + yield \(positive amount\)

PE buckets are illiquid and DO NOT participate in rebalancing \(Phase 8 liquidity overlay\).

---

## STAIRS Adapter

**File**: `src/aa_model/pe/stairs_adapter.py`

Market-state-coupled PE pacing. Adjusts commitment timing and distribution pace based on public market conditions.

Design locked at `993a751`. Implementation blocked until tests + invariants drafted.

---

## Capital Call Obligation Bridge \(Phase 19\)

**File**: `src/aa_model/pe/call_obligation.py`

Bridges PE pacing model projections to actual capital call obligations from the cash-flow workbook.

### Source Precedence

When workbook and model disagree on capital calls:

- **Default**: workbook-classified line wins; model-derived projection is cross-check
- **Override**: configurable per obligation source via reconciliation gates

### Obligation Sources \(Canonical Taxonomy\)

| Source | Description |
| --- | --- |
| explicit\_config | Manually configured obligations |
| cashflow\_workbook | From Cashflow Modeling v7.xlsx |
| pe\_pacing\_model | TA or STAIRS model output |
| investment\_summary | From Investment Summary workbook |
| synthetic\_fixture | Test fixtures |

---

## Workbook Reconciliation \(Phase 20\)

**File**: `src/aa_model/pe/call_reconciliation.py`

Compares model-projected PE calls against workbook capital-call lines per quarter per fund.

### Reconciliation Output

Per-quarter, per-source comparison showing:

- Model-projected call amount
- Workbook-classified call amount
- Delta \(absolute and percentage\)
- Reconciliation verdict

---

## Configurable Reconciliation Gates \(Phase 21\)

**File**: `src/aa_model/pe/reconciliation_gates.py`

Four-tier gate classification for reconciliation discrepancies:

| Gate Level | Behavior |
| --- | --- |
| advisory | Log only, no blocking |
| warning | Log + surface in report |
| requires\_override | Block unless operator explicitly overrides |
| hard\_fail | Block unconditionally, fail the run |

Gate thresholds are configurable per obligation type and per fund.

---

## PE Sleeve Policy

From `configs/base.yaml`:

- `pe.sleeve_target_pct`: 0.25 \(25% of total portfolio\)
- `pe.scope`: \[buyout\]
- Supported sub-strategies: buyout, venture, growth, infra, re, pc

### Illiquidity Overlay \(Phase 8\)

The liquidity overlay in `allocation/liquidity_overlay.py` enforces that PE buckets are structurally illiquid:

- PE NAV is marked but NOT available for rebalancing
- Liquid NAV residual \(public\_equity, public\_bond, cash\) absorbs all rebalancing
- This prevents the model from treating PE NAV as spendable liquidity

---

## Phase 23: PE Real-Data Commitment Input Layer

Design locked at `f81ff43`. Implementation pending.

Purpose: Replace synthetic PE commitment fixtures with real fund commitment data ingested from the Investment Summary workbook, enabling realistic call pacing against actual fund terms.

---

## Key Constraints

1. TA model is NOT behind an adapter in Phase 1 - it is the canonical implementation
2. STAIRS is optional, wrapped behind adapter interface
3. PE flows must follow canonical intra-quarter ordering on the ledger
4. PE call obligations carry provenance from the canonical source taxonomy
5. Reconciliation gates are configurable but default to workbook-wins precedence
6. All PE projections are deterministic \(seeded RNG from base.yaml\)

---

## Source Files

| Component | File |
| --- | --- |
| TA Model | `src/aa_model/pe/ta_model.py` |
| TA Adapter | `src/aa_model/pe/ta_adapter.py` |
| STAIRS Adapter | `src/aa_model/pe/stairs_adapter.py` |
| PE Pacing | `src/aa_model/pe/pacing.py` |
| Call Obligation | `src/aa_model/pe/call_obligation.py` |
| Call Reconciliation | `src/aa_model/pe/call_reconciliation.py` |
| Reconciliation Gates | `src/aa_model/pe/reconciliation_gates.py` |
| PE Factory | `src/aa_model/pe/factory.py` |
| PE Config | `configs/pe_pacing.yaml` |
| Golden CSV | `tests/golden/ta_single_fund.csv` |
