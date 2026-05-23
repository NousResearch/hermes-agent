# Spending & Liquidity Skill

## Purpose

Reference for the spending rule and liquidity management layer of the Wake Robin Liquidity Architecture. Critical context: for this multi-entity SFO, total NAV materially overstates spendable resources. Spending and liquidity decisions must be made against a separately-modeled spendable-resource base.

---



## Operator Context

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **Spending policy expertise**: Developed and maintained spending policies (~2-3% of fund) and rebalancing frameworks for the North Dakota Legacy Fund ($6B sovereign wealth) and NDRIO pension system ($6B pension + $2.3B insurance). Navigated legislative constraints on fund usage. Experience with actuarial liaison on funding ratios and contribution adequacy.
- **Liquidity management background**: Managed liquidity across eight separate pension funds, an insurance pool, and a sovereign wealth fund. Designed illiquidity budgets incorporating PE, real estate, infrastructure, and timber allocations. The four-line principle reflects direct experience managing portfolios where appraisal value, development land, and OpCo interests materially overstated spendable resources.
- **Configuration authority**: Spending base denominator selection, reserve floor adjustments, and manager terms overrides require explicit operator approval.



## Spending Rules

**ABC**: `src/aa_model/spending/base.py`

### Flat-Real Rule

Fixed real spending amount, adjusted for inflation each quarter.

```
quarterly_spend = annual_spend / 4 * (1 + inflation_rate)^quarters_elapsed
```

Simple, predictable. No NAV sensitivity.

### Smoothing Rule

Weighted average of prior spend and current NAV-implied spend.

```
spend_t = weight * (rate * NAV_t) + (1 - weight) * spend_{t-1}
```

**Known limitation \(L7\)**: When `weight=0`, spending freezes at initial level regardless of NAV changes. Documented, not a bug.

### Owl / Guyton-Klinger Guardrail Rule

**File**: `src/aa_model/spending/owl_adapter.py` \(16KB, full implementation\)

Advanced guardrail-based withdrawal strategy with prosperity and capital preservation triggers.

**Key behaviors**:

- Initial spend rate applied to spending base
- Prosperity rule: increase spend when portfolio grows above threshold
- Capital preservation rule: decrease spend when portfolio falls below threshold
- Guardrails define maximum and minimum spend adjustments per period

**Phase 11 fix \(L16\)**: Owl now uses absolute-dollar guardrail thresholds, making it scale-invariant in initial NAV. Prior version reacted to forecasted NAV, not realized NAV \(L15, resolved Phase 4a\).

---

## Spending Base \(Phase 12/12.5\)

**File**: `src/aa_model/spending/spending_base.py` \(12.6KB\)

### The Core Problem \(L19\)

Total NAV includes:

- PE fund NAV \(illiquid, not spendable\)
- Development real estate \(land carry, not income-producing\)
- OpCo interests \(not automatically liquid\)

Using total NAV as the spending rate denominator would suggest the family can spend far more than is actually accessible.

### Configurable Denominator

The spending base is configurable to use different NAV measures:

| Base Type | What It Includes | Use Case |
| --- | --- | --- |
| total\_nav | Everything | Legacy/simple models |
| liquid\_nav | Only liquid buckets | Conservative |
| distributable\_income | Income-producing assets only | Most realistic for SFO |

### Distribution Inflow \(Phase 13\)

`distribution_inflow` is a new ledger flow type representing income distributions from illiquid assets that become spendable. This bridges the gap between illiquid NAV and actual spending capacity.

---

## Liquidity Coverage

**File**: `src/aa_model/liquidity/coverage.py` \(21KB\)

### Reserve Floor

Default: 18 months of spending in cash + short-term bonds \(configurable via `liquidity.floor_months` in base.yaml\).

### Coverage Ratios

Per-period obligations vs period-available liquidity by tier.

### Five-Tier Liquidity Classification

| Tier | Granularity | Examples |
| --- | --- | --- |
| 1 | Daily | Cash, money market |
| 2 | Monthly | Public equities, liquid bonds |
| 3 | Quarterly | Hedge funds with quarterly redemption |
| 4 | At maturity | Fixed-term vehicles |
| 5 | Locked | PE, development RE, OpCo |

### Breach Alerts

Triggered when projected outflows exceed tier 1-2 capacity within the planning horizon.

---

## Manager Terms Diagnostics

**File**: `src/aa_model/liquidity/manager_terms_diagnostics.py` \(19.5KB\)

Per-manager liquidity analysis:

- Redemption notice periods
- Lock-up periods
- Gate provisions
- Side pocket exposure
- Effective liquidity tier assignment

---

## Workbook Integration

### Cashflow Modeling v7.xlsx \(Read-Only\)

The canonical entity-level cash-flow forecast. The model reads, normalizes, and reconciles to it. The model does NOT mutate the workbook.

### Investment Summary Workbook \(Read-Only\)

Canonical position universe with per-position metadata:

- Liquidity bucket \(1-5\)
- Liquidity granularity \(Daily/Monthly/Quarterly/At Maturity\)
- Cash-flow-producing flag
- Time horizon
- Expected standard deviation

---

## Key Constraints

1. Spending rate denominator must be explicitly configured, not defaulted to total NAV
2. All spending flows land on the quarterly ledger as `flow_type = "spend"`
3. Reserve floor is enforced before rebalancing
4. Liquidity tiering must honor the four-line principle
5. Workbook data is read-only; validation divergence is logged, not silently absorbed

---

## Source Files

| Component | File |
| --- | --- |
| Spending ABC | `src/aa_model/spending/base.py` |
| Rules \(flat-real, smoothing\) | `src/aa_model/spending/rules.py` |
| Owl Adapter | `src/aa_model/spending/owl_adapter.py` |
| Spending Base | `src/aa_model/spending/spending_base.py` |
| Liquidity Coverage | `src/aa_model/liquidity/coverage.py` |
| Manager Terms | `src/aa_model/liquidity/manager_terms_diagnostics.py` |
| Spending Config | `configs/spending.yaml` |
| Base Config | `configs/base.yaml` |
