---
name: validation
triggers:
  - pipeline validation
  - PIT enforcement
  - staleness gate
  - data quality gate
  - circuit breaker
  - governance metadata
  - determinism
  - score bounds
  - IC benchmark
  - pre-run checklist
  - post-run checklist
description: "Validate Wake Robin biotech screener pipelines."
---

# Validation & Governance Skill

## Purpose

Define the go/no-go gates, data quality checks, staleness windows, IC thresholds, and governance requirements that every pipeline run must satisfy before producing output. Fail-closed philosophy: uncertain or stale data triggers exclusion, not graceful degradation.

## Preconditions

- Pipeline runs MUST have an explicit `as_of_date` parameter (never `datetime.now()`).
- All validation uses `Decimal` arithmetic where scores are involved.
- PIT cutoff: `source_date <= as_of_date - 1` (standard) or `source_date < as_of_date - 2` (strict mode).

---

## Gate 1: Point-in-Time (PIT) Enforcement

| Rule | Formula | Consequence |
|------|---------|-------------|
| Standard PIT | `source_date <= as_of_date - 1 day` | Data admitted |
| Strict PIT | `source_date < as_of_date - 2 days` | Extra buffer for intraday data |
| Lookahead | `age_days < 0` (future data) | **Reject unconditionally** |

Every record must pass PIT admissibility before entering any scoring module. No exceptions.

---

## Gate 2: Data Staleness (Phase-Dependent)

### Financial Data

| Level | Age (days) | Penalty |
|-------|-----------|---------|
| PASS | <= 60 | 1.0x |
| WARN | 60-90 | 1.0x |
| SOFT_GATE | 90-120 | 0.5x |
| HARD_GATE | > 120 | **Exclude** |

### Trial Data

| Level | Phase 3 | Phase 2 | Phase 1 |
|-------|---------|---------|---------|
| PASS | <= 90d | <= 180d | <= 270d |
| WARN | 90-120d | 180-270d | 270-365d |
| SOFT_GATE | 120-180d (0.6x) | 270-365d (0.7x) | 365-545d (0.8x) |
| HARD_GATE | > 180d | > 365d | > 545d |

### Market, Short Interest, 13F Data

| Type | PASS | WARN | SOFT_GATE | HARD_GATE | Penalty |
|------|------|------|-----------|-----------|---------|
| Market | <= 3d | 3-5d | 5-10d | > 10d | 0.3x soft |
| Short Interest (FINRA 2-wk lag) | <= 20d | 20-30d | 30-45d | > 45d | 0.5x soft |
| 13F Holdings (45-day SEC lag) | <= 60d | 60-90d | 90-135d | > 135d | 0.4x soft |

**SEC_13F_FILING_LAG_DAYS**: 45 (built-in constant).

---

## Gate 3: Data Quality Hard Gates

| Gate | Threshold | Action |
|------|-----------|--------|
| Financial data age | > 90 days | Exclude from scoring |
| Market data age | > 7 days | Exclude from scoring |
| Trial data age | > 30 days | Exclude from scoring |
| Liquidity (ADV) | < $500,000/day | Exclude ticker |
| Price (penny stock) | < $5.00 | Exclude ticker |
| Market field coverage | < 80% fields present | Exclude ticker |
| Financial field coverage | < 50% fields present | Issue warning |

---

## Gate 4: Circuit Breakers

| Condition | Threshold | Action |
|----------|-----------|--------|
| Records failing validation | > 20% | Log warning |
| Records failing validation | > 50% | **Fail entire pipeline** |
| Minimum records for check | < 10 | Skip circuit breaker check |

---

## Gate 5: Input Validation

| Validation | Rule |
|-----------|------|
| Ticker format | `^[A-Z]{1,5}$` |
| Minimum date | >= 1990-01-01 |
| Cash | Non-negative |
| Market cap | Positive |
| Maximum runway | <= 1200 months |
| Valid records % | >= 10% must pass |

---

## Gate 6: Score Bounds Validation

All scores must fall within [0, 100]:

| Score Field | Module |
|------------|--------|
| financial_score | Module 2 |
| clinical_score | Module 4 |
| catalyst_score | Module 3 |
| score_blended | Module 3 v2 |
| composite_score | Module 5 |

Any score outside [0, 100] is a pipeline error. Fail-closed.

---

## Gate 7: Weight Sum Validation

Module 5 component weights must sum to 1.0 ± 0.01. Outside tolerance = configuration error. Fail-closed.

---

## Gate 8: Module Coverage Minimums

| Module | Minimum Coverage | Action if Below |
|--------|-----------------|-----------------|
| Module 2 (Financial) | 80% of universe | Warning |
| Module 3 (Catalyst) | 80% of universe | Warning |
| Module 4 (Clinical) | 80% of universe | Warning |

---

## Gate 9: Severity System

| Level | Score Multiplier | Action |
|-------|-----------------|--------|
| NONE | 1.0 | Include |
| SEV1 | 0.90 | Include with flag |
| SEV2 | 0.50 | Include, soft gate |
| SEV3 | 0.00 | **Exclude** (hard gate) |

---

## Gate 10: Pipeline Health Status

| Component | Threshold | Status if Below |
|----------|-----------|----------------|
| catalyst_raw | 10% | DEGRADED |
| momentum | 0% | OPTIONAL |
| smart_money | 0% | OPTIONAL |
| market_data | 0% | OPTIONAL |

**Status**: OK (all met) / DEGRADED (optional below threshold) / FAIL (catalyst < 5%)

---

## Gate 11: Snapshot Content Collapse Guards (2026-05-08)

Post-hash checks detecting content-level signal collapse:

| Check | Threshold | Verdict |
|-------|-----------|---------|
| coinvest_score_z SD | <= 0.10 | FAIL |
| catalyst_quality classification | < 90% among has_catalyst_signal=1 | FAIL |
| No has_catalyst_signal=1 rows | n/a | WARN |

**Tool**: `tools/verify_snapshot_integrity.py` (Section 4)

Catches silent degradation (e.g., all tickers receiving identical coinvest scores via fallback path).

---

## Gate 12: Expectation Layer Coverage Gate (Spec 105)

Hard-fails pipeline if market-expectation fields are missing or under-covered in `rankings.csv`.

| Field | Required Coverage |
|-------|------------------|
| `short_interest_pct` | 0.90 |
| `close_price` | 0.99 |
| `market_cap_mm` | 0.95 |
| `priced_move_pct` | 0.80 |
| `insider_net_buy_value_90d` | 0.30 (nonblocking / diagnostic) |

`FEATURE_COVERAGE_REQUIREMENTS` is the single source of truth. Runs at Step 5 (Gates) every execution.

---

## IC Quality Benchmarks

| Quality | IC Range | Action |
|---------|---------|--------|
| Excellent | IC > 0.05 | Deploy |
| Good | IC 0.03-0.05 | Use with confidence |
| Weak | IC 0.01-0.03 | Monitor |
| Noise | IC < 0.01 | Abandon signal |
| Negative | IC < 0 | Investigate inversion |

### IC Constants

| Constant | Value |
|----------|-------|
| MIN_OBS_IC | 10 |
| MIN_OBS_TSTAT | 20 |
| MIN_OBS_BOOTSTRAP | 30 |
| MIN_ROLLING_WINDOW | 12 weeks |
| BOOTSTRAP_ITERATIONS | 1000 |
| TSTAT_THRESHOLD_95 | 2.0 |
| TSTAT_THRESHOLD_99 | 2.58 |

### Forward Return Horizons

1w (5d), 2w (10d), 1m (20d), 1.5m (30d), 3m (60d), 4.5m (90d)

---

## Regime Data Staleness Haircuts

| Data Age | Confidence Multiplier |
|---------|----------------------|
| <= 2 days | 1.00 |
| 3-5 days | 0.85 |
| 6-10 days | 0.65 |
| > 10 days | 0.00 (force UNKNOWN regime) |

---

## Determinism Enforcement

| Setting | Required Value |
|---------|---------------|
| force_deterministic_timestamps | true |
| sort_output_keys | true |
| include_content_hashes | true |
| random_seed | 42 |

Rules:
1. Same inputs MUST produce byte-identical outputs
2. All JSON serialization uses sorted keys
3. Content hashes (SHA256) included in every output
4. No external API calls during scoring
5. All timestamps derived from `as_of_date`, never wall clock

---

## Governance Metadata Requirements

Every pipeline output MUST include:

```json
{
  "_governance": {
    "run_id": "<deterministic-hash>",
    "score_version": "<version>",
    "schema_version": "<version>",
    "parameters_hash": "sha256:<hash>",
    "pit_cutoff": "<ISO-date>",
    "as_of_date": "<ISO-date>"
  }
}
```

### Audit Stages

INIT → LOAD → ADAPT → FEATURES → RISK → SCORE → REPORT → FINAL

Status values: OK / FAIL / SKIP

### Standard Error Codes

MISSING_INPUT, SCHEMA_MISMATCH, HASH_ERROR, PARAMS_MISSING, MAPPING_MISSING, VALIDATION_ERROR, UNKNOWN_ERROR

---

## Diagnostic Fields Registry (Spec 104)

Fields tracked for observability but explicitly excluded from scoring, ranking, and selection.

| Field | Status |
|-------|--------|
| `insider_net_buy_value_90d` | DIAGNOSTIC ONLY |

### Insider Model Isolation Guard (CRITICAL)

`insider_net_buy_value_90d` must NOT enter the expectation model's `market_features` input. The model has an `insider_net_buy_z` weight that activates silently if the field flows upstream.

Rules:
- Never collapse blank (NaN) and zero (0.0) — they have different semantics
- Never impute zero for missing or blank for zero
- Promotion requires: 20+ stable snapshots, >= 60% coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval

---

## Pre-Run Checklist

1. `as_of_date` explicitly provided (never wall clock)
2. All input files exist and within size limits
3. PIT cutoff computed and logged
4. Schema versions match expected
5. Weight sums within tolerance
6. No `float` in scoring paths (only `Decimal`)
7. No `datetime.now()` in any module
8. No `random` without explicit seed
9. Audit log writer initialized
10. Run ID deterministically generated

## Post-Run Checklist

1. All output scores in [0, 100]
2. Governance metadata present in every output file
3. Content hashes match recomputed hashes
4. No SEV3 tickers in ranked output
5. Coverage metrics logged per-module
6. Circuit breaker did not trip silently
7. Staleness penalties applied where required
8. Audit log contains entries INIT through FINAL

---

## Source Files

| Component | File |
|----------|------|
| Data Quality Gates | `common/data_quality.py` |
| Staleness Gates | `common/staleness_gates.py` |
| PIT Enforcement | `common/pit_enforcement.py` |
| Input Validation | `common/input_validation.py` |
| Schema Validation | `common/schema_validation.py` |
| Production Hardening | `common/production_hardening.py` |
| IC Measurement | `backtest/ic_measurement.py` |
| Audit Log | `governance/audit_log.py` |
| Snapshot Integrity | `tools/verify_snapshot_integrity.py` |
| QA Check | `production_qa_check.py` |
| Pipeline Config | `config.yml` |
