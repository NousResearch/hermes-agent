# Validation & Governance Skill

## Purpose

Define the go/no-go gates, data quality checks, staleness windows, IC thresholds, and governance requirements that every pipeline run must satisfy before producing output. This skill encodes Wake Robin's fail-closed philosophy: uncertain or stale data triggers exclusion, not graceful degradation.

## Preconditions

- Pipeline runs MUST have an explicit `as_of_date` parameter \(never `datetime.now()`\).
- All validation uses `Decimal` arithmetic where scores are involved.
- PIT cutoff: `source_date <= as_of_date - 1` \(standard\) or `source_date < as_of_date - 2` \(strict mode\).

---

## Operator Governance Authority

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin \(Holland, MI\)
- **Governance role**: Sole authority for all pipeline governance decisions — spec approvals, spec closures, signal promotions, architecture changes, and production-hash rotations.
- **Relevant credentials**: CFA \(performance measurement, fiduciary standards, quantitative methods\), CAIA \(alternative investments, due diligence frameworks\). 30+ years institutional investment management including $14B+ AUM oversight as Deputy CIO/Interim CIO.
- **Fail-closed philosophy**: The operator's institutional governance experience \(state investment board presentations, investment policy development, regulatory compliance\) informed the DEM's fail-closed design — uncertain or stale data triggers exclusion, not graceful degradation.
- **Approval requirements**: All Checklist v2 promotion decisions, diagnostic field promotions \(Spec 104\), architecture freeze lifts, and QUARANTINE verdicts require explicit operator written approval before proceeding.

## Gate 1: Point-in-Time \(PIT\) Enforcement

| Rule | Formula | Consequence |
| --- | --- | --- |
| Standard PIT | `source_date <= as_of_date - 1 day` | Data admitted |
| Strict PIT | `source_date < as_of_date - 2 days` | Extra buffer for intraday data |
| Lookahead | `age_days < 0` \(future data\) | **Reject unconditionally** |

Every record must pass PIT admissibility before entering any scoring module. There are no exceptions.

---

## Gate 2: Data Staleness \(Phase-Dependent\)

### Financial Data

| Level | Age \(days\) | Action | Penalty |
| --- | --- | --- | --- |
| PASS | <= 60 | Green light | 1.0x |
| WARN | 60-90 | Log warning | 1.0x |
| SOFT\_GATE | 90-120 | Apply penalty | 0.5x |
| HARD\_GATE | \> 120 | **Exclude** | 0.0x |

### Trial Data - Phase 3

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 90 | 1.0x |
| WARN | 90-120 | 1.0x |
| SOFT\_GATE | 120-180 | 0.6x |
| HARD\_GATE | \> 180 | **Exclude** |

### Trial Data - Phase 2

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 180 | 1.0x |
| WARN | 180-270 | 1.0x |
| SOFT\_GATE | 270-365 | 0.7x |
| HARD\_GATE | \> 365 | **Exclude** |

### Trial Data - Phase 1

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 270 | 1.0x |
| WARN | 270-365 | 1.0x |
| SOFT\_GATE | 365-545 | 0.8x |
| HARD\_GATE | \> 545 | **Exclude** |

### Market Data

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 3 | 1.0x |
| WARN | 3-5 | 1.0x |
| SOFT\_GATE | 5-10 | 0.3x |
| HARD\_GATE | \> 10 | **Exclude** |

### Short Interest Data \(FINRA 2-week lag built in\)

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 20 | 1.0x |
| WARN | 20-30 | 1.0x |
| SOFT\_GATE | 30-45 | 0.5x |
| HARD\_GATE | \> 45 | **Exclude** |

### 13F Holdings Data \(45-day SEC filing lag\)

| Level | Age \(days\) | Penalty |
| --- | --- | --- |
| PASS | <= 60 | 1.0x |
| WARN | 60-90 | 1.0x |
| SOFT\_GATE | 90-135 | 0.4x |
| HARD\_GATE | \> 135 | **Exclude** |

**SEC\_13F\_FILING\_LAG\_DAYS**: 45 \(built-in constant\).

---

## Gate 3: Data Quality Hard Gates

> **Fix applied 2026-05-16 \(Code Review H2\):** Removed data-age thresholds for financial, market, and trial data from this gate. These conflicted with Gate 2's phase-dependent staleness system \(e.g., Gate 3 excluded trial data > 30 days while Gate 2 considers Phase 3 trial data valid up to 90 days\). **Gate 2 is authoritative for all data-age staleness decisions.** Gate 3 now contains only non-temporal quality gates.

| Gate | Threshold | Action |
| --- | --- | --- |
| Liquidity \(ADV\) | < $500,000/day | Exclude ticker |
| Price \(penny stock\) | < $5.00 | Exclude ticker |

> **Penny stock threshold note \(W7\):** This $5.00 hard gate in Gate 3 is the production exclusion threshold — any ticker below $5.00 is removed from the rankable universe before scoring begins. The financial-health skill's $2.00 penny stock penalty \(Step 6, liquidity score capped at 10\) is a SECONDARY safeguard that would only apply if Gate 3's threshold were lowered. At the current $5.00 gate, the $2.00 penalty is unreachable in production. If Gate 3 is ever revised to a lower threshold, the financial-health $2.00 penalty becomes the operative backstop. Both thresholds are intentional — Gate 3 is the hard exclusion, financial-health is the soft penalty for a lower price band.
> | Market field coverage | < 80% fields present | Exclude ticker |
> | Financial field coverage | < 50% fields present | Issue warning |

---

## Gate 4: Circuit Breakers

| Condition | Threshold | Action |
| --- | --- | --- |
| Records failing validation | \> 20% | Log warning |
| Records failing validation | \> 50% | **Fail entire pipeline** |
| Minimum records for check | < 10 | Skip circuit breaker check |

Circuit breakers prevent silent data corruption from propagating through the pipeline.

---

## Gate 5: Input Validation

| Validation | Rule | Default |
| --- | --- | --- |
| Ticker format | `^[A-Z]{1,5}$` | Max 5 uppercase alpha |
| Minimum date | >= 1990-01-01 | Historical cutoff |
| Cash | Non-negative | Required positive |
| Market cap | Positive | Required positive |
| Maximum runway | <= 1200 months | 100-year cap |
| Valid records % | >= 10% must pass | Minimum threshold |

---

## Gate 6: Score Bounds Validation

All scores must fall within \[0, 100\]:

| Score Field | Min | Max | Module |
| --- | --- | --- | --- |
| financial\_score | 0.0 | 100.0 | Module 2 |
| clinical\_score | 0.0 | 100.0 | Module 4 |
| catalyst\_score | 0.0 | 100.0 | Module 3 |
| score\_blended | 0.0 | 100.0 | Module 3 v2 |
| composite\_score | 0.0 | 100.0 | Module 5 |

Any score outside \[0, 100\] is a pipeline error. Fail-closed.

---

## Gate 7: Weight Sum Validation

Module 5 component weights must sum to 1.0 within tolerance:

| Constraint | Expected | Tolerance |
| --- | --- | --- |
| Weight sum | 1.0 | +/- 0.01 |

Weights outside tolerance are a configuration error. Fail-closed.

---

## Gate 8: Module Coverage Minimums

| Module | Minimum Coverage | Action if Below |
| --- | --- | --- |
| Module 2 \(Financial\) | 80% of universe | Warning |
| Module 3 \(Catalyst\) | 80% of universe | Warning |
| Module 4 \(Clinical\) | 80% of universe | Warning |

---

## Gate 9: Severity System

### Severity Levels

| Level | Meaning | Score Multiplier | Action |
| --- | --- | --- | --- |
| NONE | Healthy | 1.0 | Include |
| SEV1 | Caution | 0.90 \(10% penalty\) | Include with flag |
| SEV2 | Warning | 0.50 \(50% penalty\) | Include, soft gate |
| SEV3 | Critical | 0.00 | **Exclude** |

SEV3 is a hard gate. The ticker is removed from the rankable universe.

---

## Gate 10: Pipeline Health Status

| Component | Coverage Threshold | Status if Below |
| --- | --- | --- |
| catalyst\_raw | 10% | DEGRADED |
| momentum | 0% | OPTIONAL |
| smart\_money | 0% | OPTIONAL |
| market\_data | 0% | OPTIONAL |

**Run Status Classification:**

- **OK**: All thresholds met
- **DEGRADED**: Optional components below threshold
- **FAIL**: Critical catalyst pipeline broken \(< 5% with events\)

---

## IC Quality Benchmarks

### Information Coefficient Thresholds

| Quality | IC Range | Classification | Action |
| --- | --- | --- | --- |
| Excellent | IC > 0.05 | Institutional-grade | Deploy |
| Good | IC 0.03-0.05 | Tradeable | Use with confidence |
| Weak | IC 0.01-0.03 | Needs enhancement | Monitor |
| Noise | IC < 0.01 | No predictive power | Abandon signal |
| Negative | IC < 0 | Inverted signal | Investigate inversion |

### IC Measurement Constants

| Constant | Value | Purpose |
| --- | --- | --- |
| MIN\_OBS\_IC | 10 | Minimum observations for IC calculation |
| MIN\_OBS\_TSTAT | 20 | Minimum for t-statistic |
| MIN\_OBS\_BOOTSTRAP | 30 | Minimum for bootstrap CI |
| MIN\_ROLLING\_WINDOW | 12 weeks | Minimum rolling window |
| BOOTSTRAP\_ITERATIONS | 1000 | Bootstrap resampling count |
| TSTAT\_THRESHOLD\_95 | 2.0 | 95% confidence |
| TSTAT\_THRESHOLD\_99 | 2.58 | 99% confidence |

### Forward Return Horizons

| Horizon | Trading Days |
| --- | --- |
| 1w | 5 |
| 2w | 10 |
| 1m | 20 |
| 1.5m | 30 |
| 3m | 60 |
| 4.5m | 90 |

### Market Cap Buckets \(IC Analysis\)

> **Cross-reference note \(W2\):** These tier boundaries are used for IC segmentation analysis ONLY. They differ from the financial-health skill's liquidity scoring tiers \(SMALL $300M-$2B, MID $2B-$10B, LARGE >= $10B\). The two tier systems serve different purposes: IC analysis uses narrower bands to detect signal behavior across size cohorts, while liquidity scoring uses broader bands aligned to institutional trading capacity. Both are intentional. When citing market cap tiers, always specify which system \(IC analysis vs. liquidity scoring\) to avoid confusion.

| Bucket | Range |
| --- | --- |
| MICRO | < $300M |
| SMALL | $300M - $1B |
| MID | $1B - $5B |
| LARGE | \> $5B |

---

## Regime Data Staleness Haircuts

| Data Age | Confidence Multiplier |
| --- | --- |
| <= 2 days | 1.00 \(full\) |
| 3-5 days | 0.85 \(15% haircut\) |
| 6-10 days | 0.65 \(35% haircut\) |
| \> 10 days | 0.00 \(force UNKNOWN regime\) |

---

## Production Hardening Limits

### File Size Limits

| File Type | Max Size |
| --- | --- |
| JSON files | 100 MB |
| Config files | 10 MB |
| Checkpoint files | 50 MB |

### Operation Timeouts

| Operation | Timeout |
| --- | --- |
| File read | 60 seconds |
| Module execution | 600 seconds \(10 min\) |
| Full pipeline | 3600 seconds \(1 hour\) |

### Logging Sanitization

| Limit | Value |
| --- | --- |
| List items logged | 10 max |
| String length logged | 200 chars max |
| Blocked patterns | `api_key`, `password`, `secret`, `token`, `credential`, `ssn`, `account_number`, `cusip` |

---

## Determinism Enforcement

| Setting | Required Value | Purpose |
| --- | --- | --- |
| force\_deterministic\_timestamps | true | No `datetime.now()` |
| sort\_output\_keys | true | Reproducible JSON |
| include\_content\_hashes | true | Integrity verification |
| random\_seed | 42 | Reproducible randomization |

### Determinism Rules

1. Same inputs MUST produce byte-identical outputs
2. All JSON serialization uses sorted keys
3. All list operations use deterministic sort keys
4. Content hashes \(SHA256\) included in every output for verification
5. No external API calls during scoring \(stdlib only\)
6. All timestamps derived from `as_of_date`, never from wall clock

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

| Stage | When |
| --- | --- |
| INIT | Pipeline initialization |
| LOAD | Data loading |
| ADAPT | Data transformation |
| FEATURES | Feature engineering |
| RISK | Risk calculation |
| SCORE | Scoring execution |
| REPORT | Report generation |
| FINAL | Pipeline completion |

### Audit Status Values

| Status | Meaning |
| --- | --- |
| OK | Stage passed |
| FAIL | Stage failed |
| SKIP | Stage skipped |

### Standard Error Codes

| Code | Description |
| --- | --- |
| MISSING\_INPUT | Required input not found |
| SCHEMA\_MISMATCH | Schema validation failed |
| HASH\_ERROR | Integrity check failed |
| PARAMS\_MISSING | Parameters incomplete |
| MAPPING\_MISSING | Mapping not found |
| VALIDATION\_ERROR | Data validation failed |
| UNKNOWN\_ERROR | Unclassified error |

---

## Schema Version Support

| Module | Supported Versions |
| --- | --- |
| module\_1 | 1.0.0 |
| module\_2 | 1.0.0 |
| module\_3 | dynamic, m3catalyst\_vnext\_20260111 |
| module\_4 | 1.0.0 |
| module\_5 | 1.0.0, 1.1.0 |

---

## Enhancement Engine Confidence Thresholds

| Engine | Confidence Gate | Effect Below Gate |
| --- | --- | --- |
| PoS | 0.40 | PoS weight -> 0 |
| Momentum | 0.50 | Momentum not meaningful |
| Smart Money | 0.50 | Smart money signal excluded |
| Valuation | 0.40 | Valuation fallback to sector |

---

## Gate 11: Snapshot Content Collapse Guards \(added 2026-05-08\)

Post-hash/manifest checks that detect content-level signal collapse:

| Check | Threshold | Verdict | Rationale |
| --- | --- | --- | --- |
| coinvest\_score\_z SD | <= 0.10 | FAIL | Selector signal flat, pipeline fallback suspected |
| catalyst\_quality classification | < 90% classified among has\_catalyst\_signal=1 rows | FAIL | Spec 078 classification broken |
| No has\_catalyst\_signal=1 rows | n/a | WARN | Non-blocking \(possible but unusual\) |

**Tool**: `tools/verify_snapshot_integrity.py` \(Section 4\)

These guards run AFTER the existing hash/manifest checks and catch silent degradation that hash-level validation cannot detect \(e.g., all tickers receiving identical coinvest scores due to a fallback path\).

Verified against production: SD = 0.6833 \(PASS\), 261/261 \(100%\) catalyst rows classified \(PASS\).

---

## Gate 12: Expectation Layer Coverage Gate \(Spec 105, 2026-05-14\)

**QA file**: `production_qa_check.py`
**Status:** CODE-CLOSED \(commit 0ddbb509\). Pending live production snapshot QA.

Production pipeline hard-fails if market-expectation fields are missing or under-covered in `rankings.csv`. Thresholds sourced from `FEATURE_COVERAGE_REQUIREMENTS` \(not hardcoded\).

### Required Expectation Fields

| Field | Required Coverage | Source |
| --- | --- | --- |
| `short_interest_pct` | 0.90 | Market data provider |
| `close_price` | 0.99 | Market data provider |
| `market_cap_mm` | 0.95 | Market data provider |
| `priced_move_pct` | 0.80 | Derived \(catalyst pricing model\) |
| `insider_net_buy_value_90d` | 0.30 | Form 4 \(tracked nonblocking / diagnostic only\) |

### Gate Behavior

- Runs every pipeline execution at Step 5 \(Gates\)
- Hard fail if any required field is missing from DataFrame
- Hard fail if any field falls below its per-field threshold
- Error message includes: field name, actual coverage, required threshold
- Coverage stats logged every run regardless of pass/fail
- `FEATURE_COVERAGE_REQUIREMENTS` is the single source of truth

---

## Diagnostic Fields Registry \(Spec 104, 2026-05-14\)

Fields tracked for observability but explicitly excluded from scoring, ranking, and selection.

### Current Diagnostic Fields

| Field | Status | Meaning of Null | Meaning of 0.0 |
| --- | --- | --- | --- |
| `insider_net_buy_value_90d` | DIAGNOSTIC ONLY | Not fetched / no Form 4 coverage | Fetched, no insider buy activity in 90d |

### Insider Model Isolation Guard \(CRITICAL\)

`insider_net_buy_value_90d` must NOT enter the expectation model's `market_features` input. The model has an `insider_net_buy_z` weight that activates silently if the field flows upstream. Guard with at least one of:

1. **Input exclusion \(preferred\):** Runtime assert that `insider_net_buy_value_90d` is NOT in `market_features` DataFrame at inference
2. **Weight zeroing:** `insider_net_buy_z` weight = 0.0 with test
3. **Drop guard:** Pre-inference step that drops the field if present, with logged warning

### Diagnostic Field Rules

- Never collapse blank \(NaN\) and zero \(0.0\) -- they have different semantics
- Never impute zero for missing or blank for zero
- CI check: flag suspicious if column is ALL zero or ALL null
- Field must remain in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Does not affect ranks, actions, or position sizing
- Promotion requires: 20+ stable snapshots, >= 60% coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval

---

## Pre-Run Checklist

Before executing a pipeline run, verify:

1. `as_of_date` is explicitly provided \(never derived from wall clock\)
2. All input files exist and are within size limits
3. PIT cutoff is computed and logged
4. Schema versions match expected versions
5. Weight sums are within tolerance
6. No `float` arithmetic in scoring paths \(only `Decimal`\)
7. No `datetime.now()` calls in any module
8. No `random` module usage without explicit seed
9. Audit log writer is initialized
10. Run ID is deterministically generated

## Post-Run Checklist

After a pipeline run completes, verify:

1. All output scores are within \[0, 100\]
2. Governance metadata is present in every output file
3. Content hashes match recomputed hashes \(determinism check\)
4. No SEV3 tickers appear in ranked output
5. Coverage metrics are logged \(per-module and per-signal\)
6. Circuit breaker did not trip silently
7. Staleness penalties were applied where required
8. Audit log contains entries for all stages \(INIT through FINAL\)

---

## Source Files

| Component | File |
| --- | --- |
| Data Quality Gates | `common/data_quality.py` |
| Staleness Gates | `common/staleness_gates.py` |
| PIT Enforcement | `common/pit_enforcement.py` |
| Input Validation | `common/input_validation.py` |
| Integration Contracts | `common/integration_contracts.py` |
| Schema Validation | `common/schema_validation.py` |
| Production Hardening | `common/production_hardening.py` |
| Robustness Utilities | `common/robustness.py` |
| IC Measurement | `backtest/ic_measurement.py` |
| Audit Log | `governance/audit_log.py` |
| Pipeline Config | `config.yml` |

## FDA Real-Time Trial Initiative \(May 2026\)

### Proof-of-Concept Studies

FDA launched two real-time clinical trial \(RTCT\) proof-of-concept studies:

1. AstraZeneca TRAVERSE - mantle cell lymphoma
2. Amgen STREAM-SCLC - small cell lung cancer
Both use AI and cloud-based data feeds via Paradigm Health for real-time safety signal detection.

### AI-Enabled Early-Phase Trial Pilot Program

- FDA RFI published April 29, 2026 \(Federal Register\)
- Comments due May 29, 2026; pilot selections by August 2026
- Focus: AI improvements in trial efficiency, dose selection, safety monitoring, go/no-go decisions
- Aligned with NIST AI Risk Management Framework

### Projected Impact

- 20-40% trial duration reduction
- $120 million annual savings
- Continuous trials eliminating inter-phase delays

### DEM Implication

If clinical trials become continuous rather than phase-gated, the binary catalyst model \(trial readout = binary stock event\) evolves toward a continuous information release model. This would affect catalyst\_decay\_w and catalyst\_quality calibration. Monitor as Tier 4 governance question.

### ODIN Feature Comparison

ODIN's 51-feature model includes signal categories not in the DEM's clinical scoring:

- Manufacturing/CMC risk scoring
- FDA era effects \(temporal patterns in regulatory stringency\)
- Options market implied approval probability
- Sponsor historical approval rate by therapeutic area
These are evaluation candidates through the T5 promotion path \(Tier 4 design decisions\).

### AI Drug Pipeline Scale \(Q1 2026\)

- 173+ AI-originated programs in human clinical trials \(94 Phase I, 56 Phase II, 15 Phase III\)
- 7x increase since 2022
- Pre-clinical timeline compression: 4-6 years to 12-24 months
- Clinical trial timelines remain unchanged \(the phase AI has NOT yet shortened\)
- 2026 is definitive validation year - Phase III results determine if AI improves success rates beyond \~90% historical failure rate