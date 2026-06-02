---
name: clinical-scoring
description: "Score biotech clinical programs with pipeline rules."
tags:
  - biotech-screener
  - clinical
  - scoring
---

## Purpose

Score a biotech company's clinical development program to produce a normalized 0-100 clinical score. This skill encodes the exact rules, thresholds, and lookup tables from Wake Robin's pipeline (Module 4 + PoS Engine) so that any agent or analyst can reproduce the scoring deterministically.

## Preconditions

- All arithmetic MUST use `Decimal` (never `float`). Initialize from strings: `Decimal("0.40")`.
- All dates MUST be ISO 8601 (`YYYY-MM-DD`). Never call `datetime.now()`.
- PIT safety: only use data where `source_date <= as_of_date - 1`.
- Rounding: `ROUND_HALF_UP`. Scores to 2 dp (`0.01`), rates to 4 dp (`0.0001`).

---

## Operator Qualification Context

- **Operator**: Darren Schulz, CFA, CAIA — Director of Investments, Wake Robin (Holland, MI)
- **Clinical analysis background**: Conducts biotech equity due diligence including clinical pipeline analysis, phase progression evaluation, PDUFA date tracking, and catalyst timing assessment. Builds and operates automated clinical trial monitoring systems (Herald Digest, Bellringer, PDUFA alerts).
- **Manual override authority**: The operator holds sole authority for clinical scoring overrides, indication mapping corrections (ticker_overrides_v3), and PoS benchmark adjustments. CAIA credential and biotech research experience provide the domain basis for these governance decisions.
- **Indication mapping confidence**: When indication_mapper.py returns confidence < 0.65, the operator should be consulted for manual classification before scoring proceeds.

---

## Step 1: Determine Lead Phase

Map each company's most advanced trial to a canonical phase. Use the **highest** phase across all PIT-admissible trials for that ticker.

| Raw Phase String | Canonical Phase |
| --- | --- |
| "Phase 1", "PHASE1", "phase 1", "p1" | `phase_1` |
| "Phase 1/Phase 2", "Phase 1/2" | `phase_1_2` |
| "Phase 2", "PHASE2", "phase 2", "p2" | `phase_2` |
| "Phase 2/Phase 3", "Phase 2/3" | `phase_2_3` |
| "Phase 3", "PHASE3", "phase 3", "p3" | `phase_3` |
| "New Drug Application", "NDA", "BLA" | `nda_bla` |
| "Approved", "APPROVED" | `commercial` |
| anything else | `preclinical` |

**Phase ordering** (lowest to highest): `preclinical < phase_1 < phase_1_2 < phase_2 < phase_2_3 < phase_3 < nda_bla < commercial`

---

## Step 2: Look Up Base Stage Score

> **Note:** This table is used by the **PoS Engine (Step 5)** to set baseline LOA context. It is NOT used in the Module 4 clinical composite (Step 14). For the Module 4 phase score used in the clinical composite, see Step 3.

| Stage | Score (0-100) |
| --- | --- |
| preclinical | 10 |
| phase_1 | 20 |
| phase_1_2 | 30 |
| phase_2 | 40 |
| phase_2_3 | 52 |
| phase_3 | 65 |
| nda_bla | 80 |
| commercial | 90 |

---

## Step 3: Compute Phase Score (Module 4, 0-30 pts raw)

| Phase | Phase Score |
| --- | --- |
| approved | 30 |
| phase 3 | 25 |
| phase 2/3 | 22 |
| phase 2 | 18 |
| phase 1/2 | 12 |
| phase 1 | 8 |
| preclinical | 3 |
| unknown | 0 |

### Phase Progress Bonus (0-5 pts)

| Phase | Bonus |
| --- | --- |
| approved | 5.0 |
| phase 3 | 4.0 |
| phase 2/3 | 3.5 |
| phase 2 | 3.0 |
| phase 1/2 | 2.0 |
| phase 1 | 1.0 |
| preclinical | 0.0 |

---

## Step 4: Map Indication

Use `indication_mapper.py` (v2.0.0). Mapping precedence (highest to lowest):

1. **ticker_overrides_v3** (PIT-safe, has `effective_from`/`effective_until`) -> confidence 0.95
2. **ticker_overrides** (legacy, no time-bounds) -> confidence 0.85
3. **condition_patterns** (regex word-boundary matching, 2+ matches) -> confidence 0.80
4. **condition_patterns** (single match) -> confidence 0.65
5. **ta_fallback** (therapeutic area only) -> confidence 0.50
6. **phase_only** (no condition data) -> confidence 0.30

### Category Aliases

| Mapper Category | PoS Engine Category |
| --- | --- |
| cns | neurology |
| autoimmune | immunology |
| gi_hepatology | gastroenterology |

---

## Step 5: Look Up PoS Benchmarks (BIO 2011-2020)

### Phase 1 LOA (Likelihood of Approval)

| Indication | LOA |
| --- | --- |
| oncology | 0.057 |
| rare_disease | 0.106 |
| infectious_disease | 0.195 |
| neurology | 0.084 |
| cardiovascular | 0.071 |
| immunology | 0.112 |
| metabolic | 0.093 |
| respiratory | 0.089 |
| dermatology | 0.124 |
| ophthalmology | 0.117 |
| gastroenterology | 0.098 |
| hematology | 0.102 |
| urology | 0.095 |
| all_indications | 0.079 |

### Phase 2 LOA

| Indication | LOA |
| --- | --- |
| oncology | 0.131 |
| rare_disease | 0.273 |
| infectious_disease | 0.196 |
| neurology | 0.144 |
| cardiovascular | 0.126 |
| immunology | 0.218 |
| metabolic | 0.167 |
| respiratory | 0.155 |
| dermatology | 0.234 |
| ophthalmology | 0.212 |
| gastroenterology | 0.178 |
| hematology | 0.195 |
| urology | 0.172 |
| all_indications | 0.152 |

### Phase 3 LOA

| Indication | LOA |
| --- | --- |
| oncology | 0.439 |
| rare_disease | 0.649 |
| infectious_disease | 0.769 |
| neurology | 0.510 |
| cardiovascular | 0.545 |
| immunology | 0.672 |
| metabolic | 0.598 |
| respiratory | 0.567 |
| dermatology | 0.712 |
| ophthalmology | 0.687 |
| gastroenterology | 0.612 |
| hematology | 0.634 |
| urology | 0.589 |
| all_indications | 0.579 |

### Phase 3 LOA

| Indication | LOA |
| --- | --- |
| oncology | 0.439 |
| rare_disease | 0.649 |
| infectious_disease | 0.769 |
| neurology | 0.510 |
| cardiovascular | 0.545 |
| immunology | 0.672 |
| metabolic | 0.598 |
| respiratory | 0.567 |
| dermatology | 0.712 |
| ophthalmology | 0.687 |
| gastroenterology | 0.612 |
| hematology | 0.634 |
| urology | 0.589 |
| all_indications | 0.579 |

### NDA/BLA LOA

| Indication | LOA |
| --- | --- |
| all_indications | 0.903 |

### PoS Score Conversion

```
pos_score = LOA_probability * 100
```

Score range: 0.00 to 100.00.

---

## Step 6: Apply Confidence Gating

### Stage-Adjusted Base Confidence

| Stage | Confidence |
| --- | --- |
| preclinical | 0.35 |
| phase_1 | 0.45 |
| phase_1_2 | 0.50 |
| phase_2 | 0.58 |
| phase_2_3 | 0.65 |
| phase_3 | 0.75 |
| nda_bla | 0.88 |
| commercial | 0.92 |

### Indication Confidence Modifiers (additive)

| Indication | Modifier |
| --- | --- |
| rare_disease | +0.05 |
| dermatology | +0.04 |
| infectious_disease | +0.03 |
| ophthalmology | +0.03 |
| immunology | +0.02 |
| hematology | +0.02 |
| metabolic | 0.00 |
| gastroenterology | 0.00 |
| urology | 0.00 |
| respiratory | -0.02 |
| cardiovascular | -0.03 |
| all_indications | -0.03 |
| oncology | -0.05 |
| neurology | -0.08 |

### Data Quality Confidence Modifiers (additive)

| Quality State | Modifier |
| --- | --- |
| FULL | +0.05 |
| PARTIAL | 0.00 |
| MINIMAL | -0.05 |
| NONE | -0.15 |

### Confidence Bounds

```
final_confidence = clamp(base + indication_modifier + quality_modifier, 0.20, 0.95)
```

- CONFIDENCE_HIGH: >= 0.70
- CONFIDENCE_MEDIUM: >= 0.55
- CONFIDENCE_LOW: >= 0.30
- **GATING_THRESHOLD: 0.40** - Below this, PoS contributes 0 weight to composite.

---

## Step 7: Apply Optional Multipliers

| Multiplier | Clamp Range | Source |
| --- | --- | --- |
| trial_design_quality | 0.70 - 1.30 | Module 4 design scoring |
| competitive_intensity | 0.70 - 1.00 | competitive_pressure_engine.py |

---

## Step 8: Recency Scoring (0-5 pts)

Based on days since last trial update (use PIT date field priority: `first_posted` > `last_update_posted` > `source_date` > `collected_at`).

| Days Since Update | Score |
| --- | --- |
| 0-30 | 5.0 |
| 30-90 | 5.0 to 4.5 (linear) |
| 90-180 | 4.5 to 4.0 (linear) |
| 180-365 | 4.0 to 3.0 (linear) |
| 365-730 | 3.0 to 1.0 (linear) |
| >= 730 | 1.0 |

- **RECENCY_STALE_THRESHOLD**: 730 days (2 years). Triggers 20% penalty on score.
- **RECENCY_UNKNOWN_PENALTY**: 2.5 (neutral score when date unknown).

---

## Step 9: Trial Count Bonus (0-5 pts, piecewise linear)

| Trial Count | Score |
| --- | --- |
| 0 | 0.0 |
| 1 | 0.5 |
| 2 | 1.0 |
| 5 | 2.0 |
| 10 | 3.5 |
| 20 | 4.5 |
| >= 100 | 5.0 |

---

## Step 10: Indication Diversity Bonus (0-5 pts)

Based on count of unique condition tokens across all trials.

| Unique Tokens | Score |
| --- | --- |
| 0 | 0.0 |
| 2 | 0.7 |
| 5 | 1.5 |
| 10 | 3.0 |
| 20 | 4.0 |
| >= 30 | 5.0 |

---

## Step 11: Design Quality Scoring (0-25 pts)

- **Base**: 12 pts
- **Randomized**: +5 pts
- **Double-Blind**: +4 pts
- **Single-Blind**: +2 pts (mutually exclusive with double-blind)
- **Strong Endpoint**: +4 pts
- **Weak Endpoint**: -3 pts (mutually exclusive with strong)

### Strong Endpoint Patterns

`overall survival`, `OS`, `progression-free survival`, `PFS`, `complete response`, `CR`, `objective response rate`, `ORR`, `disease-free survival`, `DFS`, `event-free survival`, `EFS`, `major molecular response`, `MMR`

### Weak Endpoint Patterns

`biomarker`, `pharmacokinetic`, `PK`, `safety`, `tolerability`, `dose-finding`, `maximum tolerated dose`, `MTD`

---

## Step 12: Execution Track Record (0-25 pts)

- **Base**: 12 pts
- **Completion Rate Contribution**: `completion_rate * 10` pts
- **Termination Rate Penalty**: `termination_rate * 8` pts (subtracted)
- **Clamp**: `execution_score = clamp(execution_score, 0, 25)`

> **Code fix history (PR #288):** Runtime base remains 12 (Decimal("12")). The effective max = 12 + 10 - 0 = 22 (clamp to 25 is redundant). The denominator fix (Step 14) was the actual correction. An earlier version of this skill incorrectly claimed base was raised to 15; that change was never implemented.

### Trial Status Quality Weights

| Status | Weight |
| --- | --- |
| COMPLETED | 1.0 |
| ACTIVE | 0.8 |
| RECRUITING | 0.7 |
| NOT_YET_RECRUITING | 0.6 |
| ENROLLING_BY_INVITATION | 0.7 |
| SUSPENDED | 0.2 |
| TERMINATED | 0.0 |
| WITHDRAWN | 0.0 |
| UNKNOWN | 0.5 |

---

## Step 13: Endpoint Strength (0-20 pts)

- **Base**: 10 pts
- **Strong endpoint found**: +2 pts per occurrence
- **Weak endpoint found**: -1 pt per occurrence
- **Clamp**: `endpoint_score = clamp(endpoint_score, 0, 20)`

> **Fix applied 2026-05-16 (Code Review H5):** Added explicit clamp to [0, 20]. Previously unbounded -- a company with many strong endpoints could exceed the stated 0-20 range (e.g., 15 strong endpoints = 40 pts), causing raw_total > 120 before the final clinical_score clamp masked it.

---

## Step 14: Compute Total Clinical Score

```
raw_total = phase_score + phase_progress + trial_count_bonus
          + diversity_bonus + recency_bonus + design_score
          + execution_score + endpoint_score

clinical_score = (raw_total / 117) * 100    # Normalize to 0-100
clinical_score = clamp(clinical_score, 0, 100)
```

Effective max raw total = 117 because execution_score base is 12 (effective max = 22, not 25). Denominator = 117 per PR #288 (Codex review 2026-05-17) to produce correct 0-100 normalization.

Breakdown: 30 + 5 + 5 + 5 + 5 + 25 + 22 + 20 = 117.

**Why 117, not 120:** The stated execution_score range (0-25) is a nominal budget — the actual effective max is 22 due to base=12, making 117 the true raw total ceiling. Changing denominator from 120→117 (PR #288) was the correct fix; no code change to the execution_score base was needed.

---

## Step 15: Commercial Stage Differentiation (PoS Engine)

For companies at `commercial` stage, apply additional adjustments:

### Pipeline Tier Bonuses (LOA adjustment)

| Tier | Min Trials | LOA Bonus |
| --- | --- | --- |
| exceptional | >= 100 | 0.00 |
| strong | >= 30 | -0.02 |
| moderate | >= 10 | -0.05 |
| limited | >= 3 | -0.10 |
| minimal | 0-2 | -0.15 |

### Indication-Specific Commercial Risk (LOA adjustment)

| Indication | Risk |
| --- | --- |
| rare_disease | 0.00 |
| dermatology | -0.02 |
| ophthalmology | -0.02 |
| neurology | -0.02 |
| hematology | -0.02 |
| oncology | -0.03 |
| immunology | -0.03 |
| metabolic | -0.04 |
| respiratory | -0.04 |
| gastroenterology | -0.04 |
| urology | -0.04 |
| infectious_disease | -0.05 |
| cardiovascular | -0.05 |
| all_indications | -0.05 |

### Pipeline Diversity Bonus

+0.02 if >= 3 distinct phases represented in pipeline.

### Commercial LOA Range

Clamp commercial LOA to [0.82, 1.00].

---

## Severity Classification

| Condition | Severity |
| --- | --- |
| No trials found | SEV1 (10% penalty) |
| All trials stale (> 730 days) | SEV1 |
| No PIT-admissible data | SEV2 (50% penalty) |
| Lead phase = preclinical only | NONE (scored normally) |

---

## Composite Integration

The clinical score enters Module 5 composite with these weights:

| Weight Set | Clinical Weight |
| --- | --- |
| V3 Enhanced (all signals) | 26% |
| V3 Default (no enhancements) | 40% |
| V3 Partial (some enhancements) | 33% |
| Baker-Style Fundamental | 35% |

**PoS Delta Cap**: Maximum PoS contribution to composite = 6.0 points.

---

## Pre-Flight Checks

Before scoring, verify:

1. Trial records are PIT-admissible (`source_date <= as_of_date - 1`)
2. `pos_benchmarks_bio_2011_2020_v1.json` is loaded and contains all 14 indications
3. Indication mapping covers the ticker (log confidence tier if < 0.65)
4. At least 1 trial record exists for the ticker (otherwise assign SEV1)
5. All score components use `Decimal` arithmetic
6. Output includes `_governance` block with `score_version`, `schema_version`, `pit_cutoff`

---

## Source Files

| Component | File |
| --- | --- |
| PoS Engine | `pos_engine.py` (v1.2.0) |
| Clinical Scoring | `module_4_clinical_dev_v2.py` (v2.1.0) |
| Indication Mapper | `indication_mapper.py` (v2.0.0) |
| PoS Benchmarks | `data/pos_benchmarks_bio_2011_2020_v1.json` |
| Catalyst Scoring | `module_3_scoring_v2.py` (v2.0.0) |
