---
name: 13f-validation-coordinator
description: "Coordinate quarterly 13F validation gates."
---

# 13F Validation Coordinator Skill

## Purpose
Orchestrate the quarterly 13F data validation workflow end-to-end: filing progress monitoring, pre-refresh readiness checks, cohort quarantine runs, collapse guard verification, verdict rendering, observation window management, and IC decomposition refresh after clearance.

## When to Use
Use this skill when:
- A 13F filing quarter closes (Feb 15, May 15, Aug 15, Nov 15)
- You need to validate whether new 13F data caused cohort contamination
- A quarantine is active and you need to determine whether to clear or extend
- The user asks about "13F validation", "Q1 cohort quarantine", "13F filing progress"
- Post-quarantine clearance: refreshing IC decomposition and unlocking downstream specs

## Prerequisites
- Working directory: `/mnt/c/Projects/biotech_screener/biotech-screener`
- Python: `/usr/bin/python3`
- Tools: `tools/check_13f_cohort_quarantine.py`, `tools/prep_13f_refresh.py`
- Data: `production_data/13f_filing_status.json`, `production_data/institutional_summary.json`

---

## Workflow

### Phase 1: Filing Progress Monitoring

1. **Check filing status:**
   ```bash
   python3 -c "
   import json
   with open('production_data/13f_filing_status.json') as f:
       s = json.load(f)
   filed = s.get('filed', {})
   pending = s.get('pending', {})
   print(f'Filed: {len(filed)} managers')
   print(f'Pending: {len(pending)} managers')
   print(f'Last check: {s.get(\"last_check\")}')
   for cik, info in sorted(pending.items(), key=lambda x: x[1].get('name', '')):
       print(f'  PENDING: {info[\"name\"]}')
   "
   ```

2. **Check institutional summary:**
   ```bash
   python3 -c "
   import json
   with open('production_data/institutional_summary.json') as f:
       s = json.load(f)
   print(f'as_of_date: {s[\"as_of_date\"]}')
   print(f'cache_as_of_date: {s[\"cache_as_of_date\"]}')
   print(f'elite_managers_total: {s[\"elite_managers_total\"]}')
   print(f'elite_managers_with_filing: {s[\"elite_managers_with_filing\"]}')
   print(f'signal_coverage_pct: {s.get(\"signal_coverage_pct\",\"?\")}%')
   "
   ```

3. **Gate check:** Validation requires >= 34 filed managers (~70% of 48-manager cohort). If insufficient, log and set next check date.

### Phase 2: Pre-Refresh Baseline
If no baseline exists:
```bash
python3 tools/prep_13f_refresh.py --pre-date <LAST_CLEAN_DATE> --output <FILE>
```
Pre-date = last snapshot BEFORE bulk filing wave (e.g., 2026-05-15 for Q1 2026).

### Phase 3: Cohort Quarantine Check
```bash
python3 tools/check_13f_cohort_quarantine.py \
  --pre-date <PRE_DATE> --post-date <POST_DATE> \
  --output artifacts/13f_cohort_quarantine_YYYY_MM_DD.md
```

### Phase 4: Gate Evaluation
| Gate | Metric | PASS | Notes |
|------|--------|------|-------|
| G1 | Snapshot completeness | PASS | Tool check |
| G2 | Producer freshness | PASS | cache_as_of_date advanced |
| G3 | Manager-level context | PASS | Tool check |
| Jaccard | Top-30 overlap | >= 0.70 | Higher = better |
| coinvest_score_z KS | Distribution stability | < 0.20 | |
| inst_delta_z KS | Delta stability | < 0.30 flagged | > 0.30 expected for refresh |
| Coverage Δ | signal_coverage chg | < 10pp | -1 to -2pp typical |
| Top-30 churn | Entries/exits | Acceptable < 5 | |

If Jaccard < 0.70 or coinvest_score_z KS > 0.20: QUARANTINE ACTIVE.

### Phase 5: Collapse Guard
```bash
python3 -c "
import csv, statistics
with open('data/snapshots/YYYY-MM-DD/rankings.csv') as f:
    reader = csv.DictReader(f)
    scores = [float(r['coinvest_score_z']) for r in reader if r.get('coinvest_score_z') and r['coinvest_score_z'] != '']
sd = statistics.stdev(scores)
print(f'SD: {sd:.4f} | PASS' if sd >= 0.10 else f'SD: {sd:.4f} | FAIL')
"
```
Threshold: SD >= 0.10.

### Phase 6: Verdict Rendering
- ALL PASS -> CLEAR (no quarantine extension)
- Any FAIL -> QUARANTINE ACTIVE (document failures)

Write to `artifacts/13f_validation_verdict_YYYY_MM_DD.md`.

### Phase 7: Post-Clearance Actions
After CLEAR:
1. **5-day observation window** from post-date snapshot (scores need natural recalibration)
2. **IC decomposition refresh** after observation window
3. **Spec unblocking:** Spec 089 KG pilot, Spec 100 IC battery, Spec 094 selector-only, Spec 072 vNext diagnostic. Architecture freeze lifts (partial; ranker frozen pending Spec 096).

### Phase 8: Remaining Filing Monitor
Track unfiled managers (e.g., Broadfin Capital/Kotler, Farallon Capital for Q1 2026). Late arrivals don't block validation but should be noted.

---

## Artifact File Convention
| File | Path |
|------|------|
| Cohort quarantine diff | `artifacts/13f_cohort_quarantine_YYYY_MM_DD.md` |
| Validation verdict | `artifacts/13f_validation_verdict_YYYY_MM_DD.md` |
| Monitoring log | `artifacts/13f_qN_YYYY_monitoring_YYYY_MM_DD.md` |
| Decision tree | `artifacts/13f_decision_tree_<scope>_YYYY_MM_DD.md` |
| Baseline snapshot | `artifacts/13f_pre_refresh_baseline_YYYY_MM_DD.json` |

## Typical Timeline (Q1 2026 example)
| Date | Event |
|------|-------|
| May 15 | Filing deadline, bulk wave |
| May 15-18 | Filing monitoring |
| May 19 | >= 34 filed, cohort quarantine check, NO_QUARANTINE verdict |
| May 20 | 5-day observation window ends |
| May 21 | Formal clearance, specs unblock |

## Pitfalls
1. **Not every date has snapshots.** Weekend snapshots archived under `data/snapshots/_archive_weekends/`.
2. **inst_delta_z KS > 0.30 is EXPECTED** during refresh — not a quarantine trigger unless Jaccard < 0.70 or coinvest_score_z KS > 0.20.
3. **Coverage drops ~1pp** post-refresh normally — threshold is 10pp.
4. **No pre-2026-05-14 snapshot** exists for Q1 2026 cycle — May 15 is the earliest pre-refresh date.
5. **coinvest_score_z collapse guard SD < 0.10** = selector signal flatlining.
6. **coinvest_score_z only covers ~144 tickers** with active institutional holdings, not all 298 universe tickers.

## Verification
1. Cohort quarantine artifact exists
2. Verdict artifact exists
3. If CLEAR: note observation window end date
4. If QUARANTINE_ACTIVE: document specific failures, set next check date
