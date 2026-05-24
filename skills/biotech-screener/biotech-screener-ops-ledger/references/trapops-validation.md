# TrapOps Validation — Rankings.csv Data Audit
# Absorbed from: de-schema-gate-fix skill

## What TrapOps Is

`tools/trapops_monitor.py` — read-only daily monitoring tool for the Trap T20 → B6 → sizing pipeline.
Runs as Step 7 (non-blocking) inside `run_daily_production.py:6023` after every production run.
Also has watchdog recovery in `cron_watchdog.sh:171-175`.

**4 modules:**
- **A. selection_diff** — B6 top-30 change vs yesterday: overlap, added, removed, trap-veto list
- **B. execution_stress** — reads `execution_stress_base/stress.json`: ADV participation, tail concentration
- **C. trap_attribution** — reads `ees_gate_performance.json`: eligible vs trap-removed forward returns
- **D. health_alerts** — thresholds: GREEN / YELLOW / RED

**Output:** `data/snapshots/<date>/trapops_daily_summary.json`

## Key Validation Checks

### EES Eligibility Cross-Check

The gate diagnostics (`ees_gate_diagnostics.json`) only store the TOP-N filtered names (30 each for quality, 30 for trap), not all filtered tickers. The CSV `ees_eligible` field is computed correctly from `quality_gate=False OR trap_gate=False`.

```python
# Quick cross-check to verify CSV vs gate are consistent
quality_filtered = set(ees_diag.get('quality_filtered_names', []))
trap_filtered = set(ees_diag.get('trap_filtered_names', []))
all_filtered_gate = quality_filtered | trap_filtered

csv_ineligible = set(r['ticker'] for r in rows if r.get('ees_eligible','').lower() == 'false')

# These should all be in csv_ineligible (subset, not exact match — gate truncates to top-N)
missing = all_filtered_gate - csv_ineligible
# If missing > 0: gate filtered a name but CSV says eligible — REAL BUG

# Extra in csv_ineligible beyond gate top-N: EXPECTED (gate only stores top 30 per bucket)
```

**Universe formula (verified 2026-05-06):**
`quality_fail + trap_fail - both_fail = total_ineligible`
Example: 49 + 60 - 18 = 91 ineligible → 299 - 91 = 208 eligible (69.6%) ✅

### Trap-Removed Verification

Trap-removed = names in B6 top-30 by `selector_score` that have `ees_eligible=False`:

```python
all_b6 = [(r['ticker'], float(r.get('selector_score','') or 0))
          for r in rows if r.get('selector_score','').strip()]
all_b6.sort(key=lambda x: -x[1])
top30_by_selector = [t for t,_ in all_b6[:30]]
ees_eligible_set = set(r['ticker'] for r in rows if r.get('ees_eligible','').lower() == 'true')
trap_removed = sorted(set(top30_by_selector) - ees_eligible_set)
```

Verified accurate against 2026-05-05 reported output: exact match.

### TrapOps Staleness Pattern

Typical coverage gaps:
- Weekends never run (no production run → no Step 7)
- Sleep-roulette misses on evenings: if laptop sleeps before 16:30 production, no trapops written
- Second production runs (watchdog recovery 16:30 ET) do NOT always execute Step 7

Staleness threshold: >1 business day = stale. Check:
```bash
find data/snapshots -name 'trapops_daily_summary.json' | sort | tail -5
```

## Known Data Gap Classification (rankings.csv)

Run this to distinguish bugs from design:

| Gap | Expected? | Reason |
|-----|-----------|--------|
| `event_ev_*` columns all empty | YES (until next run after bd777483) | Fixed: now in SNAPSHOT_COLUMNS |
| `clinical_quality_*` columns all empty | YES (until next run after bd777483) | Fixed: now in SNAPSHOT_COLUMNS |
| `tier_dev` empty for ~51 eligible | YES | Only drug_developer archetype gets tier_dev |
| `catalyst_days` empty for ~23 eligible | YES | `cat_mode=no_upcoming` by design |
| `clinical_score_z` empty for ~20 eligible | YES | platform_* archetypes don't get clinical z-score |
| `runway_buffer_months` empty for approved/commercial | YES | Runway model only fires for pre-revenue drug devs |
| `missing_components` 100% empty | YES | Field appears unused/always blank in v1.14.0 |
| MS Morningstar fields 100% empty | YES | Passive feed, not connected |
| OVF fields 100% empty | YES | Options verdict feed not active |
| `ranker_v2_score` empty ~80% | YES | ranker_v2 cohort gated by membership |
| `target_weight_pct` empty ~90% | YES | Only top-30 portfolio names get weights |

## Execution Stress Thresholds (health_alerts)

| Threshold | Level |
|-----------|-------|
| n_above_20pct_adv > 0 | RED |
| n_above_5pct_adv > 3 | YELLOW |
| top3_participation_weight_pct > 15% | YELLOW |
| trap_pass_rate shifts > 15pp from 20d rolling avg | YELLOW |
| quality_trap_correlation > 0.40 | YELLOW |
| trap_fail outperforms eligible (mean_ret) | RED |

Healthy baseline (2026-05-05/06):
- ADV: 0 above 5%, 0 above 20%
- top3_weight: 8-11%
- quality_trap_correlation: 0.06-0.10 (well below 0.40)
