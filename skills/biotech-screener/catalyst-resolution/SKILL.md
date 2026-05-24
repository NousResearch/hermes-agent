---
name: catalyst-resolution
description: >-
  Reference for the catalyst and event resolution pipeline — from multi-source
  event ingestion through catalyst timing/quality signals used in the screener's
  production scoring.
---

# Catalyst & Event Resolution Skill

## Purpose

Reference for the catalyst and event resolution pipeline - from multi-source event ingestion through catalyst timing/quality signals used in the screener's production scoring.

This skill is organized into two sections:

1. **Framework Reference** — Stable architecture, sources, and signal definitions (changes only with code updates)
2. **Operational State** — Volatile snapshots that require periodic refresh

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Architecture Overview

```
7+ Event Sources (CTGov, SEC 8-K, FDA ADCOM, FDA Regulatory, PDUFA, EMA, merged trials)
  -> event_ledger.py (unified event ledger)
  -> catalyst_resolution_tracker.py (per-ticker resolution files)
  -> catalyst_decay_w (timing signal, production)
  -> catalyst_quality / binary_quality_score (quality signal, Spec 078)
  -> event_ev_p_hit (EV binder, Spec 077, prospective accumulation)
```

## Event Ledger

**Builder**: `event_ledger.py` — `build_event_ledger()`

### Sources (7+)

| Source | Data |
| --- | --- |
| ClinicalTrials.gov (AACT) | Trial status changes, phase transitions |
| SEC 8-K | Material events (earnings, FDA actions) |
| FDA ADCOM | Advisory committee meetings |
| FDA Regulatory | Approval/CRL/priority review decisions |
| PDUFA Manual | Target action dates |
| EMA | European regulatory decisions |
| Merged Trials | Cross-registry deduplicated (NCT/EudraCT) |

### EU/EEA Registry Collectors

- `euctr_collector.py` — EU Clinical Trials Register
- `ctis_collector.py` — Clinical Trials Information System
- `isrctn_collector.py` — ISRCTN Registry
- `trial_registry_merger.py` — Cross-registry dedup by NCT/EudraCT IDs

---

## catalyst_decay_w (Timing Signal)

Measures proximity to the next known catalyst event. Near-term catalyst = higher weight.

### Key Properties

- Production signal in rankings.csv
- Signal primarily discriminates in lower quartile (~15-18 tickers); top-60 tends toward ceiling effect (median = 1.000)
- IC tests blocked on Spec 071 Lane 2 + Gate 4
- Requires >= 30 post-PIT HIT/MISS outcomes for formal evaluation

### Monitoring (updated 2026-05-13)

Shadow-track catalyst_decay_w + binary_quality_score distributions in top-60 monthly (Spec 097). No formal IC claims until gates clear.

**Spec 097 monitoring framework** (canonicalized 2026-05-13):

- Event-EV prospective monitoring with Brier score gate (Brier <= 0.08)
- Minimum n >= 30 calibration threshold required
- Tier-wise validation

**Spec 098 monitoring framework** (canonicalized 2026-05-13):

- Catalyst timing prospective monitor
- Correlation > 0.15 gate required
- Tier-wise validation

---

## catalyst_quality / binary_quality_score (Quality Signal)

Classification of catalyst event quality (Spec 078).

### Key Properties

- binary_quality_score has meaningful variability (IQR ~0.2)
- Joint opportunity (timing + quality): typically ~38% of top-60 tickers

### CTGOV_CALENDAR Dependency

A material share of top-60 catalysts are sourced from ClinicalTrials.gov calendar. Lane 2 dependency confirmed material - some false catalysts expected in any given top-60.

---

## event_ev_p_hit (EV Binder, Spec 077)

Bayesian expected value estimate for catalyst events, binding EV artifacts to resolution outcomes.

### Design

- Forward-only (no backfill)
- Writes null where no EV artifact match exists (correct behavior)
- Prospective sample accumulation required before evaluation

### Gate Requirements

| Gate | Requirement |
| --- | --- |
| Gate 3 | >= 15 non-null event_ev_p_hit records |
| Gate 4 | >= 30 post-PIT HIT/MISS with non-null |
| Spec 079 | Calibration review at n >= 30 |

### Calibration Bias Risk

EV Bayesian priors derived from FDA historical precedent and endpoint type, fit before post-PIT-fix period. If priors are miscalibrated (e.g., FDA accelerated-approval scrutiny shift), values may be systematically biased. Risk level: MEDIUM-HIGH.

---

## Catalyst Resolution Tracker (CRT)

### Architecture Overview

The CRT has two decoupled pipeline steps:

| Step | Location | Code | Purpose |
|------|----------|------|---------|
| **5k.21c** | `run_daily_production.py:5537` | `scripts/research/build_crt_options_join.py` | Builds `crt_options_join.json` — CRT data joined with option snapshots + price history (120s timeout, subprocess) |
| **5m** | `run_daily_production.py:5834` | `tools.catalyst_resolution_tracker.run_crt()` | Updates per-ticker resolution files, watchlist_current.json (direct call) |

**Critical property**: Steps 5k and 5m are **decoupled** — one can fail without blocking the other. The CRT tracker does NOT depend on the join table.

### CRT Options Join Table (`output/catalyst_ev/crt_options_join.json`)

Research artifact joining CRT resolution records with option-state snapshot rankings and price history from `production_data/price_history.csv`. Timestamp is the primary freshness indicator.

**Timeout risk**: Runs as subprocess with 120s timeout. `_load_prices()` reads entire `price_history.csv` (~491K lines) per run. On slow I/O (WSL cold-start), processing 40-50 resolution records against snapshot dirs can exceed 120s.

### Per-ticker resolution files

**Location**: `data/snapshots/resolutions/{YYYY-MM}/`

Individual per-ticker resolution files tracking catalyst event outcomes.

### Resolution States

| State | Meaning |
| --- | --- |
| HIT | Catalyst event occurred and was positive |
| MISS | Catalyst event occurred and was negative |
| PENDING | Event not yet resolved |
| EXPIRED | Event window passed without resolution |

### watchlist_current.json

- Today-only aggregator regenerated on every cron run
- NOT tracked in git (gitignored after contaminating commits)
- History captured by per-ticker resolution files
- Freshness check: as_of_date within 3 days (WARN if stale)

### CRT Join Table Staleness Diagnosis

When investigating a stale `crt_options_join.json`:

**Step 1 — Check timestamp:**
```bash
stat output/catalyst_ev/crt_options_join.json
```
If >72h old (not counting weekends), investigate further.

**Step 2 — Search production logs for timeout:**
```bash
grep -i "CRT join table" logs/daily_production_2026-05-*.log
```
Look for: `WARNING - CRT join table refresh failed: Command ... timed out after 120 seconds`

**Step 3 — Check if the CRT tracker (step 5m) ran independently:**
```bash
grep "CRT →" logs/daily_production_2026-05-*.log
```
If the tracker ran but the join table is stale, the issue is isolated to step 5k.21c.

**Step 4 — Check if production aborted before CRT steps:**
```bash
grep -n "Aborting before screen run" logs/daily_production_2026-05-*.log
```
If market data is stale (4+ days), the pipeline aborts at step 3 before ever reaching CRT.

**Step 5 — Check price_history.csv size** (common timeout cause):
```bash
wc -l production_data/price_history.csv
```
If >400K lines, the full CSV load in `_load_prices()` is a likely contributor to the timeout.

**Common root causes:**
1. **Timeout** (most likely) — 120s insufficient when price CSV is large or I/O is slow
2. **Pipeline abortion** — stale market data prevents pipeline from reaching step 5k
3. **No new resolutions** — if no pending catalysts resolved, the join table doesn't need updating (stale is acceptable)

**Remediation options** (requires operator approval, blocked during architecture freezes):
1. Increase timeout (e.g., 120s → 300s) in `run_daily_production.py` line ~5545
2. Optimize `build_crt_options_join.py` — use indexed price access instead of full CSV load, or early-exit if no new resolutions since last run
3. Both

See `references/crt-join-table-timeout-diagnosis.md` for full reproduction recipe.

---

## AACT Pipeline

ClinicalTrials.gov data ingestion via AACT database.

### Timing

- Pipeline timeout: 6000s (100 min) to cover worst-case AACT + tail steps
- Monday runs are typically longest (weekend AACT batch)
- Previous 4500s (75 min) timeout was killing the pipeline mid-AACT

### Cache Warming

```bash
warm_caches.py --sources sec_8k,ctgov,sec_13f,fda_adcom,fda_regulatory,euctr,ctis,isrctn,merged_trials
```

Always warm 8-K cache BEFORE running screen.

---

## Composite Integration

Catalyst signals enter Module 5 composite via Module 3:

| Weight Set | Catalyst Weight |
| --- | --- |
| V3 Enhanced | Part of remaining allocation |
| V3 Default | 25% (legacy) |

---

## Source Files

| Component | File |
| --- | --- |
| CRT Join Table Builder | `scripts/research/build_crt_options_join.py` |
| Event Ledger Builder | `event_ledger.py` |
| Catalyst Resolution Tracker | `catalyst_resolution_tracker.py` |
| Module 3 Scoring | `module_3_scoring_v2.py` |
| Snapshot Column Spec | `run_screen_columns.py` |
| Cache Warmer | `warm_caches.py` |
| AACT Collector | `wake_robin_data_pipeline/collectors/` |
| Trial Registry Merger | `trial_registry_merger.py` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** — The values below are point-in-time and go stale. Verify against current pipeline output before citing.

---

## event_ev_p_hit Gate Progress

*Last reviewed: 2026-05-18*

- Binder shipped and operational (forward-only, confirmed present 2026-05-08)
- Spec 087 B1b: PASS (first-fire validation complete 2026-05-13)
- Non-null records accumulated: 0 (as of 2026-05-08)
- Gate 3 (n >= 15): **0/15** — accumulating
- Gate 4 (n >= 30): **0/30** — blocked on Gate 3
- Next monthly check: 2026-06-08

## BioShort / Hedge Report Forward Analysis (Spec 092)

*Added: 2026-05-13. Spec 092 Phases A-D all complete.*

Historical backfill of hedge report features across 146 snapshots with forward return analysis (pseudo-PIT):

| Metric | Value |
| --- | --- |
| DEFER verdict accuracy (T+5) | 60.5% (129 samples) |
| Median T+5 return | +0.63% |
| Median T+20 return | +2.49% |
| Median 20d max drawdown | -2.86% |

Research-mode isolation verified: 100% success rate, 0 writes to live output/hedge_report/ path. All Phase D outputs in `artifacts/research/bioshort_backfill/forward_analysis/`.

**Caveat**: Pseudo-PIT (features computed with current logic on historical snapshots). No promotion claims supported per Spec 092 section A6 - descriptive analysis only. Candidate for independent overlay signal or pre-trade timing filter, but requires true forward evidence before any production use.

## binary_quality_score Coverage

*Last reviewed: 2026-05-08*

- 261/261 (100%) catalyst rows classified in current snapshot
- Rising trend in May: n(>0.7) grew from 24 to 34 tickers in top-60
- Joint opportunity (timing + quality): median 23/60 tickers (38%)

## CTGOV_CALENDAR Dependency

*Last reviewed: 2026-05-08*

- ~48% of top-60 catalysts sourced from ClinicalTrials.gov calendar
- ~6 estimated false catalysts in current top-60 (Lane 2 dependency)
- BCRX excluded from monitoring

## catalyst_decay_w Coverage

*Last reviewed: 2026-05-08*

- 299/299 coverage in recent snapshots
- Median = 1.000 in top-60 (ceiling effect confirmed)

## External Catalyst Platforms and FDA Initiative (May 2026)

### Catalyst Calendar Platforms

- BiotechSigns: 970 companies, 74,988 active signals (PDUFA, Phase readouts, insider filings)
- CatalystAlert: 1,624 companies, 14,310 drug pipelines, 3,815 upcoming catalysts
- BioCatalysts.AI: Bio-Score algorithm predicting volatility magnitude per catalyst event
- PDUFA.BIO: 200+ PDUFA events scored by ODIN (96.2% verified accuracy)

### ODIN Confidence Tiers

ODIN assigns binary confidence tiers that correlate with post-approval stock performance:

- TIER_1: >85% approval probability
- TIER_2: 70-85%
- TIER_3: 40-70%
- TIER_4: <40%
Potential external benchmark for CRT catalyst_quality classification.

### FDA Real-Time Clinical Trial Initiative

If the FDA's RTCT initiative succeeds (20-40% trial duration reduction projected), the binary catalyst model evolves:

- Faster time-to-market increases NPV of pipeline assets
- Reduced trial costs improve capital efficiency for small-cap biotechs
- Real-time safety data reduces binary event risk for investors
- Adaptive designs enable mid-trial pivots preserving option value
Monitor as Tier 4 governance question for catalyst_decay_w and catalyst_quality calibration.
