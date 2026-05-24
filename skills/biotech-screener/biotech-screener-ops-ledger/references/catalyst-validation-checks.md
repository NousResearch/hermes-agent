# Catalyst Data Validation Checks
# Biotech Screener — rankings.csv catalyst field audit
# Run after daily production against data/snapshots/<date>/rankings.csv
# Absorbed from: de-schema-gate-fix skill

## Quick validation script (Python)

```python
import csv
from datetime import datetime

AS_OF = "2026-05-06"  # set to snapshot date
SNAP = f"/mnt/c/Projects/biotech_screener/biotech-screener/data/snapshots/{AS_OF}/rankings.csv"
as_of_dt = datetime.strptime(AS_OF, "%Y-%m-%d")

with open(SNAP) as f:
    rows = list(csv.DictReader(f))

eligible = [r for r in rows if r.get("eligible","").strip() == "1"]
portfolio = [r for r in eligible if r.get("target_weight_pct","").strip() not in ("","nan","")]

# --- CHECK 1: catalyst_days vs next_catalyst_date consistency ---
# Far-window hydration can leave next_catalyst_date stale (M3 date) while
# catalyst_days reflects the near CTGov PCD. Delta > 5d = stale date bug.
print("=== CHECK 1: catalyst_days vs next_catalyst_date ===")
issues = []
for r in eligible:
    if r.get("catalyst_mode","") not in ("specific_days", "far_window"):
        continue
    days_str = r.get("catalyst_days","").strip()
    ncd_str = r.get("next_catalyst_date","").strip()
    if not days_str or not ncd_str:
        continue
    try:
        days = int(float(days_str))
        ncd = datetime.strptime(ncd_str[:10], "%Y-%m-%d")
        implied = (ncd - as_of_dt).days
        if abs(implied - days) > 5:
            issues.append(f"  {r['ticker']}: catalyst_days={days} ncd={ncd_str} implied={implied}d delta={abs(implied-days)}")
    except Exception:
        pass
if issues:
    print(f"  INCONSISTENCIES ({len(issues)}):")
    for i in issues: print(i)
else:
    print("  OK")

# --- CHECK 2: portfolio negative/zero catalyst_days ---
print("\n=== CHECK 2: Portfolio catalyst_days > 0 ===")
stale = [(r["ticker"], r.get("catalyst_days","")) for r in portfolio
         if r.get("catalyst_days","").strip() and float(r.get("catalyst_days","0") or 0) <= 0]
print(f"  WARN: {stale}" if stale else "  OK")

# --- CHECK 3: priority/event_type alignment ---
print("\n=== CHECK 3: Priority/event_type alignment ===")
for r in portfolio:
    evt = r.get("catalyst_event_type","")
    pri = r.get("cat_priority","")
    if "PDUFA" in evt and pri != "1":
        print(f"  WARN {r['ticker']}: PDUFA event but cat_priority={pri}")
    if evt == "DATA_READOUT" and pri == "1":
        print(f"  WARN {r['ticker']}: DATA_READOUT with cat_priority=1")
print("  (no output = OK)")

# --- CHECK 4: catalyst_mode distribution ---
print("\n=== CHECK 4: catalyst_mode distribution (eligible) ===")
modes = {}
for r in eligible:
    m = r.get("catalyst_mode","") or "EMPTY"
    modes[m] = modes.get(m, 0) + 1
for k,v in sorted(modes.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} {v}")

# --- CHECK 5: far_window names in portfolio ---
print("\n=== CHECK 5: far_window portfolio names ===")
fw = [r for r in portfolio if r.get("catalyst_mode","") == "far_window"]
if fw:
    for r in fw:
        print(f"  {r['ticker']}: days={r.get('catalyst_days')} lower={r.get('catalyst_date_lower')} upper={r.get('catalyst_date_upper')} src={r.get('catalyst_source')}")
else:
    print("  None (all portfolio names have specific_days)")
```

## Known structural gaps (not bugs)

- `catalyst_days` empty for tickers with `cat_mode=no_upcoming` — by design
- `clinical_score_z` empty for `platform_*` archetypes — clinical z only for drug_developers
- `runway_buffer_months` empty for approved/commercial names — data unavailable for these
- `tier_dev` empty for non-drug-developer archetypes — by design

## Fixed bugs (historical)

| Bug | Fix | Commit |
|-----|-----|--------|
| 11 Spec 057/061 columns missing from rankings.csv | Added to SNAPSHOT_COLUMNS in run_screen_columns.py | bd777483 |
| False rank mismatch WARNs (DNTH/PHVS/RCUS/RVMD) | Set-based rank check in run_daily_production.py | bd777483 |
| far_window next_catalyst_date stale (NGNE/CABA/HALO) | Recompute in _hydrate_far_horizon_catalysts | 91c383bf |
