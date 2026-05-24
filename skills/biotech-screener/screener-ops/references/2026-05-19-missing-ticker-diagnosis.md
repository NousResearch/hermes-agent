# Missing Ticker Investigation — DRUG & KYNB (2026-05-19)

## Discovery

During full financial data collection (337-ticker universe), `collect_financial_data.py` produced 333 entries in `financial_data.json`. Four tickers missing: DRUG, KYNB, TARA, TARS. TARA and TARS were resolved by `--ticker` fetch (they simply needed their first collection). DRUG and KYNB had deeper causes.

## Investigation Workflow for Missing Tickers

### Step 1: Targeted Fetch (rule out transient)

```bash
cd /mnt/c/Projects/biotech_screener/biotech-screener
python3 collect_financial_data.py --ticker DRUG,KYNB
```

If the ticker already got an entry from the bulk run, this would update `collected_at`. If still missing after this, proceed to Step 2.

### Step 2: Check CIK Mapping

Query authoritative SEC mapping:

```python
import requests
resp = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers={"User-Agent": "YourName email@domain.com"},
    timeout=30
)
tmap = resp.json()
for entry in tmap.values():
    if entry.get("ticker", "").upper() in ("DRUG", "KYNB"):
        print(entry["ticker"], "→ CIK", entry["cik_str"], entry["title"])
```

### Step 3: Compare Universe CIK vs SEC CIK

```python
import json
# Read universe
u = json.load(open("production_data/universe.json"))
for ticker in ("DRUG", "KYNB"):
    entry = next((e for e in u if e.get("ticker", "").upper() == ticker), None)
    if entry:
        print(f"{ticker}: universe_CIK={entry.get('cik')}, SEC CIK=<from_step_2>")
```

### Step 4: Query CompanyFacts by CIK

For each candidate CIK, check what data exists:

```python
cik_padded = str(cik).zfill(10)
resp = requests.get(
    f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
    headers={"User-Agent": "..."},
    timeout=30
)
if resp.status_code == 200:
    facts = resp.json().get("facts", {})
    print("GAAP keys:", len(facts.get("us-gaap", {})))
    print("IFRS keys:", len(facts.get("ifrs-full", {})))
    # Check specific fields
    cash = facts.get("ifrs-full", {}).get("CashAndCashEquivalents", {})
    if cash:
        units = cash.get("units", {})
        print("Available units:", list(units.keys()))
        # Latest entry
        for unit, entries in units.items():
            print(f"Latest {unit}:", entries[-1] if entries else "none")
```

### Step 5: Check Freshness Cutoff

Financial data has a 4-year (1461-day) freshness cutoff. Compute whether the latest filing `end` date is within the window.

### Step 6: Check IFRS Currency List

If the ticker is an IFRS filer, check whether the reporting currency is in the script's supported list. `collect_financial_data.py` had this list: EUR, USD, GBP, JPY, AUD, NZD, HKD, SGD, KRW, INR, ZAR.

**Confirmed missing: CAD** — DRUG reports in CAD. The fix is to add "CAD" to this list.

## Findings for DRUG (Bright Minds Biosciences)

- **CIK**: 0001836311 (correct in both universe and SEC mapping)
- **Taxonomy**: IFRS (`ifrs-full`), not US-GAAP
- **Currency**: CAD (Canadian Dollars) — NOT in script's supported IFRS currency list
- **Latest filing**: 2024-09-30 (within 4-year cutoff)
- **Data exists**: Cash $5.7M CAD, Assets $6.1M CAD, R&D $1.2M CAD, Net Loss -$2.8M CAD
- **Root cause**: Script silently skipped all IFRS-CAD data because CAD wasn't in the currency whitelist
- **Fix**: Add `"CAD"` to the IFRS currencies list in `collect_financial_data.py`

## Findings for KYNB (Kyntra Bio / formerly FibroGen)

- **Universe CIK**: 0001609702 (legacy entity — last data 2022-03-31, stale)
- **SEC-mapped CIK**: 0000921299 (correct — Kyntra Bio, active)
- **Data at wrong CIK**: Filing exists but data is stale past 4-year cutoff → silently excluded
- **Data at correct CIK**: $37M USD cash, $159M USD assets as of 2026-03-31
- **Root cause**: `collect_financial_data.py` checks universe CIK first, never reaches SEC-mapped CIK
- **Fix**: Update KYNB's CIK in `production_data/universe.json` from 0001609702 to 0000921299

## Summary Table

| Ticker | Universe CIK | SEC CIK | Categorization | Tax | Currency | Latest Data | Reason Missing |
|--------|-------------|---------|----------------|-----|----------|-------------|----------------|
| DRUG | 0001836311 | 0001836311 | IFRS Filer | ifrs-full | CAD | 2024-09-30 | CAD missing from currency list |
| KYNB | 0001609702 | **0000921299** | Stale CIK | us-gaap | USD | 2022-03-31 | CIK mismatch in universe.json |

## Broader Implications

1. **Canadian/TSX biotechs on EDGAR**: Many Canadian biotechs (cross-listed on NASDAQ/TSX) report under IFRS in CAD. The currency list should include CAD preemptively.
2. **Renamed tickers**: When a company changes its ticker (e.g., FIBROGEN → KYNB), the universe.json CIK entry may be a legacy CIK. Ticker-rename events are the primary source of CIK drift.
3. **Script fixes needed (if/when authorized)**:
   - `collect_financial_data.py`: Add "CAD" to IFRS currency list
   - `production_data/universe.json`: Update KYNB CIK to 0000921299
