---
name: sec-edgar-mechanics
description: "Parse SEC EDGAR filings for biotech signals."
---

# SEC EDGAR Mechanics

## Purpose

Encode the mechanical rules for interacting with SEC EDGAR — API conventions, filing type taxonomy, CIK resolution, XBRL financial data extraction quirks, and dedup logic. This skill supports the institutional-signal pipeline, financial data collection, and the three SEC filing routines (13F Monitor, 13D/13G Monitor, Full Registry).

---

## EDGAR API Conventions

### Fair-Use Policy

SEC requires a valid `User-Agent` header identifying the requester. Format:
```
User-Agent: CompanyName AdminContact@company.com
```

SEC_USER_AGENT preflight check added 2026-05-13 to all EDGAR requests. Requests without a valid User-Agent are throttled or blocked.

### Rate Limiting

- Maximum 10 requests per second to EDGAR
- Respect `Retry-After` headers
- Exponential backoff on 429/503 responses
- EDGAR is a public service — do not abuse it

### Key Endpoints

| Endpoint | Use |
| --- | --- |
| `efts.sec.gov/LATEST/search-index` | Full-text search across filings |
| `data.sec.gov/submissions/CIK{cik}.json` | Filing history by CIK |
| `www.sec.gov/cgi-bin/browse-edgar` | Legacy search interface |
| `efts.sec.gov/LATEST/xbrl-inline` | Inline XBRL data |
| `data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json` | XBRL company facts (financial data) |
| `www.sec.gov/files/company_tickers.json` | Ticker → CIK mapping (authoritative) |

---

## CIK Resolution

CIK (Central Index Key) is the unique SEC identifier for each filer.

### Lookup Methods

1. **Manager registry:** `production_data/manager_registry.json` maps firm names to CIKs
2. **EDGAR company search:** `www.sec.gov/cgi-bin/browse-edgar?company=&CIK={name}&type=13F`
3. **Submissions API:** `data.sec.gov/submissions/CIK{cik_padded}.json` (10-digit zero-padded)
4. **Company tickers (authoritative):** `www.sec.gov/files/company_tickers.json` — the SEC's own ticker→CIK map

### CIK Resolution Order (ticker → CIK)

When resolving a ticker's CIK across multiple sources, the resolution order matters:

1. **Universe CIK** (e.g., `universe.json` or `universe_ciks` dict) — checked FIRST
2. **SEC company_tickers.json** — checked SECOND (fallback), authoritative for ticker→CIK mapping

**Critical pitfall**: If universe.json has a stale CIK for a renamed ticker, the correct SEC-mapped CIK is never reached. Example: KYNB (Kyntra Bio, formerly FibroGen) — universe.json had CIK 0001609702 (old entity, last data 2022-03-31, stale), while SEC maps KYNB → CIK 0000921299 (active, $37M cash as of 2026-03-31). The script fetched from the stale CIK, got data past the 4-year freshness cutoff, and reported "No filings."

**Diagnosis**: Compare CIKs between universe and SEC ticker map:
```python
import requests
headers = {"User-Agent": "YourName email@domain.com"}
resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
tmap = resp.json()
for entry in tmap.values():
    if entry.get("ticker", "").upper() in ("KYNB",):
        print(entry["ticker"], "→ CIK", entry["cik_str"], "name=", entry["title"])
```

### CIK Mismatch Investigation (missing filings)

When a ticker returns "No filings (EDGAR + yfinance)":

1. Verify CIK in SEC company_tickers.json — does the ticker map to a CIK at all?
2. If CIK exists, check CompanyFacts endpoint: `data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json`
3. Check HTTP status (200 = data exists, 404 = no filings ever)
4. Check fact categories: `facts.get("us-gaap", {})` for US filers, `facts.get("ifrs-full", {})` for foreign filers
5. Check data freshness: compare latest `end` dates against the 4-year (1461-day) freshness cutoff
6. Check DEI fields for entity name identification
7. If universe CIK differs from SEC-mapped CIK, the universe CIK is likely stale — update it

### Key CIKs (Priority Firms)

| Firm | CIK |
| --- | --- |
| Fairmount Funds Management | 0001802528 |
| Deep Track Capital | 0001856083 |
| Logos Global Management | 0001792126 |

### CIK Rules

- Always zero-pad to 10 digits for API calls: `CIK0001802528`
- CIKs are stable — they do not change when firms rename
- One CIK per legal entity; multiple funds under one manager share a CIK for 13F purposes
- Some managers file through multiple CIKs (e.g., Foresite Capital IV/V/VI have separate CIKs)

---

## Financial Data Extraction from XBRL

### GAAP vs IFRS

Filings use one of two fact taxonomies:

| Taxonomy | Standard | Prefix |
| --- | --- | --- |
| `us-gaap` | US Generally Accepted Accounting Principles | US filers |
| `ifrs-full` | International Financial Reporting Standards | Foreign filers (F-1, F-3, etc.) |

**Detection**: Check `facts` dict keys. If `"ifrs-full"` is present and `"us-gaap"` is absent (or sparse), the company is an IFRS filer.

### IFRS Currency Handling

IFRS filers report in their domestic currency, not necessarily USD. The script's IFRS fallback must include ALL possible reporting currencies, not just common ones.

**Known gap (confirmed 2026-05-19)**: `collect_financial_data.py` supports IFRS fallback but only checks: EUR, USD, GBP, JPY, AUD, NZD, HKD, SGD, KRW, INR, ZAR. Missing currencies cause silent data drop — the script finds IFRS data but skips it because the currency doesn't match.

**Confirmed gap**: **CAD** was missing from the IFRS currency list. DRUG (Bright Minds Biosciences) reports in CAD under IFRS — "ifrs-full" facts exist with "Cash and cash equivalents" = 5,696,000 CAD as of 2024-09-30, but the currency filter silently dropped all entries. To fix: add `"CAD"` to the IFRS currency list in the script.

**Currency-diverse IFRS filers to watch**: Canadian biotechs (TSX-listed or US cross-listed), UK-based ADRs, Australian ADRs may report in CAD, GBP, AUD respectively.

### XBRL Fact Field Mapping

Common financial data fields under each taxonomy:

| Concept | GAAP (`us-gaap`) | IFRS (`ifrs-full`) |
| --- | --- | --- |
| Cash & Equivalents | `CashAndCashEquivalentsAtCarryingValue` | `CashAndCashEquivalents` |
| Revenue | `RevenueFromContractWithCustomerExcludingAssessedTax` | `Revenue` |
| R&D Expense | `ResearchAndDevelopmentExpense` | `ResearchAndDevelopmentExpense` |
| Assets (Total) | `Assets` | `Assets` |
| Long-term Debt | `LongTermDebtNoncurrent` | `LongTermBorrowings` |
| Net Income | `NetIncomeLoss` | `ProfitLoss` |
| Operating Cash Flow | `NetCashProvidedByUsedInOperatingActivities` | `CashFlowsFromUsedInOperatingActivities` |

### CIK → Entity Name Check via DEI

Use DEI (Document and Entity Information) facts to confirm you have the right company:
```python
dei_facts = facts.get("dei", {})
entity_name = dei_facts.get("EntityRegistrantName", {}).get("description")
current_fiscal_year = dei_facts.get("CurrentFiscalYearEndDate", {}).get("description")
```
Cross-check the entity name against your ticker's expected company name.

### Data Freshness Cutoff

Financial data has a **4-year (1461-day) freshness cutoff**. Any filing whose `end` date is older than 1461 days before the collection date is considered stale and excluded. This means a ticker with a stale CIK (still filing but for an old/dormant entity) will silently return no data.

### Debugging Missing Financial Data

When a ticker exists in the universe but produces no financial data:

1. **Check CIK source**: Is the universe CIK different from SEC's ticker map CIK? If yes, the universe CIK is the problem.
2. **Check GAAP vs IFRS**: Query CompanyFacts. If only `ifrs-full` exists, the company is a foreign filer.
3. **Check currency**: What currency does the IFRS data report in? Cross-reference against the script's currency list.
4. **Check freshness**: What is the latest `end` date in the available facts? Is it within the 1461-day window?
5. **Check entity name**: Do DEI facts confirm we're looking at the right company?

---

## Filing Type Taxonomy

### Institutional Holdings (Quarterly)

| Filing | Full Name | Who Files | When | Threshold |
| --- | --- | --- | --- | --- |
| 13F-HR | Institutional Investment Manager Report | Managers with $100M+ AUM | 45 days after quarter-end | $100M AUM |
| 13F-HR/A | Amendment to 13F-HR | Same | After original, correcting errors | Same |

**Filing lag:** SEC_13F_FILING_LAG_DAYS = 45 (built-in constant). Q1 deadline ~May 15, Q2 ~Aug 14, Q3 ~Nov 14, Q4 ~Feb 14.

**Filing pattern:** Filings cluster in the final 3 business days before the deadline. Monitor EDGAR starting ~3 days before deadline.

### Ownership Crossings (Event-Driven)

| Filing | Trigger | Timing | Key Signal |
| --- | --- | --- | --- |
| 13D | Acquire > 5% with activist intent | 10 days after crossing | Activist position, potential campaign |
| 13D/A | Amendment to 13D | Material changes | Position change, intent update |
| 13G | Acquire > 5%, passive intent | 45 days after calendar year-end (or 10 days after crossing 10%) | Passive large holder |
| 13G/A | Amendment to 13G | Annual or upon material change | Position update |

### Insider Transactions

| Filing | Who Files | When | Key Signal |
| --- | --- | --- | --- |
| Form 4 | Officers, directors, 10% holders | 2 business days after transaction | Insider buy/sell activity |

**DEM usage:** `insider_net_buy_value_90d` (Spec 104, DIAGNOSTIC ONLY). Tracks net insider purchase value over trailing 90 days. Blank = not fetched; 0.0 = fetched, no activity.

### Regulatory

| Filing | Use |
| --- | --- |
| Form ADV | Investment advisor registration, AUM disclosure |
| Form ADV/A | Annual amendment |

---

## CUSIP-First Reasoning

**Rule from institutional-signal:** Always reason from CUSIP to canonical ticker, never the reverse.

- Holdings in 13F-HR XML are identified by CUSIP, not ticker
- CUSIP-to-ticker mapping can change (ticker changes, corporate actions)
- The canonical summary (`institutional_summary.json`) maintains the CUSIP-to-ticker mapping
- Raw EDGAR XML is debug-only — never build narratives from raw filing parses

---

## Accession Number Dedup

Accession numbers uniquely identify each filing. Format: `{filer-id}-{year}-{sequence}` (e.g., `0001104659-26-062419`).

### Dedup Protocol

1. When a filing is processed and alerted, store its accession number in a memory
2. Before alerting on a filing, check if its accession number has already been processed
3. If the accession number exists in memories, skip the alert (do not re-alert)
4. Memories include the date processed and the alert target (email recipient)

### Current Dedup State

Stored in global memories. Categories:
- Logos 13G/A filings (May 15, 2026): 4 accession numbers
- Deep Track 13G/A filings (May 15/May 8/May 12, 2026): 12 accession numbers
- Fairmount 13G/A and 13D/A filings (May 15/May 4, 2026): 3 accession numbers
- Q1 2026 13F-HR mass filing (May 15, 2026): 45 accession numbers across 48 registry managers

---

## 13F-HR Parsing

### XML Structure

13F-HR filings contain an information table in XML format listing all holdings:

| Field | Description |
| --- | --- |
| nameOfIssuer | Company name |
| titleOfClass | Security class (e.g., COM, SHS) |
| cusip | 9-character CUSIP |
| value | Market value in thousands ($) |
| sshPrnamt | Share count |
| sshPrnamtType | Share type (SH = shares) |
| investmentDiscretion | SOLE, SHARED, or DEFINED |
| votingAuthority | Sole, shared, none vote counts |

### Holdings Truth Source

`production_data/institutional_summary.json` is canonical. If raw EDGAR XML count differs from summary count, investigate the summary pipeline first (not the raw filing).

---

## Filing Cycle Operations

### Pre-Refresh Readiness (5 guards)

Run `tools/prep_13f_refresh.py` before processing new filings:

1. Most recent snapshot has valid `institutional_summary_delta.json`
2. `coinvest_score_z` has healthy variance (SD > 0.10)
3. PIT cache has entries within 3 days of today
4. SEC EDGAR endpoint is reachable
5. Dry-run: `build_institutional_summary()` produces valid output (>=80% coverage)

### Post-Filing Action Sequence

1. Warm 13F cache (`tools/warm_13f_cache.py`)
2. Run cohort quarantine (`tools/check_13f_cohort_quarantine.py`)
3. Check collapse guards (coinvest_score_z SD)
4. Refresh IC decomposition
5. 5-day observation window before treating as production-grade

### Manager Registry

**File:** `production_data/manager_registry.json`
**Never edit directly** -- use `tools/onboard_manager.py`
**Current state:** 48 managers (42 elite_core + 6 conditional)

---

## Source Files

| Component | File |
| --- | --- |
| 13F Cache Warmer | `tools/warm_13f_cache.py` |
| Institutional Summary Builder | `build_institutional_summary.py` |
| Manager Onboarding | `tools/onboard_manager.py` |
| 13F Refresh Readiness | `tools/prep_13f_refresh.py` |
| Cohort Quarantine | `tools/check_13f_cohort_quarantine.py` |
| Snapshot Integrity | `tools/verify_snapshot_integrity.py` |
| Financial Data Collection | `collect_financial_data.py` |
