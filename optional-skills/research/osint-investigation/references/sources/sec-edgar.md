# SEC EDGAR — Corporate Filings

## 1. Summary

EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the SEC's system
for corporate disclosure filings: 10-K (annual), 10-Q (quarterly), 8-K (current
events), DEF 14A (proxy), Form 4 (insider trading), 13F (institutional holdings).

## 2. Access Methods

- **API:** `https://data.sec.gov/submissions/CIK<10-digit-padded>.json` (no auth)
- **Filing index:** `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=...`
- **Full-text search:** `https://efts.sec.gov/LATEST/search-index?q=...`
- **Auth:** None — requires `User-Agent` header with contact info per SEC policy
- **Rate limit:** 10 requests/second per IP (enforced)

## 3. Data Schema

Key fields emitted by `fetch_sec_edgar.py` (filings index):

| Column | Type | Description |
|--------|------|-------------|
| `cik` | str | Central Index Key (10-digit padded) |
| `company_name` | str | Registrant name |
| `form_type` | str | 10-K, 10-Q, 8-K, etc. |
| `filing_date` | str | YYYY-MM-DD |
| `accession_number` | str | Filing accession (e.g. 0000320193-24-000123) |
| `primary_document` | str | Filename of main document |
| `filing_url` | str | Direct URL to filing index |
| `reporting_period` | str | Period of report (where applicable) |

## 4. Coverage

- All public US registrants from 1993 → present
- 1993-2000 has spotty coverage of older filings (paper-to-electronic migration)
- ~12M filings cumulative
- Updated within minutes of filing acceptance

## 5. Cross-Reference Potential

- **USAspending** ↔ `company_name` (public companies as federal contractors)
- **Senate LD** ↔ `company_name` (public companies hire lobbyists)
- **OFAC SDN** ↔ `company_name` (sanctions screening of public registrants)

Join key: company name OR CIK if you have it. CIK is canonical and stable.

## 6. Data Quality

- Subsidiaries often filed under parent CIK — be careful with name matches
- Name changes over time (rebrands, acquisitions) — CIK remains constant
- 10-K Item 1A Risk Factors are free-form text — useful for `web_extract`-style
  parsing, not structured queries
- Foreign private issuers file 20-F instead of 10-K

## 7. Acquisition Script

Path: `scripts/fetch_sec_edgar.py`

```bash
# By CIK
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --cik 0000320193 \
    --types 10-K,10-Q --out data/edgar_filings.csv

# By company name (resolves to CIK first via name search)
python3 SKILL_DIR/scripts/fetch_sec_edgar.py --company "APPLE INC" \
    --types 8-K --since 2024-01-01 --out data/edgar_filings.csv
```

Set `SEC_USER_AGENT` env var with your contact email (SEC requirement).
Example: `SEC_USER_AGENT="Research example@example.com"`.

## 8. Legal & Licensing

- Public record under SEC Rule 24b-2 / 17 CFR § 230.401
- No commercial use restrictions on filing content
- SEC asks all bulk users to include a `User-Agent` with contact info and to
  respect 10 req/s — failure to do so can result in IP blocking

## 9. References

- Developer docs: https://www.sec.gov/edgar/sec-api-documentation
- EDGAR full-text search: https://efts.sec.gov/LATEST/search-index
- Fair access policy: https://www.sec.gov/os/accessing-edgar-data
