# Senate LD — Lobbying Disclosure (LD-1 / LD-2)

## 1. Summary

The Senate Office of Public Records publishes lobbying disclosures under the
Lobbying Disclosure Act of 1995 (LDA, as amended by HLOGA 2007). LD-1 is
registration of a new client-lobbyist relationship; LD-2 is the quarterly
activity report.

## 2. Access Methods

- **API:** `https://lda.senate.gov/api/v1/` (no auth required for read-only)
- **Bulk download:** `https://lda.senate.gov/api/v1/filings/?format=csv` (paginated)
- **Auth:** Token required for >120 req/hour — register at https://lda.senate.gov/api/auth/register/
- **Rate limit:** 120 req/hour unauthenticated, 1,200 req/hour authenticated

## 3. Data Schema

Key fields emitted by `fetch_senate_ld.py`:

| Column | Type | Description |
|--------|------|-------------|
| `filing_uuid` | str | Unique filing ID |
| `filing_type` | str | LD-1, LD-2, LD-203, etc. |
| `filing_year` | int | Year |
| `filing_period` | str | Q1/Q2/Q3/Q4 or annual |
| `registrant_name` | str | Lobbying firm or organization |
| `registrant_id` | str | Senate-assigned registrant ID |
| `client_name` | str | Client being represented |
| `client_id` | str | Senate-assigned client ID |
| `client_general_description` | str | Client industry / business |
| `income` | float | LD-2 income from client this quarter (USD) |
| `expenses` | float | LD-2 expenses (in-house lobbying) |
| `lobbyists` | str | Semicolon-separated lobbyist names |
| `issues` | str | Semicolon-separated issue areas |
| `government_entities` | str | Agencies/chambers contacted |
| `filing_date` | str | YYYY-MM-DD |

## 4. Coverage

- US federal lobbying only (state lobbying handled by individual state ethics offices)
- 1999 → present (full electronic coverage from 2008)
- Quarterly reporting cycle (LD-2)
- ~1M+ filings cumulative

## 5. Cross-Reference Potential

- **USAspending** ↔ `client_name` (clients lobbying for contracts)
- **SEC EDGAR** ↔ `client_name` (public companies as lobbying clients)
- **OFAC SDN** ↔ `client_name` (sanctions screening of lobbying clients)

Join key: normalized client_name. registrant_id and client_id are canonical
when joining Senate-internal records.

## 6. Data Quality

- Many lobbyist names appear in multiple registrants over time (job changes)
- `issues` and `government_entities` are free-text — Inconsistent capitalization
- Foreign agents register under FARA (Department of Justice), NOT here
- Income/expenses are reported in $10,000 brackets in some older filings

## 7. Acquisition Script

Path: `scripts/fetch_senate_ld.py`

```bash
# By client
python3 SKILL_DIR/scripts/fetch_senate_ld.py --client "EXAMPLE CORP" \
    --year 2024 --out data/lobbying.csv

# By registrant (lobbying firm)
python3 SKILL_DIR/scripts/fetch_senate_ld.py --registrant "BIG K STREET LLP" \
    --year 2024 --out data/lobbying.csv
```

Set `SENATE_LDA_TOKEN` env var if you have one (or pass `--token`).
Defaults to anonymous (120 req/hour).

## 8. Legal & Licensing

- Public record under 2 U.S.C. § 1604 (LDA)
- No commercial use restrictions
- Reuse is unconditional — see Senate Public Records Office disclaimer

## 9. References

- API docs: https://lda.senate.gov/api/redoc/v1/
- LDA guidance: https://lobbyingdisclosure.house.gov/ld_guidance.pdf
- Senate Public Records: https://lda.senate.gov/
