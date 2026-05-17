# CourtListener — Free Law Project

## 1. Summary

CourtListener (Free Law Project) aggregates court opinions, dockets, oral
arguments, and judge data. Covers ~10M federal and state court opinions
back to colonial America, plus PACER docket data from RECAP submissions.

## 2. Access Methods

- **REST API v4:** `https://www.courtlistener.com/api/rest/v4/`
- **Auth:** Anonymous reads allowed on most endpoints; token raises rate
  limits and unlocks bulk export
- **Rate limit:** ~5,000 req/hour unauthenticated for search; higher with token

Set `COURTLISTENER_TOKEN` env var. Get a free token at
https://www.courtlistener.com/sign-in/ then create an API key.

## 3. Data Schema

Key fields emitted by `fetch_courtlistener.py`:

| Column | Type | Description |
|--------|------|-------------|
| `case_name` | str | Case name |
| `court` | str | Court name |
| `court_id` | str | Court ID (e.g. `nysd`, `scotus`, `ca9`) |
| `date_filed` | str | YYYY-MM-DD |
| `docket_number` | str | Court docket number |
| `judge` | str | Judge name(s) |
| `citation` | str | Reporter citation(s) |
| `result_type` | str | opinions / dockets / oral / people |
| `snippet` | str | Search-match snippet (up to 500 chars) |
| `absolute_url` | str | Direct CourtListener URL |

## 4. Coverage

- Federal: all circuit and district courts, SCOTUS
- State: all 50 state supreme/appellate courts, many trial courts
- Opinions: ~10M back to 1600s (colonial), full coverage 1950 → present
- Dockets via RECAP: ~3M+ from user-submitted PACER PDFs
- Updated continuously

## 5. Cross-Reference Potential

- **OpenCorporates** ↔ `case_name` (corporate litigation)
- **SEC EDGAR** ↔ `case_name` (securities class actions)
- **OFAC SDN** ↔ `case_name` (sanctions-related civil/criminal cases)

Join key: party name from `case_name`. Note: `case_name` often abbreviates
("Smith v. Jones" rather than full party names) — use the full case URL
to get all parties.

## 6. Data Quality

- Older opinions (pre-1990) often lack docket numbers and judges
- State coverage is more uneven than federal
- PACER docket coverage depends on RECAP user submissions — not exhaustive
- Sealed documents are excluded
- Party names in case captions don't always match filing names exactly

## 7. Acquisition Script

Path: `scripts/fetch_courtlistener.py`

```bash
# Search opinions for a party / keyword
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Example Corp" \
    --out data/cl.csv

# PACER dockets (best for recent litigation)
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Example Corp" \
    --type dockets --out data/cl_dockets.csv

# Restrict to a court
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Microsoft" \
    --court ca9 --out data/cl_9th.csv

# Date range
python3 SKILL_DIR/scripts/fetch_courtlistener.py --query "Example Corp" \
    --date-from 2020-01-01 --date-to 2024-12-31 --out data/cl.csv
```

Pass `--token` or set `COURTLISTENER_TOKEN`.

## 8. Legal & Licensing

- Court opinions are public domain
- Free Law Project provides the data under CC0 / public domain dedication
- No commercial use restrictions on opinion text or metadata
- Some PACER PDFs have copyright on layout (not text) — fair use applies

## 9. References

- API docs: https://www.courtlistener.com/help/api/rest/
- Court IDs: https://www.courtlistener.com/api/jurisdictions/
- RECAP archive: https://www.courtlistener.com/recap/
- Bulk data: https://www.courtlistener.com/help/api/bulk-data/
