# NYC ACRIS — NYC Real Property Records

## 1. Summary

The Automated City Register Information System (ACRIS) is NYC's index of
recorded property documents: deeds, mortgages, satisfactions, liens, UCC
filings. Covers Manhattan, Bronx, Brooklyn, Queens, Staten Island.
Published as 4 linked Socrata datasets on the NYC Open Data portal.

## 2. Access Methods

- **Socrata API:** `https://data.cityofnewyork.us/resource/636b-3b5g.json` (Parties)
- **Other datasets:** `bnx9-e6tj` (Master), `8h5j-fqxa` (Legal), `uqqa-hym2` (References)
- **Auth:** None for read access (Socrata `$app_token` raises rate limits if needed)
- **Rate limit:** Generous (~1000 req/hour unauthenticated)

## 3. Data Schema

Key fields emitted by `fetch_nyc_acris.py` (Parties joined to Master):

| Column | Type | Description |
|--------|------|-------------|
| `document_id` | str | ACRIS document ID |
| `name` | str | Party name as recorded (often "LAST, FIRST" but varies) |
| `party_type` | str | 1=grantor, 2=grantee, 3=other |
| `party_role` | str | Human-readable role label |
| `address_1` | str | Property or party address line 1 |
| `city`, `state`, `zip`, `country` | str | Address parts |
| `doc_type` | str | DEED, MTGE (mortgage), SAT (satisfaction), AGMT, etc. |
| `doc_date`, `recorded_date` | str | YYYY-MM-DD |
| `borough` | str | Manhattan / Bronx / Brooklyn / Queens / Staten Island |
| `amount` | str | Document amount (USD, when applicable) |
| `filing_url` | str | Direct ACRIS DocumentImageView link |

## 4. Coverage

- NYC 5 boroughs only — other counties have their own recorders
- 1966 → present (older filings exist on microfilm at the County Clerk)
- Updated nightly
- ~70M+ party records cumulative

## 5. Cross-Reference Potential

- **SEC EDGAR** ↔ `name` (insider filers with NYC property)
- **USAspending** ↔ `name` (federal contractors with NYC property)
- **Senate LDA** ↔ `name` (lobbyists / clients with NYC property)
- **ICIJ Offshore** ↔ `name` (NYC properties owned via offshore vehicles)

Join key: normalized party name. NYC property records typically store names
as "LAST, FIRST" or full LLC names — use `entity_resolution.py`.

## 6. Data Quality

- Same person appears with multiple name formats over time
- LLC and trust ownership obscures beneficial owners
- Recording lag can be 2-4 weeks after closing
- Older documents have spottier address data
- Sealed records (e.g. domestic violence shelters) are excluded by law

## 7. Acquisition Script

Path: `scripts/fetch_nyc_acris.py`

```bash
# By party name
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "ROLNICK" --out data/acris.csv

# By address (useful when you know the property but not the names)
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --address "571 HUDSON" --out data/acris.csv

# Restrict to grantees (buyers / mortgagees)
python3 SKILL_DIR/scripts/fetch_nyc_acris.py --name "ROLNICK" --party-type 2 \
    --out data/acris_buyers.csv
```

The script joins Parties → Master to populate doc_type, dates, borough, and
amount. Pass `--no-enrich` to skip the join (faster, fewer columns).

## 8. Legal & Licensing

- Public record under NYS Real Property Law and NYC Charter
- No commercial use restrictions on the data
- All ACRIS data is public information by statute

## 9. References

- ACRIS portal: https://a836-acris.nyc.gov/CP/
- NYC Open Data: https://data.cityofnewyork.us/
- Parties dataset: https://data.cityofnewyork.us/City-Government/ACRIS-Real-Property-Parties/636b-3b5g
- Document type codes: https://www1.nyc.gov/site/finance/taxes/acris.page
