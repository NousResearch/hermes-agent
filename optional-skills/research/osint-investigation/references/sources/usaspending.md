# USAspending — Federal Government Contracts and Grants

## 1. Summary

USAspending.gov is the official source of federal spending data. Coverage:
contracts, grants, loans, direct payments, sub-awards. Required by the DATA Act
of 2014 — all federal agencies must report to a single schema.

## 2. Access Methods

- **API v2:** `https://api.usaspending.gov/api/v2/` (no auth, no key)
- **Bulk:** `https://files.usaspending.gov/` (CSV / Parquet by award type)
- **Auth:** None
- **Rate limit:** Not strictly enforced, but be polite — keep to <10 req/s

## 3. Data Schema

Key fields emitted by `fetch_usaspending.py` (prime awards):

| Column | Type | Description |
|--------|------|-------------|
| `award_id` | str | Federal award ID (PIID for contracts, FAIN for grants) |
| `recipient_name` | str | Awardee legal name |
| `recipient_uei` | str | Unique Entity Identifier (replaced DUNS in 2022) |
| `recipient_duns` | str | Legacy DUNS number (historical only) |
| `recipient_parent_name` | str | Ultimate parent organization |
| `recipient_state` | str | Recipient state |
| `awarding_agency` | str | Department / agency name |
| `awarding_sub_agency` | str | Sub-tier (e.g. DoD → Army) |
| `award_type` | str | Contract / Grant / Loan / Direct Payment |
| `award_amount` | float | Current total obligation in USD |
| `award_date` | str | Action / signed date YYYY-MM-DD |
| `period_of_performance_start` | str | YYYY-MM-DD |
| `period_of_performance_end` | str | YYYY-MM-DD |
| `naics_code` | str | Industry classification |
| `psc_code` | str | Product / Service Code |
| `competition_extent` | str | Full / limited / sole-source |
| `description` | str | Award description (free-text) |

## 4. Coverage

- US federal awards only (state/local not included)
- FY 2008 → present (full coverage from FY 2017)
- Updated bi-weekly from agency reporting
- ~100M+ transaction records cumulative

## 5. Cross-Reference Potential

- **SEC EDGAR** ↔ `recipient_name` (public companies as contractors)
- **Senate LD** ↔ `recipient_name` (lobbying clients winning contracts)
- **OFAC SDN** ↔ `recipient_name` (sanctions screening of contractors — must be
  filtered out by SAM.gov but verify)
- **ICIJ Offshore** ↔ `recipient_name` (offshore-linked contractors)

Join key: normalized recipient name. UEI is canonical when present.

## 6. Data Quality

- DUNS → UEI transition (April 2022) — old records have DUNS, new records have UEI
- Some sub-awards aren't reported (FFATA threshold is $30k)
- Award amount changes over time (mod actions) — fetch script reports current total
- `competition_extent` field is free-text in older records — `fetch_usaspending.py`
  normalizes to canonical values
- Recipient name variations are extensive — "ACME LLC", "Acme L.L.C.", "ACME, INC"
  all appear. Use `entity_resolution.py`.

## 7. Acquisition Script

Path: `scripts/fetch_usaspending.py`

```bash
# By recipient name
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --out data/contracts.csv

# By awarding agency
python3 SKILL_DIR/scripts/fetch_usaspending.py --agency "Department of Defense" \
    --fy 2024 --out data/contracts.csv

# Filter to sole-source only
python3 SKILL_DIR/scripts/fetch_usaspending.py --recipient "EXAMPLE CORP" \
    --fy 2024 --sole-source-only --out data/contracts.csv
```

## 8. Legal & Licensing

- Public record under the Federal Funding Accountability and Transparency Act
  (FFATA, 2006) and DATA Act (2014)
- No commercial use restrictions on the data
- Personal information of award recipients (e.g. small business owners' addresses
  in some grants) should be handled per the source agency's privacy notice

## 9. References

- API docs: https://api.usaspending.gov/
- Data dictionary: https://www.usaspending.gov/data-dictionary
- Award schema: https://files.usaspending.gov/docs/Data_Dictionary_Crosswalk.xlsx
