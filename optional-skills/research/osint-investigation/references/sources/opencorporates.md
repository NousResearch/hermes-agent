# OpenCorporates — Global Corporate Registry

## 1. Summary

OpenCorporates aggregates corporate registry data from 130+ jurisdictions
worldwide (~200M companies). Covers US state-level filings (NY DOS, Delaware
DOC, California SOS, etc.), UK Companies House, EU registries, and most
common-law jurisdictions.

## 2. Access Methods

- **REST API:** `https://api.opencorporates.com/v0.4/`
- **HTML fallback:** `https://opencorporates.com/companies?q=...`
- **Auth:** API token required (free tier 500 calls/month, paid plans available)
- **Rate limit:** Token-bound; un-tokened requests return 401

Set `OPENCORPORATES_API_TOKEN` env var. Get a free token at
https://opencorporates.com/api_accounts/new.

## 3. Data Schema

Key fields emitted by `fetch_opencorporates.py`:

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Company legal name |
| `company_number` | str | Registry-assigned number |
| `jurisdiction_code` | str | e.g. `us_ny`, `us_de`, `gb` |
| `jurisdiction_name` | str | Human-readable jurisdiction |
| `incorporation_date` | str | YYYY-MM-DD |
| `dissolution_date` | str | YYYY-MM-DD (empty if active) |
| `company_type` | str | Domestic LLC / Foreign Corp / etc. |
| `status` | str | Active / Inactive / Dissolved |
| `registered_address` | str | Registered office address |
| `opencorporates_url` | str | Link to OpenCorporates entity page |
| `officers_count` | str | Total officers on record |
| `source` | str | `api`, `html`, or `html-fallback` |

## 4. Coverage

- US: all 50 states + DC at state level (LLCs, corps, LPs)
- International: UK, EU, Canada, Australia, NZ, many APAC + LATAM jurisdictions
- ~200M company records cumulative
- Update frequency varies by jurisdiction (UK CH is near-realtime; some
  state registries lag months)

## 5. Cross-Reference Potential

- **NYC ACRIS** ↔ `name` (LLC/corp owners of NYC property)
- **USAspending** ↔ `name` (corporate federal contractors)
- **SEC EDGAR** ↔ `name` (public companies + their subsidiaries)
- **ICIJ Offshore** ↔ `name` (international corporate structures)

Join key: normalized company name. Some entries have `previous_names` arrays
which are not currently exported by the fetch script — query OC directly
for that.

## 6. Data Quality

- Company-name spellings vary across re-incorporations and renames
- Officer records are spottier than company records (many jurisdictions
  don't require officer disclosure)
- Beneficial-ownership data is generally NOT here — most jurisdictions
  don't require it. UK Companies House has PSC (people with significant
  control) but that's not universal.
- Cross-jurisdictional links (parent / subsidiary) are based on registry
  filings only; corporate trees are often incomplete

## 7. Acquisition Script

Path: `scripts/fetch_opencorporates.py`

```bash
# Search globally by name
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --out data/oc.csv

# Restrict to a jurisdiction
python3 SKILL_DIR/scripts/fetch_opencorporates.py --query "Example Corp" \
    --jurisdiction us_ny --out data/oc_ny.csv

# Set token via env or flag
OPENCORPORATES_API_TOKEN=xxx python3 SKILL_DIR/scripts/fetch_opencorporates.py \
    --query "Microsoft" --out data/oc.csv
```

Without a token the script falls back to scraping the HTML search page.
The fallback is brittle and only fills in `name`, `jurisdiction_code`,
`opencorporates_url` — set the token for serious work.

## 8. Legal & Licensing

- OpenCorporates aggregates public records — the underlying facts are
  public domain
- OpenCorporates own database is licensed CC-BY-SA-4.0; attribution required
- API ToS prohibits redistributing the full dataset; per-record reference
  is fine

## 9. References

- API docs: https://api.opencorporates.com/documentation/API-Reference
- Jurisdiction codes: https://api.opencorporates.com/v0.4/jurisdictions.json
- Schema: https://opencorporates.com/info/our_data
