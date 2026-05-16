# FEC — Federal Election Commission (Individual Contributions)

## 1. Summary

The FEC publishes individual contributions to federal candidates, parties, and
committees. Filings are required for any contributor giving more than $200 in a
cycle. Coverage: presidential, House, Senate, party committees, PACs.

## 2. Access Methods

- **API:** `https://api.open.fec.gov/v1/` (free, requires API key — `DEMO_KEY` works for low volume)
- **Bulk:** `https://www.fec.gov/data/browse-data/?tab=bulk-data` (CSV/ZIP per cycle)
- **Auth:** Get a key at https://api.data.gov/signup/ — pass via `?api_key=...`
- **Rate limit:** 1,000 requests/hour with a registered key, 30/hour with `DEMO_KEY`

## 3. Data Schema

Key fields emitted by `fetch_fec.py`:

| Column | Type | Description |
|--------|------|-------------|
| `contributor_name` | str | Donor full name (LAST, FIRST) |
| `contributor_employer` | str | Self-reported employer |
| `contributor_occupation` | str | Self-reported occupation |
| `contributor_city` | str | City of donor |
| `contributor_state` | str | 2-letter state |
| `contributor_zip` | str | ZIP code |
| `recipient_name` | str | Candidate or committee name |
| `recipient_committee_id` | str | FEC committee ID (e.g. C00580100) |
| `amount` | float | USD |
| `date` | str | YYYY-MM-DD contribution date |
| `cycle` | int | Election cycle (e.g. 2024) |
| `transaction_id` | str | FEC transaction ID |

## 4. Coverage

- US federal elections only (state/local handled by state systems like MA OCPF)
- 1980 → present, cycle-by-cycle
- Updated continuously during filing periods
- ~30M+ individual contribution records cumulative

## 5. Cross-Reference Potential

- **USAspending** ↔ `contributor_employer` (donor's employer may be a contractor)
- **SEC EDGAR** ↔ `contributor_employer` (executives at public companies)
- **Senate LD** ↔ `contributor_employer` (lobbyist firms or their clients)
- **OFAC SDN** ↔ `contributor_name` (sanctions screening of donors — rare but possible)

Join key: normalized entity name. Use `entity_resolution.py` since employer
strings are user-typed and vary heavily.

## 6. Data Quality

- Employer/occupation are self-reported and inconsistent ("self-employed",
  "n/a", "retired", varied capitalizations of the same firm)
- Older cycles (pre-2000) have spottier address data
- Some itemized contributions are amendments to earlier filings — same donor
  can appear multiple times for one logical contribution
- Aggregated contributions (under $200) are NOT itemized

## 7. Acquisition Script

Path: `scripts/fetch_fec.py`

```bash
# By contributor name (use uppercase "LAST, FIRST" — names are stored this way)
python3 SKILL_DIR/scripts/fetch_fec.py --contributor "SMITH, JOHN" --state NY --cycle 2024 \
    --out data/fec_donations.csv

# By committee ID
python3 SKILL_DIR/scripts/fetch_fec.py --committee C00580100 --cycle 2024 \
    --out data/fec_donations.csv

# By employer (find all donations from people at a given employer)
python3 SKILL_DIR/scripts/fetch_fec.py --employer "EXAMPLE CORP" --cycle 2024 \
    --out data/fec_donations.csv
```

Set `FEC_API_KEY` env var (or pass `--api-key`). Defaults to `DEMO_KEY`
(40 calls/hour — exhausted very quickly during real investigations).

**Note**: `--candidate` is a deprecated alias for `--contributor` — FEC searches
by contributor (donor) name, not candidate name. To search donations TO a
specific candidate, use `--committee <ID>` after looking up their principal
campaign committee ID.

## 8. Legal & Licensing

- Public record under 52 U.S.C. § 30104 (FECA disclosure)
- No commercial use restrictions on the data
- Cannot be used for solicitation of contributions or commercial purposes
  (11 CFR § 104.15) — this restricts USE of the names for fundraising, not
  research

## 9. References

- API docs: https://api.open.fec.gov/developers/
- Bulk data: https://www.fec.gov/data/browse-data/?tab=bulk-data
- Data dictionary: https://www.fec.gov/campaign-finance-data/contributions-individuals-file-description/
