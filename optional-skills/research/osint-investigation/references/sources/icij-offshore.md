# ICIJ Offshore Leaks Database

## 1. Summary

The International Consortium of Investigative Journalists (ICIJ) publishes a
combined database of offshore entities from the Panama Papers, Paradise Papers,
Pandora Papers, Bahamas Leaks, and Offshore Leaks. ~800,000+ offshore entities
with their officers, intermediaries, and addresses.

## 2. Access Methods

- **Bulk download (primary):** `https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip` (~70 MB ZIP, refreshed periodically)
- **Search UI (human):** `https://offshoreleaks.icij.org/`
- **Auth:** None
- **Note:** The previous Open Refine reconciliation endpoint at
  `/reconcile` now returns 404. ICIJ has removed it. The bulk ZIP is the
  remaining stable access path. The skill's `fetch_icij_offshore.py` caches
  the ZIP locally (default `~/.cache/hermes-osint/icij/`, refreshes after
  30 days) and searches it offline.

## 3. Data Schema

Key fields emitted by `fetch_icij_offshore.py`:

| Column | Type | Description |
|--------|------|-------------|
| `node_id` | int | ICIJ canonical node ID |
| `name` | str | Entity / officer / intermediary name |
| `node_type` | str | entity / officer / intermediary / address |
| `country_codes` | str | Semicolon-separated ISO codes |
| `countries` | str | Country names |
| `jurisdiction` | str | Offshore jurisdiction (BVI, Panama, etc.) |
| `incorporation_date` | str | YYYY-MM-DD |
| `inactivation_date` | str | YYYY-MM-DD (if struck) |
| `source` | str | Panama Papers / Paradise Papers / Pandora Papers / etc. |
| `entity_url` | str | Link to ICIJ page |
| `connections` | str | Semicolon-separated node IDs of related entities |

## 4. Coverage

- Worldwide offshore entity records
- Earliest records: 1970s (Bahamas Leaks). Most data 1990–2018.
- NOT updated in real-time — new leaks added when ICIJ publishes them
- ~810,000 offshore entities + ~750,000 officers + ~150,000 intermediaries

## 5. Cross-Reference Potential

- **SEC EDGAR** ↔ `name` (public companies with offshore arms)
- **USAspending** ↔ `name` (federal contractors with offshore structure)
- **OFAC SDN** ↔ `name` (sanctioned entities using offshore vehicles)

Join key: normalized entity/officer name. `node_id` is canonical for cross-
referencing within ICIJ. Connections graph traversal is in-script (BFS over
`connections`).

## 6. Data Quality

- Offshore entity names sometimes appear in multiple leaks with slight variations
- Officers may be nominees (front persons), not beneficial owners
- Some entries have minimal info (just a name + jurisdiction)
- The connections graph is incomplete — some relationships are documented in
  source materials but not in the structured database
- Inactive/struck-off entities are still included with `inactivation_date`

## 7. Acquisition Script

Path: `scripts/fetch_icij_offshore.py`

```bash
# Search by entity name (case-insensitive substring across the bulk DB)
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --out data/icij.csv

# Search by officer (individual person)
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --officer "SMITH JOHN" \
    --out data/icij.csv

# Search by jurisdiction (filter on cached results)
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --officer "SMITH" \
    --jurisdiction "BRITISH VIRGIN ISLANDS" --out data/icij_bvi.csv

# Force a fresh download (default refresh window is 30 days)
python3 SKILL_DIR/scripts/fetch_icij_offshore.py --entity "EXAMPLE CORP" \
    --force-refresh --out data/icij.csv
```

First call downloads the ~70 MB ZIP under `~/.cache/hermes-osint/icij/`
(or `$HERMES_OSINT_CACHE/icij/`). Subsequent calls reuse the cache for 30 days.

## 8. Legal & Licensing

- Public record as published by ICIJ under explicit publication
- No copyright on the underlying facts (entity names, jurisdictions)
- ICIJ asks for attribution if used in derivative reporting
- **Ethical note**: Presence in this database does NOT imply wrongdoing. Many
  offshore structures are legal. The database is a research tool, not a list of
  criminals.

## 9. References

- Database: https://offshoreleaks.icij.org/
- About the data: https://offshoreleaks.icij.org/pages/about
- Methodology: https://www.icij.org/investigations/panama-papers/
- API hints: Open Refine reconciliation endpoint at `https://offshoreleaks.icij.org/reconcile`
