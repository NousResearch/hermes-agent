# OFAC SDN — Specially Designated Nationals List

## 1. Summary

The Office of Foreign Assets Control (OFAC) publishes the Specially Designated
Nationals and Blocked Persons List (SDN). US persons are generally prohibited
from dealing with individuals and entities on this list. Also published:
non-SDN consolidated lists (BIS Denied Persons, FSE, etc.).

## 2. Access Methods

- **Full XML:** `https://www.treasury.gov/ofac/downloads/sdn.xml`
- **Delimited:** `https://www.treasury.gov/ofac/downloads/sdn.csv`
- **Consolidated:** `https://www.treasury.gov/ofac/downloads/consolidated/consolidated.xml`
- **Auth:** None
- **Rate limit:** None (static file downloads). Updated continuously.

## 3. Data Schema

Key fields emitted by `fetch_ofac_sdn.py`:

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | int | OFAC unique ID |
| `name` | str | Primary name |
| `entity_type` | str | individual / entity / vessel / aircraft |
| `program_list` | str | Semicolon-separated sanctions programs (e.g. SDGT;IRAN) |
| `title` | str | For individuals: title/role |
| `nationalities` | str | Semicolon-separated country codes |
| `aka_list` | str | Semicolon-separated "also known as" names |
| `addresses` | str | Semicolon-separated known addresses |
| `dob` | str | Date of birth (individuals) |
| `pob` | str | Place of birth (individuals) |
| `remarks` | str | OFAC's free-text remarks |
| `last_updated` | str | YYYY-MM-DD (publication date) |

## 4. Coverage

- Worldwide — all entities sanctioned by US Treasury
- ~10,000 entries on SDN, ~15,000 on consolidated lists
- Updated continuously (sometimes daily during active enforcement)
- Includes AKAs (very common, can be 10+ per entity)

## 5. Cross-Reference Potential

- **SEC EDGAR** ↔ `name` (public companies sanctioned)
- **USAspending** ↔ `name` (sanctioned entity as federal contractor — should
  be impossible but verify)
- **ICIJ Offshore** ↔ `name` (offshore entities also sanctioned)

Join key: normalized name. **CRITICAL**: must match against `aka_list` too.
Many sanctioned entities are caught only via aliases.

## 6. Data Quality

- Names are transliterated from many scripts — multiple romanizations possible
- AKAs often differ wildly from primary name
- Some entries have minimal info (no DOB, no address) for individuals
- Free-text `remarks` contain critical context — read them
- "Specially Designated Global Terrorists" (SDGT) and "Cyber-related" (CYBER2)
  programs add and remove entries frequently

## 7. Acquisition Script

Path: `scripts/fetch_ofac_sdn.py`

```bash
# Full snapshot
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --out data/ofac_sdn.csv

# Filter to specific program
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --program SDGT --out data/sdn_sdgt.csv

# Entities only (skip individuals, vessels, aircraft)
python3 SKILL_DIR/scripts/fetch_ofac_sdn.py --entity-type entity --out data/sdn_entities.csv
```

## 8. Legal & Licensing

- Public record under Executive Order authority and statutory sanctions programs
- US persons MUST screen against this list — it is enforced
- No restrictions on the data itself; restrictions are on transactions with
  the listed entities
- ZERO penalty for "over-matching" — false positives must be cleared but are not
  prohibited

## 9. References

- OFAC home: https://ofac.treasury.gov/
- SDN list: https://ofac.treasury.gov/specially-designated-nationals-and-blocked-persons-list-sdn-human-readable-lists
- Data formats: https://ofac.treasury.gov/sdn-list/sanctions-list-search-tool
- Compliance guidance: https://ofac.treasury.gov/recent-actions
