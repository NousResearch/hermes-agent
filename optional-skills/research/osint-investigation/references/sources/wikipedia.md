# Wikipedia + Wikidata

## 1. Summary

Wikipedia is the canonical narrative-bio source for notable people, places,
and organizations. Wikidata is its structured-data counterpart: ~110M
items, each with claims, dates, identifiers, and cross-references to
external authorities (VIAF, ISNI, ORCID, GRID, etc.).

Together they're a high-precision entity-resolution layer — the bar for
inclusion is real, but anything past that bar is well-cross-referenced.

## 2. Access Methods

- **Wikipedia OpenSearch:** `https://en.wikipedia.org/w/api.php?action=opensearch`
- **Wikipedia REST summary:** `https://en.wikipedia.org/api/rest_v1/page/summary/<title>`
- **Wikidata Action API:** `https://www.wikidata.org/w/api.php?action=wbgetentities`
- **Wikidata SPARQL:** `https://query.wikidata.org/sparql` (more powerful but aggressively rate-limited)
- **Auth:** None, but **a meaningful User-Agent is required**

Set `HERMES_OSINT_UA` to something identifying (e.g. `your-app/1.0 (you@example.com)`).
Wikimedia returns HTTP 429 to generic UAs.

## 3. Data Schema

Key fields emitted by `fetch_wikipedia.py`:

| Column | Type | Description |
|--------|------|-------------|
| `source` | str | `wikipedia` or `wikipedia+wikidata` |
| `label` | str | Wikipedia article title |
| `description` | str | Short Wikidata description |
| `qid` | str | Wikidata QID (e.g. Q2283 for Microsoft) |
| `wikipedia_title`, `wikipedia_url` | str | Article identifier + URL |
| `wikidata_url` | str | Wikidata entity URL |
| `instance_of` | str | What kind of thing it is (P31) |
| `country` | str | Country (P17 for orgs/places, P27 for people) |
| `occupation` | str | P106 |
| `employer` | str | P108 |
| `date_of_birth` | str | P569, YYYY-MM-DD |
| `place_of_birth` | str | P19 |
| `summary` | str | Wikipedia REST extract (~1000 chars) |

The fetch script uses Wikidata's Action API (NOT SPARQL) for structured
facts — far more lenient on rate limits.

## 4. Coverage

- Wikipedia EN: ~7M articles
- Wikidata: ~110M items, ~1.5B statements
- Updated continuously; abuse filters and bots run constantly
- High notability bar — most private individuals are not in Wikipedia

## 5. Cross-Reference Potential

- **All sources** ↔ `label` (entity identity resolution)
- **SEC EDGAR** ↔ `label` (public companies)
- **CourtListener** ↔ `label` (parties to notable litigation)
- **Wikidata external identifiers** (not currently in this fetcher's output)
  link to VIAF, ISNI, ORCID, GRID, GitHub, Twitter, IMDb, ...

Join key: Wikidata QID is canonical. Wikipedia titles are stable for
most articles but can be renamed.

## 6. Data Quality

- Notability filter — only notable entities (criteria vary by topic)
- Recency lag — current events take days to weeks to be reflected
- POV / vandalism — moderated, but edits between sweeps can be bad
- Living-persons biographies have stricter sourcing requirements
- Wikidata claims have qualifiers and references — the fetch script
  doesn't currently export them

## 7. Acquisition Script

Path: `scripts/fetch_wikipedia.py`

```bash
# Look up a notable entity
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Microsoft" --out data/wp.csv

# A specific person
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Bill Gates" --out data/wp_bg.csv

# Skip the Wikidata enrichment for speed
python3 SKILL_DIR/scripts/fetch_wikipedia.py --query "Microsoft" --no-wikidata \
    --limit 5 --out data/wp.csv
```

The OpenSearch is fuzzy — `--limit 5` returns the top 5 Wikipedia article
matches. Each is enriched with the QID + structured facts unless
`--no-wikidata` is passed.

## 8. Legal & Licensing

- Wikipedia text: CC-BY-SA-3.0 / GFDL
- Wikidata claims: CC0 (public domain)
- API ToS: respect rate limits, identify your agent
- Commercial use allowed with attribution

## 9. References

- Wikipedia OpenSearch: https://www.mediawiki.org/wiki/API:Opensearch
- Wikipedia REST: https://en.wikipedia.org/api/rest_v1/
- Wikidata Action API: https://www.wikidata.org/wiki/Wikidata:Data_access
- Wikidata SPARQL: https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service
- User-Agent policy: https://meta.wikimedia.org/wiki/User-Agent_policy
