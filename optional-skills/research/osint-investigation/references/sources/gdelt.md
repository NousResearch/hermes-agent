# GDELT — Global News Monitoring

## 1. Summary

GDELT (Global Database of Events, Language, and Tone) monitors world news
in 100+ languages with full-text indexing. Updated every 15 minutes.
~2015 → present, ~1B+ articles indexed. Free anonymous access.

GDELT is wider than Google News (more international, more long-tail
sources) and indexed by tone/sentiment, themes (CAMEO codes), people, and
organizations.

## 2. Access Methods

- **DOC 2.0 API:** `https://api.gdeltproject.org/api/v2/doc/doc`
- **Events / GKG 2.0:** `https://api.gdeltproject.org/api/v2/events/events`
- **Auth:** None
- **Rate limit:** **1 request per 5 seconds** for the DOC API — strict

The fetch script automatically retries after a 6-second sleep when a
429 is received.

## 3. Data Schema

Key fields emitted by `fetch_gdelt.py`:

| Column | Type | Description |
|--------|------|-------------|
| `title` | str | Article title |
| `url` | str | Article URL |
| `seen_date` | str | When GDELT first saw the article (UTC) |
| `domain` | str | Publisher domain |
| `language` | str | Source language |
| `source_country` | str | 2-letter country code |
| `tone` | str | GDELT-computed tone score (negative = negative coverage) |
| `social_image` | str | Open Graph image URL when available |

## 4. Coverage

- Worldwide news in 100+ languages
- ~2015 → present (Events back to 1979 via a separate stream)
- Update frequency: 15 minutes
- Bias: heavily Anglophone in volume but very wide source list overall

## 5. Cross-Reference Potential

- **All sources** ↔ `title` / `url` (news context for any subject)
- **Wikipedia** ↔ event timeline for notable entities
- **Wayback Machine** ↔ recover articles whose URLs have died
- **OFAC SDN** ↔ news context for sanctions designations
- **SEC EDGAR** ↔ news context for 8-K material events

Join key: entity name appearing in article title or full-text. GDELT also
extracts named entities into a separate stream (GKG) not exposed by this
fetcher — query GDELT directly for entity-level filtering.

## 6. Data Quality

- Title extraction is automated and can be wrong (sometimes captures the
  site name + delimiter + article title; sometimes a generic page title)
- Sentiment / tone is computed by GDELT, not source-supplied
- Some domains are oversampled (newswires, aggregators)
- Source country is inferred from domain registration / TLD — can be
  wrong for international news sites with country-neutral domains
- Article URLs can rot — pair with Wayback Machine to preserve content

## 7. Acquisition Script

Path: `scripts/fetch_gdelt.py`

```bash
# Recent news mentioning an entity
python3 SKILL_DIR/scripts/fetch_gdelt.py --query "Nous Research" \
    --timespan 6m --out data/gdelt.csv

# Phrase-exact (use double quotes inside single quotes for the shell)
python3 SKILL_DIR/scripts/fetch_gdelt.py --query '"Dillon Rolnick"' \
    --timespan 1y --out data/gdelt.csv

# Filter to a country / language
python3 SKILL_DIR/scripts/fetch_gdelt.py --query "Microsoft" \
    --source-country US --source-lang English --out data/gdelt.csv

# Date range
python3 SKILL_DIR/scripts/fetch_gdelt.py --query "Microsoft" \
    --start 2024-01-01 --end 2024-12-31 --out data/gdelt.csv
```

GDELT supports its own query operators: phrase quoting, AND/OR/NOT,
`sourcecountry:US`, `theme:ECON_BANKRUPTCY`, `tone<-5`, etc.
See https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ for syntax.

## 8. Legal & Licensing

- GDELT data is provided free for academic and journalistic use
- Article URLs link out to original publishers — copyright remains with
  the publisher
- GDELT is NOT a content archive; it's a metadata index

## 9. References

- DOC 2.0 API: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- Themes & query syntax: https://blog.gdeltproject.org/gkg-2-0-our-global-knowledge-graph-2-0-amazing-data-at-your-fingertips/
- Project home: https://www.gdeltproject.org/
