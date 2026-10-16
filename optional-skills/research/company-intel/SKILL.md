---
name: company-intel
description: Company intelligence research via free public APIs. Wikipedia company profiles, Wikidata structured data (revenue, employees, stock info), SEC EDGAR filings for US public companies, Hacker News discussion tracking, and entity resolution. No API keys required.
platforms: [linux, macos, windows]
---

# Company Intelligence Research

Research companies using free public APIs. **No API keys required.**

## Helper script

This skill includes `scripts/company_intel.py` — a complete CLI tool for all company research operations.

```bash
# Company profile (Wikipedia + Wikidata)
python3 SKILL_DIR/scripts/company_intel.py profile "Apple Inc."

# Search for companies by name
python3 SKILL_DIR/scripts/company_intel.py search "Tesla"

# SEC EDGAR filings
python3 SKILL_DIR/scripts/company_intel.py sec "Apple Inc."

# Hacker News mentions
python3 SKILL_DIR/scripts/company_intel.py news "Apple Inc." --days 30

# Full research report (all sources combined)
python3 SKILL_DIR/scripts/company_intel.py research "Microsoft Corporation"

# Wikidata structured query (revenue, employees, founded)
python3 SKILL_DIR/scripts/company_intel.py wikidata "Apple Inc."
```

`SKILL_DIR` is the directory containing this SKILL.md file. All output is structured JSON.

## Commands

| Command | What it does | Example |
|---------|-------------|---------|
| `profile` | Wikipedia summary + Wikidata quick facts | `profile "Apple Inc."` |
| `search` | Search for companies by keyword | `search "pharmaceutical" --num 5` |
| `wikidata` | Structured data from Wikidata SPARQL | `wikidata "Apple Inc."` |
| `sec` | SEC EDGAR filings and CIK lookup | `sec "Apple Inc."` |
| `news` | Recent Hacker News mentions | `news "Apple Inc." --days 30` |
| `research` | Full report combining all sources | `research "Microsoft"` |

## APIs Used

### Primary: Wikipedia REST API (free, no auth)

Company summaries, descriptions, and extracts. Fast and reliable.

```
GET https://en.wikipedia.org/api/rest_v1/page/summary/{title}
```

### Secondary: Wikidata SPARQL (free, no auth)

Structured query over 100M+ entities. Returns revenue, employees, founding date, stock ticker, headquarters, and more.

```
GET https://query.wikidata.org/sparql?format=json&query={SPARQL}
```

### Tertiary: SEC EDGAR (free, no auth)

CIK lookup and filing metadata for US public companies. Uses SEC's public XHR API.

```
GET https://efts.sec.gov/LATEST/search-index?q={company}
```

### News: Hacker News Algolia API (free, no auth)

Search recent discussions and stories mentioning the company. Useful for sentiment and trending topics.

```
GET https://hn.algolia.com/api/v1/search?query={company}&tags=story
```

## Examples

### Company Profile

```bash
python3 company_intel.py profile "Nvidia Corporation"
```

Returns Wikipedia summary + Wikidata quick facts (founded, revenue, employees, stock ticker).

### SEC Filings

```bash
python3 company_intel.py sec "Nvidia Corporation" --limit 10
```

Returns recent 10-K, 10-Q, 8-K filings with dates and form types.

### Full Research Report

```bash
python3 company_intel.py research "Nvidia Corporation"
```

Combines Wikipedia profile, Wikidata facts, SEC filings, and recent HN mentions into one comprehensive report.

## Tips

- For public US companies, `sec` command provides the most detailed financial data
- `wikidata` command works for both public and private companies worldwide
- Use `news` to gauge recent sentiment and community discussion
- Combine with `osint-investigation` skill for deeper SEC EDGAR analysis
- Wikipedia titles are case-sensitive — use `search` first to find the exact title