---
name: semantic-scholar
description: "Citations, references, and author metrics via S2 API."
version: 1.0.0
author: Sam27
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Papers, Academic, Citations, Science, API]
    related_skills: [arxiv]
---

# Semantic Scholar Skill

Query the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/)
for citation data, references, author profiles, and paper recommendations.
No API key required for basic use. Complements `arxiv` (full text) with
structured metadata arXiv cannot provide.

## When to Use

- User needs citation counts or influential citation metrics for a paper
- User asks "who cited this paper?" or "what does this paper cite?"
- User wants paper recommendations based on a seed paper
- User needs author h-index, publication list, or affiliation
- User wants to search papers and get structured JSON (not XML)

## Prerequisites

- `curl` and `python3` (stdlib only, no pip packages)
- Optional: set `SEMANTIC_SCHOLAR_API_KEY` in env for higher rate limits

## How to Run

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=transformer+attention&fields=title,year,citationCount&limit=5" | python3 -m json.tool
```

## Quick Reference

| Goal | Endpoint |
|------|----------|
| Search papers | `GET /paper/search?query=QUERY&fields=...` |
| Paper details | `GET /paper/{id}?fields=...` |
| Citations (who cited it) | `GET /paper/{id}/citations?fields=...` |
| References (what it cites) | `GET /paper/{id}/references?fields=...` |
| Recommendations | `GET /recommendations/v1/papers/forpaper/{s2id}` |
| Author search | `GET /author/search?query=NAME&fields=...` |
| Author papers | `GET /author/{id}/papers?fields=...` |
| Batch lookup | `POST /paper/batch?fields=...` |

**Base URL:** `https://api.semanticscholar.org/graph/v1`
**Paper ID formats:** S2 hash, `ARXIV:2106.15928`, `DOI:10.18653/v1/N19-1423`, `CorpusID:12345`

## Procedure

### Search papers

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=QUERY\
&fields=title,year,authors,citationCount,tldr\
&limit=10" | python3 -m json.tool

# With year filter and citation sort
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=QUERY&fields=title,year,citationCount\
&limit=10&year=2022-2025&sort=citationCount" | python3 -m json.tool
```
{% endraw %}

**Useful fields:** `title`, `year`, `abstract`, `authors`, `citationCount`,
`influentialCitationCount`, `tldr`, `openAccessPdf`, `externalIds`, `publicationVenue`

### Paper details

```bash
# By arXiv ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762\
?fields=title,year,abstract,citationCount,influentialCitationCount,openAccessPdf" | python3 -m json.tool

# By DOI
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.18653/v1/N19-1423\
?fields=title,year,citationCount,tldr" | python3 -m json.tool
```

### Citations (who cited this paper?)

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/citations\
?fields=title,year,citationCount&limit=10" | \
python3 -c "
import json, sys
for item in json.load(sys.stdin)['data']:
    p = item['citingPaper']
    print(f\"{p.get('year','?')} | {p.get('citationCount',0):>6} cites | {p['title']}\")
"
```
{% endraw %}

### References (what does this paper cite?)

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/references\
?fields=title,year,citationCount&limit=10" | \
python3 -c "
import json, sys
for item in json.load(sys.stdin)['data']:
    p = item['citedPaper']
    print(f\"{p.get('year','?')} | {p.get('citationCount',0):>6} cites | {p['title']}\")
"
```
{% endraw %}

### Recommendations

```bash
# Single paper (use S2 paper ID from a prior lookup)
curl -s "https://api.semanticscholar.org/recommendations/v1/papers/\
forpaper/204e3073870fae3d05bcbc2f6a8e263d9b72e776\
?fields=title,year,citationCount&limit=5" | python3 -m json.tool

# Multi-paper (accepts ARXIV:/DOI: prefixes)
curl -s -X POST "https://api.semanticscholar.org/recommendations/v1/papers\
?fields=title,year,citationCount&limit=5" \
  -H "Content-Type: application/json" \
  -d '{"positivePaperIds": ["ARXIV:1706.03762", "ARXIV:2005.14165"], "negativePaperIds": []}' | python3 -m json.tool
```

### Author search and papers

```bash
# Find an author
curl -s "https://api.semanticscholar.org/graph/v1/author/search\
?query=Yoshua+Bengio&fields=name,hIndex,citationCount,paperCount&limit=5" | python3 -m json.tool

# Get papers by author ID
curl -s "https://api.semanticscholar.org/graph/v1/author/1741101/papers\
?fields=title,year,citationCount&limit=10&sort=citationCount" | python3 -m json.tool
```

### Batch lookup

```bash
curl -s -X POST "https://api.semanticscholar.org/graph/v1/paper/batch\
?fields=title,year,citationCount" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["ARXIV:1706.03762", "ARXIV:2005.14165"]}' | python3 -m json.tool
```

## Pitfalls

- **Rate limit:** 1 request/second without a key. Always `sleep 1` between
  calls in loops. Bursts trigger an IP-level ban lasting several minutes.
- **DOI coverage:** Not all DOIs resolve. If a DOI returns "not found", search
  by title instead.
- **Recommendations:** The single-paper endpoint (`/forpaper/{id}`) requires
  the S2 paper ID (hex hash), not `ARXIV:` or `DOI:` prefixes. The multi-paper
  POST endpoint accepts both.
- **`influentialCitationCount`** is often more useful than `citationCount` — it
  counts only citations with significant methodological impact.
- **`tldr`** is an AI-generated summary; not available for all papers.

## Verification

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762?fields=title,citationCount" | python3 -m json.tool
```

A successful response returns JSON with `paperId`, `title`, and `citationCount` fields.
If you see a 429 error, wait 60 seconds and retry — you've hit the rate limit.
