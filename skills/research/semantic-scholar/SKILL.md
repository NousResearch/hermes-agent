---
name: semantic-scholar
description: "Search papers, citations, references, authors, and recommendations via Semantic Scholar API."
version: 1.0.0
author: Sam27
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, Papers, Academic, Citations, Science, API]
    related_skills: [arxiv]
---

# Semantic Scholar

Search and explore academic papers, citations, references, author profiles,
and paper recommendations via the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/api-docs/).
No API key required for basic use. Complements the `arxiv` skill: use arXiv to
find and read papers; use this skill when you need **citation data, references,
author metrics, or recommendations** — things arXiv cannot provide.

---

## Quick reference

| Goal | Command |
|------|---------|
| Search papers by keyword | See [Search papers](#1-search-papers) |
| Get paper details + abstract | See [Paper details](#2-paper-details) |
| List papers that cite a paper | See [Citations](#3-citations) |
| List papers a paper cites | See [References](#4-references) |
| Find similar papers | See [Recommendations](#5-recommendations) |
| Look up an author | See [Author search](#6-author-search) |
| Get all papers by an author | See [Author papers](#7-author-papers) |

**Base URL:** `https://api.semanticscholar.org/graph/v1`
**Rate limit (no key):** 1 request/second — always add `sleep 1` between calls in loops.
**Optional API key:** set `SEMANTIC_SCHOLAR_API_KEY` in env for higher limits; pass as `-H "x-api-key: $SEMANTIC_SCHOLAR_API_KEY"`.

---

## 1. Search papers

```bash
# Basic keyword search (returns 10 results)
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=QUERY\
&fields=title,year,authors,citationCount,tldr\
&limit=10" | python3 -m json.tool

# With year filter and more results
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=QUERY\
&fields=title,year,authors,citationCount,abstract,openAccessPdf\
&limit=20\
&year=2022-2025" | python3 -m json.tool

# Sort by citation count (most cited first)
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=QUERY\
&fields=title,year,citationCount,influentialCitationCount\
&limit=10\
&sort=citationCount" | python3 -m json.tool
```

**Useful `fields` values:**
- `title`, `year`, `abstract` — basics
- `authors` — author names + IDs
- `citationCount` — total citations
- `influentialCitationCount` — citations where this paper had significant methodological impact (more meaningful than raw count)
- `tldr` — AI-generated one-sentence summary (not available for all papers)
- `openAccessPdf` — link to free PDF if available
- `externalIds` — DOI, arXiv ID, etc.
- `publicationVenue` — journal or conference name

**Parse results cleanly with Python:**

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search\
?query=attention+is+all+you+need\
&fields=title,year,citationCount,authors" | \
python3 -c "
import json, sys
data = json.load(sys.stdin)
for p in data['data']:
    authors = ', '.join(a['name'] for a in p.get('authors', [])[:3])
    print(f\"{p['year']} | {p['citationCount']:>6} citations | {authors} | {p['title']}\")
"
```
{% endraw %}

---

## 2. Paper details

Accepted paper ID formats: S2 paper ID, `DOI:10.xxx`, `ARXIV:2106.15928`,
`CorpusID:12345`, `URL:https://...`

```bash
# By arXiv ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762\
?fields=title,year,abstract,authors,citationCount,influentialCitationCount,\
references,openAccessPdf,publicationVenue" | python3 -m json.tool

# By DOI
curl -s "https://api.semanticscholar.org/graph/v1/paper/DOI:10.18653/v1/N19-1423\
?fields=title,year,abstract,citationCount,tldr" | python3 -m json.tool

# By S2 paper ID
curl -s "https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b\
?fields=title,year,abstract,citationCount,authors" | python3 -m json.tool
```

---

## 3. Citations

Papers that **cite** a given paper (who built on this work?):

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/citations\
?fields=title,year,authors,citationCount\
&limit=20" | \
python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data['data']:
    p = item['citingPaper']
    authors = ', '.join(a['name'] for a in p.get('authors', [])[:2])
    print(f\"{p.get('year','?')} | {p.get('citationCount',0):>6} cites | {authors} | {p['title']}\")
"
```
{% endraw %}

---

## 4. References

Papers **referenced by** a given paper (what did this paper build on?):

{% raw %}
```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762/references\
?fields=title,year,authors,citationCount\
&limit=20" | \
python3 -c "
import json, sys
data = json.load(sys.stdin)
for item in data['data']:
    p = item['citedPaper']
    print(f\"{p.get('year','?')} | {p.get('citationCount',0):>6} cites | {p['title']}\")
"
```
{% endraw %}

---

## 5. Recommendations

Find papers similar to a known paper (or a set of papers):

```bash
# Recommendations for a single paper (use the S2 paper ID from a prior lookup)
curl -s "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/204e3073870fae3d05bcbc2f6a8e263d9b72e776\
?fields=title,year,authors,citationCount\
&limit=10" | python3 -m json.tool

# Recommendations from a set of papers (positive = like these, negative = unlike these)
curl -s -X POST "https://api.semanticscholar.org/recommendations/v1/papers\
?fields=title,year,citationCount,authors\
&limit=10" \
  -H "Content-Type: application/json" \
  -d '{
    "positivePaperIds": ["ARXIV:1706.03762", "ARXIV:2005.14165"],
    "negativePaperIds": []
  }' | python3 -m json.tool
```

---

## 6. Author search

```bash
# Search for an author by name
curl -s "https://api.semanticscholar.org/graph/v1/author/search\
?query=Yoshua+Bengio\
&fields=name,hIndex,citationCount,paperCount,affiliations\
&limit=5" | python3 -m json.tool
```

Key author fields: `name`, `hIndex`, `citationCount`, `paperCount`, `affiliations`, `homepage`, `externalIds`

---

## 7. Author papers

{% raw %}
```bash
# Get papers by author ID (from step 6 results)
curl -s "https://api.semanticscholar.org/graph/v1/author/1741101/papers\
?fields=title,year,citationCount,venue\
&limit=20\
&sort=citationCount" | \
python3 -c "
import json, sys
data = json.load(sys.stdin)
for p in data['data']:
    print(f\"{p.get('year','?')} | {p.get('citationCount',0):>6} cites | {p['title']}\")
"
```
{% endraw %}

---

## Batch lookups

Fetch details for multiple papers in one request (more efficient than looping):

```bash
curl -s -X POST "https://api.semanticscholar.org/graph/v1/paper/batch\
?fields=title,year,citationCount,authors" \
  -H "Content-Type: application/json" \
  -d '{"ids": ["ARXIV:1706.03762", "ARXIV:2005.14165", "DOI:10.18653/v1/N19-1423"]}' | \
python3 -m json.tool
```

---

## Rate limiting

The public API allows **1 request/second** without a key. Always sleep between
calls in a loop:

```bash
for ID in ARXIV:1706.03762 ARXIV:2005.14165; do
  curl -s "https://api.semanticscholar.org/graph/v1/paper/${ID}?fields=title,citationCount"
  sleep 1
done
```

For bulk work, request a free API key at https://www.semanticscholar.org/product/api
and pass it as a header: `-H "x-api-key: $SEMANTIC_SCHOLAR_API_KEY"`.

---

## When to use this vs arXiv

| Need | Use |
|------|-----|
| Find and read a paper's full text | `arxiv` skill |
| Search by topic, browse abstracts | Either (arXiv for latest preprints) |
| Citation counts, influential citations | **This skill** |
| Who cited this paper? | **This skill** |
| What papers does this cite? | **This skill** |
| Find papers similar to X | **This skill** (recommendations) |
| Author h-index, publication list | **This skill** |

---

## Notes

- `influentialCitationCount` is often more useful than `citationCount` — it counts
  only citations where the paper had significant methodological impact, not just
  passing mentions.
- The `tldr` field contains an AI-generated single-sentence summary; not available
  for all papers.
- Paper IDs are stable and cross-format: `ARXIV:1706.03762`, `DOI:10.xxx`, and
  the native S2 ID all resolve to the same record.
- Semantic Scholar covers 200M+ papers across all fields, not just CS/ML.
- The single-paper recommendations endpoint (`/forpaper/{id}`) works best with
  S2 paper IDs (the hex hash from a prior lookup). The multi-paper POST endpoint
  accepts `ARXIV:` and `DOI:` prefixes directly.
- Not all DOIs resolve in Semantic Scholar. If a DOI lookup returns "not found",
  try searching by title instead.
