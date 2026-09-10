---
name: openalex-scholar
description: Academic research via OpenAlex API — search 250M+ scholarly works, authors, institutions, concepts, sources. Filters by year, OA status, citations, field of study. Citation analysis, author disambiguation, concept hierarchy traversal. Free, no API key needed.
platforms: [linux, macos, windows]
---

# OpenAlex Scholar

Academic research via the [OpenAlex API](https://docs.openalex.org/) — a free, open index of 250M+ scholarly works. **No API key required.**

## Helper script

This skill includes `scripts/openalex_cli.py` — a complete CLI tool for all research operations.

```bash
# Search works
python3 SKILL_DIR/scripts/openalex_cli.py search "quantum computing"

# Get author profile
python3 SKILL_DIR/scripts/openalex_cli.py author "Geoffrey Hinton"

# Get specific work by OpenAlex ID
python3 SKILL_DIR/scripts/openalex_cli.py work W4255852847

# Get work by DOI
python3 SKILL_DIR/scripts/openalex_cli.py doi 10.1038/nature12373

# Concept exploration
python3 SKILL_DIR/scripts/openalex_cli.py concept "machine learning"

# Institution research
python3 SKILL_DIR/scripts/openalex_cli.py institution "Stanford University"

# Citation analysis
python3 SKILL_DIR/scripts/openalex_cli.py citations W4255852847

# Advanced search with filters
python3 SKILL_DIR/scripts/openalex_cli.py search "transformer neural network" --since 2020 --min-citations 100 --open-access

# Filter by concept
python3 SKILL_DIR/scripts/openalex_cli.py search "reinforcement learning" --concept-id C126700600

# Top-cited works in a field
python3 SKILL_DIR/scripts/openalex_cli.py search "protein folding" --sort cited_by_count --per-page 10
```

## Commands

| Command | What it does | Example |
|---------|-------------|---------|
| `search` | Search scholarly works | `search "llm reasoning" --since 2023` |
| `work` | Get work details by OpenAlex ID | `work W4255852847` |
| `doi` | Look up work by DOI | `doi 10.1038/nature12373` |
| `author` | Author profile + works | `author "Yoshua Bengio"` |
| `concept` | Concept hierarchy + related | `concept "deep learning"` |
| `institution` | Institution profile + stats | `institution "MIT"` |
| `source` | Journal/conference info | `source "Nature"` |
| `citations` | Citation analysis for a work | `citations W4255852847` |

## API Overview

Base URL: `https://api.openalex.org/`

### Rate Limits

| Pool | Rate | How to get |
|------|------|------------|
| Public | 10 req/s | Default (no header) |
| Polite | 100 req/s | Add `mailto=you@example.com` to requests |

**Always use Polite pool** when possible — add `?mailto=your@example.com` to API calls. It's faster and helps OpenAlex track usage for funding.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/works` | Scholarly works (papers, books, datasets, theses) |
| `/authors` | Disambiguated author profiles |
| `/concepts` | Hierarchical topic taxonomy |
| `/institutions` | Universities and research organizations |
| `/sources` | Journals, repositories, conferences |

### Search Parameters

| Param | Type | Example |
|-------|------|---------|
| `search` | string | `search=neural+network` |
| `filter` | comma-separated | `filter=from_publication_date:2023-01-01,is_oa:true` |
| `sort` | field + order | `sort=cited_by_count:desc` |
| `per_page` | int (max 200) | `per_page=50` |
| `page` | int (1-based) | `page=2` |
| `group_by` | field | `group_by=publication_year` |

### Filter Syntax

| Filter | Description |
|--------|-------------|
| `from_publication_date:2023-01-01` | Published after date |
| `is_oa:true` | Open access only |
| `type:article` | Type: article, review, book, chapter, dataset, etc. |
| `language:en` | Language code |
| `authorships.author.id:A123` | Specific author |
| `primary_location.source.id:S123` | Specific journal |
| `concepts.id:C123` | Specific concept |
| `cited_by_count:>100` | Citation count threshold |
| `institutions.id:I123` | Affiliated institution |
| `open_access.oa_url:null` | Has OA URL |

## Response Format

### Work

```json
{
  "id": "https://openalex.org/W4255852847",
  "doi": "https://doi.org/10.1016/b978-0-444-53643-3.00223-5",
  "title": "DIATOM RECORDS | Antarctic Waters",
  "publication_year": 2013,
  "cited_by_count": 1,
  "is_retracted": false,
  "is_paratext": false,
  "type": "book-chapter",
  "open_access": { "is_oa": false, "oa_status": "closed" },
  "authorships": [{"author": {"display_name": "Catherine E Stickley"}}],
  "concepts": [{"display_name": "Oceanography", "score": 0.8}]
}
```

### Author

```json
{
  "id": "https://openalex.org/A123",
  "display_name": "Geoffrey E. Hinton",
  "cited_by_count": 447546,
  "works_count": 637,
  "h_index": 132,
  "2yr_mean_citedness": 234.5,
  "last_known_institutions": [{"display_name": "University of Toronto"}],
  "concepts": [{"display_name": "Artificial intelligence", "score": 0.9}]
}
```

## Filters Reference

### Work types
`article`, `book-chapter`, `dataset`, `dissertation`, `edited-book`, `encyclopedia-entry`, `grant`, `monograph`, `paratext`, `peer-review`, `posted-content`, `proceedings`, `proceedings-reference`, `reference-book`, `reference-entry`, `report`, `review`, `standard`, `supplementary-materials`

### OA statuses
`gold`, `green`, `hybrid`, `bronze`, `closed`

### Sortable fields
`publication_year`, `cited_by_count`, `title`, `publication_date`, `relevance_score`, `created_date`

## Tips

- **Polite pool**: Always include `mailto=you@example.com` in requests for 10x faster rate limits
- **Selective fields**: Use `?select=id,title,cited_by_count` to reduce response size
- **Pagination**: Use `cursor` parameter for deep pagination (beyond 10K results)
- **Group by**: `group_by=publication_year` gives yearly counts — great for trend analysis
- **Concept IDs**: Get full concept hierarchy by fetching a concept's `ancestors` and `descendants`
- **Work → Author → Works**: Traverse from a work to its authors, then to each author's other works
- **OpenAlex IDs**: Always use the numeric part (e.g., `W4255852847` not the full URL)

## Platform compatibility

Pure Python stdlib (`urllib.request`, `json`, `argparse`). Works identically on Linux, macOS, and Windows.

## When to use this vs arxiv

| Task | Better tool | Why |
|------|-------------|-----|
| "Papers about X across all disciplines" | **This skill** | OpenAlex indexes 250M+ works from ALL publishers |
| "Author's publication record + metrics" | **This skill** | Author disambiguation, h-index, citation counts |
| "CS/AI/ML preprints only" | `arxiv` skill | arxiv is faster for preprint-specific searches |
| "Citation network analysis" | **This skill** | referenced_works + cited_by_count + group_by year |
| "Find top institutions researching X" | **This skill** | Filter by concept, group by institution |
| "Is this paper open access?" | **This skill** | OpenAlex has OA status per work |
| "Trend analysis for a research field" | **This skill** | group_by=publication_year for yearly trends |

## Pitfalls

- **Rate limits**: 10 req/s public, 100 req/s polite (with mailto). Exceed → 429 errors
- **Data freshness**: OpenAlex updates weekly. Recent papers may be missing
- **Author disambiguation**: Not perfect. Verify by institution + co-authors
- **Citation counts**: Lag behind Google Scholar. Use as relative metric, not absolute
- **Large responses**: Use `select` parameter to limit fields. Use `cursor` pagination beyond 10K
- **API stability**: OpenAlex is actively developed. Endpoints may change
- **Mailto privacy**: Your email is visible in API logs. Use a disposable email if concerned

## Workflows

### Research field mapping
1. Search for a concept: `python3 SKILL_DIR/scripts/openalex_cli.py concept "machine learning"`
2. Get top-cited works in that concept: `python3 SKILL_DIR/scripts/openalex_cli.py search "machine learning" --sort cited_by_count --per-page 20`
3. Identify leading authors from results
4. Check each author's other works
5. Group by year to see field growth

### Author impact analysis
1. Look up author: `python3 SKILL_DIR/scripts/openalex_cli.py author "Name"`
2. Note h-index, citation count, 2yr mean citedness
3. Check affiliated institutions
4. Get their top-cited works
5. Cross-reference with co-authors for collaboration network

### Literature review
1. Search with keywords and date range
2. Filter by work type (review, article) and OA status
3. Sort by citation count or relevance
4. Check each work's referenced_works for backward citations
5. Use concepts to find related works outside initial search terms