---
name: pubmed
description: Search PubMed via NCBI E-utilities for biomedical and clinical literature. Retrieve PMIDs, summaries, abstracts, MeSH-guided matches, and related papers without an API key.
version: 1.0.0
author: sgaofen
license: MIT
metadata:
  hermes:
    tags: [Research, PubMed, NCBI, Biomedical, Clinical, Papers, Literature]
    related_skills: [arxiv, ocr-and-documents, research-paper-writing]
---

# PubMed Research

Search PubMed through NCBI E-utilities when the task involves biomedical,
clinical, public health, pharmacology, genetics, or life-science literature.
This skill is optimized for the common workflow: find papers, inspect metadata,
read abstracts, and follow related-paper links.

No API key is required. If the user has an `NCBI_API_KEY`, the helper script
will use it automatically for higher rate limits. `NCBI_EMAIL` is also
optional and will be sent when present.

## When to Use

Load this skill when the user asks for:

- biomedical or clinical literature searches
- disease, drug, gene, pathway, or trial-related papers
- PMID lookup or abstract retrieval
- MeSH-guided searches instead of general web search
- related-paper expansion from a seed PubMed article

Prefer this skill over the `arxiv` skill when the source of truth should be
PubMed indexing rather than preprints.

## Quick Reference

| Action | Command |
|--------|---------|
| Search PubMed | `python3 scripts/search_pubmed.py "crispr base editing" --max 5` |
| Search by author | `python3 scripts/search_pubmed.py --author "Jennifer Doudna" --max 5` |
| Search with MeSH | `python3 scripts/search_pubmed.py "immunotherapy" --mesh "Neoplasms" --since 2022` |
| Summarize known PMIDs | `python3 scripts/search_pubmed.py --pmid 41256272,41248061` |
| Fetch an abstract | `python3 scripts/search_pubmed.py --abstract 41256272` |
| Find related papers | `python3 scripts/search_pubmed.py --related 41256272 --max 10` |
| Open PubMed page | `web_extract(urls=["https://pubmed.ncbi.nlm.nih.gov/41256272/"])` |

## Helper Script

Define a shorthand first:

```bash
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PUBMED_SKILL_DIR="$HERMES_HOME/skills/research/pubmed"
PYTHON_BIN="${HERMES_PYTHON:-python3}"
PUBMED="$PYTHON_BIN $PUBMED_SKILL_DIR/scripts/search_pubmed.py"
```

Then use:

```bash
$PUBMED "crispr base editing" --max 5 --sort date
$PUBMED --author "Jennifer Doudna" --max 5
$PUBMED "glioblastoma immunotherapy" --mesh "Neoplasms" --since 2021 --max 8
$PUBMED --pmid 41256272,41248061
$PUBMED --abstract 41256272
$PUBMED --related 41256272 --max 10
```

## Query Patterns

The helper script covers the most common structured filters:

- `QUERY` positional text: general PubMed search with automatic term mapping
- `--author "Name"`: restrict to author field
- `--journal "Nature"`: restrict to journal title
- `--title "exact words"`: title-only narrowing
- `--mesh "Term"`: add a MeSH term constraint
- `--since YEAR` and `--until YEAR`: publication-date filter via `pdat`
- `--sort relevance|date`: relevance or newest-first

Examples:

```bash
# General discovery
$PUBMED "base editing liver delivery" --max 10

# Title + author narrowing
$PUBMED --title "single cell atlas" --author "Satija" --max 5

# MeSH-backed search
$PUBMED "checkpoint inhibitors" --mesh "Neoplasms" --since 2023 --max 10

# Journal-specific
$PUBMED "foundation models pathology" --journal "Nature Medicine" --max 5
```

## Common NCBI Endpoints

Use the raw APIs when you need exact control:

```bash
# Search PMIDs
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=crispr+editing&retmax=5&retmode=json"

# Get article metadata for PMIDs
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=41256272,41248061&retmode=json"

# Fetch full abstract text
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=41256272&retmode=xml"

# Find related papers
curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pubmed&id=41256272&linkname=pubmed_pubmed&retmode=json"
```

## Typical Workflow

1. Search broadly with a disease, gene, drug, method, or clinical question.
2. Inspect the returned PMIDs, journal, year, and DOI.
3. Fetch the abstract for the most relevant papers.
4. Expand via `--related` if one seed paper is especially good.
5. If the user wants the article page rendered, pass the PubMed URL into `web_extract`.
6. If the paper has a free full-text PDF elsewhere, combine with `ocr-and-documents`.

## Output Shape

Search and PMID summary modes print:

- PMID
- title
- journal / source
- publication date
- authors
- DOI / PMCID when present
- a short snippet when available

Abstract mode prints:

- PMID and title
- journal/date
- the structured abstract text, keeping section labels when PubMed exposes them

Related mode prints the same summary layout as search mode.

## Pitfalls

- `esummary` does **not** include the full abstract. Use `--abstract` for that.
- PubMed and PMC are different. A paper can have a PMID without a free PMC full text.
- Some recent records are indexed incompletely right after publication; DOI or MeSH may be missing.
- PubMed query syntax is powerful but unforgiving. If a restrictive fielded query returns nothing, retry with a broader general-text search first.
- NCBI rate limits anonymous traffic. Keep bursts small; use `NCBI_API_KEY` if the user already has one.

## Verification

A good result should satisfy all of these:

- the query returns PMIDs that match the intended disease / method / drug
- the top records show sane journal names and publication dates
- `--abstract PMID` returns non-empty text for the selected article
- if the user asked for related work, `--related PMID` returns papers that are topically adjacent rather than random keyword matches
