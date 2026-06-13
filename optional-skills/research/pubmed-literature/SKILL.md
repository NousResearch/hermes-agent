---
name: pubmed-literature
description: Search and analyze biomedical literature via PubMed/NCBI E-utilities. Free API, no key required. Search by keyword, author, journal, date range. Fetch abstracts, citation counts, author info, MeSH terms, and full publication metadata.
platforms: [linux, macos, windows]
---

# PubMed Literature Research

Search biomedical literature via PubMed/NCBI E-utilities. **No API key required.**

## Helper script

This skill includes `scripts/pubmed_search.py` — a complete CLI tool.

```bash
# Search by keyword
python3 SKILL_DIR/scripts/pubmed_search.py search "CRISPR gene therapy"

# Get publication details
python3 SKILL_DIR/scripts/pubmed_search.py detail 38271494

# Search by author
python3 SKILL_DIR/scripts/pubmed_search.py author "Yoshua Bengio"

# Search by journal
python3 SKILL_DIR/scripts/pubmed_search.py journal "Nature" --since 2024

# Advanced search with filters
python3 SKILL_DIR/scripts/pubmed_search.py search "machine learning" --since 2023 --num 20
```

Commands: search, detail, author, journal. Output is structured JSON.