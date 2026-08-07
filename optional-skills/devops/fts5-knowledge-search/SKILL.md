---
name: fts5-knowledge-search
description: "FTS5 knowledge search: trigram CJK index + LIKE fallback."
version: 1.0.1
author: ligl0325
license: MIT
platforms:
  - linux
  - macos
  - windows
metadata:
  hermes:
    tags:
      - search
      - fts5
      - sqlite
      - knowledge-base
      - retrieval
      - cjk
    category: devops
triggers:
  - search knowledge base
  - find in knowledge
  - query knowledge
  - kb search
  - fts5 search
toolsets:
  - terminal
  - file
---

# FTS5 Knowledge Base Search

Search local knowledge bases using SQLite FTS5 full-text search with trigram
tokenizer for CJK support and LIKE fallback.

## When to Use

- You need to search markdown/JSON knowledge files with keyword queries
- You have a structured local knowledge base and need ranked results
- Your knowledge base contains CJK (Chinese/Japanese/Korean) text
- You want to build a simple, file-based search without external services

Do NOT use for:
- Real-time search on rapidly changing data — rebuilding the index is needed
- Very large corpora (>100k docs) — consider dedicated search engines

## Prerequisites

- Python 3.8+ with `sqlite3` module (FTS5 support)
- SQLite compiled with FTS5 enabled (verify with `python3 -c "import sqlite3; c = sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(content)')"`)
- Knowledge files in JSON, Markdown, or plain text format

## How to Run

```bash
python3 build_fts5_index.py --dir /path/to/knowledge --db knowledge.db
python3 search_fts5.py --db knowledge.db --query "your search terms"
```

## Quick Reference

| Step | Action | Key Command |
|------|--------|-------------|
| Build index | Load files into FTS5 table | `CREATE VIRTUAL TABLE docs USING fts5(content, tokenize='trigram')` |
| Search (ASCII) | FTS5 MATCH query | `SELECT * FROM docs WHERE docs MATCH ?` |
| Search (CJK) | Trigram FTS5 MATCH | Same as above — trigram tokenizer handles CJK natively |
| Search (fallback) | LIKE for edge cases | `SELECT * FROM docs WHERE content LIKE '%keyword%'` |
| Rank results | BM25 scoring | `SELECT rank FROM docs WHERE docs MATCH ? ORDER BY rank` |

## Procedure

### Phase 1: Build the Index

1. Discover knowledge files (JSON, MD, TXT) in the target directory
2. Define schema: `id`, `module`, `title`, `content`
3. Create SQLite table with FTS5 virtual table using trigram tokenizer:
   ```sql
   CREATE VIRTUAL TABLE knowledge_fts USING fts5(
     module, title, content,
     tokenize='trigram'
   );
   ```
4. Insert documents (handle nested JSON by flattening to text)
5. For CJK text, the trigram tokenizer handles character trigrams — no special
   handling needed

### Phase 2: Keyword Extraction

1. Extract crop names, symptom keywords, domain terms from the query
2. Try FTS5 MATCH first with trigram tokenizer (works for CJK and ASCII)
3. If FTS5 returns no results, fall back to `LIKE '%keyword%'` for each term
4. Combine results with dedup (seen_ids hash set)
5. Filter out non-domain modules (e.g., exclude 'general' or 'product' categories)
6. Sort: domain knowledge first, limit to top N results

### Phase 3: Context Assembly

1. Build context string from matched documents
2. Include: module name, title, content snippet (~200 chars)
3. Format as numbered list with clear source attribution
4. Pass to LLM for synthesis

### Phase 4: Search Tuning

1. If no results, retry with shorter query (first 15 chars)
2. If still none, try single-character keyword expansion
3. If zero matches after all, return 'no relevant knowledge found'

## Pitfalls

- **Trigram tokenizer availability**: Not all SQLite builds include the trigram
  tokenizer. Verify with `SELECT * FROM pragma_compile_options WHERE
  compile_options LIKE '%TRIGRAM%'`.
- **LIKE for UTF-8 CJK**: `LIKE` is case-insensitive for ASCII but
  case-sensitive for UTF-8 Chinese. Use `LIKE` with lowercase queries for CJK.
- **Large knowledge bases (>1000 docs)**: Use FTS5 MATCH with trigram, not
  LIKE. LIKE scans the entire table.
- **Module filtering is critical**: Product docs in the same DB will drown out
  domain knowledge. Always filter by category/module.
- **Persistent database**: SQLite `:memory:` databases are lost on restart. Use
  a persistent file path.
- **FTS5 syntax restrictions**: Colons (`:`) in queries can be misinterpreted
  as FTS5 special syntax. Sanitize or double-quote them.

## Verification

- [ ] FTS5 trigram tokenizer is available: `python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(c, tokenize=\"trigram\")')"`
- [ ] Index builds successfully and documents are queryable
- [ ] CJK queries return expected results via trigram FTS5
- [ ] LIKE fallback returns results when FTS5 MATCH returns empty
- [ ] Results are ranked by relevance (BM25)
- [ ] No false positives from cross-module contamination