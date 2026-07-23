---
name: fts5-knowledge-search
description: Keyword extraction + SQLite FTS5 full-text search for local knowledge bases. Extracts key terms from queries, searches structured documents, returns ranked results. Useful for any project with markdown/JSON knowledge files.
version: 1.0.0
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
    category: devops
triggers:
  - search knowledge base
  - find in knowledge
  - query knowledge
  - kb search
toolsets:
  - terminal
  - file
---

## Overview
Generic FTS5 knowledge search pattern: load JSON/markdown files into SQLite FTS5, extract keywords from user queries, search with LIKE or FTS5, rank results, and return with context.

## Phase 1: Build Index
- Discover knowledge files (JSON, MD, TXT) in specified directory
- Define schema: id, module, title, content
- Create SQLite table with FTS5 virtual table
- Insert documents (handle nested JSON by flattening to text)
- Chinese text: use LIKE instead of FTS5 (FTS5 doesn't tokenize CJK by default)

## Phase 2: Keyword Extraction
- Extract crop names, symptom keywords, domain terms from query
- For each keyword, run LIKE '%keyword%' search
- Combine results with dedup (seen_ids hash set)
- Filter out non-domain modules (e.g., exclude 'general' or 'product' categories)
- Sort: domain knowledge first, limit to top N results

## Phase 3: Context Assembly
- Build context string from matched documents
- Include: module name, title, content snippet (~200 chars)
- Format: numbered list with clear source attribution
- Pass to LLM for synthesis

## Phase 4: Search Tuning
- If no results, retry with shorter query (first 15 chars)
- If still none, try single-character keyword expansion
- If zero matches after all, return 'no relevant knowledge found'

## Pitfalls
- LIKE is case-insensitive for ASCII but sensitive for UTF-8 Chinese
- Large knowledge bases (>1000 docs) should use FTS5 MATCH, not LIKE
- Module filtering is critical — product docs in the same DB will drown out domain knowledge
- SQLite :memory: databases are lost on restart; use persistent file path
