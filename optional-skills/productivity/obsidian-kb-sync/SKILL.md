---
name: obsidian-kb-sync
description: Sync Obsidian notes into a searchable SQLite knowledge base.
version: 1.0.0
author: ligl0325
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [obsidian, knowledge-base, notes, vault, sync, search]
    category: productivity
triggers: ['sync obsidian', 'index vault', 'update knowledge base', 'obsidian kb', 'vault sync']
toolsets: [terminal, file]
---

## Overview
Indexes markdown notes from an Obsidian vault into a SQLite database that an AI agent can search during conversations. Uses the `scripts/obsidian_kb_sync.py` helper for all database operations.

## Phase 1: Configure Vault Path
- Ask user for Obsidian vault path or detect common locations (~/Documents/Obsidian Vault/, ~/Obsidian/, ~/vault/)
- Verify path contains .obsidian/ (or at least markdown files with frontmatter)
- Determine the `HERMES_HOME` path for storing the database (default: `~/.hermes/data/obsidian_kb.db`)

## Phase 2: Scan & Index (via helper script)
Run the helper script to scan and index all markdown files:
```
python3 scripts/obsidian_kb_sync.py /path/to/vault ~/.hermes/data/obsidian_kb.db
```

The script handles:
- Finding all .md files recursively (skipping hidden directories)
- Parsing YAML frontmatter for title, tags, aliases
- Extracting H1/H2 headings as section titles
- Using filename as doc_id if no frontmatter title
- Module = directory name relative to vault root (filename if no subdir)
- Inserting into SQLite FTS5 table with trigram tokenizer
- Tracking file modification time (mtime) for incremental sync

Output: JSON summary `{"added": N, "updated": M, "removed": K, "total": T}`

## Phase 3: Search
Query the FTS5 index using SQLite MATCH with trigram tokenizer:
```sql
SELECT d.doc_id, d.module, d.title, d.file_path,
       rank as relevance
FROM documents_fts
JOIN documents d ON d.rowid = documents_fts.rowid
WHERE documents_fts MATCH ?
ORDER BY rank;
```
- **Tokenizer**: FTS5 trigram (supports CJK + English without separate word segmentation)
- **Search columns**: title and content
- **Ranking**: FTS5 built-in bm25() relevance scoring (lower rank = better match)
- Bind the search query as a single `?` parameter; FTS5 trigram parses it as contiguous n-grams

## Phase 4: Incremental Update
Run the script again — it automatically reconciles mtimes:
```
python3 scripts/obsidian_kb_sync.py /path/to/vault ~/.hermes/data/obsidian_kb.db
```
- Only re-indexes files whose mtime changed
- Removes docs for deleted files
- For full reindex (ignore cache): add `--reindex` flag

## Pitfalls
- Large vaults (>1000 files) take time; offer to index subdirectory first
- Binary files (.png, .pdf attachments) are skipped by the script's .md filter
- Frontmatter parsing: handle missing, malformed, or empty frontmatter gracefully
- Vault with symlinked files: resolve symlinks before checking mtime