---
name: obsidian-kb-sync
description: Index Obsidian Vault markdown notes into a searchable SQLite knowledge base. Reads markdown files, extracts title/headings/content, inserts into FTS5 database. Supports incremental updates via file modification time.
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
Indexes markdown notes from an Obsidian vault into a SQLite database that an AI agent can search during conversations. Each file becomes one or more knowledge documents.

## Phase 1: Configure Vault Path
- Ask user for Obsidian vault path or detect common locations (~/Documents/Obsidian Vault/, ~/Obsidian/, ~/vault/)
- Verify path contains .obsidian/ (or at least markdown files with frontmatter)

## Phase 2: Scan & Index
- Find all .md files recursively
- For each file:
  - Parse YAML frontmatter for title, tags, aliases
  - Extract H1/H2 headings as section titles
  - Use filename as doc_id if no frontmatter title
  - Module = directory name (filename if no subdir)
- Insert into SQLite: doc_id, module, title, content
- Track: file modification time (mtime) for incremental sync

## Phase 3: Search
- Same keyword extraction + LIKE search pattern as FTS5 knowledge search
- Search across title and content
- Return ranked results with source file path

## Phase 4: Incremental Update
- On re-run, compare mtime of each file against stored timestamp
- Only re-index changed files
- Remove docs for deleted files
- Report: updated N, added M, removed K

## Pitfalls
- Large vaults (>1000 files) take time; offer to index subdirectory first
- Binary files (.png, .pdf attachments) should be skipped
- Frontmatter parsing: handle missing, malformed, or empty frontmatter gracefully
- Vault with symlinked files: resolve symlinks before checking mtime
