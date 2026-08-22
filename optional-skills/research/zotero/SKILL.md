---
name: zotero
description: Manage a Zotero reference library — add books and papers by DOI, ISBN, arXiv ID, or URL; read PDF content; create and organise reading notes; manage collections and tags; set up a dedicated hermes-agent workspace folder inside the library. Use when the user mentions Zotero, wants to add references, read papers or books, take reading notes, manage a research library, search their bibliography, export citations (APA, MLA, BibTeX), or organise research collections.
version: 1.0.0
author: ed
license: MIT
required_environment_variables:
  - name: ZOTERO_API_KEY
    prompt: Zotero API key
    help: Get a key at https://www.zotero.org/settings/keys — enable read/write access to your library
    required_for: all operations
  - name: ZOTERO_USER_ID
    prompt: Zotero user ID (numeric)
    help: Found at https://www.zotero.org/settings/keys alongside your API key
    required_for: all operations
metadata:
  hermes:
    tags: [Research, Zotero, Papers, Books, PDF, Notes, References, Bibliography, Citations]
    related_skills: [arxiv, ocr-and-documents]
    category: research
---

# Zotero

Manage a Zotero reference library via the Web API v3. Add items, read PDFs, write notes, organise collections.

## Setup

**Required environment variables** (in `~/.hermes/.env` or shell):
```
ZOTERO_API_KEY=your_key_here      # from zotero.org/settings/keys (needs write access)
ZOTERO_USER_ID=1234567            # numeric ID shown on the same page
```

**One-time workspace bootstrap** (creates the hermes-agent collection tree):
```bash
python scripts/zotero_setup.py
```

This creates inside your Zotero library:
```
hermes-agent/
├── Books/
├── Papers/
├── Notes/
└── Reading List/
```

Re-running is safe — it is idempotent. Use `--show` to print existing keys without modifying.

**Base URL for all API calls:**
```
https://api.zotero.org/users/{ZOTERO_USER_ID}
```

Always send headers:
```
Zotero-API-Version: 3
Zotero-API-Key: {ZOTERO_API_KEY}
```

## Adding Items

Use `scripts/zotero_add.py`. It fetches metadata from external sources, then POSTs to Zotero and places the item in the hermes-agent sub-collection.

```bash
# Add by DOI (fetches from CrossRef)
python scripts/zotero_add.py --doi 10.1145/3290605.3300786

# Add by ISBN (fetches from Open Library)
python scripts/zotero_add.py --isbn 978-0-13-468599-1

# Add by arXiv ID (PDF readable on demand via URL — no upload to Zotero needed)
python scripts/zotero_add.py --arxiv 2301.07041

# Add by URL (creates webpage item)
python scripts/zotero_add.py --url https://example.com/article

# Add to a specific sub-collection (use key from zotero_setup output)
python scripts/zotero_add.py --doi 10.1145/xxx --collection BOOKS_KEY

# Bulk import from BibTeX file (batches ≤50 per request)
python scripts/zotero_add.py --bibtex refs.bib
```

## Browsing the Library

```bash
# List hermes-agent sub-collections (with keys)
python scripts/zotero_search.py --collections

# List all items in a collection
python scripts/zotero_search.py --collection COLLECTION_KEY

# Get full metadata for one item
python scripts/zotero_search.py --item ITEM_KEY

# Collection stats: item counts by type and top tags
python scripts/zotero_search.py --stats --collection COLLECTION_KEY
```

## Searching

```bash
# Full-text keyword search
python scripts/zotero_search.py "transformer attention mechanism"

# Filter by tag
python scripts/zotero_search.py --tag "machine-learning"

# Filter by item type (book, journalArticle, conferencePaper, webpage…)
python scripts/zotero_search.py --type book

# Items added after a date
python scripts/zotero_search.py --since 2025-01-01

# Detect duplicates by DOI/ISBN/title
python scripts/zotero_search.py --dupes

# Combined filters
python scripts/zotero_search.py "deep learning" --tag "unread" --type journalArticle

# Export collection to BibTeX
python scripts/zotero_search.py --export bibtex COLLECTION_KEY > refs.bib
```

## Reading PDFs

`zotero_read.py` tries methods in order (fastest first):

1. Zotero fulltext index (`GET /items/{key}/fulltext`) — instant if indexed by desktop app
2. Download from Zotero cloud storage + extract with `pdfplumber`
3. **Fetch directly from source URL** (arXiv, open-access DOI via Unpaywall) — works even if no PDF is stored in Zotero

This means arXiv papers are always readable even if the PDF was never uploaded.

```bash
# Read the PDF for an item (auto-finds attachment or fetches from source)
python scripts/zotero_read.py ITEM_KEY

# Limit to first N pages (good for long papers)
python scripts/zotero_read.py ITEM_KEY --pages 10

# Save extracted text to a file
python scripts/zotero_read.py ITEM_KEY --out summary.txt

# Metadata and abstract only — no PDF fetch
python scripts/zotero_read.py ITEM_KEY --metadata-only

# Get a formatted citation
python scripts/zotero_read.py ITEM_KEY --cite apa
python scripts/zotero_read.py ITEM_KEY --cite mla
python scripts/zotero_read.py ITEM_KEY --cite bibtex
```

`pdfplumber` must be installed for PDF extraction:
```bash
pip install pdfplumber
```

## Notes

Notes are Zotero items with `itemType: "note"` attached as children of a parent item.

```bash
# Create a reading note on an item
python scripts/zotero_note.py create ITEM_KEY --title "Reading notes" --body "Key points..."

# Create note from a file
python scripts/zotero_note.py create ITEM_KEY --title "Summary" --file notes.md

# Use a template (reading, book, quick)
python scripts/zotero_note.py create ITEM_KEY --template reading

# List all notes for an item
python scripts/zotero_note.py list ITEM_KEY

# Show note content
python scripts/zotero_note.py show NOTE_KEY

# Update an existing note
python scripts/zotero_note.py update NOTE_KEY --body "Replacement text"
python scripts/zotero_note.py update NOTE_KEY --append "...additional text"

# Set reading progress tag on an item (replaces previous status)
python scripts/zotero_note.py status ITEM_KEY unread
python scripts/zotero_note.py status ITEM_KEY reading
python scripts/zotero_note.py status ITEM_KEY done
```

### Structured note template

When creating a reading note, use this structure:

```markdown
## Summary
[2-3 sentence overview]

## Key Points
- Point 1
- Point 2

## Quotes
> "Notable quote" (p. X)

## Questions / Gaps
- Question raised

## Related
- [Related item or concept]
```

## Organisation

```bash
# Add an item to a collection (can be in multiple)
python scripts/zotero_add.py --move ITEM_KEY --collection COLLECTION_KEY

# Create a new sub-collection under hermes-agent
python scripts/zotero_setup.py --add-collection "Philosophy"

# Bulk-tag items in a collection
python scripts/zotero_search.py --collection COLLECTION_KEY --add-tag "reading-sprint-2026"
```

## Export & Citations

```bash
# Export full collection to BibTeX
python scripts/zotero_search.py --export bibtex HERMES_COLLECTION_KEY > library.bib

# Export to RIS format
python scripts/zotero_search.py --export ris HERMES_COLLECTION_KEY > library.ris

# Single-item citation
python scripts/zotero_read.py ITEM_KEY --cite apa
```

Citation styles supported: `apa`, `mla`, `chicago`, `bibtex`.

## Workflow Recipes

### Add a paper and take notes
```bash
python scripts/zotero_add.py --doi 10.1234/example --collection PAPERS_KEY
# Note the ITEM_KEY printed
python scripts/zotero_read.py ITEM_KEY --pages 5
python scripts/zotero_note.py create ITEM_KEY --template reading
python scripts/zotero_note.py status ITEM_KEY reading
```

### Literature review on a topic
```bash
python scripts/zotero_search.py "your topic" --type journalArticle
python scripts/zotero_search.py --export bibtex COLLECTION_KEY > review.bib
```

### Reading sprint
```bash
python scripts/zotero_search.py --tag "unread" --collection HERMES_KEY
# For each item:
python scripts/zotero_read.py ITEM_KEY --pages 10
python scripts/zotero_note.py create ITEM_KEY --title "Sprint notes" --body "..."
python scripts/zotero_note.py status ITEM_KEY done
```

### Add a book
```bash
python scripts/zotero_add.py --isbn 978-0-06-112008-4 --collection BOOKS_KEY
# Zotero will store title, author, publisher, year from Open Library
```

## Deletions

Deletion is **not available** through these scripts. To delete items, collections, or tags, use the Zotero desktop app. This prevents accidental data loss.

## Notes

- All script output prints the Zotero item key — save it for follow-up operations
- `zotero_setup.py --show` prints all hermes-agent collection keys without modifying anything
- Rate limit: Zotero API allows ~15 requests/second with a valid API key
- Items can belong to multiple collections simultaneously
- The fulltext index is populated when you open a PDF in the Zotero desktop app; use the download fallback if not yet indexed
- Date strings like "December 27, 2017" are handled correctly — year is extracted by regex

## API Reference

For full endpoint details, see [api-reference.md](api-reference.md).
