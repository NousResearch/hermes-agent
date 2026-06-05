# Books Pipeline — PDF & ePub → Knowledge Base

> **Superseded by `book-extraction.md`** — use that file as the canonical books pipeline.

Full extraction pipeline for long-form documents (books, papers, manuals).

## ePub Extraction with ebooklib

```bash
pip install ebooklib
```

```python
# extract_epub.py — Extract all text from an ePub file
from ebooklib import epub
from bs4 import BeautifulSoup
import sys

book = epub.read_epub(sys.argv[1])
for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        soup = BeautifulSoup(item.get_body_content(), 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        if text.strip():
            print(text)
            print('\n---\n')
```

Usage: `python3 extract_epub.py book.epub > /tmp/book_text.txt`

## PDF Extraction (text-based)

```bash
python3 -c "
import pymupdf
doc = pymupdf.open('file.pdf')
for page in doc:
    print(page.get_text())
" > /tmp/book_text.txt
```

## PDF Extraction (scanned/OCR)

```bash
pip install marker-pdf  # ~3-5GB, one-time
marker_single file.pdf --output_dir /tmp/marker_out
cat /tmp/marker_out/file/file.md | head -20000  # first 20K chars
```

## Summarization Approach

Books are 50K-150K words — can't fit in a single note. Strategy:

1. **Extract full text** → `/tmp/book_text.txt`
2. **Archive to MinIO** (BEFORE note creation):
   ```bash
   mc cp /tmp/book_text.txt minio/knowledge-base/books/<slug>.txt
   mc cp book.epub minio/knowledge-base/books/<slug>.epub  # or .pdf
   ```
   See `references/minio-storage.md` for MinIO setup.
3. **Identify structure** — chapter boundaries via `grep -n "^Chapter\|^CHAPITRE\|^[0-9]+\." /tmp/book_text.txt`
4. **Per-chapter summary** — for each chapter, extract first 2K chars + last 500 chars + any bolded/emphasized text. Feed to LLM summarizer.
5. **Key claims extraction** — look for declarative statements with citations or statistical claims
6. **Fact-check top claims** via `web_search` (see `references/fact-check-workflow.md`)
7. **Save structured note** in `Knowledge base/` with `source_file` in frontmatter pointing to the MinIO archive

## Note Template for Books

```markdown
---
date: YYYY-MM-DD
source: <Book: Title by Author, Year, Publisher>
source_url: <optional ISBN or URL>
source_file: <MinIO URL — http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/books/<slug>.epub>
confidence: varies per claim — see individual ratings
tags: [tag1, tag2, tag3]
format: book
pages: <N>
---

# Title — Author

## Metadata
- **Author:** <name>
- **Year:** <YYYY>
- **Publisher:** <name>
- **ISBN:** <if available>
- **Pages:** <N>

## Overview
2-3 paragraph summary of the book's thesis and structure.

## Key Claims
### Claim 1: <title>
**Confidence:** ✅/⚠️/🔬/❌
- What the book asserts
- Evidence provided
- External verification

### Claim 2: ...
[...]

## Chapter Summaries
### Ch. 1 — <title>
- Main argument
- Key evidence
- Notable quotes

### Ch. 2 — ...
[...]

## Notable Quotes
- "..." (Ch. X, p. Y)
- "..." (Ch. X, p. Y)

## Critical Assessment
- Strengths
- Weaknesses / omissions
- Author's potential biases
- Conflicts with other sources

## Sources
- The book itself
- External sources consulted for fact-checking

## Voir aussi
- [[related-note-1]]
- [[related-note-2]]
```

## Pitfalls

- **ePub images** — ebooklib extracts text only. Images/charts are skipped — note this in the summary if the book relies heavily on visual data.
- **DRM-protected ePubs** — ebooklib can't decrypt DRM. User must provide an unlocked copy.
- **Multi-column PDFs** — pymupdf reads in reading order but may interleave columns. marker-pdf handles this better but requires 3-5GB disk.
- **Chapter detection** — not all books use standard chapter markers. Fall back to page ranges if regex fails.
- **Token limits** — a 100K-word book will exceed any single LLM call. Always chunk by chapter, never try to summarize the whole book in one pass.
