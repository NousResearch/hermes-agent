# Book Extraction — ePub & PDF

Pipeline for processing books into the knowledge base. Books are 50-150K words — never read the full text in context. Read chapter by chapter, extract key claims, synthesize.

## Supported formats

| Format | Tool | Install |
|--------|------|---------|
| ePub | `ebooklib` + `beautifulsoup4` | `pip install ebooklib beautifulsoup4` |
| PDF (text) | `pymupdf` | `pip install pymupdf` |
| PDF (scanned/OCR) | `marker-pdf` | `pip install marker-pdf` (~5GB) |

## ePub pipeline

### Step 1: User sends the file

Telegram blocks `.epub` — user must rename to `.zip` before uploading, or use `scp`.

```bash
unzip -o /tmp/book.zip -d /tmp/book_output
```

### Step 2: Extract full text

```python
from ebooklib import epub
from bs4 import BeautifulSoup
import os

epub_file = [f for f in os.listdir('/tmp/book_output') if f.endswith('.epub')][0]
book = epub.read_epub(f'/tmp/book_output/{epub_file}')

items = list(book.get_items_of_type(9))  # ITEM_DOCUMENT
all_text = []
total_chars = 0

for i, item in enumerate(items):
    soup = BeautifulSoup(item.get_content(), 'html.parser')
    text = soup.get_text(separator='\n').strip()
    chars = len(text)
    total_chars += chars
    all_text.append((i, chars, text))

# Sort by size (biggest = main content chapters)
all_text.sort(key=lambda x: -x[1])

# Save full text for reference — PRESERVE CHAPTER ORDER, don't sort by size
# Keep the size-sorted list for analysis but save in original order
with open('/tmp/book_full.txt', 'w') as f:
    for i, item in enumerate(items):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text = soup.get_text(separator='\n').strip()
        chars = len(text)
        if chars > 100:
            f.write(f'\n\n=== CHAPTER {i} ({chars:,} chars) ===\n\n{text}')

# Also print chapter sizes sorted for quick analysis
sorted_chapters = sorted(all_text, key=lambda x: -x[1])
print(f'\nChapter sizes (sorted):')
for idx, chars, text in sorted_chapters[:10]:
    preview = text[:80].replace('\n', ' ')
    print(f'  Ch.{idx}: {chars:,} chars — {preview}...')

# Metadata
title = book.get_metadata('DC', 'title')
author = book.get_metadata('DC', 'creator')
print(f'Title: {title[0][0] if title else "?"}')
print(f'Author: {author[0][0] if author else "?"}')
print(f'Total: {total_chars:,} chars (~{total_chars//5:,} words)')
print(f'Chapters: {len(items)}')
```

### Step 3: Read EVERY chapter in full — no sampling, no exceptions

**Read the entire book.** Every chapter, every page. No sampling, no "key passages only," no "start with the top 5 by size."

**How:** Use `read_file` with `offset` and `limit` to read the full extracted text (`/tmp/book_full.txt`) in chunks if needed. For short books, read it all in one pass. For long books (100K+ words), read across multiple turns — the user will wait.

**Never sample.** The user has been explicit: "On lit tout pas d'échantillonnage." Sampling means you will miss content, miss quotes, and produce a note that's too shallow. The user will ask you to redo it. Read everything the first time.

**Focus on extracting:**
- The author's own words — direct quotes with chapter numbers
- Claims the author makes (especially controversial or novel ones)
- Statistics, studies cited
- Clinical advice, treatment protocols, patient case studies
- The author's framing: why they wrote the book, what they think they discovered

**Verify coverage:** After reading, confirm you've read every chapter. Count them against the epub metadata.

### Step 4: Create structured note

**Load the template first:** `skill_view(name='knowledge-base', file_path='templates/book-note-template.md')`

Save to `Knowledge base/<slug>.md`. The template defines the required structure. Key sections that MUST be populated:

### Step 5: Push to Git

```bash
cd "$OBSIDIAN_VAULT_PATH"
git add -A
git commit -m "add: <slug> — <author> (livre, ~XK mots)"
git push
```

## PDF pipeline

For PDFs, use the `ocr-and-documents` skill. `pymupdf` for text-based PDFs, `marker-pdf` for scanned/OCR. Then follow the same reading and structuring steps as ePub (Step 3-5 above).

## Language rule

**Everything in English.** Template labels, section headers, analysis — all English. Book quotes stay in their original language but all surrounding text is English. Never mix French and English in the same note.

## Minimum quality bar — MUST meet ALL before pushing

**Stop and self-audit before `git push`. If any of these are missing, rework the note.**

1. **≥4 direct quotes** from the book in the "Key Claims" section, each preceded by `>`, with chapter number
2. **≥2 fact-checks** via `web_search` — at least one factual verification and one context check (author, reception)
3. **A "Critical Analysis" section** that critiques the book's method/bias/limitations, not just summarizes what critics said
4. **The "Who Is the Author?" section** includes background, credentials, conflicts of interest or known biases
5. **Note ≥ 60 lines** (excluding frontmatter) — anything less is a summary, not a note

## Pitfalls

- **Don't try to read the whole book in context.** 97K words = ~200K tokens. Read chapter by chapter from the extracted text file.
- **Claims need verification.** Book authors are often polemical and may cite studies loosely. `web_search` the key claims. Fact-checking is mandatory, not optional — also search the author's background and the book's critical reception.
- **ePub chapter ordering is unreliable.** Sort by size to find the actual content chapters (TOC/cover/copyright are tiny).
- **Telegram blocks .epub** — user must rename to .zip before sending.
- **Don't deliver a reception summary as a book note.** The user expects deep notes: direct quotes from the book, chapter-by-chapter claims with citations, fact-checking of key assertions, author credentials and biases, and critical reception. A note that only summarizes what Wikipedia or critics said about the book is insufficient — extract the author's own words from the text. The "Key Claims" section with `>` blockquotes is the heart of the note, not optional.
- **The first version is never good enough.** Budget 2-3 rounds on any book note: v1 = extract quotes, v2 = fact-check + structure, v3 = polish. Never push v1. The user will always ask for a deeper version — skip that step and deliver v2 or v3 directly.
- **Sorting chapters by size loses order.** The extraction code above saves chapters in original epub order. If you wrote code that sorts by size, the resulting file is unreadable as a book — chapters 1-5 end up after chapter 19. Always preserve the epub's chapter sequence in the saved output file. See Step 2 code for the correct pattern.
- **Small chapters are not optional.** Prefaces, introductions, and short concluding chapters often contain the author's thesis statement, methodology, and key definitions. Reading only the "big" chapters is like skipping the introduction and conclusion of a paper — you miss the argument's framing entirely.
