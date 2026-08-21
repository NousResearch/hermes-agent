---
name: book-to-notes
description: Read a book (PDF/EPUB) incrementally, extract key concepts chapter by chapter, and store findings into a Zettelkasten or wiki. Optimized for long texts that exceed context window limits.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [reading, knowledge-extraction, zettelkasten, pdf, epubs, learning]
    category: note-taking
    related_skills: [read-book, ocr-and-documents, obsidian, llm-wiki]
    requires_toolsets: [terminal, files, delegate]
---

# Book-to-Notes Workflow

Read a book incrementally and extract structured knowledge for your note system.

## Configuration

This skill writes to a configurable vault. Set your preferred path via environment variable:

```bash
# In ~/.hermes/.env
BOOK_VAULT_PATH="~/notes"
```

If unset, defaults to `~/notes/`.

Typical vault structure:
```
~/notes/
├── books/              # Source PDFs/EPUBs and extracted text
├── reading-journal/    # Chapter summaries and reading logs
├── zettelkasten/       # Atomic notes (one idea per file)
└── concepts/           # Wiki-style concept pages linking atomic notes
```

## Step 1: Acquire the Book

**Prefer local files over remote URLs.** URLs rot. If the user provides a URL, download it immediately to a persistent location.

```bash
# Download to persistent storage (NOT /tmp/)
curl -sL "$URL" -o ~/books/bookname.pdf
```

**Pitfall:** `/tmp/` gets cleared between sessions or on reboot. Always use `~/books/` or another persistent path.

## Step 2: Extract Text

For **text-based PDFs**, `pdftotext` (from poppler-utils) is often pre-installed and faster than Python libraries:

```bash
pdftotext book.pdf book.txt
```

For **scanned PDFs**, **EPUBs**, or **complex layouts**, use the `ocr-and-documents` skill (pymupdf or marker-pdf).

For **EPUB** specifically:
```bash
# EPUBs are ZIP files with HTML chapters
unzip -q book.epub -d book_extracted/
# Text is usually in OEBPS/Text/ or similar
find book_extracted/ -name "*.html" -o -name "*.xhtml" | sort
```

## Step 3: Split into Chapters

**Goal:** Create manageable chunks (~2K-10K lines each) that fit in context windows.

```bash
# Split by "CHAPTER N" markers (case-insensitive)
python3 -c "
import re, sys
with open('book.txt') as f:
    text = f.read()
# Match CHAPTER followed by number or roman numeral
chapters = re.split(r'\n\s*(CHAPTER\s+[0-9IVX]+|Chapter\s+[0-9]+)\s*\n', text)
# chapters[0] = front matter, chapters[1] = 'CHAPTER 1', chapters[2] = content, etc.
for i in range(1, len(chapters), 2):
    num = chapters[i].strip().replace(' ', '_')
    content = chapters[i+1] if i+1 < len(chapters) else ''
    with open(f'ch_{num}.txt', 'w') as out:
        out.write(content)
        print(f'Wrote {num}: {len(content)} chars')
"
```

**Alternative:** Split by page ranges if chapter markers are unreliable:
```bash
# Extract pages 1-50
pdftotext -f 1 -l 50 book.pdf ch1.txt
```

**Store splits persistently** alongside the source PDF.

## Step 4: Incremental Reading

Read **one chapter per tool call** (or two if they're short). For each chapter:

1. **Read the chapter** with `read_file`
2. **Extract immediately** — don't defer synthesis. Extract:
   - Core concepts (name + definition)
   - Models/frameworks (list components)
   - Key arguments and evidence
   - Notable quotes (verbatim, with page numbers if available)
   - Connections to other chapters or ideas

3. **Write a chapter summary** as you go — a 5-10 bullet summary per chapter prevents re-reading.

### Quality Checklist for Subagent Extraction

When delegating chapter extraction to a subagent, provide the **Reading Tool** (`templates/reading-tool.md`) as its instructions. This enforces:
- Two-pass extraction (structural → conceptual)
- Concept crystallization with boundaries and implications
- Cross-chapter threading
- Explicit failure-mode guards against parroting, compressing, and inventing

Load the tool with: `skill_view(name="book-to-notes", file_path="templates/reading-tool.md")`

## Step 5: Synthesize and Store

After reading all chapters (or enough for the user's purpose):

1. **Create atomic notes** in the Zettelkasten (one idea per note, sentence-titled)
2. **Create a concept page** in the wiki linking to atomic notes
3. **Tag and link** — connect new concepts to existing notes

**Storage location:** Use your configured vault (default `~/notes/`):
- Source: `~/notes/books/Book_Title.md`
- Notes: `~/notes/zettelkasten/`
- Wiki: `~/notes/concepts/`

## Pitfalls

| Pitfall | Solution |
|---------|----------|
| `/tmp/` cleared | Use `~/books/` or your vault's books directory |
| URL dies | Download immediately; keep local copy |
| Context overflow | One chapter per call; extract as you read |
| Re-reading same content | Write chapter summaries immediately |
| Chapter regex fails | Inspect first 100 lines of text; adjust regex |
| PDF is scanned/image-based | Fall back to `ocr-and-documents` skill (marker-pdf) |
| EPUB has no clear chapter files | Unzip and inspect structure; often `OEBPS/Text/ch*.html` |

## Example: Full Session

```bash
# 1. Acquire
curl -sL "$URL" -o ~/books/flow.pdf

# 2. Extract
pdftotext ~/books/flow.pdf ~/books/flow.txt

# 3. Split
python3 scripts/split_book.py ~/books/flow.txt ~/books/flow_chapters/

# 4. Read & extract (repeat per chapter)
# ... read_file calls ...

# 5. Store
# ... write_file to vault ...
```

## Agent Architecture for Autonomous Reading

When building a *reusable system* for an agent to read books unattended, prefer **subagent delegation with disk-state** over cron-based autonomous loops for books under ~400 pages.

### The Pattern (v0.1)

```
state/
├── status.json      # {"book":"Title","total_chapters":N,"current":3,"phase":"extracting"}
└── resume.md        # Laser-focus context for the next subagent run

chapters/
├── ch_001.txt
├── ch_002.txt
└── ...

outputs/
├── chapter_summaries/
├── atomic_notes/
└── wiki_drafts/
```

**Why this works:**
- Each subagent run reads `resume.md` + 1-2 chapters, writes outputs, updates `status.json`
- No persistent memory needed between runs — the disk is the state machine
- The parent agent spawns a subagent per work unit, then regains control
- Human can inspect `status.json` at any time to see progress

### Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **Reject flush tool for < 400 pages** | Context management via `resume.md` is sufficient; flush adds complexity without benefit |
| **Defer cron jobs to v1.0** | Premature autonomy hides failures; manual subagent runs let you verify quality first |
| **One chapter per subagent call** | Prevents context overflow; keeps each unit focused and debuggable |
| **Resume.md as laser-focus state** | Contains only: book thesis so far, open threads, next task, nothing else |
| **Status.json as ground truth** | Machine-readable; cheap to parse; survives across sessions |

### Incremental Roadmap

- **v0.1** (today): Manual subagent runs, disk state, one chapter at a time
- **v0.2**: Add synthesis pass — subagent reads all chapter summaries and builds the concept map
- **v0.3**: Add triage — pre-read TOC + intro to generate `questions_to_answer.md`, skip irrelevant chapters
- **v1.0**: Cron schedule + automatic flushing only when books exceed 400 pages or context limits demand it

### When to Stop Reading

You don't need to read every chapter. Stop when you have:
- The core framework/model (usually Chapters 1-4)
- The applied techniques (usually later chapters)
- Enough to answer the user's specific question

If the user says "use what you learn to help me do X," prioritize chapters relevant to X over completeness.
