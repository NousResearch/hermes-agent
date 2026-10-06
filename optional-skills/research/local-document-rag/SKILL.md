---
name: local-document-rag
description: Offline per-project RAG over PDF/DOCX docs with page cites.
version: 1.0.0
author: Paweł Suchocki (pawelsu8) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
dependencies: [sqlite-vec, fastembed, pymupdf, python-docx]
metadata:
  hermes:
    tags: [RAG, Documents, PDF, OCR, Retrieval, Offline, sqlite-vec]
    related_skills: [pdf]
---

# Local Document RAG Skill

Indexes a project's own documents (specs, tenders, contracts, manuals — PDF incl. scans, DOCX, Markdown, text) into one SQLite file per project and answers questions with `file + page` citations, fully offline. It exists because a document pasted into chat lives only in the context window: after compression the full text is gone and the agent stops quoting it. The index keeps the text outside the context and re-fetches the relevant fragment on demand. This is not conversation memory (memory providers / LCM cover that).

## When to Use

- The user attaches or points at a project document and will ask about it later.
- The user asks about the content of an already indexed document (requirements, figures, deadlines).
- After context compression, when the document text is no longer visible in the window.
- The user complains the agent "forgets the PDF" or "stops referring to the attachment".

## Prerequisites

- Python packages: `pip install sqlite-vec fastembed pymupdf python-docx`.
- First run downloads the embedding model (~220 MB, `paraphrase-multilingual-MiniLM-L12-v2`, 384-dim, multilingual) into the fastembed cache.
- Optional OCR for scanned PDFs: Tesseract plus the language model (`<lang>.traineddata`) in `RAG_TESSDATA`, `TESSDATA_PREFIX` or `$HERMES_HOME/tessdata`.
- Environment (all optional): `RAG_EMBEDDING_MODEL` + `RAG_EMBEDDING_DIM` (must match each other and the existing DB), `RAG_OCR_LANG` (default `eng`, e.g. `pol`), `RAG_USAGE_LOG` (JSONL path; unset = no logging).

## How to Run

Invoke through the `terminal` tool, with the database inside the project directory:

```bash
python scripts/doc_rag.py ingest ./spec.pdf --db ./rag.db --copy --project my-project
python scripts/doc_rag.py query "warranty period for the servers" --db ./rag.db -k 6
```

## Quick Reference

| Command | Purpose |
|---|---|
| `ingest FILE... --db DB [--copy] [--project NAME]` | Index files; `--copy` also stores them in `<db dir>/assets/` |
| `query "TEXT" --db DB [-k N] [--json]` | Hybrid search, top N fragments with file + page |
| `list --db DB` | Indexed documents with page/chunk counts |
| `stats --db DB` | Model, dimension, document and chunk totals |

Exit code of `ingest` is `1` when any file was skipped (missing, unsupported, no text).

## Procedure

1. Pick the project directory; the DB is `<project>/rag.db`. One DB per project or client — never a shared index, so fragments of one client's documents cannot surface in another's conversation.
2. On a new document run `ingest ... --copy`. Re-ingesting the same file is a no-op (SHA-256 dedup).
3. Confirm to the user in one line: pages and chunks indexed, OCR pages if any.
4. Before answering a question about the document, run `query`; quote the fragment and cite `file p.N`.
5. When a hit is truncated or the answer needs a whole table, open the copied original in `assets/` with `read_file` or the `pdf` skill.

## Pitfalls

- Changing `RAG_EMBEDDING_MODEL` on an existing DB makes old vectors meaningless — rebuild the DB.
- DOCX is indexed as a single "page" (python-docx has no pagination); cite the section heading instead.
- A scanned PDF without Tesseract installed prints `[OCR unavailable]` and those pages are skipped; `ingest` exits 1 if the whole file yielded no text.
- Prefix matching (first ~75% of each long token) approximates stemming for inflected languages; vector search covers paraphrases. Neither replaces reading the source for exact legal wording.
- `--copy` keeps the file name; two different files with the same name keep the first copy.

## Verification

```bash
python scripts/doc_rag.py stats --db ./rag.db
```

Shows the model, dimension and non-zero document/chunk counts after a successful ingest.
