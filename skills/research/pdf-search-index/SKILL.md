---
name: pdf-search-index
description: "Index and semantically search local PDF collections using embeddings and FAISS — no external services, no API keys."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Research, PDF, Search, Embeddings, FAISS, Vector, Index]
    related_skills: [ocr-and-documents, arxiv]
    requires_install: true
---

# PDF Search & Index

Build a searchable vector index from local PDF collections. Extracts text with pymupdf, embeds with sentence-transformers (`all-MiniLM-L6-v2`), stores in FAISS. Sub-second semantic search over thousands of papers — no API keys, no cloud services, everything runs locally.

## Quickstart

```bash
# 1. Install dependencies (one-time)
pip install pymupdf sentence-transformers faiss-cpu

# 2. Index a directory
python skills/research/pdf-search-index/scripts/index_pdfs.py ~/papers/

# 3. Search
python skills/research/pdf-search-index/scripts/search_pdfs.py \
    "differential privacy convergence bound" --top-k 5
```

## Indexing

Scan a directory recursively for PDFs, extract text, chunk into paragraphs, embed, and store in a FAISS vector index.

```
python scripts/index_pdfs.py INPUT_DIR [OPTIONS]

Arguments:
  INPUT_DIR            Directory containing PDFs (scanned recursively)

Options:
  --index-dir DIR      Where to store index files (default: ./pdf_index)
  --chunk-size N       Characters per text chunk (default: 500)
  --extensions EXT     Comma-separated file extensions (default: .pdf)
  --model NAME         Sentence-transformers embedding model
                       (default: all-MiniLM-L6-v2)
  --force              Re-index even if index already exists
  --quiet              Suppress progress bars
```

Rerun after adding new PDFs — only new files are processed (use `--force` to rebuild).

Output files in `--index-dir`:
- `index.faiss` — FAISS vector index (inner-product)
- `chunks.jsonl` — Text chunks with source PDF, page, position metadata
- `files.json` — List of indexed file paths
- `model_name.txt` — Embedding model used for search consistency

## Searching

Query the index with natural language.

```
python scripts/search_pdfs.py QUERY [OPTIONS]

Options:
  --index-dir DIR      Index directory (default: ./pdf_index)
  --top-k N            Number of results (default: 5)
  --show-context       Print full chunk text instead of first 200 chars
  --json               Output results as JSON
```

Each result includes: source filename, page number, similarity score, and chunk text.

## Python API

```python
from pdf_index import PDFIndex

idx = PDFIndex(index_dir="./pdf_index")

# Build or update index
idx.index_directory("/path/to/papers/", chunk_size=500)

# Search
results = idx.search("gradient inversion attack", top_k=5)
for r in results:
    print(f"[{r['filename']}:p{r['page']}] score={r['score']:.3f}")
    print(f"  {r['text'][:200]}")

# Inspect
info = idx.info()
print(f"{info['indexed_files']} files, {info['total_chunks']} chunks")
```

## Model selection

| Model | Size | Best for |
|-------|------|----------|
| `all-MiniLM-L6-v2` (default) | 80 MB | General English, scientific text |
| `all-mpnet-base-v2` | 420 MB | Higher quality, slower |
| `paraphrase-multilingual-MiniLM-L12-v2` | 420 MB | Multi-language papers |
| `mixedbread-ai/mxbai-embed-large-v1` | 670 MB | State-of-the-art retrieval |

Change with `--model` flag or `model_name=` in the Python API. Must use the same model for indexing and searching.

## Dependencies

- `pymupdf` — PDF text extraction (handles most PDFs, fast)
- `sentence-transformers` — embedding models with caching
- `faiss-cpu` — vector similarity search (no GPU needed)
- `numpy` — array operations

## How it works

1. **Extract**: pymupdf reads each PDF page, extracts text blocks
2. **Chunk**: Text split into paragraph-aligned chunks (~500 chars) with sliding-window overlap
3. **Embed**: Sentence-transformer converts each chunk to a normalized 384-dim vector
4. **Index**: FAISS `IndexFlatIP` stores vectors for cosine-similarity (inner-product) search
5. **Search**: Query is embedded identically; FAISS returns top-k by similarity

## Limitations

- Scanned/image-only PDFs need OCR preprocessing (use `ocr-and-documents` skill first)
- Default model is English-only; use multilingual model for non-English papers
- Large collections (10k+ papers, 1M+ chunks) may benefit from GPU FAISS (`faiss-gpu`) or IVF indexing
- Memory: ~1 GB per 10k pages with default model

## Testing

```bash
# Unit tests
python -m pytest skills/research/pdf-search-index/tests/ -v

# Smoke test (creates a small index from test PDFs)
python skills/research/pdf-search-index/tests/test_integration.py
```
