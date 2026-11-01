"""
pdf_index.py — General-purpose PDF indexing and semantic search library.

Build a searchable FAISS vector index from local PDFs. Extracts text via
pymupdf, embeds with sentence-transformers, stores in FAISS for sub-second
cosine-similarity search. No API keys, no cloud services.

Usage:
    from pdf_index import PDFIndex

    idx = PDFIndex(index_dir="./pdf_index")
    idx.index_directory("~/papers/", chunk_size=500)
    results = idx.search("differential privacy convergence", top_k=5)

    for r in results:
        print(f"[{r['filename']}:p{r['page']}] {r['score']:.3f}")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


__version__ = "1.0.0"


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


class PDFIndex:
    """Manage a FAISS vector index over a collection of PDFs.

    Parameters
    ----------
    index_dir : str or Path
        Directory to store index files (faiss index, chunks, metadata).
        Created if it doesn't exist.
    """

    INDEX_FILE = "index.faiss"
    CHUNKS_FILE = "chunks.jsonl"
    FILES_FILE = "files.json"
    MODEL_FILE = "model_name.txt"

    def __init__(self, index_dir: str | Path = "./pdf_index"):
        self._index_dir = _expand(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)

        # Lazily loaded
        self._faiss_index: Any = None
        self._model: Any = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_directory(
        self,
        input_dir: str | Path,
        chunk_size: int = 500,
        extensions: Sequence[str] = (".pdf",),
        model_name: str = "all-MiniLM-L6-v2",
        force: bool = False,
        progress: bool = True,
    ) -> int:
        """Scan *input_dir* recursively for PDFs, extract, embed, and index.

        Parameters
        ----------
        input_dir : str or Path
            Root directory to scan (recursive).
        chunk_size : int
            Target characters per text chunk (default 500).
        extensions : sequence of str
            File extensions to include (default ``(".pdf",)``).
        model_name : str
            Sentence-transformers embedding model.
        force : bool
            If True, re-index all files even if already in the index.
        progress : bool
            Show progress bars during embedding.

        Returns
        -------
        int
            Total number of chunks in the index after this call.
        """
        import fitz  # pymupdf

        input_path = _expand(input_dir)

        # Collect PDF files
        pdf_files: List[Path] = []
        for ext in extensions:
            pdf_files.extend(input_path.rglob(f"*{ext}"))
        pdf_files = sorted(set(pdf_files))

        if not pdf_files:
            print(f"No PDFs found in {input_path}")
            return self._total_chunks()

        print(f"Found {len(pdf_files)} PDFs in {input_path}")

        # Determine which files need indexing
        chunks_file = self._index_dir / self.CHUNKS_FILE
        existing_sources: set = set()
        if chunks_file.exists() and not force:
            existing_sources = self._collect_sources()
            print(f"Existing index: {len(existing_sources)} files")

        new_files = [f for f in pdf_files if str(f) not in existing_sources]
        if force:
            new_files = pdf_files
            chunks_file.unlink(missing_ok=True)
            existing_sources.clear()

        if not new_files:
            print("All files already indexed. Use force=True to re-index.")
            return self._total_chunks()

        print(f"Processing {len(new_files)} new PDFs...")

        # Extract text
        raw_chunks = self._extract_chunks(new_files, chunk_size)
        if not raw_chunks:
            print("No extractable text found.")
            return self._total_chunks()

        print(f"Extracted {len(raw_chunks)} text chunks")

        # Embed
        from sentence_transformers import SentenceTransformer

        if progress:
            print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        texts = [c["text"] for c in raw_chunks]
        embeddings = model.encode(
            texts,
            show_progress_bar=progress,
            batch_size=64,
            normalize_embeddings=True,
        )

        # Build / update FAISS index
        import faiss

        dim = embeddings.shape[1]
        index_file = self._index_dir / self.INDEX_FILE
        if index_file.exists() and self._faiss_index is None:
            self._faiss_index = faiss.read_index(str(index_file))
        elif self._faiss_index is None:
            self._faiss_index = faiss.IndexFlatIP(dim)

        self._faiss_index.add(embeddings.astype(np.float32))
        faiss.write_index(self._faiss_index, str(index_file))

        # Persist chunks
        with open(chunks_file, "a") as f:
            for ch in raw_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        # Persist file list
        files_list = sorted(self._collect_sources())
        with open(self._index_dir / self.FILES_FILE, "w") as f:
            json.dump(list(files_list), f, indent=2)

        # Persist model name
        (self._index_dir / self.MODEL_FILE).write_text(model_name)

        total = self._total_chunks()
        print(f"Indexed {len(raw_chunks)} new chunks ({total} total).")
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed PDFs.

        Parameters
        ----------
        query : str
            Natural language query.
        top_k : int
            Number of results to return.
        model_name : str, optional
            Override embedding model (must match the one used for indexing).

        Returns
        -------
        list of dict
            Each result has keys: source, filename, page, text, score.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        index_file = self._index_dir / self.INDEX_FILE
        if not index_file.exists():
            raise FileNotFoundError(
                f"No index found at {index_file}. Run index_directory() first."
            )

        faiss_idx = faiss.read_index(str(index_file))
        chunks = self._load_chunks()
        if not chunks:
            return []

        # Determine model
        if model_name is None:
            mn_file = self._index_dir / self.MODEL_FILE
            model_name = mn_file.read_text().strip() if mn_file.exists() else "all-MiniLM-L6-v2"

        model = SentenceTransformer(model_name)
        q_emb = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = faiss_idx.search(q_emb, min(top_k, len(chunks)))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            ch = dict(chunks[idx])
            ch["score"] = float(score)
            results.append(ch)

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return index metadata."""
        files_file = self._index_dir / self.FILES_FILE
        model_file = self._index_dir / self.MODEL_FILE

        info: Dict[str, Any] = {
            "index_dir": str(self._index_dir),
            "has_index": (self._index_dir / self.INDEX_FILE).exists(),
            "total_chunks": self._total_chunks(),
            "indexed_files": 0,
            "model": model_file.read_text().strip() if model_file.exists() else None,
        }

        if files_file.exists():
            with open(files_file) as f:
                files = json.load(f)
            info["indexed_files"] = len(files)
            info["files"] = files[:50]

        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chunks(
        pdf_files: List[Path],
        chunk_size: int,
    ) -> List[Dict[str, Any]]:
        """Extract text chunks from PDFs via pymupdf."""
        import fitz

        raw: List[Dict[str, Any]] = []
        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text("text")
                    if not text.strip():
                        continue
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for pi, para in enumerate(paragraphs):
                        if len(para) < 20:
                            continue
                        step = max(chunk_size // 5, 50)
                        for start in range(0, len(para), step):
                            chunk = para[start : start + chunk_size]
                            if len(chunk) < 50:
                                continue
                            raw.append({
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "page": page_num + 1,
                                "para_idx": pi,
                                "chunk_start": start,
                                "text": chunk,
                            })
                doc.close()
            except Exception as e:
                print(f"  WARN: {pdf_path.name}: {e}")
        return raw

    def _load_chunks(self) -> List[Dict[str, Any]]:
        chunks_file = self._index_dir / self.CHUNKS_FILE
        if not chunks_file.exists():
            return []
        chunks: List[Dict[str, Any]] = []
        with open(chunks_file) as f:
            for line in f:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return chunks

    def _total_chunks(self) -> int:
        chunks_file = self._index_dir / self.CHUNKS_FILE
        if not chunks_file.exists():
            return 0
        return sum(1 for _ in open(chunks_file))

    def _collect_sources(self) -> set:
        sources: set = set()
        for ch in self._load_chunks():
            sources.add(ch.get("source", ""))
        return sources


# ----------------------------------------------------------------------
# Quick sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        idx = PDFIndex(index_dir=tmp)
        info = idx.info()
        assert info["total_chunks"] == 0
        assert not info["has_index"]
        print(f"pdf_index.py v{__version__} — OK (no index yet)")
