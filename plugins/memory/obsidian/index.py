"""Persistent vault index: note metadata + chunk embeddings cached to disk.

The index is stored in ``{vault}/.obsidian/ai-index/`` so it survives
restarts without a full re-embed. Only changed files (by mtime+size hash)
are re-embedded on load.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from plugins.memory.obsidian.embeddings import EmbeddingProvider, get_embedding_provider
from plugins.memory.obsidian.graph import WikilinkGraph
from plugins.memory.obsidian.search import BM25Index, HybridSearch, SemanticIndex
from plugins.memory.obsidian.vault import Note, VaultReader, format_note_for_context

logger = logging.getLogger(__name__)

_INDEX_DIR = ".obsidian/ai-index"
_META_FILE = "meta.json"
_VECS_FILE = "embeddings.npy"
_IDS_FILE = "chunk_ids.json"


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_mtime:.3f}:{st.st_size}"


# ---------------------------------------------------------------------------
# Chunk registry: maps chunk_id → (note_path, chunk_text)
# ---------------------------------------------------------------------------

class _ChunkRegistry:
    def __init__(self) -> None:
        self._chunks: Dict[int, Tuple[str, str]] = {}  # id → (path_str, text)
        self._note_chunks: Dict[str, List[int]] = {}    # path_str → [chunk_ids]
        self._next_id: int = 0

    def add(self, path_str: str, texts: List[str]) -> List[Tuple[int, str]]:
        ids = []
        for text in texts:
            cid = self._next_id
            self._next_id += 1
            self._chunks[cid] = (path_str, text)
            ids.append((cid, text))
        self._note_chunks[path_str] = [cid for cid, _ in ids]
        return ids

    def remove(self, path_str: str) -> List[int]:
        ids = self._note_chunks.pop(path_str, [])
        for cid in ids:
            self._chunks.pop(cid, None)
        return ids

    def text(self, chunk_id: int) -> Optional[str]:
        entry = self._chunks.get(chunk_id)
        return entry[1] if entry else None

    def path_of(self, chunk_id: int) -> Optional[str]:
        entry = self._chunks.get(chunk_id)
        return entry[0] if entry else None

    def all_chunks(self) -> List[Tuple[int, str]]:
        return [(cid, txt) for cid, (_, txt) in self._chunks.items()]


# ---------------------------------------------------------------------------
# VaultIndex
# ---------------------------------------------------------------------------

class VaultIndex:
    """Full index over an Obsidian vault: BM25 + optional semantic + graph."""

    def __init__(self, vault_path: Path) -> None:
        self.vault = vault_path
        self._index_dir = vault_path / _INDEX_DIR
        self._lock = threading.RLock()

        self._reader = VaultReader(vault_path)
        self._registry = _ChunkRegistry()
        self._graph = WikilinkGraph()

        self._embed: Optional[EmbeddingProvider] = None
        self._bm25 = BM25Index()
        self._semantic: Optional[SemanticIndex] = None
        self._search: Optional[HybridSearch] = None

        # sig cache: path_str → file sig string (mtime:size)
        self._sigs: Dict[str, str] = {}

        self._built = False

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, *, force: bool = False) -> None:
        """Index the full vault. Incremental if cached embeddings exist."""
        with self._lock:
            self._embed = get_embedding_provider()
            if self._embed:
                self._semantic = SemanticIndex(self._embed)

            cached_vecs, cached_ids, cached_sigs = self._load_cache()
            notes = self._reader.load_all()
            changed_notes: List[Note] = []
            unchanged_paths: set[str] = set()

            for note in notes:
                ps = str(note.path)
                sig = _file_sig(note.path)
                if not force and ps in cached_sigs and cached_sigs[ps] == sig:
                    unchanged_paths.add(ps)
                else:
                    changed_notes.append(note)
                self._sigs[ps] = sig

            # Re-use unchanged chunks from cache
            if cached_vecs is not None and cached_ids and unchanged_paths:
                for cid_str, (path_str, text) in cached_ids.items():
                    if path_str in unchanged_paths:
                        cid = int(cid_str)
                        self._registry._chunks[cid] = (path_str, text)
                        self._registry._note_chunks.setdefault(path_str, []).append(cid)
                        if cid >= self._registry._next_id:
                            self._registry._next_id = cid + 1

            # Index changed/new notes
            for note in changed_notes:
                self._index_note(note, rebuild=False)

            # Build the graph over all notes
            all_notes = self._reader.load_all()
            self._graph.build(all_notes, self.vault)

            # Build search indices
            all_chunks = self._registry.all_chunks()
            self._bm25.build(all_chunks)

            if self._semantic is not None:
                # Stitch cached embeddings + new embeddings
                new_paths = {str(n.path) for n in changed_notes}
                new_chunks = [(cid, txt) for cid, txt in all_chunks
                              if self._registry.path_of(cid) in new_paths]
                old_chunks = [(cid, txt) for cid, txt in all_chunks
                              if cid not in {c for c, _ in new_chunks}]

                if cached_vecs is not None and old_chunks:
                    old_ids_set = {c for c, _ in old_chunks}
                    old_indices = [i for i, cid_str in enumerate(cached_ids)
                                   if int(cid_str) in old_ids_set]
                    old_matrix = cached_vecs[old_indices] if old_indices else np.zeros((0, self._embed.dim), dtype=np.float32)
                    if new_chunks:
                        new_matrix = self._embed.encode([t for _, t in new_chunks])
                        full_matrix = np.vstack([old_matrix, new_matrix])
                    else:
                        full_matrix = old_matrix
                    self._semantic._ids = [c for c, _ in old_chunks] + [c for c, _ in new_chunks]
                    self._semantic._matrix = full_matrix
                else:
                    self._semantic.build(all_chunks)

                self._save_cache()

            self._search = HybridSearch(self._bm25, self._semantic)
            self._built = True
            logger.info(
                "obsidian-index: indexed %d notes (%d chunks, embed=%s)",
                len(notes),
                len(all_chunks),
                type(self._embed).__name__ if self._embed else "none",
            )

    def _index_note(self, note: Note, *, rebuild: bool = True) -> None:
        ps = str(note.path)
        self._registry.remove(ps)
        # Chunk by heading (max 1200 chars each)
        header = f"# {note.title}\ntags: {', '.join(note.tags)}\n\n"
        chunks_text = note.chunk_by_heading(max_chars=1200)
        if not chunks_text:
            chunks_text = [note.body[:1200]] if note.body else []
        full_chunks = [(0, header + c) for c in chunks_text] if chunks_text else [(0, header)]
        added = self._registry.add(ps, [t for _, t in full_chunks])

        if rebuild:
            all_chunks = self._registry.all_chunks()
            self._bm25.build(all_chunks)
            if self._semantic is not None and self._embed is not None:
                for cid, text in added:
                    self._semantic.update_chunk(cid, text)
                self._search = HybridSearch(self._bm25, self._semantic)

    # ------------------------------------------------------------------
    # Incremental updates (called by watcher)
    # ------------------------------------------------------------------

    def update_note(self, path: Path) -> None:
        with self._lock:
            note = self._reader._load(path)
            self._index_note(note, rebuild=True)
            self._graph.add_note(note)
            self._sigs[str(path)] = _file_sig(path)
            if self._embed:
                self._save_cache()

    def remove_note(self, path: Path) -> None:
        with self._lock:
            ps = str(path)
            removed_ids = self._registry.remove(ps)
            if self._semantic is not None:
                self._semantic.remove_chunks(removed_ids)
            all_chunks = self._registry.all_chunks()
            self._bm25.build(all_chunks)
            self._sigs.pop(ps, None)
            if self._embed:
                self._save_cache()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, top_k: int = 8) -> List[str]:
        """Return formatted note snippets ranked by relevance."""
        with self._lock:
            if self._search is None:
                return []
            results = self._search.search(query)
            seen_paths: set[str] = set()
            formatted: List[str] = []
            for cid, _ in results[:top_k]:
                path_str = self._registry.path_of(cid)
                if path_str in seen_paths:
                    continue
                seen_paths.add(path_str)
                note = self._reader._load(Path(path_str))
                text = format_note_for_context(note, self.vault, max_chars=600)
                formatted.append(text)
                # Graph augmentation: include top backlink
                neighbours = self._graph.backlinks(note.title)
                if neighbours:
                    n_path = self._graph.path_for(neighbours[0])
                    if n_path and str(n_path) not in seen_paths:
                        n_note = self._reader._load(n_path)
                        formatted.append(format_note_for_context(n_note, self.vault, max_chars=300))
                        seen_paths.add(str(n_path))
            return formatted

    def graph_context(self, title: str, *, depth: int = 2) -> List[str]:
        """Return notes linked to/from `title` within `depth` hops."""
        with self._lock:
            neighbours = self._graph.neighbours(title, depth=depth)
            results = []
            for stem in neighbours[:6]:
                p = self._graph.path_for(stem)
                if p and p.exists():
                    note = self._reader._load(p)
                    results.append(format_note_for_context(note, self.vault, max_chars=400))
            return results

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self) -> Tuple[Optional[np.ndarray], dict, dict]:
        idx_dir = self._index_dir
        try:
            vecs_path = idx_dir / _VECS_FILE
            ids_path = idx_dir / _IDS_FILE
            meta_path = idx_dir / _META_FILE
            if not (vecs_path.exists() and ids_path.exists() and meta_path.exists()):
                return None, {}, {}
            vecs = np.load(str(vecs_path))
            with open(ids_path) as f:
                ids = json.load(f)
            with open(meta_path) as f:
                meta = json.load(f)
            sigs = meta.get("sigs", {})
            return vecs, ids, sigs
        except Exception as exc:
            logger.debug("obsidian-index: cache load failed: %s", exc)
            return None, {}, {}

    def _save_cache(self) -> None:
        if self._semantic is None or self._semantic._matrix is None:
            return
        try:
            idx_dir = self._index_dir
            idx_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(idx_dir / _VECS_FILE), self._semantic._matrix)
            ids_map = {
                str(cid): (self._registry.path_of(cid), self._registry.text(cid))
                for cid in self._semantic._ids
            }
            with open(idx_dir / _IDS_FILE, "w") as f:
                json.dump(ids_map, f)
            with open(idx_dir / _META_FILE, "w") as f:
                json.dump({"sigs": self._sigs}, f)
        except Exception as exc:
            logger.warning("obsidian-index: cache save failed: %s", exc)
