#!/usr/bin/env python3
"""
Nexus Store — Persistent document storage with DualRetriever.

Provides:
- add/remove/update documents with metadata
- BM25 + ChromaDB semantic + keyword overlap late fusion at search time
- Wing/room organization (like MemPalace)
- Persistent ChromaDB backend

Usage:
    from .store import MemoryStore
    store = MemoryStore("~/.hermes/nexus")
    store.add("Marius prefers Norwegian", wing="user", room="preferences")
    results = store.search("what does Marius prefer")
"""

import chromadb
import datetime
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# =============================================================================
# BM25
# =============================================================================

class BM25:
    """BM25 scorer with O(1) per-document scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_doc_freqs: dict[str, int] = {}
        self.avgdl: float = 0
        self.N: int = 0
        self.doc_lengths: list[int] = []
        self.doc_term_freqs: list[dict[str, int]] = []

    def fit(self, documents: list[str]):
        self.N = len(documents)
        self.term_doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.doc_term_freqs = []
        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            for term in tf:
                self.term_doc_freqs[term] += 1
        self.avgdl = total_len / max(self.N, 1)

    def score_batch(self, query: str) -> list[float]:
        tokens = self._tokenize(query)
        scores = []
        for doc_idx in range(self.N):
            doc_tf = self.doc_term_freqs[doc_idx]
            doc_len = self.doc_lengths[doc_idx]
            score = 0.0
            for term in tokens:
                if term in doc_tf:
                    tf = doc_tf[term]
                    df = self.term_doc_freqs[term]
                    idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * numerator / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b[a-z0-9]{3,}\b', text.lower())


# =============================================================================
# Stop words & keyword extraction
# =============================================================================

STOP_WORDS = {
    'what', 'when', 'where', 'who', 'how', 'which', 'did', 'do', 'was', 'were',
    'have', 'has', 'had', 'is', 'are', 'the', 'a', 'an', 'my', 'me', 'i',
    'you', 'your', 'their', 'it', 'its', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'ago', 'last', 'that', 'this', 'there',
    'about', 'get', 'got', 'give', 'gave', 'buy', 'bought', 'made', 'make',
}


def extract_keywords(text: str) -> list[str]:
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]


def keyword_overlap(query: str, document: str) -> float:
    query_kws = set(extract_keywords(query))
    if not query_kws:
        return 0.0
    doc_lower = document.lower()
    hits = sum(1 for kw in query_kws if kw in doc_lower)
    return hits / len(query_kws)


# =============================================================================
# Fusion
# =============================================================================

def late_fusion_rrf(scores_list: list[list[float]], k: int = 60) -> list[float]:
    fused = [0.0] * len(scores_list[0])
    for scores in scores_list:
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_idx, _) in enumerate(indexed_scores, 1):
            fused[doc_idx] += 1.0 / (k + rank)
    return fused


def normalize_scores(scores: list[float]) -> list[float]:
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


# =============================================================================
# MemoryStore
# =============================================================================

class MemoryStore:
    """
    Persistent memory store with BM25 + semantic late fusion.

    Uses ChromaDB for semantic search, BM25 for keyword search,
    and keyword overlap as a third signal — all fused with RRF.

    Corpus is kept in-memory for BM25 scoring, while ChromaDB
    provides persistent storage and semantic search.
    """

    def __init__(
        self,
        palace_path: str = "~/.hermes/nexus",
        embed_fn=None,  # ChromaDB embedding function (default uses built-in)
    ):
        self.palace_path = os.path.expanduser(palace_path)
        self.embed_fn = embed_fn
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._semantic_col: Optional[chromadb.Collection] = None
        self._kg_col: Optional[chromadb.Collection] = None
        self._bm25 = BM25()
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []
        self._metadata: list[dict] = []
        self._loaded = False

    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------

    def _ensure_client(self):
        if self._chroma_client is None:
            os.makedirs(self.palace_path, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.palace_path)

    def _get_semantic_col(self) -> chromadb.Collection:
        self._ensure_client()
        if self._semantic_col is None:
            try:
                self._semantic_col = self._chroma_client.get_collection("hermes_drawers")
            except Exception:
                self._semantic_col = self._chroma_client.create_collection(
                    "hermes_drawers",
                    embedding_function=self.embed_fn,
                )
        return self._semantic_col

    def _get_kg_col(self) -> chromadb.Collection:
        self._ensure_client()
        if self._kg_col is None:
            try:
                self._kg_col = self._chroma_client.get_collection("hermes_kg")
            except Exception:
                self._kg_col = self._chroma_client.create_collection("hermes_kg")
        return self._kg_col

    def load(self):
        """Load all documents from ChromaDB into BM25 index."""
        col = self._get_semantic_col()
        all_data = col.get(include=["documents", "metadatas"])

        self._corpus = []
        self._corpus_ids = []
        self._metadata = []

        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            self._corpus.append(doc)
            self._corpus_ids.append(meta.get("doc_id", "unknown"))
            self._metadata.append(meta)

        if self._corpus:
            self._bm25.fit(self._corpus)

        self._loaded = True
        return self

    # --------------------------------------------------------------------------
    # Document operations
    # --------------------------------------------------------------------------

    def add(
        self,
        text: str,
        wing: str,
        room: str,
        source: str = "manual",
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Add a document to the memory store.
        Returns the document ID.
        """
        if not self._loaded:
            self.load()

        col = self._get_semantic_col()
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        metadata = {
            "doc_id": doc_id,
            "wing": wing,
            "room": room,
            "source": source,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        col.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata],
        )

        # Update in-memory corpus for BM25
        self._corpus.append(text)
        self._corpus_ids.append(doc_id)
        self._metadata.append(metadata)
        self._bm25.fit(self._corpus)

        return doc_id

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if found and deleted."""
        col = self._get_semantic_col()
        try:
            col.delete(ids=[doc_id])
        except Exception:
            return False

        # Rebuild in-memory index
        idx = self._corpus_ids.index(doc_id) if doc_id in self._corpus_ids else -1
        if idx >= 0:
            self._corpus.pop(idx)
            self._corpus_ids.pop(idx)
            self._metadata.pop(idx)
            if self._corpus:
                self._bm25.fit(self._corpus)

        return True

    def update(self, doc_id: str, text: str) -> bool:
        """Update document text. Returns True if found and updated."""
        col = self._get_semantic_col()
        try:
            col.update(documents=[text], ids=[doc_id])
        except Exception:
            return False

        idx = self._corpus_ids.index(doc_id) if doc_id in self._corpus_ids else -1
        if idx >= 0:
            self._corpus[idx] = text
            self._bm25.fit(self._corpus)

        return True

    # --------------------------------------------------------------------------
    # Search — dual retriever (BM25 + semantic + keyword overlap)
    # --------------------------------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> list[dict]:
        """
        Search using BM25 + ChromaDB semantic + keyword overlap late fusion.

        Returns top-n results sorted by fused RRF score.
        """
        if not self._loaded:
            self.load()

        n = len(self._corpus)
        if n == 0:
            return []

        # 1. BM25 scores
        bm25_scores = self._bm25.score_batch(query)

        # 2. ChromaDB semantic search
        col = self._get_semantic_col()
        where_filter = {}
        if wing and room:
            where_filter = {"$and": [{"wing": wing}, {"room": room}]}
        elif wing:
            where_filter = {"wing": wing}
        elif room:
            where_filter = {"room": room}

        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results * 2, n),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = col.query(**kwargs)
        chroma_ids = results["ids"][0]
        chroma_dists = results["distances"][0]

        # Map ChromaDB results back to full corpus indices
        sem_scores = [0.0] * n
        for rid, dist in zip(chroma_ids, chroma_dists):
            # ChromaDB IDs are our doc_ids
            if rid in self._corpus_ids:
                doc_idx = self._corpus_ids.index(rid)
                sem_scores[doc_idx] = 1.0 - dist  # convert L2 distance to similarity

        # 3. Keyword overlap
        kw_scores = [keyword_overlap(query, doc) for doc in self._corpus]

        # 4. RRF fusion of normalized scores
        fused = late_fusion_rrf([
            normalize_scores(bm25_scores),
            normalize_scores(sem_scores),
            normalize_scores(kw_scores),
        ])

        # 5. Sort and return top-n
        indexed = list(enumerate(fused))
        indexed.sort(key=lambda x: x[1], reverse=True)

        hits = []
        for doc_idx, score in indexed[:n_results]:
            meta = self._metadata[doc_idx]
            hits.append({
                "doc_id": self._corpus_ids[doc_idx],
                "text": self._corpus[doc_idx],
                "score": round(score, 4),
                "bm25": round(bm25_scores[doc_idx], 4),
                "semantic": round(sem_scores[doc_idx], 4),
                "keyword_overlap": round(kw_scores[doc_idx], 4),
                "wing": meta.get("wing", "unknown"),
                "room": meta.get("room", "unknown"),
                "source": meta.get("source", "unknown"),
                "timestamp": meta.get("timestamp", ""),
            })

        return hits

    # --------------------------------------------------------------------------
    # List / stats
    # --------------------------------------------------------------------------

    def list_wings(self) -> list[str]:
        """Return all unique wings."""
        if not self._loaded:
            self.load()
        return sorted(set(m.get("wing", "unknown") for m in self._metadata))

    def list_rooms(self, wing: Optional[str] = None) -> list[str]:
        """Return all unique rooms, optionally filtered by wing."""
        if not self._loaded:
            self.load()
        rooms = set()
        for m in self._metadata:
            if wing is None or m.get("wing") == wing:
                rooms.add(m.get("room", "unknown"))
        return sorted(rooms)

    def count(self) -> int:
        """Total number of documents."""
        if not self._loaded:
            self.load()
        return len(self._corpus)

    def stats(self) -> dict:
        """Return store statistics."""
        if not self._loaded:
            self.load()
        wings = self.list_wings()
        by_wing = {}
        for w in wings:
            by_wing[w] = len([m for m in self._metadata if m.get("wing") == w])
        return {
            "total": len(self._corpus),
            "wings": wings,
            "by_wing": by_wing,
            "palace_path": self.palace_path,
        }

    # --------------------------------------------------------------------------
    # Knowledge graph (simple subject-predicate-object triples)
    # --------------------------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> str:
        """Store a structured fact (subject-predicate-object)."""
        col = self._get_kg_col()
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        metadata = {
            "subject": subject,
            "predicate": predicate,
            "valid_from": valid_from or datetime.datetime.now().isoformat(),
            "valid_to": valid_to or "",
        }
        col.add(
            documents=[obj],
            ids=[fact_id],
            metadatas=[metadata],
        )
        return fact_id

    def query_facts(self, subject: str) -> list[dict]:
        """Query all facts about a subject."""
        col = self._get_kg_col()
        try:
            results = col.get(
                where={"subject": subject},
                include=["documents", "metadatas"],
            )
            facts = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                facts.append({
                    "subject": meta.get("subject", subject),
                    "predicate": meta.get("predicate", "unknown"),
                    "object": doc,
                    "valid_from": meta.get("valid_from"),
                    "valid_to": meta.get("valid_to"),
                })
            return facts
        except Exception:
            return []


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nexus Store CLI")
    parser.add_argument("--palace", default="~/.hermes/nexus", help="Palace path")
    sub = parser.add_subparsers(dest="cmd")

    add_p = sub.add_parser("add")
    add_p.add_argument("--text", required=True)
    add_p.add_argument("--wing", required=True)
    add_p.add_argument("--room", required=True)
    add_p.add_argument("--source", default="manual")

    sub.add_parser("stats")
    sub.add_parser("list-wings")

    args = parser.parse_args()

    store = MemoryStore(args.palace)

    if args.cmd == "add":
        doc_id = store.add(args.text, args.wing, args.room, args.source)
        print(f"Added: {doc_id}")
    elif args.cmd == "stats":
        import json
        print(json.dumps(store.stats(), indent=2))
    elif args.cmd == "list-wings":
        print(", ".join(store.list_wings()))
    else:
        parser.print_help()
