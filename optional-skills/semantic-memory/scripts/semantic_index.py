#!/usr/bin/env python3
"""
Semantic Index — Phase 3: Local embeddings for the memory engine.

Uses fastembed (BAAI/bge-small-en-v1.5) for local, free, zero-token embeddings.
384 dimensions, ~6ms per embedding.

Integrates with quantum_index as Phase 3 of the retriever:
  Phase 1: Keyword bitmap (O(1), 0 tokens, ~50μs)
  Phase 2: BM25 (0 tokens, ~5ms)
  Phase 3: Semantic similarity (0 tokens, ~6ms) ← THIS

Storage: embeddings stored in SQLite as binary blobs (1.5KB per fact).
Search: brute-force cosine similarity (fast enough for <100K facts).
"""

import json
import sqlite3
import struct
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"

# Lazy-load model (heavy import)
_model = None

def get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS semantic_embeddings (
    fact_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    text_hash TEXT,
    model TEXT DEFAULT 'bge-small-en-v1.5',
    dimensions INTEGER DEFAULT 384,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_se_fact ON semantic_embeddings(fact_id);
"""


@dataclass
class SemanticResult:
    fact_id: str
    summary: str
    similarity: float
    keywords: Dict
    status: str
    tier: str


class SemanticIndex:
    """Local embedding index with cosine similarity search."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.conn = None
        self.dims = 384

    def connect(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    # ── Embedding helpers ────────────────────────────────────────

    @staticmethod
    def encode_texts(texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a list of texts. Local, free."""
        model = get_model()
        return list(model.embed(texts))

    @staticmethod
    def encode_one(text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        model = get_model()
        return list(model.embed([text]))[0]

    @staticmethod
    def vec_to_blob(vec: np.ndarray) -> bytes:
        """Convert numpy vector to binary blob for SQLite storage."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vec(blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy vector."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    # ── Index operations ─────────────────────────────────────────

    def embed_fact(self, fact_id: str, text: str) -> bool:
        """Generate and store embedding for a single fact."""
        # Check if already embedded
        existing = self.conn.execute(
            "SELECT 1 FROM semantic_embeddings WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if existing:
            return False

        vec = self.encode_one(text)
        blob = self.vec_to_blob(vec)

        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()[:12]

        self.conn.execute(
            "INSERT OR REPLACE INTO semantic_embeddings (fact_id, embedding, text_hash) VALUES (?, ?, ?)",
            (fact_id, blob, text_hash)
        )
        self.conn.commit()
        return True

    def embed_all_facts(self, batch_size: int = 50) -> Dict:
        """Embed all quantum_facts that don't have embeddings yet."""
        # Get facts without embeddings
        rows = self.conn.execute("""
            SELECT qf.id, qf.summary
            FROM quantum_facts qf
            LEFT JOIN semantic_embeddings se ON qf.id = se.fact_id
            WHERE se.fact_id IS NULL
        """).fetchall()

        if not rows:
            return {"embedded": 0, "total": 0, "message": "All facts already embedded"}

        total = len(rows)
        embedded = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]
            texts = [r[1] for r in batch]
            ids = [r[0] for r in batch]

            # Batch embed
            vecs = self.encode_texts(texts)

            for fid, vec, text in zip(ids, vecs, texts):
                blob = self.vec_to_blob(vec)
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()[:12]

                self.conn.execute(
                    "INSERT OR REPLACE INTO semantic_embeddings (fact_id, embedding, text_hash) VALUES (?, ?, ?)",
                    (fid, blob, text_hash)
                )
                embedded += 1

            self.conn.commit()

        return {"embedded": embedded, "total": total}

    # ── Search ───────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[SemanticResult]:
        """
        Semantic search: embed query, compare against all stored embeddings.
        Brute-force cosine — fast enough for <100K facts.
        """
        query_vec = self.encode_one(query)

        # Get all embeddings + fact metadata
        rows = self.conn.execute("""
            SELECT se.fact_id, se.embedding, qf.summary, qf.keywords_readable,
                   qf.status, qf.storage_tier
            FROM semantic_embeddings se
            JOIN quantum_facts qf ON se.fact_id = qf.id
        """).fetchall()

        if not rows:
            return []

        # Score all facts
        scored = []
        for fid, blob, summary, kw_json, status, tier in rows:
            vec = self.blob_to_vec(blob)
            sim = self.cosine_similarity(query_vec, vec)
            if sim >= min_similarity:
                keywords = json.loads(kw_json) if kw_json else {}
                scored.append(SemanticResult(
                    fact_id=fid,
                    summary=summary,
                    similarity=sim,
                    keywords=keywords,
                    status=status or "unknown",
                    tier=tier or "warm",
                ))

        # Sort by similarity descending
        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_k]

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Embedding index statistics."""
        total_embeddings = self.conn.execute(
            "SELECT COUNT(*) FROM semantic_embeddings"
        ).fetchone()[0]

        total_facts = self.conn.execute(
            "SELECT COUNT(*) FROM quantum_facts"
        ).fetchone()[0]

        # Storage size
        storage = self.conn.execute(
            "SELECT SUM(LENGTH(embedding)) FROM semantic_embeddings"
        ).fetchone()[0] or 0

        return {
            "total_embeddings": total_embeddings,
            "total_facts": total_facts,
            "coverage": f"{total_embeddings}/{total_facts} ({100*total_embeddings/max(total_facts,1):.0f}%)",
            "storage_bytes": storage,
            "storage_kb": f"{storage/1024:.1f}KB",
            "bytes_per_embedding": 384 * 4,  # float32
            "model": "BAAI/bge-small-en-v1.5",
            "dimensions": 384,
        }


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Index (Phase 3)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("backfill", help="Embed all facts that don't have embeddings")

    p = sub.add_parser("search", help="Semantic search")
    p.add_argument("query", nargs="+")
    p.add_argument("--top-k", type=int, default=5)

    sub.add_parser("stats", help="Embedding stats")

    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    idx = SemanticIndex()
    idx.connect()

    try:
        if args.command == "backfill":
            start = time.time()
            result = idx.embed_all_facts()
            elapsed = time.time() - start
            print(f"  Embedded: {result['embedded']} facts in {elapsed:.1f}s")
            if result['embedded'] > 0:
                print(f"  Rate: {result['embedded']/elapsed:.0f} facts/sec")

        elif args.command == "search":
            query = " ".join(args.query)
            start = time.time()
            results = idx.search(query, top_k=args.top_k)
            elapsed = time.time() - start

            if args.json:
                print(json.dumps([{
                    "id": r.fact_id, "summary": r.summary,
                    "similarity": round(r.similarity, 4),
                    "status": r.status, "keywords": r.keywords,
                } for r in results], indent=2))
            else:
                print(f"\n  SEMANTIC SEARCH: {query} ({elapsed*1000:.0f}ms)")
                print(f"  {'─' * 55}\n")
                for i, r in enumerate(results, 1):
                    bar = "█" * int(r.similarity * 20)
                    print(f"  [{i}] {r.summary}")
                    print(f"      Similarity: {r.similarity:.3f} {bar}")
                    print(f"      Status: {r.status} | Tier: {r.tier}")
                    domains = r.keywords.get("domain", [])
                    if domains:
                        print(f"      Domain: {', '.join(domains)}")
                    print()

        elif args.command == "stats":
            stats = idx.get_stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"\n  SEMANTIC INDEX STATS")
                print(f"  {'─' * 55}")
                print(f"  Coverage:    {stats['coverage']}")
                print(f"  Storage:     {stats['storage_kb']}")
                print(f"  Model:       {stats['model']}")
                print(f"  Dimensions:  {stats['dimensions']}")
                print()

        else:
            parser.print_help()

    finally:
        idx.close()


if __name__ == "__main__":
    main()
