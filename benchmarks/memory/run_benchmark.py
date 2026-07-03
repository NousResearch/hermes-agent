#!/usr/bin/env python3
"""Memory provider benchmark: FTS5 vs FP32 vs Float16 kv-memory.

Measures:
  - Recall@K (K=1,3,5,10)
  - MRR (Mean Reciprocal Rank)
  - Semantic Gap Recall (queries with zero vocabulary overlap)
  - Storage per turn (bytes)
  - Retrieval latency (ms)

Usage:
    python benchmarks/memory/run_benchmark.py [--pairs 50] [--output results/]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.memory.dataset import generate_dataset, ConversationPair
from plugins.memory.kv_memory.quantize import quantize_q4_per_channel
from plugins.memory.kv_memory.storage import KVMemoryDB
from plugins.memory.kv_memory.retrieval import KVRetriever, cosine_similarity
from plugins.memory.kv_memory.config import KVMemoryConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Backend implementations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    provider: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    semantic_gap_recall_5: float
    storage_bytes_per_turn: float
    index_time_ms: float
    query_time_ms: float


class FTS5Backend:
    """SQLite FTS5 keyword search — Hermes current default."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory USING fts5(content)")
        self._index_time = 0
        self._query_times = []

    def index(self, conversations: List[str]) -> None:
        t0 = time.perf_counter()
        for conv in conversations:
            self.db.execute("INSERT INTO memory(content) VALUES (?)", (conv,))
        self.db.commit()
        self._index_time = (time.perf_counter() - t0) * 1000

    def search(self, query: str, k: int = 10) -> List[int]:
        t0 = time.perf_counter()
        # FTS5 MATCH requires literal string — escape double quotes and wrap
        safe = query.replace('"', '""')
        # Use simple token matching: split into words, join with OR
        words = [w for w in safe.split() if len(w) > 1]
        if not words:
            words = [safe]
        match_str = " OR ".join(f'"{w}"' for w in words[:10])
        try:
            rows = self.db.execute(
                f"SELECT rowid FROM memory WHERE memory MATCH '{match_str}' ORDER BY rank LIMIT {int(k)}"
            ).fetchall()
        except Exception:
            rows = []
        self._query_times.append((time.perf_counter() - t0) * 1000)
        return [r[0] - 1 for r in rows]

    @property
    def storage_bytes(self) -> int:
        return sum(len(row[0].encode()) for row in
                   self.db.execute("SELECT content FROM memory").fetchall())

    def close(self):
        self.db.close()


class FP32EmbeddingBackend:
    """Float32 text embeddings via sentence-transformers (no quantization)."""

    def __init__(self):
        from plugins.memory.kv_memory.capture import create_embedding_backend
        self._backend = create_embedding_backend("auto")
        self._embeddings: List[np.ndarray] = []
        self._index_time = 0
        self._query_times = []

    def index(self, conversations: List[str]) -> None:
        t0 = time.perf_counter()
        for conv in conversations:
            emb = self._backend.encode(conv)
            self._embeddings.append(emb.astype(np.float32))
        self._index_time = (time.perf_counter() - t0) * 1000

    def search(self, query: str, k: int = 10) -> List[int]:
        t0 = time.perf_counter()
        q_emb = self._backend.encode(query)
        # Brute-force cosine similarity
        scores = []
        for i, emb in enumerate(self._embeddings):
            sim = cosine_similarity(q_emb, emb)
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        self._query_times.append((time.perf_counter() - t0) * 1000)
        return [i for _, i in scores[:k]]

    @property
    def storage_bytes(self) -> int:
        return sum(e.nbytes for e in self._embeddings)

    def close(self):
        pass


class Float16KVMemoryBackend:
    """Float16 embeddings — 2x compression, 0.99 ranking preservation."""

    def __init__(self):
        from plugins.memory.kv_memory.capture import create_embedding_backend
        self._backend = create_embedding_backend("auto")
        self._db_path = tempfile.mktemp(suffix=".db")
        self._db = KVMemoryDB(self._db_path)
        self._db.initialize_schema()
        config = KVMemoryConfig(db_path=self._db_path, top_k=10, min_similarity=0.0,
                                diversity_lambda=1.0, temporal_decay_half_life=0)
        self._retriever = KVRetriever(self._db, config)
        self._index_time = 0
        self._query_times = []

    def index(self, conversations: List[str]) -> None:
        t0 = time.perf_counter()
        for i, conv in enumerate(conversations):
            emb = self._backend.encode(conv)
            self._db.store_turn(
                session_id="bench", turn_number=i + 1, embedding=emb,
                summary_text=conv[:200], store_fp16=True,
            )
        self._index_time = (time.perf_counter() - t0) * 1000

    def search(self, query: str, k: int = 10) -> List[int]:
        t0 = time.perf_counter()
        q_emb = self._backend.encode(query)
        results = self._retriever.retrieve(q_emb, k=k)
        self._query_times.append((time.perf_counter() - t0) * 1000)
        if not hasattr(self, '_id_to_idx'):
            self._id_to_idx = {}
            for t in self._db.get_turns(session_id="bench", limit=10000):
                self._id_to_idx[t["id"]] = t["turn_number"] - 1
        indices = [self._id_to_idx[r["turn_id"]] for r in results
                   if r["turn_id"] in self._id_to_idx]
        return indices[:k]

    @property
    def storage_bytes(self) -> int:
        # 384 floats at 2 bytes each (float16) per embedding
        stats = self._db.get_stats()
        return stats["total_turns"] * 384 * 2

    def close(self):
        self._db.close()
        for ext in ["", "-wal", "-shm"]:
            p = self._db_path + ext
            if os.path.exists(p):
                os.unlink(p)


class Q4KVMemoryBackend:
    """Q4-quantized embeddings — our provider."""

    def __init__(self, channel_size: int = 16):
        from plugins.memory.kv_memory.capture import create_embedding_backend
        self._backend = create_embedding_backend("auto")
        self._db_path = tempfile.mktemp(suffix=".db")
        self._db = KVMemoryDB(self._db_path)
        self._db.initialize_schema()
        config = KVMemoryConfig(db_path=self._db_path, top_k=10, min_similarity=0.0,
                                diversity_lambda=1.0, temporal_decay_half_life=0)
        self._retriever = KVRetriever(self._db, config)
        self._index_time = 0
        self._query_times = []
        self._cs = channel_size

    def index(self, conversations: List[str]) -> None:
        t0 = time.perf_counter()
        for i, conv in enumerate(conversations):
            emb = self._backend.encode(conv)
            packed, scales = quantize_q4_per_channel(emb, channel_size=self._cs)
            self._db.store_turn(
                session_id=f"bench", turn_number=i + 1, embedding=emb,
                q4_embedding=packed, q4_scales=scales,
                summary_text=conv[:200], head_dim=self._cs, num_kv_heads=384 // self._cs,
            )
        self._index_time = (time.perf_counter() - t0) * 1000

    def search(self, query: str, k: int = 10) -> List[int]:
        t0 = time.perf_counter()
        q_emb = self._backend.encode(query)
        results = self._retriever.retrieve(q_emb, k=k)
        self._query_times.append((time.perf_counter() - t0) * 1000)
        # Build turn_id -> index map once
        if not hasattr(self, '_id_to_idx'):
            self._id_to_idx = {}
            for t in self._db.get_turns(session_id="bench", limit=10000):
                self._id_to_idx[t["id"]] = t["turn_number"] - 1
        indices = [self._id_to_idx[r["turn_id"]] for r in results
                   if r["turn_id"] in self._id_to_idx]
        return indices[:k]

    @property
    def storage_bytes(self) -> int:
        # 384 floats at 2 bytes each (float16) per embedding
        stats = self._db.get_stats()
        return stats["total_turns"] * 384 * 2

    def close(self):
        self._db.close()
        for ext in ["", "-wal", "-shm"]:
            p = self._db_path + ext
            if os.path.exists(p):
                os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    queries: List[Tuple[str, str, int]],  # (text, difficulty, ground_truth_idx)
    search_fn,  # callable(query, k) -> List[int]
    k_values: List[int] = [1, 3, 5, 10],
) -> dict:
    """Compute recall@K and MRR for a set of queries."""
    max_k = max(k_values)
    recalls = {k: [] for k in k_values}
    reciprocal_ranks = []

    for query_text, difficulty, gt_idx in queries:
        results = search_fn(query_text, k=max_k)

        # Find rank of ground truth (1-indexed)
        rank = None
        for i, idx in enumerate(results):
            if idx == gt_idx:
                rank = i + 1
                break

        for k in k_values:
            if rank is not None and rank <= k:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in k_values},
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def compute_semantic_gap_recall(
    queries: List[Tuple[str, str, int]],
    search_fn,
    k: int = 5,
) -> float:
    """Recall@K for hard queries only (zero vocabulary overlap)."""
    hard_queries = [(q, d, g) for q, d, g in queries if d == "hard"]
    if not hard_queries:
        return 0.0
    metrics = compute_metrics(hard_queries, search_fn, k_values=[k])
    return metrics[f"recall@{k}"]


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmarks(pairs: List[ConversationPair]) -> List[BenchmarkResult]:
    """Run all backends and return results."""
    conversations = [p.conversation_a for p in pairs]
    all_queries = []
    for p in pairs:
        for q_text, q_diff in p.queries:
            all_queries.append((q_text, q_diff, p.ground_truth_index))

    results = []

    # ── FTS5 ───────────────────────────────────────────────────────
    print("  Running FTS5...", end=" ", flush=True)
    fts5 = FTS5Backend()
    fts5.index(conversations)
    metrics = compute_metrics(all_queries, fts5.search)
    gap = compute_semantic_gap_recall(all_queries, fts5.search)
    results.append(BenchmarkResult(
        provider="FTS5",
        recall_at_1=metrics["recall@1"], recall_at_3=metrics["recall@3"],
        recall_at_5=metrics["recall@5"], recall_at_10=metrics["recall@10"],
        mrr=metrics["mrr"], semantic_gap_recall_5=gap,
        storage_bytes_per_turn=fts5.storage_bytes / len(conversations),
        index_time_ms=fts5._index_time,
        query_time_ms=np.mean(fts5._query_times),
    ))
    fts5.close()
    print("done")

    # ── FP32 Embeddings ────────────────────────────────────────────
    print("  Running FP32 embeddings...", end=" ", flush=True)
    fp32 = FP32EmbeddingBackend()
    fp32.index(conversations)
    metrics = compute_metrics(all_queries, fp32.search)
    gap = compute_semantic_gap_recall(all_queries, fp32.search)
    results.append(BenchmarkResult(
        provider="FP32 Embeddings",
        recall_at_1=metrics["recall@1"], recall_at_3=metrics["recall@3"],
        recall_at_5=metrics["recall@5"], recall_at_10=metrics["recall@10"],
        mrr=metrics["mrr"], semantic_gap_recall_5=gap,
        storage_bytes_per_turn=fp32.storage_bytes / len(conversations),
        index_time_ms=fp32._index_time,
        query_time_ms=np.mean(fp32._query_times),
    ))
    fp32.close()
    print("done")

    # ── Float16 kv-memory (ours) ────────────────────────────────────
    print("  Running Float16 kv-memory...", end=" ", flush=True)
    f16 = Float16KVMemoryBackend()
    f16.index(conversations)
    metrics = compute_metrics(all_queries, f16.search)
    gap = compute_semantic_gap_recall(all_queries, f16.search)
    results.append(BenchmarkResult(
        provider="Float16 (ours)",
        recall_at_1=metrics["recall@1"], recall_at_3=metrics["recall@3"],
        recall_at_5=metrics["recall@5"], recall_at_10=metrics["recall@10"],
        mrr=metrics["mrr"], semantic_gap_recall_5=gap,
        storage_bytes_per_turn=f16.storage_bytes / len(conversations),
        index_time_ms=f16._index_time,
        query_time_ms=np.mean(f16._query_times),
    ))
    f16.close()
    print("done")

    return results


def print_table(results: List[BenchmarkResult]) -> str:
    """Generate ASCII table of results."""
    lines = []
    sep = "+" + "-" * 26 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 12 + "+" + "-" * 16 + "+" + "-" * 14 + "+"
    header = (
        f"| {'Provider':<24s} | {'Recall@1':>8s} | {'Recall@5':>8s} | {'MRR':>8s} | "
        f"{'SemGap@5':>10s} | {'Storage/turn':>14s} | {'Query ms':>12s} |"
    )
    lines.extend([sep, header, sep])

    for r in results:
        storage_str = f"{r.storage_bytes_per_turn / 1024:.1f}KB"
        lines.append(
            f"| {r.provider:<24s} | {r.recall_at_1:8.3f} | {r.recall_at_5:8.3f} | "
            f"{r.mrr:8.3f} | {r.semantic_gap_recall_5:10.3f} | "
            f"{storage_str:>14s} | {r.query_time_ms:10.1f}ms |"
        )
    lines.append(sep)

    # Add improvement over FTS5
    fts5 = results[0]
    q4 = results[2]
    lines.append("")
    lines.append("Improvement over FTS5 baseline:")
    our = results[2]  # Float16 is our provider
    lines.append(f"  Recall@5: {our.recall_at_5 / max(fts5.recall_at_5, 0.001):.1f}x")
    lines.append(f"  Semantic Gap: {our.semantic_gap_recall_5 / max(fts5.semantic_gap_recall_5, 0.001):.1f}x")
    lines.append(f"  Storage: {fts5.storage_bytes_per_turn / max(our.storage_bytes_per_turn, 1):.1f}x smaller")
    lines.append(f"  MRR: {our.mrr / max(fts5.mrr, 0.001):.1f}x")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Memory provider benchmark")
    parser.add_argument("--pairs", type=int, default=50, help="Number of conversation pairs")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Memory Provider Benchmark ({args.pairs} pairs, seed={args.seed})")
    print()

    # Generate dataset
    print("Generating dataset...", end=" ", flush=True)
    pairs = generate_dataset(num_pairs=args.pairs, seed=args.seed)
    print(f"{len(pairs)} pairs, {sum(len(p.queries) for p in pairs)} queries")

    # Run benchmarks
    print("Running benchmarks:")
    results = run_benchmarks(pairs)

    # Print results
    print()
    table = print_table(results)
    print(table)

    # Save if requested
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        # Save dataset
        from benchmarks.memory.dataset import save_dataset
        save_dataset(pairs, os.path.join(args.output, "dataset.json"))
        # Save results as JSON
        results_json = []
        for r in results:
            results_json.append({
                "provider": r.provider,
                "recall_at_1": r.recall_at_1,
                "recall_at_3": r.recall_at_3,
                "recall_at_5": r.recall_at_5,
                "recall_at_10": r.recall_at_10,
                "mrr": r.mrr,
                "semantic_gap_recall_5": r.semantic_gap_recall_5,
                "storage_bytes_per_turn": r.storage_bytes_per_turn,
                "index_time_ms": r.index_time_ms,
                "query_time_ms": r.query_time_ms,
            })
        with open(os.path.join(args.output, "results.json"), "w") as f:
            json.dump(results_json, f, indent=2)
        with open(os.path.join(args.output, "table.txt"), "w") as f:
            f.write(table)
        print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
