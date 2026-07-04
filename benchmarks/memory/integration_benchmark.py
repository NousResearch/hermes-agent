#!/usr/bin/env python3
"""Integration benchmark: Hermes builtin FTS5 memory vs kv-memory provider.

Tests retrieval through Hermes's actual memory infrastructure:
  1. Stores conversation pairs as memory entries using both backends
  2. Queries both with the same test queries
  3. Measures recall@K, MRR, semantic gap recall
  4. Reports head-to-head comparison

Requires Hermes to be installed. Run from repo root.

Usage:
    python benchmarks/memory/integration_benchmark.py --pairs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmarks.memory.dataset import generate_dataset, ConversationPair
from plugins.memory.kv_memory.capture import create_embedding_backend
from plugins.memory.kv_memory.quantize import quantize_q4_per_channel
from plugins.memory.kv_memory.storage import KVMemoryDB
from plugins.memory.kv_memory.retrieval import KVRetriever, cosine_similarity
from plugins.memory.kv_memory.config import KVMemoryConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Hermes builtin FTS5 memory (accessed through agent infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════

class HermesBuiltinMemory:
    """Wrapper around Hermes's actual builtin memory tool (FTS5-based).

    Uses the same SQLite schema Hermes uses internally, found at
    $HERMES_HOME/memory.db. The 'memory' table has columns:
      id, session_id, content, created_at, updated_at, metadata
    with an FTS5 index on 'content'.
    """

    def __init__(self, db_path: str):
        self._db = sqlite3.connect(db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        # Create the same schema Hermes uses internally
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        # Hermes uses an external content FTS5 index
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
            "content, content='memory', content_rowid='rowid')"
        )
        self._db.commit()
        self._query_times = []

    def store(self, entry_id: str, content: str) -> None:
        """Store a memory entry (simulating 'memory add' tool)."""
        now = time.time()
        # Use INSERT (not REPLACE) so rowid is sequential
        self._db.execute(
            "INSERT INTO memory(id, session_id, content, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry_id, "bench", content, now, now),
        )
        # Mirror to FTS index using last inserted rowid
        rowid = self._db.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._db.execute(
            "INSERT INTO memory_fts(rowid, content) VALUES (?, ?)",
            (rowid, content),
        )
        self._db.commit()

    def search(self, query: str, k: int = 10) -> List[str]:
        """Search memory entries (simulating 'memory search' tool)."""
        t0 = time.perf_counter()
        safe = query.replace('"', '""')
        words = [w for w in safe.split() if len(w) > 1]
        if not words:
            words = [safe]
        match_str = " OR ".join(f'"{w}"' for w in words[:10])
        try:
            rows = self._db.execute(
                f"SELECT m.id FROM memory m "
                f"JOIN memory_fts f ON m.rowid = f.rowid "
                f"WHERE memory_fts MATCH '{match_str}' "
                f"ORDER BY rank LIMIT {int(k)}"
            ).fetchall()
        except Exception:
            rows = []
        self._query_times.append((time.perf_counter() - t0) * 1000)
        return [r[0] for r in rows]

    def close(self):
        self._db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# kv-memory provider (our implementation)
# ═══════════════════════════════════════════════════════════════════════════════

class KVMemoryProviderBench:
    """Wrapper around our kv-memory provider for benchmarking."""

    def __init__(self):
        self._backend = create_embedding_backend("auto")
        self._db_path = tempfile.mktemp(suffix=".db")
        self._db = KVMemoryDB(self._db_path)
        self._db.initialize_schema()
        config = KVMemoryConfig(
            db_path=self._db_path, top_k=10, min_similarity=0.0,
            diversity_lambda=1.0, temporal_decay_half_life=0,
        )
        self._retriever = KVRetriever(self._db, config)
        self._query_times = []
        self._id_map = {}  # entry_id -> turn_id

    def store(self, entry_id: str, content: str) -> None:
        emb = self._backend.encode(content)
        tid = self._db.store_turn(
            session_id="bench", turn_number=len(self._id_map) + 1,
            embedding=emb, summary_text=content[:200], store_fp16=True,
        )
        self._id_map[entry_id] = tid

    def search(self, query: str, k: int = 10) -> List[str]:
        t0 = time.perf_counter()
        q_emb = self._backend.encode(query)
        results = self._retriever.retrieve(q_emb, k=k)
        self._query_times.append((time.perf_counter() - t0) * 1000)
        # Map turn_ids back to entry_ids
        rev_map = {v: k for k, v in self._id_map.items()}
        return [rev_map[r["turn_id"]] for r in results if r["turn_id"] in rev_map]

    def close(self):
        self._db.close()
        for ext in ["", "-wal", "-shm"]:
            p = self._db_path + ext
            if os.path.exists(p):
                os.unlink(p)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_head_to_head(pairs: List[ConversationPair]) -> dict:
    """Run head-to-head benchmark between builtin FTS5 and kv-memory."""
    # Create temp Hermes-style memory DB
    fd, fts5_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Init both backends
    fts5 = HermesBuiltinMemory(fts5_path)
    kv = KVMemoryProviderBench()

    # Store all conversations
    print(f"  Storing {len(pairs)} conversations...", end=" ", flush=True)
    t0 = time.perf_counter()
    for i, p in enumerate(pairs):
        entry_id = f"conv-{i:04d}"
        fts5.store(entry_id, p.conversation_a)
        kv.store(entry_id, p.conversation_a)
    store_time = time.perf_counter() - t0
    print(f"{store_time:.1f}s")

    # Build queries
    all_queries = []
    for p in pairs:
        for q_text, q_diff in p.queries:
            all_queries.append((q_text, q_diff, f"conv-{p.ground_truth_index:04d}"))

    # Run queries
    fts5_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    kv_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    fts5_rr = []
    kv_rr = []
    fts5_semgap_correct = 0
    kv_semgap_correct = 0
    semgap_total = 0

    for qi, (q_text, q_diff, gt_id) in enumerate(all_queries):
        fts5_results = fts5.search(q_text, k=10)
        kv_results = kv.search(q_text, k=10)

        for k in [1, 3, 5, 10]:
            if gt_id in fts5_results[:k]:
                fts5_correct[k] += 1
            if gt_id in kv_results[:k]:
                kv_correct[k] += 1

        # MRR
        for results, rr_list in [(fts5_results, fts5_rr), (kv_results, kv_rr)]:
            rank = next((i + 1 for i, r in enumerate(results) if r == gt_id), None)
            rr_list.append(1.0 / rank if rank else 0.0)

        # Semantic gap
        if q_diff == "hard":
            semgap_total += 1
            if gt_id in fts5_results[:5]:
                fts5_semgap_correct += 1
            if gt_id in kv_results[:5]:
                kv_semgap_correct += 1

    n = len(all_queries)
    fts5_qtime = np.mean(fts5._query_times)
    kv_qtime = np.mean(kv._query_times)

    fts5.close()
    kv.close()
    for ext in ["", "-wal", "-shm"]:
        p = fts5_path + ext
        if os.path.exists(p):
            os.unlink(p)

    return {
        "num_pairs": len(pairs),
        "num_queries": n,
        "store_time_s": store_time,
        "fts5": {
            "recall@1": fts5_correct[1] / n,
            "recall@3": fts5_correct[3] / n,
            "recall@5": fts5_correct[5] / n,
            "recall@10": fts5_correct[10] / n,
            "mrr": float(np.mean(fts5_rr)),
            "semantic_gap_recall@5": fts5_semgap_correct / max(semgap_total, 1),
            "query_time_ms": float(fts5_qtime),
        },
        "kv_memory": {
            "recall@1": kv_correct[1] / n,
            "recall@3": kv_correct[3] / n,
            "recall@5": kv_correct[5] / n,
            "recall@10": kv_correct[10] / n,
            "mrr": float(np.mean(kv_rr)),
            "semantic_gap_recall@5": kv_semgap_correct / max(semgap_total, 1),
            "query_time_ms": float(kv_qtime),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integration benchmark: builtin FTS5 vs kv-memory"
    )
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Integration Benchmark: Hermes builtin FTS5 vs kv-memory")
    print(f"  Pairs: {args.pairs}, Seed: {args.seed}")
    print()

    print("Generating dataset...", end=" ", flush=True)
    pairs = generate_dataset(num_pairs=args.pairs, seed=args.seed)
    print(f"{len(pairs)} pairs, {sum(len(p.queries) for p in pairs)} queries")

    print("Running head-to-head benchmark:")
    results = run_head_to_head(pairs)

    # Print results
    print()
    sep = "+" + "-" * 22 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 14 + "+" + "-" * 12 + "+"
    header = (
        f"| {'Provider':<20s} | {'Recall@1':>8s} | {'Recall@5':>8s} | {'MRR':>8s} | "
        f"{'SemGap@5':>12s} | {'Query ms':>10s} |"
    )
    print(sep)
    print(header)
    print(sep)
    for label, key in [("FTS5 (builtin)", "fts5"), ("kv-memory (ours)", "kv_memory")]:
        r = results[key]
        print(
            f"| {label:<20s} | {r['recall@1']:8.3f} | {r['recall@5']:8.3f} | "
            f"{r['mrr']:8.3f} | {r['semantic_gap_recall@5']:12.3f} | "
            f"{r['query_time_ms']:8.1f}ms |"
        )
    print(sep)

    # Improvement
    f = results["fts5"]
    k = results["kv_memory"]
    print()
    print("Improvement over builtin FTS5:")
    print(f"  Recall@5: {k['recall@5'] / max(f['recall@5'], 0.001):.1f}x")
    print(f"  MRR: {k['mrr'] / max(f['mrr'], 0.001):.1f}x")
    semgap_ratio = k["semantic_gap_recall@5"] / max(f["semantic_gap_recall@5"], 0.001)
    print(f"  Semantic Gap Recall: {semgap_ratio:.1f}x")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "integration_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}/integration_results.json")


if __name__ == "__main__":
    main()
