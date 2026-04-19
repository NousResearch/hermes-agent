#!/usr/bin/env python3
"""
Dream Worker — Standalone full-stack dream engine for MSSQL.

Reads memories from MSSQL, uses sentence-transformers for embeddings,
runs all 3 dream phases, and writes results back to MSSQL.

Usage:
    python dream_worker.py              # one-shot dream cycle
    python dream_worker.py --daemon     # background loop (idle-based)
    python dream_worker.py --phase nrem # single phase only

Config: reads from config.yaml or env vars.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure plugin dir is on path
_PLUGIN_DIR = Path(__file__).parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

logger = logging.getLogger("dream_worker")


# ---------------------------------------------------------------------------
# Embedding Provider (sentence-transformers only — full stack)
# ---------------------------------------------------------------------------

class EmbedProvider:
    """Sentence-transformers embedding provider for dream worker.

    Uses the same cache as embed_provider.py: ~/.neural_memory/models/
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    MODEL_DIR = Path.home() / ".neural_memory" / "models"

    _shared_model = None

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        # Reuse shared model if already loaded
        if EmbedProvider._shared_model is not None:
            self._model = EmbedProvider._shared_model
            return

        from sentence_transformers import SentenceTransformer

        cached = self.MODEL_DIR / f"models--sentence-transformers--{self.MODEL_NAME}"
        is_cached = cached.exists()

        if is_cached:
            logger.info("Loading %s from local cache (%s)...", self.MODEL_NAME, cached)
        else:
            logger.info("Downloading %s (first time, ~87MB)...", self.MODEL_NAME)

        self._model = SentenceTransformer(
            self.MODEL_NAME,
            cache_folder=str(self.MODEL_DIR),
            local_files_only=is_cached,
        )
        EmbedProvider._shared_model = self._model
        logger.info("Embedding model ready: %s", self.MODEL_NAME)

    def embed(self, text: str) -> List[float]:
        self._load()
        vec = self._model.encode(text, show_progress_bar=False)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load()
        vecs = self._model.encode(texts, show_progress_bar=False,
                                   batch_size=64)
        return [v.tolist() for v in vecs]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Dream Worker
# ---------------------------------------------------------------------------

class DreamWorker:
    """Full-stack dream engine operating on MSSQL with sentence-transformers."""

    def __init__(self, mssql_config: Optional[dict] = None):
        from dream_mssql_store import DreamMSSQLStore

        if mssql_config:
            self.store = DreamMSSQLStore.from_config(mssql_config)
        else:
            self.store = DreamMSSQLStore()

        self.embedder = EmbedProvider()
        self._embedding_cache: Dict[int, List[float]] = {}

    def close(self):
        self.store.close()

    # -- Embedding helpers ---------------------------------------------------

    def _get_embedding(self, memory_id: int, content: str) -> Optional[List[float]]:
        """Get embedding for a memory, with caching."""
        if memory_id in self._embedding_cache:
            return self._embedding_cache[memory_id]
        if not content or not content.strip():
            return None
        try:
            emb = self.embedder.embed(content[:512])
            self._embedding_cache[memory_id] = emb
            return emb
        except Exception as e:
            logger.debug("Embed failed for memory %d: %s", memory_id, e)
            return None

    def _similarity(self, a: List[float], b: List[float]) -> float:
        return EmbedProvider.cosine_similarity(a, b)

    # -- Phase 1: NREM -------------------------------------------------------

    def phase_nrem(self, limit: int = 100) -> Dict[str, Any]:
        """NREM: Replay recent memories, strengthen active connections.

        1. Load recent memories from MSSQL
        2. Compute embeddings batch
        3. For each memory, find connections where the OTHER end is similar
        4. Strengthen those connections (active replay reinforces them)
        5. Weak connections (< 0.1) get pruned
        """
        logger.info("NREM: loading %d recent memories...", limit)
        memories = self.store.get_recent_memories(limit)
        if not memories:
            return {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}

        session_id = self.store.start_session("nrem")
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}

        # Build embedding index
        logger.info("NREM: computing %d embeddings...", len(memories))
        contents = [m["content"][:512] for m in memories if m["content"]]
        mem_ids = [m["id"] for m in memories if m["content"]]

        embeddings = self.embedder.embed_batch(contents)
        embed_map = dict(zip(mem_ids, embeddings))

        # For each memory, find which of its connections lead to similar content
        logger.info("NREM: finding activated connections...")
        activated_edges = set()

        cursor = self.store.conn.cursor()
        for mid in mem_ids:
            if mid not in embed_map:
                continue
            mid_emb = embed_map[mid]

            # Get connections involving this memory
            cursor.execute(
                "SELECT source_id, target_id, weight FROM connections "
                "WHERE source_id = ? OR target_id = ?",
                mid, mid
            )
            for row in cursor.fetchall():
                src, tgt, w = row
                other_id = tgt if src == mid else src
                if other_id in embed_map:
                    sim = self._similarity(mid_emb, embed_map[other_id])
                    if sim > 0.4:
                        key = (min(src, tgt), max(src, tgt))
                        activated_edges.add(key)

            stats["processed"] += 1

        # Strengthen activated connections
        logger.info("NREM: strengthening %d connections...", len(activated_edges))
        for (src, tgt) in activated_edges:
            cursor.execute(
                "UPDATE connections SET weight = CASE "
                "WHEN weight + 0.05 > 1.0 THEN 1.0 ELSE weight + 0.05 END "
                "WHERE source_id = ? AND target_id = ?",
                src, tgt
            )
        self.store.conn.commit()
        stats["strengthened"] = len(activated_edges)

        # Prune
        stats["pruned"] = self.store.prune_weak(0.05)

        self.store.finish_session(session_id, stats)
        logger.info("NREM: %d processed, %d strengthened, %d weakened, %d pruned",
                    stats["processed"], stats["strengthened"],
                    stats["weakened"], stats["pruned"])
        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def phase_rem(self, limit: int = 100) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Compute embeddings for all isolated memories
        3. Compare against ALL memories to find unconnected but similar
        4. Create bridge connections
        """
        logger.info("REM: finding isolated memories...")
        isolated = self.store.get_isolated_memories(max_connections=3, limit=limit)
        if not isolated:
            return {"explored": 0, "bridges": 0, "rejected": 0}

        session_id = self.store.start_session("rem")
        stats = {"explored": 0, "bridges": 0, "rejected": 0}

        # Embed isolated memories
        logger.info("REM: embedding %d isolated memories...", len(isolated))
        iso_embeds: Dict[int, List[float]] = {}
        for mem in isolated:
            emb = self._get_embedding(mem["id"], mem["content"])
            if emb:
                iso_embeds[mem["id"]] = emb

        # Get a sample of ALL memories for comparison
        all_recent = self.store.get_recent_memories(500)
        logger.info("REM: embedding %d comparison memories...", len(all_recent))
        comp_embeds: Dict[int, List[float]] = {}
        for mem in all_recent:
            if mem["id"] in iso_embeds:
                continue
            emb = self._get_embedding(mem["id"], mem["content"])
            if emb:
                comp_embeds[mem["id"]] = emb

        # Find bridges: for each isolated memory, find similar unconnected
        for iso_id, iso_emb in iso_embeds.items():
            similarities = []
            for comp_id, comp_emb in comp_embeds.items():
                sim = self._similarity(iso_emb, comp_emb)
                if 0.3 < sim < 0.95:
                    similarities.append((comp_id, sim))

            # Sort by similarity, take top 3
            similarities.sort(key=lambda x: -x[1])
            for comp_id, sim in similarities[:3]:
                bridge_weight = round(sim * 0.3, 3)
                self.store.add_bridge(iso_id, comp_id, bridge_weight)
                self.store.log_connection_change(
                    iso_id, comp_id, 0.0, bridge_weight, "rem_bridge"
                )
                stats["bridges"] += 1

            stats["explored"] += 1

        self.store.finish_session(session_id, stats)
        logger.info("REM: %d explored, %d bridges created",
                    stats["explored"], stats["bridges"])
        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction."""
        logger.info("Insight: building graph from MSSQL connections...")
        stats = {"communities": 0, "bridges": 0, "insights": 0}
        session_id = self.store.start_session("insight")

        edges = self.store.get_connections()
        if not edges:
            return stats

        # Build adjacency
        adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        nodes = set()
        for e in edges:
            s, t, w = e["source_id"], e["target_id"], e["weight"]
            if w >= 0.3:
                adj[s].append((t, w))
                adj[t].append((s, w))
                nodes.add(s)
                nodes.add(t)

        # Connected components (BFS)
        visited = set()
        communities: List[List[int]] = []
        for node in nodes:
            if node in visited:
                continue
            component = []
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.append(curr)
                for neighbor, _ in adj.get(curr, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            communities.append(component)

        stats["communities"] = len(communities)
        logger.info("Insight: found %d communities", len(communities))

        # Map nodes to communities
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i

        # Find bridge nodes
        bridge_nodes = set()
        for e in edges:
            if e["weight"] < 0.3:
                continue
            s_comm = node_to_comm.get(e["source_id"], -1)
            t_comm = node_to_comm.get(e["target_id"], -1)
            if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                bridge_nodes.add(e["source_id"])
                bridge_nodes.add(e["target_id"])

        stats["bridges"] = len(bridge_nodes)

        # Cluster insights (only for communities >= 5 members)
        for i, comm in enumerate(communities):
            if len(comm) < 5:
                continue
            theme = self._extract_theme(comm)
            confidence = min(len(comm) / 20.0, 1.0)
            content = f"Cluster of {len(comm)} memories: {theme}"
            self.store.add_insight(session_id, "cluster", comm[0], content, confidence)
            stats["insights"] += 1

        # Bridge insights
        for bnode in bridge_nodes:
            bridging_communities = set()
            for e in edges:
                if e["source_id"] == bnode or e["target_id"] == bnode:
                    other = e["target_id"] if e["source_id"] == bnode else e["source_id"]
                    bridging_communities.add(node_to_comm.get(other, -1))
            bridging_communities.discard(-1)

            if len(bridging_communities) >= 3:
                content = f"Bridge connecting {len(bridging_communities)} communities, memory #{bnode}"
                self.store.add_insight(session_id, "bridge", bnode, content, 0.8)
                stats["insights"] += 1

        self.store.finish_session(session_id, stats)
        logger.info("Insight: %d communities, %d bridges, %d insights",
                    stats["communities"], stats["bridges"], stats["insights"])
        return stats

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs via keyword frequency."""
        # Fetch content for these nodes
        placeholders = ",".join(str(n) for n in node_ids[:100])
        try:
            import pyodbc
            cursor = self.store.conn.cursor()
            cursor.execute(
                f"SELECT TOP 50 content FROM memories WHERE id IN ({placeholders})"
            )
            contents = [row[0] for row in cursor.fetchall() if row[0]]
        except Exception:
            return f"{len(node_ids)} memories"

        word_freq: Dict[str, int] = defaultdict(int)
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "it", "its", "this", "that", "user", "assistant", "and", "or",
            "but", "not", "if", "then", "so", "just", "also", "very",
            "really", "like", "get", "got", "want", "need", "think",
            "know", "see", "look", "make", "let", "use", "still",
        }
        for c in contents:
            for w in c.lower().split():
                w = w.strip(".,!?;:'\"()[]{}#@")
                if len(w) > 3 and w not in stopwords:
                    word_freq[w] += 1

        top = sorted(word_freq.items(), key=lambda x: -x[1])[:5]
        return ", ".join(w for w, _ in top) if top else "mixed topics"

    # -- Full cycle ----------------------------------------------------------

    def dream(self, phase: str = "all") -> Dict[str, Any]:
        """Run a dream cycle (or specific phase)."""
        start = time.time()
        result: Dict[str, Any] = {}

        if phase in ("all", "nrem"):
            result["nrem"] = self.phase_nrem()

        if phase in ("all", "rem"):
            result["rem"] = self.phase_rem()

        if phase in ("all", "insight"):
            result["insights"] = self.phase_insights()

        result["duration"] = time.time() - start
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dream Worker — MSSQL + sentence-transformers")
    parser.add_argument("--phase", default="all", choices=["all", "nrem", "rem", "insight"])
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon")
    parser.add_argument("--idle", type=int, default=300, help="Idle threshold in seconds")
    parser.add_argument("--limit", type=int, default=200, help="Max memories per phase")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    worker = DreamWorker()

    try:
        if args.daemon:
            logger.info("Dream daemon started (idle=%ds)", args.idle)
            last_activity = time.time()
            while True:
                time.sleep(30)
                idle = time.time() - last_activity
                if idle >= args.idle:
                    logger.info("Dream triggered (idle=%.0fs)", idle)
                    result = worker.dream()
                    logger.info("Dream complete: %.1fs", result["duration"])
                    last_activity = time.time()
        else:
            result = worker.dream(args.phase)
            print(f"\nDream complete in {result['duration']:.1f}s:")
            for phase, stats in result.items():
                if phase == "duration":
                    continue
                print(f"  {phase}: {stats}")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
