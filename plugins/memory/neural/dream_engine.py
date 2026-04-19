"""Dream Engine — autonomous background memory consolidation.

Implements three phases inspired by biological sleep:
  1. NREM — Replay & consolidation (strengthen, prune)
  2. REM  — Exploration & bridge discovery
  3. Insight — Abstraction & community detection

Runs as a background daemon during idle periods. Stores results
in the same SQLite DB as the main neural memory, extended with
dream-specific tables.

MSSQL support: if mssql_store is configured, dreams run against
the shared MSSQL backend (sneaky multi-agent consolidation).
Otherwise falls back to SQLite.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema extensions for dream tables (SQLite)
# ---------------------------------------------------------------------------

_DREAM_SCHEMA = """
CREATE TABLE IF NOT EXISTS dream_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    phase TEXT NOT NULL,
    memories_processed INTEGER DEFAULT 0,
    connections_strengthened INTEGER DEFAULT 0,
    connections_pruned INTEGER DEFAULT 0,
    bridges_found INTEGER DEFAULT 0,
    insights_created INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    insight_type TEXT NOT NULL,
    source_memory_id INTEGER,
    content TEXT,
    confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS connection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    changed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dream_insights_type
    ON dream_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_dream_insights_session
    ON dream_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_conn_history_nodes
    ON connection_history(source_id, target_id);
"""


# ---------------------------------------------------------------------------
# Abstract Dream Backend
# ---------------------------------------------------------------------------

class DreamBackend:
    """Interface for dream storage backends (SQLite or MSSQL)."""

    def start_session(self, phase: str) -> int:
        raise NotImplementedError

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_connections(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        raise NotImplementedError

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        raise NotImplementedError

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05) -> int:
        """Bulk strengthen. Returns count updated."""
        count = 0
        for src, tgt in edges:
            self.strengthen_connection(src, tgt, delta)
            count += 1
        return count

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01) -> int:
        """Bulk weaken all connections above threshold. Returns count updated."""
        raise NotImplementedError

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        raise NotImplementedError

    def prune_weak(self, threshold: float = 0.05) -> int:
        raise NotImplementedError

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        raise NotImplementedError

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        raise NotImplementedError

    def get_dream_stats(self) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SQLite Dream Backend
# ---------------------------------------------------------------------------

class SQLiteDreamBackend(DreamBackend):
    """Dream backend using the existing neural memory SQLite DB."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(_DREAM_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def start_session(self, phase: str) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO dream_sessions (started_at, phase) VALUES (?, ?)",
                (time.time(), phase)
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def finish_session(self, session_id: int, stats: Dict[str, Any]) -> None:
        if session_id < 0:
            return
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE dream_sessions SET "
                "finished_at = ?, memories_processed = ?, "
                "connections_strengthened = ?, connections_pruned = ?, "
                "bridges_found = ?, insights_created = ? WHERE id = ?",
                (
                    time.time(),
                    stats.get("processed", stats.get("explored", 0)),
                    stats.get("strengthened", 0),
                    stats.get("pruned", 0),
                    stats.get("bridges", 0),
                    stats.get("insights", 0),
                    session_id,
                )
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content FROM memories "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [{"id": r["id"], "content": r["content"] or ""} for r in rows]
        finally:
            conn.close()

    def get_isolated_memories(self, max_connections: int = 3,
                               limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT m.id, m.content,
                       (SELECT COUNT(*) FROM connections
                        WHERE source_id = m.id OR target_id = m.id) as cnt
                FROM memories m
                WHERE (SELECT COUNT(*) FROM connections
                       WHERE source_id = m.id OR target_id = m.id) < ?
                ORDER BY m.created_at DESC LIMIT ?
            """, (max_connections, limit)).fetchall()
            return [
                {"id": r["id"], "content": r["content"] or "", "connection_count": r["cnt"]}
                for r in rows
            ]
        finally:
            conn.close()

    def get_connections(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT source_id, target_id, weight FROM connections "
                "WHERE weight >= 0.05"
            ).fetchall()
            return [
                {"source_id": r["source_id"], "target_id": r["target_id"], "weight": r["weight"]}
                for r in rows
            ]
        finally:
            conn.close()

    def strengthen_connection(self, source_id: int, target_id: int,
                               delta: float = 0.05) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def weaken_connection(self, source_id: int, target_id: int,
                           delta: float = 0.01) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE source_id = ? AND target_id = ?",
                (delta, source_id, target_id)
            )
            conn.commit()
        finally:
            conn.close()

    def batch_strengthen_connections(self, edges: List[Tuple[int, int]],
                                      delta: float = 0.05) -> int:
        """Bulk strengthen via executemany. Returns count updated."""
        if not edges:
            return 0
        conn = self._connect()
        try:
            conn.executemany(
                "UPDATE connections SET weight = MIN(weight + ?, 1.0) "
                "WHERE source_id = ? AND target_id = ?",
                [(delta, src, tgt) for src, tgt in edges]
            )
            conn.commit()
            return len(edges)
        finally:
            conn.close()

    def batch_weaken_connections(self, threshold: float = 0.05,
                                  delta: float = 0.01) -> int:
        """Bulk weaken all connections above threshold in one UPDATE."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE connections SET weight = MAX(weight - ?, 0.0) "
                "WHERE weight > ?",
                (delta, threshold)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def add_bridge(self, source_id: int, target_id: int,
                    weight: float = 0.3) -> None:
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id FROM connections "
                "WHERE (source_id = ? AND target_id = ?) "
                "OR (source_id = ? AND target_id = ?)",
                (source_id, target_id, target_id, source_id)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO connections (source_id, target_id, weight, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (source_id, target_id, weight, time.time())
                )
                conn.commit()
        finally:
            conn.close()

    def prune_weak(self, threshold: float = 0.05) -> int:
        conn = self._connect()
        try:
            count = conn.execute(
                "DELETE FROM connections WHERE weight < ?",
                (threshold,)
            ).rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def log_connection_change(self, source_id: int, target_id: int,
                               old_weight: float, new_weight: float,
                               reason: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO connection_history "
                "(source_id, target_id, old_weight, new_weight, reason, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, target_id, old_weight, new_weight, reason, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def add_insight(self, session_id: int, insight_type: str,
                    source_memory_id: int, content: str,
                    confidence: float = 0.0) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO dream_insights "
                "(session_id, insight_type, source_memory_id, content, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, insight_type, source_memory_id, content, confidence, time.time())
            )
            conn.commit()
        finally:
            conn.close()

    def get_dream_stats(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            s = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(memories_processed), 0), "
                "COALESCE(SUM(connections_strengthened), 0), "
                "COALESCE(SUM(connections_pruned), 0), "
                "COALESCE(SUM(bridges_found), 0), "
                "COALESCE(SUM(insights_created), 0) "
                "FROM dream_sessions"
            ).fetchone()

            insights = conn.execute(
                "SELECT insight_type, COUNT(*) FROM dream_insights GROUP BY insight_type"
            ).fetchall()

            return {
                "sessions": s[0] if s else 0,
                "total_processed": s[1] if s else 0,
                "total_strengthened": s[2] if s else 0,
                "total_pruned": s[3] if s else 0,
                "total_bridges": s[4] if s else 0,
                "total_insights": s[5] if s else 0,
                "insight_types": {r[0]: r[1] for r in insights},
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """Autonomous background consolidation for neural memory.

    Three phases:
      NREM  — Replay recent memories, strengthen active, prune dead
      REM   — Explore isolated memories, discover bridges via embedding
      Insight — Community detection, bridge identification, abstraction
    """

    def __init__(
        self,
        backend: DreamBackend,
        neural_memory: Optional[Any] = None,
        idle_threshold: float = 300.0,     # 5 min idle
        memory_threshold: int = 50,         # dream every N new memories
        max_memories_per_cycle: int = 100,
    ):
        self._backend = backend
        self._memory = neural_memory        # NeuralMemory instance for think/recall
        self._idle_threshold = idle_threshold
        self._memory_threshold = memory_threshold
        self._max_memories = max_memories_per_cycle

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_activity = time.time()
        self._memory_count_at_last_dream = 0
        self._dream_count = 0

    @classmethod
    def sqlite(cls, db_path: str, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with SQLite backend."""
        backend = SQLiteDreamBackend(db_path)
        return cls(backend, neural_memory, **kwargs)

    @classmethod
    def mssql(cls, mssql_config: dict, neural_memory: Optional[Any] = None, **kwargs) -> 'DreamEngine':
        """Create a DreamEngine with MSSQL backend."""
        from dream_mssql_store import DreamMSSQLStore
        backend = DreamMSSQLStore.from_config(mssql_config)
        return cls(backend, neural_memory, **kwargs)

    def start(self) -> None:
        """Start the background dream daemon."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dream_loop, daemon=True, name="dream-engine"
        )
        self._thread.start()
        logger.info(
            "Dream engine started: idle=%ss, threshold=%d",
            self._idle_threshold, self._memory_threshold,
        )

    def stop(self) -> None:
        """Stop the dream daemon."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Dream engine stopped after %d cycles", self._dream_count)

    def touch(self) -> None:
        """Signal activity — resets idle timer."""
        self._last_activity = time.time()

    def dream_now(self) -> Dict[str, Any]:
        """Force an immediate dream cycle. Returns stats."""
        return self._run_dream_cycle()

    # -- Main loop -----------------------------------------------------------

    def _dream_loop(self) -> None:
        """Background daemon: dream when idle or threshold reached."""
        while self._running:
            try:
                time.sleep(30)
                if not self._running:
                    break

                idle = time.time() - self._last_activity
                try:
                    stats = self._memory.stats() if self._memory else {"memories": 0}
                    total = stats.get("memories", 0)
                except Exception:
                    total = 0
                new_since_last = total - self._memory_count_at_last_dream

                should_dream = (
                    idle >= self._idle_threshold
                    or new_since_last >= self._memory_threshold
                )

                if should_dream:
                    logger.info(
                        "Dream cycle triggered: idle=%.0fs, new=%d",
                        idle, new_since_last,
                    )
                    self._run_dream_cycle()

            except Exception as e:
                logger.debug("Dream loop error: %s", e)
                time.sleep(60)

    # -- Dream Cycle ---------------------------------------------------------

    def _run_dream_cycle(self) -> Dict[str, Any]:
        """Execute a full NREM → REM → Insight cycle."""
        with self._lock:
            start = time.time()
            total_stats: Dict[str, Any] = {"nrem": {}, "rem": {}, "insights": {}}

            try:
                total_stats["nrem"] = self._phase_nrem()
                total_stats["rem"] = self._phase_rem()
                total_stats["insights"] = self._phase_insights()

                self._dream_count += 1
                if self._memory:
                    try:
                        s = self._memory.stats()
                        self._memory_count_at_last_dream = s.get("memories", 0)
                    except Exception:
                        pass

                total_stats["duration"] = time.time() - start
                total_stats["dream_id"] = self._dream_count

                logger.info(
                    "Dream #%d complete: %.1fs | NREM: %d+/ %d- / %d pruned | REM: %d bridges | Insights: %d",
                    self._dream_count, total_stats["duration"],
                    total_stats["nrem"].get("strengthened", 0),
                    total_stats["nrem"].get("weakened", 0),
                    total_stats["nrem"].get("pruned", 0),
                    total_stats["rem"].get("bridges", 0),
                    total_stats["insights"].get("insights", 0),
                )

            except Exception as e:
                logger.error("Dream cycle failed: %s", e)
                total_stats["error"] = str(e)

            return total_stats

    # -- Phase 1: NREM -------------------------------------------------------

    def _phase_nrem(self) -> Dict[str, Any]:
        """NREM: Replay, strengthen active, weaken inactive, prune dead.

        For each recent memory:
          1. Fire spreading activation
          2. Activated edges: batch strengthen
          3. Non-activated edges: bulk weaken (single SQL UPDATE)
          4. Edges below threshold: prune
        """
        stats = {"processed": 0, "strengthened": 0, "weakened": 0, "pruned": 0}
        session_id = self._backend.start_session("nrem")

        try:
            memories = self._backend.get_recent_memories(self._max_memories)
            if not memories:
                return stats

            activated_edges: Set[Tuple[int, int]] = set()

            for mem in memories:
                mid = mem["id"]
                if self._memory:
                    try:
                        activated = self._memory.think(mid, depth=2)
                        for a in activated:
                            aid = a.get("id")
                            if aid and aid != mid:
                                activated_edges.add((min(mid, aid), max(mid, aid)))
                    except Exception:
                        pass
                stats["processed"] += 1

            # Batch strengthen activated edges (usually small set)
            if activated_edges:
                stats["strengthened"] = self._backend.batch_strengthen_connections(
                    list(activated_edges), 0.05
                )

            # Bulk weaken ALL non-activated connections above threshold
            # Single SQL UPDATE instead of per-row loop
            stats["weakened"] = self._backend.batch_weaken_connections(
                threshold=0.05, delta=0.01
            )

            # Prune dead connections
            stats["pruned"] = self._backend.prune_weak(0.05)

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("NREM phase error: %s", e)

        return stats

    # -- Phase 2: REM --------------------------------------------------------

    def _phase_rem(self) -> Dict[str, Any]:
        """REM: Explore isolated memories, discover bridges.

        1. Find isolated memories (few connections)
        2. Search via embedding similarity for unconnected but similar
        3. Create tentative bridge connections (weight 0.1-0.3)
        """
        stats = {"explored": 0, "bridges": 0, "rejected": 0}
        session_id = self._backend.start_session("rem")

        try:
            isolated = self._backend.get_isolated_memories(max_connections=3, limit=50)
            if not isolated:
                return stats

            for mem in isolated:
                mid = mem["id"]
                content = mem["content"]

                if not self._memory or not content:
                    continue

                try:
                    similar = self._memory.recall(content[:200], k=10)
                    stats["explored"] += 1

                    for sim in similar:
                        sim_id = sim.get("id")
                        sim_score = sim.get("similarity", 0)

                        if not sim_id or sim_id == mid:
                            continue
                        if sim_score < 0.3 or sim_score > 0.95:
                            continue

                        bridge_weight = round(sim_score * 0.3, 3)
                        self._backend.add_bridge(mid, sim_id, bridge_weight)
                        self._backend.log_connection_change(
                            mid, sim_id, 0.0, bridge_weight, "rem_bridge"
                        )
                        stats["bridges"] += 1

                except Exception:
                    pass

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("REM phase error: %s", e)

        return stats

    # -- Phase 3: Insights ---------------------------------------------------

    def _phase_insights(self) -> Dict[str, Any]:
        """Insight: Community detection, bridge identification, abstraction.

        1. Find connected components (communities)
        2. Identify bridge nodes connecting communities
        3. Create insight memories for dense clusters
        """
        stats = {"communities": 0, "bridges": 0, "insights": 0}
        session_id = self._backend.start_session("insight")

        try:
            edges = self._backend.get_connections()
            if not edges:
                return stats

            # Build adjacency
            adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            nodes = set()
            for e in edges:
                s, t, w = e["source_id"], e["target_id"], e["weight"]
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

            # Map nodes to communities
            node_to_comm = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = i

            # Find bridge nodes
            bridge_nodes = set()
            for e in edges:
                s_comm = node_to_comm.get(e["source_id"], -1)
                t_comm = node_to_comm.get(e["target_id"], -1)
                if s_comm != t_comm and s_comm >= 0 and t_comm >= 0:
                    bridge_nodes.add(e["source_id"])
                    bridge_nodes.add(e["target_id"])
            stats["bridges"] = len(bridge_nodes)

            # Create cluster insights
            for i, comm in enumerate(communities):
                if len(comm) < 3:
                    continue
                theme = self._extract_theme(comm)
                confidence = min(len(comm) / 10.0, 1.0)
                content = f"Cluster of {len(comm)} related memories: {theme}"
                self._backend.add_insight(session_id, "cluster", comm[0], content, confidence)
                stats["insights"] += 1

            # Create bridge insights
            for bnode in bridge_nodes:
                bridging_communities = set()
                for e in edges:
                    if e["source_id"] == bnode or e["target_id"] == bnode:
                        other = e["target_id"] if e["source_id"] == bnode else e["source_id"]
                        bridging_communities.add(node_to_comm.get(other, -1))
                bridging_communities.discard(-1)

                if len(bridging_communities) >= 2:
                    content = (
                        f"Bridge connecting {len(bridging_communities)} communities, "
                        f"memory #{bnode}"
                    )
                    self._backend.add_insight(session_id, "bridge", bnode, content, 0.8)
                    stats["insights"] += 1

            self._backend.finish_session(session_id, stats)

        except Exception as e:
            logger.debug("Insight phase error: %s", e)

        return stats

    # -- Helpers -------------------------------------------------------------

    def _extract_theme(self, node_ids: List[int]) -> str:
        """Extract common themes from node IDs (simple keyword frequency)."""
        # If we have memory access, get contents via store (thread-safe)
        contents = []
        if self._memory and hasattr(self._memory, 'store'):
            try:
                placeholders = ",".join("?" * len(node_ids))
                with self._memory.store._lock:
                    rows = self._memory.store.conn.execute(
                        f"SELECT content FROM memories WHERE id IN ({placeholders})",
                        tuple(node_ids)
                    ).fetchall()
                    contents = [r[0] for r in rows if r[0]]
            except Exception:
                pass

        if not contents:
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

    def get_stats(self) -> Dict[str, Any]:
        """Get dream engine statistics."""
        base = self._backend.get_dream_stats()
        base["engine_running"] = self._running
        base["dream_cycles"] = self._dream_count
        return base
