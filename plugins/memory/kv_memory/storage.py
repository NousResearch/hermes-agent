"""SQLite storage backend for kv-memory provider.

Schema (see schema.sql):
  - sessions: session-level metadata
  - turns: Q4-quantized embeddings + text summaries per turn
  - session_links: cross-session semantic/causal relationships
  - embeddings_idx: ANN search via sqlite-vec (if available)

Features:
  - WAL mode for concurrent reads
  - Connection pooling for thread safety
  - Automatic schema migration
  - Compaction: merges adjacent turns, prunes low-importance entries
  - Pruning: removes turns older than retention period
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import KVMemoryConfig

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ═══════════════════════════════════════════════════════════════════════════════
# Database class
# ═══════════════════════════════════════════════════════════════════════════════

class KVMemoryDB:
    """SQLite database for kv-memory provider.

    Thread-safe. Uses WAL mode for concurrent reads. All write operations
    acquire a short-lived lock to prevent corruption.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._local = threading.local()

    # ── Connection management ─────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        """Thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ── Schema ───────────────────────────────────────────────────────

    def initialize_schema(self):
        """Create tables and indexes if they don't exist.

        The vec0 virtual table (sqlite-vec ANN index) is optional —
        if the extension isn't available, we fall back to brute-force
        cosine similarity in the retriever.
        """
        if not _SCHEMA_PATH.exists():
            logger.warning("schema.sql not found at %s", _SCHEMA_PATH)
            return
        schema = _SCHEMA_PATH.read_text()
        with self._lock:
            self.conn.executescript(schema)

        # Try to create the vec0 ANN index (optional — requires sqlite-vec)
        try:
            with self._lock:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_idx "
                    "USING vec0(turn_id TEXT PRIMARY KEY, embedding FLOAT[4096])"
                )
        except Exception:
            logger.debug(
                "sqlite-vec not available — using brute-force cosine similarity "
                "(linear scan). Install sqlite-vec for ANN speedup on large DBs."
            )

    def migrate_schema(self):
        """Run any pending schema migrations."""
        # V1 → current: ensure sessions/links tables exist
        # Future migrations go here
        self.initialize_schema()

    # ── Sessions ─────────────────────────────────────────────────────

    def ensure_session(self, session_id: str, metadata: dict | None = None):
        """Insert or update a session record."""
        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self._lock:
            self.conn.execute(
                """INSERT INTO sessions (id, created_at, last_accessed, metadata)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       last_accessed = excluded.last_accessed,
                       metadata = excluded.metadata""",
                (session_id, now, now, meta_json),
            )
            self.conn.commit()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # ── Turns ────────────────────────────────────────────────────────

    def store_turn(
        self,
        session_id: str,
        turn_number: int,
        embedding: np.ndarray,
        q4_embedding: np.ndarray | None = None,
        q4_scales: np.ndarray | None = None,
        summary_text: str = "",
        tool_calls: list | None = None,
        model_id: str = "",
        num_kv_heads: int = 0,
        head_dim: int = 0,
        num_layers: int = 0,
        metadata: dict | None = None,
        store_fp16: bool = False,
    ) -> str:
        """Store a turn with its embedding(s).

        Args:
            session_id: Session this turn belongs to.
            turn_number: Monotonic turn number within the session.
            embedding: FP32/FP16 embedding vector.
            q4_embedding: Q4-packed uint8 array (optional).
            q4_scales: Per-channel float32 scales for Q4 (optional).
            summary_text: Lightweight text summary for fallback.
            tool_calls: List of tool call dicts this turn.
            model_id: Model used for inference.
            num_kv_heads, head_dim, num_layers: KV-cache metadata.
            metadata: Arbitrary JSON-serializable metadata.
            store_fp16: Also store raw FP16 embedding (for fidelity checks).

        Returns:
            The turn UUID.
        """
        turn_id = str(uuid.uuid4())
        now = time.time()

        # Ensure session exists
        self.ensure_session(session_id)

        # Serialize blobs
        q4_blob = q4_embedding.tobytes() if q4_embedding is not None else None
        scales_blob = q4_scales.tobytes() if q4_scales is not None else None
        fp16_blob = embedding.astype(np.float16).tobytes() if store_fp16 else None
        tool_json = json.dumps(tool_calls or [])
        meta_json = json.dumps(metadata or {})

        with self._lock:
            self.conn.execute(
                """INSERT INTO turns
                   (id, session_id, turn_number, timestamp, model_id,
                    num_layers, num_kv_heads, head_dim, embedding_dim,
                    q4_embedding, q4_scales, fp16_embedding,
                    summary_text, tool_calls, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn_id, session_id, turn_number, now, model_id,
                    num_layers, num_kv_heads, head_dim, int(embedding.shape[0]),
                    q4_blob, scales_blob, fp16_blob,
                    summary_text, tool_json, meta_json,
                ),
            )
            self.conn.commit()

        return turn_id

    def get_turns(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all turns for a session, ordered by turn_number."""
        rows = self.conn.execute(
            """SELECT id, session_id, turn_number, timestamp, model_id,
                      num_layers, num_kv_heads, head_dim, embedding_dim,
                      summary_text, tool_calls, user_feedback, metadata
               FROM turns
               WHERE session_id = ?
               ORDER BY turn_number
               LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_embedding(
        self,
        turn_id: str,
        dequantize: bool = True,
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """Retrieve a turn's embedding.

        Returns (embedding: np.ndarray, metadata: dict) or None.
        If dequantize=True and Q4 data exists, returns dequantized FP32.
        Otherwise returns the raw FP16 embedding if stored.
        """
        row = self.conn.execute(
            """SELECT q4_embedding, q4_scales, fp16_embedding,
                      embedding_dim, num_kv_heads, head_dim
               FROM turns WHERE id = ?""",
            (turn_id,),
        ).fetchone()

        if row is None:
            return None

        meta = {
            "embedding_dim": row["embedding_dim"],
            "num_kv_heads": row["num_kv_heads"],
            "head_dim": row["head_dim"],
        }

        if dequantize and row["q4_embedding"] is not None and row["q4_scales"] is not None:
            from .quantize import dequantize_q4_per_channel
            packed = np.frombuffer(row["q4_embedding"], dtype=np.uint8)
            scales = np.frombuffer(row["q4_scales"], dtype=np.float32)
            channel_size = row["head_dim"] or 128
            embedding = dequantize_q4_per_channel(
                packed, scales, channel_size, row["embedding_dim"]
            )
            return embedding, meta
        elif row["fp16_embedding"] is not None:
            embedding = np.frombuffer(row["fp16_embedding"], dtype=np.float16).astype(np.float32)
            return embedding, meta

        return None

    def get_all_embeddings(
        self,
        session_id: str | None = None,
        dequantize: bool = True,
    ) -> List[Tuple[str, np.ndarray, dict]]:
        """Get all stored embeddings, optionally filtered by session.

        Returns list of (turn_id, embedding, metadata_dict).
        """
        if session_id:
            rows = self.conn.execute(
                """SELECT id, q4_embedding, q4_scales, fp16_embedding,
                          embedding_dim, num_kv_heads, head_dim,
                          summary_text, timestamp, session_id
                   FROM turns WHERE session_id = ?
                   ORDER BY timestamp""",
                (session_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT id, q4_embedding, q4_scales, fp16_embedding,
                          embedding_dim, num_kv_heads, head_dim,
                          summary_text, timestamp, session_id
                   FROM turns
                   ORDER BY timestamp""",
            ).fetchall()

        results = []
        for row in rows:
            meta = {
                "embedding_dim": row["embedding_dim"],
                "num_kv_heads": row["num_kv_heads"],
                "head_dim": row["head_dim"],
                "summary_text": row["summary_text"] or "",
                "timestamp": row["timestamp"],
                "session_id": row["session_id"],
            }

            if dequantize and row["q4_embedding"] is not None and row["q4_scales"] is not None:
                from .quantize import dequantize_q4_per_channel
                packed = np.frombuffer(row["q4_embedding"], dtype=np.uint8)
                scales = np.frombuffer(row["q4_scales"], dtype=np.float32)
                channel_size = row["head_dim"] or 128
                embedding = dequantize_q4_per_channel(
                    packed, scales, channel_size, row["embedding_dim"]
                )
                results.append((row["id"], embedding, meta))
            elif row["fp16_embedding"] is not None:
                embedding = np.frombuffer(row["fp16_embedding"], dtype=np.float16).astype(np.float32)
                results.append((row["id"], embedding, meta))

        return results

    def update_feedback(self, turn_id: str, feedback: int):
        """Update user feedback for a turn (-1, 0, 1)."""
        with self._lock:
            self.conn.execute(
                "UPDATE turns SET user_feedback = ? WHERE id = ?",
                (feedback, turn_id),
            )
            self.conn.commit()

    # ── Session links ─────────────────────────────────────────────────

    def link_sessions(
        self,
        source_id: str,
        target_id: str,
        similarity: float,
        link_type: str = "semantic",
    ):
        """Create or update a cross-session link."""
        now = time.time()
        with self._lock:
            existing = self.conn.execute(
                """SELECT id FROM session_links
                   WHERE source_session_id = ? AND target_session_id = ?""",
                (source_id, target_id),
            ).fetchone()
            if existing:
                self.conn.execute(
                    """UPDATE session_links
                       SET similarity = ?, created_at = ?
                       WHERE id = ?""",
                    (similarity, now, existing["id"]),
                )
            else:
                self.conn.execute(
                    """INSERT INTO session_links
                       (source_session_id, target_session_id, similarity, link_type, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source_id, target_id, similarity, link_type, now),
                )
            self.conn.commit()

    def get_linked_sessions(
        self,
        session_id: str,
        min_similarity: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Get sessions linked to the given session."""
        rows = self.conn.execute(
            """SELECT * FROM session_links
               WHERE (source_session_id = ? OR target_session_id = ?)
                 AND similarity >= ?
               ORDER BY similarity DESC""",
            (session_id, session_id, min_similarity),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Governance ────────────────────────────────────────────────────

    def count_turns(self) -> int:
        """Total number of stored turns."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM turns").fetchone()
        return row["cnt"] if row else 0

    def prune_old_turns(self, retention_days: int) -> int:
        """Delete turns older than retention_days. Returns count deleted."""
        cutoff = time.time() - (retention_days * 86400)
        with self._lock:
            cursor = self.conn.execute(
                "DELETE FROM turns WHERE timestamp < ?", (cutoff,)
            )
            self.conn.commit()
            return cursor.rowcount

    def compact_session(self, session_id: str) -> int:
        """Merge adjacent turns within a session.

        Compaction strategy:
          1. Keep the most recent turn per adjacent pair
          2. Merge their summary_text
          3. Recompute embedding for merged turn (caller's responsibility)

        Returns number of turns removed.
        """
        turns = self.get_turns(session_id, limit=1000)
        if len(turns) < 3:
            return 0

        removed = 0
        with self._lock:
            # Remove every other turn, keeping the more recent one
            for i in range(len(turns) - 1):
                if i % 2 == 0:
                    older = turns[i]
                    newer = turns[i + 1]
                    # Merge summary text into the newer turn
                    merged_summary = (
                        (older.get("summary_text") or "")
                        + " | "
                        + (newer.get("summary_text") or "")
                    )[:1000]
                    self.conn.execute(
                        "UPDATE turns SET summary_text = ? WHERE id = ?",
                        (merged_summary, newer["id"]),
                    )
                    self.conn.execute(
                        "DELETE FROM turns WHERE id = ?", (older["id"],)
                    )
                    removed += 1
            self.conn.commit()

        return removed

    # ── Stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        total_turns = self.count_turns()
        total_sessions = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions"
        ).fetchone()["cnt"]

        # Total storage
        try:
            db_size = os.path.getsize(self._db_path)
        except Exception:
            db_size = 0

        # Average embedding dim
        avg_dim_row = self.conn.execute(
            "SELECT AVG(embedding_dim) as avg_dim FROM turns"
        ).fetchone()
        avg_dim = avg_dim_row["avg_dim"] if avg_dim_row else 0

        return {
            "total_turns": total_turns,
            "total_sessions": total_sessions,
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "avg_embedding_dim": int(avg_dim) if avg_dim else 0,
        }
