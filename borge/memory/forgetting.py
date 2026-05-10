"""
Forgetting Engine — Active Memory Decay

Implements Ebbinghaus-inspired forgetting to prevent memory accumulation
from degrading retrieval quality.

Forgetting is tiered by encoding depth:
  SHALLOW  + forget_score > PRUNE_THRESHOLD  → delete
  SEMANTIC + forget_score > COMPRESS_THRESHOLD → compress to entity tag only
  SCHEMATIC / META                            → never delete, only compress
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

PRUNE_THRESHOLD    = 2.0   # SHALLOW entries above this are deleted
COMPRESS_THRESHOLD = 3.0   # SEMANTIC entries above this are compressed

# Borge DB columns added to existing Hermes messages table
BORGE_COLUMNS_SQL = """
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_valence REAL DEFAULT 0.0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_arousal REAL DEFAULT 0.5;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS emotional_significance REAL DEFAULT 0.0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS encoding_depth INTEGER DEFAULT 1;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS entity_tags TEXT DEFAULT '[]';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS graph_node_ids TEXT DEFAULT '[]';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS retrieval_count INTEGER DEFAULT 0;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS last_retrieved TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS importance_score REAL DEFAULT 0.5;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS forget_score REAL DEFAULT 0.0;
"""


class ForgettingEngine:
    """
    Periodically recomputes forget_score for all messages and
    removes or compresses entries that exceed thresholds.
    """

    def __init__(
        self,
        prune_threshold: float = PRUNE_THRESHOLD,
        compress_threshold: float = COMPRESS_THRESHOLD,
    ):
        self.prune_threshold = prune_threshold
        self.compress_threshold = compress_threshold

    def run_forgetting_pass(self, db_path: str) -> dict:
        """
        Execute one forgetting pass over the messages table.
        Returns {"deleted": N, "compressed": M}.
        """
        deleted = 0
        compressed = 0

        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                self._ensure_columns(conn)

                rows = conn.execute(
                    """SELECT id, timestamp, last_retrieved, retrieval_count,
                              importance_score, graph_node_ids, encoding_depth, content
                       FROM messages
                       WHERE role IN ('user', 'assistant', 'tool')"""
                ).fetchall()

                now = datetime.now()

                for row in rows:
                    score = self._compute_score(row, now)
                    depth = row["encoding_depth"] or 1

                    if depth <= 1 and score > self.prune_threshold:
                        conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
                        deleted += 1

                    elif depth == 2 and score > self.compress_threshold:
                        stub = f"[compressed:{row['id'][:8]}]"
                        conn.execute(
                            "UPDATE messages SET content = ? WHERE id = ?",
                            (stub, row["id"]),
                        )
                        compressed += 1

                    else:
                        conn.execute(
                            "UPDATE messages SET forget_score = ? WHERE id = ?",
                            (round(score, 4), row["id"]),
                        )

        except sqlite3.OperationalError as e:
            log.warning(f"[Forgetting] DB error (possibly no borge columns yet): {e}")

        return {"deleted": deleted, "compressed": compressed}

    @staticmethod
    def _compute_score(row: sqlite3.Row, now: datetime) -> float:
        """Ebbinghaus-inspired forget score."""
        import json
        import math

        ts_str = row["last_retrieved"] or row["timestamp"]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            ts = now

        days_since    = max(0.0, (now - ts).total_seconds() / 86400.0)
        retrieval_cnt = row["retrieval_count"] or 0
        importance    = row["importance_score"] or 0.5

        try:
            graph_ids = json.loads(row["graph_node_ids"] or "[]")
            graph_n   = len(graph_ids)
        except (json.JSONDecodeError, TypeError):
            graph_n = 0

        recency_decay    = days_since ** 0.7
        usage_penalty    = 1.0 / (1.0 + retrieval_cnt)
        importance_res   = 1.0 / (1.0 + importance)
        graph_resistance = 1.0 / (1.0 + graph_n)

        return recency_decay * usage_penalty * importance_res * graph_resistance

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection) -> None:
        """Silently add Borge columns to existing Hermes messages table."""
        for stmt in BORGE_COLUMNS_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists
