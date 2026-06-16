"""Summary DAG — hierarchical compaction graph.

Each node is a summary of source material (raw messages or lower-depth
summaries). Nodes form a directed acyclic graph where edges point from
a summary to its sources.

Depth semantics:
  D0 — leaf summaries of raw messages (minutes timescale)
  D1 — condensation of D0 nodes (hours)
  D2 — condensation of D1 nodes (days)
  D3+ — further condensation (weeks/months)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    ExternalContentFtsSpec,
    configure_connection,
    ensure_external_content_fts,
    run_versioned_migrations,
)
from .search_query import (
    AGE_DECAY_RATE,
    compute_search_candidate_cap,
    compute_directness_rank_bonus_upper_bound,
    compute_directness_score,
    compute_like_fallback_fetch_limit,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    requires_like_fallback,
    sanitize_fts5_query,
    should_apply_directness_rank_adjustment,
)
from .store import _normalize_source_value, _UNKNOWN_SOURCE, _legacy_blank_source_clause
from .config import LCMConfig
from .ingest_protection import redact_sensitive_value
from .storage_security import ensure_lcm_file_permissions


logger = logging.getLogger(__name__)


def _build_search_order_by(sort: str | None, recency_expr: str) -> str:
    normalized = normalize_search_sort(sort)
    if normalized == "relevance":
        return f"rank ASC, {recency_expr} DESC"
    if normalized == "hybrid":
        return (
            f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {recency_expr}) / 3600.0)) * {AGE_DECAY_RATE}))) ASC, "
            f"{recency_expr} DESC"
        )
    return f"{recency_expr} DESC"


def _fallback_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(node.search_rank or 0.0) * -1.0
    recency = float(node.latest_at or node.created_at or 0.0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (-score, -directness, -recency)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, -recency)
    return (-recency, -score, -directness)


def _fts_result_sort_key(node: "SummaryNode", sort: str | None) -> tuple[float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    recency = float(node.latest_at or node.created_at or 0.0)
    directness = float(node.search_directness or 0.0)

    if normalized == "relevance":
        return (rank_value, -directness, -recency)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return (-blended_strength, -directness, -recency)
    return (-recency, rank_value, 0.0)


def _fts_primary_value(node: "SummaryNode", sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = node.search_rank
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        recency = float(node.latest_at or node.created_at or 0.0)
        age_hours = max(0.0, (time.time() - recency) / 3600.0)
        strength = (-rank_value) if rank is not None else float("-inf")
        blended_strength = strength / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("-inf")
        return -blended_strength
    return rank_value


def build_nodes_fts_spec() -> ExternalContentFtsSpec:
    return ExternalContentFtsSpec(
        table_name="nodes_fts",
        content_table="summary_nodes",
        content_rowid="node_id",
        indexed_column="summary",
        trigger_sqls=(
            """
            CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
                AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO nodes_fts(rowid, summary)
                    VALUES (new.node_id, new.summary);
            END;
            """,
            """
            CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                AFTER DELETE ON summary_nodes BEGIN
                INSERT INTO nodes_fts(nodes_fts, rowid, summary)
                    VALUES('delete', old.node_id, old.summary);
            END;
            """,
        ),
    )


@dataclass
class SummaryNode:
    """A single node in the summary DAG."""
    node_id: int = 0
    session_id: str = ""
    depth: int = 0
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0  # total tokens of source material
    source_ids: List[int] = field(default_factory=list)  # store_ids or node_ids
    source_type: str = "messages"  # "messages" or "nodes"
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    expand_hint: str = ""  # "Expand for details about: ..."
    search_rank: float | None = None
    search_directness: float = 0.0


class SummaryDAG:
    """SQLite-backed DAG of summary nodes."""

    def __init__(self, db_path: str | Path, *, ingest_protection_config=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        ensure_lcm_file_permissions(self.db_path)
        self._ingest_protection_config = ingest_protection_config or LCMConfig(database_path=str(self.db_path))
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                earliest_at REAL,
                latest_at REAL,
                expand_hint TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_session_depth
                ON summary_nodes(session_id, depth, created_at);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        ensure_external_content_fts(
            self._conn,
            build_nodes_fts_spec(),
        )
        run_versioned_migrations(self._conn)
        self._ensure_source_window_columns()
        self._conn.commit()
        ensure_lcm_file_permissions(self.db_path)

    def _ensure_source_window_columns(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(summary_nodes)").fetchall()
        }
        if "earliest_at" not in columns:
            self._conn.execute("ALTER TABLE summary_nodes ADD COLUMN earliest_at REAL")
        if "latest_at" not in columns:
            self._conn.execute("ALTER TABLE summary_nodes ADD COLUMN latest_at REAL")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_session_latest ON summary_nodes(session_id, latest_at, created_at)"
        )

    # -- Write --------------------------------------------------------------

    def add_node(self, node: SummaryNode) -> int:
        """Insert a summary node and return its node_id."""
        summary = str(redact_sensitive_value(node.summary, self._ingest_protection_config) or "")
        expand_hint = str(redact_sensitive_value(node.expand_hint, self._ingest_protection_config) or "")
        cur = self._conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, earliest_at, latest_at, expand_hint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.session_id,
                node.depth,
                summary,
                node.token_count,
                node.source_token_count,
                json.dumps(node.source_ids),
                node.source_type,
                node.created_at or time.time(),
                node.earliest_at,
                node.latest_at,
                expand_hint,
            ),
        )
        self._conn.commit()
        ensure_lcm_file_permissions(self.db_path)
        node.summary = summary
        node.expand_hint = expand_hint
        node.node_id = cur.lastrowid
        return node.node_id

    def delete_below_depth(self, session_id: str, min_depth: int) -> int:
        """Delete all nodes for a session with depth < min_depth.

        Returns the number of deleted nodes. Used during session reset
        to retain only high-level summaries across sessions.
        """
        cur = self._conn.execute(
            """DELETE FROM summary_nodes
               WHERE session_id = ? AND depth < ?""",
            (session_id, min_depth),
        )
        deleted = cur.rowcount
        self._conn.commit()
        return deleted

    def delete_session_nodes(self, session_id: str) -> int:
        """Delete all nodes for a session. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM summary_nodes WHERE session_id = ?",
            (session_id,),
        )
        deleted = cur.rowcount
        self._conn.commit()
        return deleted

    def reassign_session_nodes(self, old_session_id: str, new_session_id: str) -> int:
        """Move all nodes from one session_id to another.

        Used for /new carry-over where retained summaries should become part of
        the fresh session while preserving node IDs and node-to-node links.
        """
        cur = self._conn.execute(
            "UPDATE summary_nodes SET session_id = ? WHERE session_id = ?",
            (new_session_id, old_session_id),
        )
        moved = cur.rowcount
        self._conn.commit()
        return moved

    # -- Read ---------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[SummaryNode]:
        row = self._conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_session_nodes(self, session_id: str,
                          depth: int | None = None,
                          limit: int = 1000) -> List[SummaryNode]:
        """Get nodes for a session, optionally filtered by depth."""
        if depth is not None:
            rows = self._conn.execute(
                """SELECT * FROM summary_nodes
                   WHERE session_id = ? AND depth = ?
                   ORDER BY created_at LIMIT ?""",
                (session_id, depth, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM summary_nodes
                   WHERE session_id = ?
                   ORDER BY depth, created_at LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_at_depth(self, session_id: str, depth: int) -> int:
        """Count nodes at a specific depth for a session."""
        row = self._conn.execute(
            """SELECT COUNT(*) FROM summary_nodes
               WHERE session_id = ? AND depth = ?""",
            (session_id, depth),
        ).fetchone()
        return row[0] if row else 0

    def get_uncondensed_at_depth(self, session_id: str, depth: int,
                                  limit: int = 100) -> List[SummaryNode]:
        """Get nodes at a depth that haven't been condensed yet.

        A node is 'uncondensed' if it's not referenced as a source by
        any higher-depth node.
        """
        rows = self._conn.execute(
            """SELECT n.* FROM summary_nodes n
               WHERE n.session_id = ? AND n.depth = ?
               AND n.node_id NOT IN (
                   SELECT json_each.value FROM summary_nodes p,
                   json_each(p.source_ids)
                   WHERE p.session_id = ? AND p.depth > ? AND p.source_type = 'nodes'
               )
               ORDER BY n.created_at LIMIT ?""",
            (session_id, depth, session_id, depth, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None,
               source: str | None = None) -> List[SummaryNode]:
        """FTS5 search across summary nodes.

        Retrieval contract:
        - ``session_id`` limits which sessions are eligible
        - ``session_id=None`` means all sessions; an empty string is treated as
          a literal session id
        - ``source`` filters summaries by descendant raw-message lineage, not by
          session-level source presence
        - mixed-source nodes may match more than one ``source`` filter
        """
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if requires_like_fallback(query):
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

        order_by = _build_search_order_by(sort, "COALESCE(n.latest_at, n.created_at)")
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        candidate_cap = compute_search_candidate_cap(limit)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 2e-7
        offset = 0
        scanned_rows = 0
        results: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            try:
                if session_id is not None:
                    rows = self._conn.execute(
                        f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                           JOIN summary_nodes n ON n.node_id = fts.rowid
                           WHERE nodes_fts MATCH ? AND n.session_id = ?
                           ORDER BY {order_by} LIMIT ? OFFSET ?""",
                        (safe_query, session_id, fetch_limit, offset),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        f"""SELECT n.*, rank as search_rank FROM nodes_fts fts
                           JOIN summary_nodes n ON n.node_id = fts.rowid
                           WHERE nodes_fts MATCH ?
                           ORDER BY {order_by} LIMIT ? OFFSET ?""",
                        (safe_query, fetch_limit, offset),
                    ).fetchall()
                scanned_rows += len(rows)
            except sqlite3.Error as exc:
                logger.warning("FTS node search failed, falling back to LIKE: %s", exc)
                return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

            raw_nodes = [self._row_to_node(r) for r in rows]
            for node in raw_nodes:
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                if apply_directness_adjustment and node.search_rank is not None:
                    rank_adjustment = max(float(node.search_directness), 0.0)
                    node.search_rank = float(node.search_rank) - (rank_adjustment * 2e-7)
                results.append(node)
            results.sort(key=lambda node: _fts_result_sort_key(node, sort))

            exhausted = len(rows) < fetch_limit or scanned_rows >= candidate_cap
            if source and not exhausted:
                offset += len(rows)
                remaining = candidate_cap - scanned_rows
                if remaining <= 0:
                    return results[:limit]
                fetch_limit = min(fetch_limit * 2, remaining)
                continue

            if exhausted or not apply_directness_adjustment or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = _fts_primary_value(raw_nodes[-1], sort)
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return results[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None) -> List[SummaryNode]:
        safe_query = sanitize_fts5_query(query)
        terms = extract_search_terms(safe_query)
        phrases = extract_quoted_phrases(safe_query)
        if not terms:
            return []
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)

        where: list[str] = ["summary IS NOT NULL"]
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        like_clauses = []
        for term in terms:
            like_clauses.append("summary LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")
        fetch_limit = compute_like_fallback_fetch_limit(limit, terms, phrases)
        base_args = list(args)
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        candidate_cap = compute_search_candidate_cap(limit)
        offset = 0
        scanned_rows = 0
        nodes: list[SummaryNode] = []
        source_match_cache: dict[int, bool] = {}
        while True:
            rows = self._conn.execute(
                f"""SELECT * FROM summary_nodes
                    WHERE {' AND '.join(where)}
                    LIMIT ? OFFSET ?""",
                [*base_args, fetch_limit, offset],
            ).fetchall()
            scanned_rows += len(rows)
            for row in rows:
                node = self._row_to_node(row)
                if source and not self._node_matches_source(node.node_id, source, cache=source_match_cache):
                    continue
                score = sum(
                    min(count_term_matches(node.summary, term), 1) if collapse_risky_repeats else count_term_matches(node.summary, term)
                    for term in terms
                )
                if score <= 0:
                    continue
                node.search_rank = -float(score)
                node.search_directness = compute_directness_score(node.summary, terms, phrases)
                nodes.append(node)

            nodes.sort(key=lambda node: _fallback_result_sort_key(node, sort))
            if not source or len(rows) < fetch_limit or scanned_rows >= candidate_cap:
                return nodes[:limit]

            offset += len(rows)
            remaining = candidate_cap - scanned_rows
            if remaining <= 0:
                return nodes[:limit]
            fetch_limit = min(fetch_limit * 2, remaining)

    # -- DAG traversal ------------------------------------------------------

    def get_source_nodes(self, node: SummaryNode) -> List[SummaryNode]:
        """Get the immediate child nodes of a summary node."""
        if node.source_type != "nodes" or not node.source_ids:
            return []
        placeholders = ",".join("?" * len(node.source_ids))
        rows = self._conn.execute(
            f"""SELECT * FROM summary_nodes
                WHERE node_id IN ({placeholders})
                ORDER BY created_at""",
            node.source_ids,
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def _node_matches_source(
        self,
        node_id: int,
        source: str,
        *,
        cache: dict[int, bool] | None = None,
    ) -> bool:
        if not source:
            return True
        normalized_source = _normalize_source_value(source)
        if cache is not None and node_id in cache:
            return cache[node_id]
        legacy_blank_clause = _legacy_blank_source_clause("m.source")
        row = self._conn.execute(
            f"""
            WITH RECURSIVE source_walk(source_type, source_id) AS (
                SELECT n.source_type, CAST(j.value AS INTEGER)
                FROM summary_nodes n, json_each(n.source_ids) j
                WHERE n.node_id = ?

                UNION ALL

                SELECT child.source_type, CAST(j.value AS INTEGER)
                FROM summary_nodes child
                JOIN source_walk walk
                  ON walk.source_type = 'nodes'
                 AND child.node_id = walk.source_id
                JOIN json_each(child.source_ids) j
            )
            SELECT 1
            FROM source_walk walk
            JOIN messages m
              ON walk.source_type = 'messages'
             AND m.store_id = walk.source_id
            WHERE CASE
                    WHEN ? = ? THEN (m.source = ? OR {legacy_blank_clause})
                    ELSE m.source = ?
                  END
            LIMIT 1
            """,
            (node_id, normalized_source, _UNKNOWN_SOURCE, normalized_source, normalized_source),
        ).fetchone()
        matched = row is not None
        if cache is not None:
            cache[node_id] = matched
        return matched

    def get_source_time_window(self, node_ids: List[int]) -> tuple[float | None, float | None]:
        if not node_ids:
            return None, None
        placeholders = ",".join("?" * len(node_ids))
        row = self._conn.execute(
            f"""SELECT
                    MIN(COALESCE(earliest_at, created_at)),
                    MAX(COALESCE(latest_at, created_at))
                FROM summary_nodes
                WHERE node_id IN ({placeholders})""",
            node_ids,
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    def describe_subtree(self, node_id: int) -> Dict[str, Any]:
        """Return metadata about a node's subtree without loading content."""
        node = self.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        children = []
        if node.source_type == "nodes":
            for child_node in self.get_source_nodes(node):
                children.append({
                    "node_id": child_node.node_id,
                    "depth": child_node.depth,
                    "token_count": child_node.token_count,
                    "source_token_count": child_node.source_token_count,
                    "expand_hint": child_node.expand_hint,
                })

        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "token_count": node.token_count,
            "source_token_count": node.source_token_count,
            "source_type": node.source_type,
            "num_sources": len(node.source_ids),
            "earliest_at": node.earliest_at,
            "latest_at": node.latest_at,
            "expand_hint": node.expand_hint,
            "children": children,
        }

    # -- Helpers ------------------------------------------------------------

    def _row_to_node(self, row) -> SummaryNode:
        return SummaryNode(
            node_id=row[0],
            session_id=row[1],
            depth=row[2],
            summary=row[3],
            token_count=row[4],
            source_token_count=row[5],
            source_ids=json.loads(row[6]) if row[6] else [],
            source_type=row[7],
            created_at=row[8],
            earliest_at=row[9],
            latest_at=row[10],
            expand_hint=row[11] or "",
            search_rank=row[12] if len(row) > 12 else None,
        )

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
