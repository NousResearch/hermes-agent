"""SQLite persistence helpers for Hermes DAG context metadata.

This module is deliberately storage-only for PR1: it creates no runtime context
engine behavior and only wraps the additive tables initialized by SessionDB.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from agent.context_dag_models import (
    Checkpoint,
    MutationLogEntry,
    Projection,
    SummaryNode,
    SummarySource,
)
from hermes_state import SessionDB


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Optional[str], default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


class ContextDAGStore:
    """Session-scoped store for summary DAG, projection, and checkpoints."""

    def __init__(self, session_db: SessionDB):
        self.db = session_db

    @staticmethod
    def deterministic_message_hash(message: Dict[str, Any]) -> str:
        """Return stable SHA-256 for the PR1 message identity contract.

        Canonical raw identity is still SQLite ``messages.id``. This hash is
        reconciliation metadata built from normalized OpenAI/Hermes message
        fields: role, content, tool_calls, tool_call_id, name/tool_name, and
        reasoning (including common provider-specific reasoning fields when
        present).
        """

        canonical = {
            "role": message.get("role"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name", message.get("tool_name")),
            "reasoning": message.get("reasoning"),
            "reasoning_content": message.get("reasoning_content"),
            "reasoning_details": message.get("reasoning_details"),
        }
        return hashlib.sha256(_json_dumps(canonical).encode("utf-8")).hexdigest()

    @staticmethod
    def deterministic_source_hash(parts: List[Dict[str, Any]]) -> str:
        """Return stable SHA-256 for a list of source descriptors/messages."""

        return hashlib.sha256(_json_dumps(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _summary_id(session_id: str, kind: str, source_hash: str, prompt_version: Optional[str]) -> str:
        return ContextDAGStore.compute_summary_id(session_id, kind, source_hash, prompt_version)

    @staticmethod
    def compute_summary_id(session_id: str, kind: str, source_hash: str, prompt_version: Optional[str]) -> str:
        """Return the deterministic id for a summary identity tuple."""

        raw = _json_dumps(
            {
                "session_id": session_id,
                "kind": kind,
                "source_hash": source_hash,
                "prompt_version": prompt_version,
            }
        )
        return "ctxsum_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _row_to_summary(row) -> SummaryNode:
        return SummaryNode(
            id=row["id"],
            session_id=row["session_id"],
            kind=row["kind"],
            summary_text=row["summary_text"],
            status=row["status"],
            source_hash=row["source_hash"],
            summary_hash=row["summary_hash"],
            prompt_version=row["prompt_version"],
            summary_model=row["summary_model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            token_estimate=row["token_estimate"],
            metadata=_json_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _row_to_source(row) -> SummarySource:
        return SummarySource(
            id=row["id"],
            summary_id=row["summary_id"],
            source_type=row["source_type"],
            source_id=row["source_id"] or "",
            start_message_id=row["start_message_id"],
            end_message_id=row["end_message_id"],
            start_offset=None if row["start_offset"] == -1 else row["start_offset"],
            end_offset=None if row["end_offset"] == -1 else row["end_offset"],
            metadata=_json_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_projection(row) -> Projection:
        return Projection(
            id=row["id"],
            session_id=row["session_id"],
            engine_version=row["engine_version"],
            status=row["status"],
            projection=_json_loads(row["projection_json"], []),
            fresh_tail_start_message_id=row["fresh_tail_start_message_id"],
            latest_raw_message_id=row["latest_raw_message_id"],
            token_estimate=row["token_estimate"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _row_to_checkpoint(row) -> Checkpoint:
        return Checkpoint(
            session_id=row["session_id"],
            last_ingested_message_id=row["last_ingested_message_id"],
            last_projection_message_id=row["last_projection_message_id"],
            last_anchor_message_id=row["last_anchor_message_id"],
            anchor_hash=row["anchor_hash"],
            updated_at=row["updated_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _row_to_mutation(row) -> MutationLogEntry:
        return MutationLogEntry(
            id=row["id"],
            session_id=row["session_id"],
            operation=row["operation"],
            status=row["status"],
            idempotency_key=row["idempotency_key"],
            payload=_json_loads(row["payload_json"], {}),
            error=row["error"],
            attempts=row["attempts"] if "attempts" in row.keys() else 0,
            max_attempts=row["max_attempts"] if "max_attempts" in row.keys() else 3,
            claimed_at=row["claimed_at"] if "claimed_at" in row.keys() else None,
            claimed_by=row["claimed_by"] if "claimed_by" in row.keys() else None,
            available_at=row["available_at"] if "available_at" in row.keys() else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_json_loads(row["metadata_json"], {}),
        )

    @staticmethod
    def _validate_session_message_ids(conn, session_id: str, message_ids, context: str) -> None:
        """Ensure every non-null message id exists and belongs to ``session_id``."""

        unique_message_ids = {message_id for message_id in message_ids if message_id is not None}
        if not unique_message_ids:
            return
        placeholders = ", ".join("?" for _ in unique_message_ids)
        rows = conn.execute(
            f"""
            SELECT id FROM messages
            WHERE session_id = ? AND id IN ({placeholders})
            """,
            (session_id, *unique_message_ids),
        ).fetchall()
        found_message_ids = {row["id"] for row in rows}
        missing_or_cross_session = unique_message_ids - found_message_ids
        if missing_or_cross_session:
            raise ValueError(
                f"{context} message ids must exist and belong to "
                f"session {session_id!r}: {sorted(missing_or_cross_session)!r}"
            )

    def _create_summary_node_conn(
        self,
        conn,
        *,
        session_id: str,
        summary_text: str,
        kind: str = "leaf",
        status: str = "valid",
        source_hash: Optional[str] = None,
        summary_hash: Optional[str] = None,
        prompt_version: Optional[str] = None,
        summary_model: Optional[str] = None,
        token_estimate: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ):
        if node_id is None:
            if source_hash:
                node_id = self.compute_summary_id(session_id, kind, source_hash, prompt_version)
            else:
                node_id = "ctxsum_" + hashlib.sha256(
                    f"{session_id}:{kind}:{summary_text}".encode("utf-8")
                ).hexdigest()[:32]
        if summary_hash is None:
            summary_hash = hashlib.sha256(summary_text.encode("utf-8")).hexdigest()
        metadata_json = _json_dumps(metadata or {})
        now = time.time()

        effective_node_id = node_id
        if source_hash:
            existing = conn.execute(
                """
                SELECT id FROM context_summary_nodes
                WHERE session_id = ? AND kind = ? AND source_hash = ?
                  AND prompt_version IS ?
                """,
                (session_id, kind, source_hash, prompt_version),
            ).fetchone()
            if existing is not None:
                effective_node_id = existing["id"]
        conn.execute(
            """
            INSERT INTO context_summary_nodes (
                id, session_id, kind, summary_text, status, source_hash,
                summary_hash, prompt_version, summary_model, created_at,
                updated_at, token_estimate, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary_text = excluded.summary_text,
                status = excluded.status,
                source_hash = excluded.source_hash,
                summary_hash = excluded.summary_hash,
                prompt_version = excluded.prompt_version,
                summary_model = excluded.summary_model,
                updated_at = excluded.updated_at,
                token_estimate = excluded.token_estimate,
                metadata_json = excluded.metadata_json
            """,
            (
                effective_node_id,
                session_id,
                kind,
                summary_text,
                status,
                source_hash,
                summary_hash,
                prompt_version,
                summary_model,
                now,
                now,
                token_estimate,
                metadata_json,
            ),
        )
        return conn.execute(
            "SELECT * FROM context_summary_nodes WHERE id = ? AND session_id = ?",
            (effective_node_id, session_id),
        ).fetchone()

    def create_summary_node(
        self,
        *,
        session_id: str,
        summary_text: str,
        kind: str = "leaf",
        status: str = "valid",
        source_hash: Optional[str] = None,
        summary_hash: Optional[str] = None,
        prompt_version: Optional[str] = None,
        summary_model: Optional[str] = None,
        token_estimate: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> SummaryNode:
        """Idempotently create/update a session-scoped summary node."""

        def _do(conn):
            return self._create_summary_node_conn(
                conn,
                session_id=session_id,
                summary_text=summary_text,
                kind=kind,
                status=status,
                source_hash=source_hash,
                summary_hash=summary_hash,
                prompt_version=prompt_version,
                summary_model=summary_model,
                token_estimate=token_estimate,
                metadata=metadata,
                node_id=node_id,
            )

        return self._row_to_summary(self.db._execute_write(_do))

    def create_summary_node_with_links(
        self,
        *,
        session_id: str,
        summary_text: str,
        kind: str,
        source_hash: str,
        prompt_version: Optional[str],
        summary_model: Optional[str] = None,
        token_estimate: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        summary_hash: Optional[str] = None,
        status: str = "valid",
    ) -> SummaryNode:
        """Atomically upsert a summary node and all required DAG links.

        This is the compactor's preferred write path: if any source or edge
        insert fails, the node upsert is rolled back too, preventing a valid
        orphan summary from being persisted. Replays are idempotent because the
        node id, sources, and edges all use deterministic/upsert identities.
        """

        def _do(conn):
            row = self._create_summary_node_conn(
                conn,
                session_id=session_id,
                summary_text=summary_text,
                kind=kind,
                status=status,
                source_hash=source_hash,
                summary_hash=summary_hash,
                prompt_version=prompt_version,
                summary_model=summary_model,
                token_estimate=token_estimate,
                metadata=metadata,
            )
            summary_id = row["id"]
            for edge in edges or []:
                self._add_summary_edge_conn(
                    conn,
                    session_id=session_id,
                    parent_id=edge["parent_id"],
                    child_id=edge["child_id"],
                    edge_order=edge.get("edge_order", 0),
                )
            for source in sources or []:
                self._link_summary_source_conn(
                    conn,
                    session_id=session_id,
                    summary_id=source.get("summary_id") or summary_id,
                    source_type=source["source_type"],
                    start_message_id=source.get("start_message_id"),
                    end_message_id=source.get("end_message_id"),
                    source_id=source.get("source_id", ""),
                    start_offset=source.get("start_offset"),
                    end_offset=source.get("end_offset"),
                    metadata=source.get("metadata"),
                )
            return conn.execute(
                "SELECT * FROM context_summary_nodes WHERE id = ? AND session_id = ?",
                (summary_id, session_id),
            ).fetchone()

        return self._row_to_summary(self.db._execute_write(_do))

    def get_summary_node(self, session_id: str, summary_id: str) -> Optional[SummaryNode]:
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT * FROM context_summary_nodes WHERE id = ? AND session_id = ?",
                (summary_id, session_id),
            ).fetchone()
        return self._row_to_summary(row) if row else None

    def list_summary_nodes(self, session_id: str, status: str = "valid") -> List[SummaryNode]:
        """Return summary nodes for a session, ordered deterministically."""

        with self.db._lock:
            rows = self.db._conn.execute(
                """
                SELECT * FROM context_summary_nodes
                WHERE session_id = ? AND status = ?
                ORDER BY created_at, id
                """,
                (session_id, status),
            ).fetchall()
        return [self._row_to_summary(row) for row in rows]

    def _link_summary_source_conn(
        self,
        conn,
        *,
        session_id: str,
        summary_id: str,
        source_type: str,
        start_message_id: Optional[int] = None,
        end_message_id: Optional[int] = None,
        source_id: str = "",
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        metadata_json = _json_dumps(metadata or {})
        now = time.time()
        owner = conn.execute(
            "SELECT 1 FROM context_summary_nodes WHERE id = ? AND session_id = ?",
            (summary_id, session_id),
        ).fetchone()
        if owner is None:
            raise ValueError(f"summary {summary_id!r} does not belong to session {session_id!r}")
        self._validate_session_message_ids(
            conn,
            session_id,
            (start_message_id, end_message_id),
            "summary source",
        )
        conn.execute(
            """
            INSERT INTO context_summary_sources (
                summary_id, source_type, source_id, start_message_id,
                end_message_id, start_offset, end_offset, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET metadata_json = excluded.metadata_json
            """,
            (
                summary_id,
                source_type,
                source_id or "",
                start_message_id,
                end_message_id,
                start_offset if start_offset is not None else -1,
                end_offset if end_offset is not None else -1,
                metadata_json,
                now,
            ),
        )
        return conn.execute(
            """
            SELECT * FROM context_summary_sources
            WHERE summary_id = ? AND source_type = ? AND source_id = ?
              AND start_message_id IS ? AND end_message_id IS ?
              AND start_offset IS ? AND end_offset IS ?
            """,
            (
                summary_id,
                source_type,
                source_id or "",
                start_message_id,
                end_message_id,
                start_offset if start_offset is not None else -1,
                end_offset if end_offset is not None else -1,
            ),
        ).fetchone()

    def link_summary_source(
        self,
        *,
        session_id: str,
        summary_id: str,
        source_type: str,
        start_message_id: Optional[int] = None,
        end_message_id: Optional[int] = None,
        source_id: str = "",
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SummarySource:
        """Idempotently attach a raw source span to a summary in one session."""

        def _do(conn):
            return self._link_summary_source_conn(
                conn,
                session_id=session_id,
                summary_id=summary_id,
                source_type=source_type,
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                source_id=source_id,
                start_offset=start_offset,
                end_offset=end_offset,
                metadata=metadata,
            )

        return self._row_to_source(self.db._execute_write(_do))

    def get_summary_sources(self, session_id: str, summary_id: str) -> List[SummarySource]:
        with self.db._lock:
            rows = self.db._conn.execute(
                """
                SELECT s.* FROM context_summary_sources s
                JOIN context_summary_nodes n ON n.id = s.summary_id
                WHERE s.summary_id = ? AND n.session_id = ?
                ORDER BY s.id
                """,
                (summary_id, session_id),
            ).fetchall()
        return [self._row_to_source(row) for row in rows]

    def _add_summary_edge_conn(
        self, conn, *, session_id: str, parent_id: str, child_id: str, edge_order: int = 0
    ) -> None:
        count = conn.execute(
            "SELECT COUNT(*) FROM context_summary_nodes WHERE session_id = ? AND id IN (?, ?)",
            (session_id, parent_id, child_id),
        ).fetchone()[0]
        if count != 2:
            raise ValueError("parent and child summaries must belong to the same session")
        conn.execute(
            """
            INSERT INTO context_summary_edges (parent_id, child_id, edge_order)
            VALUES (?, ?, ?)
            ON CONFLICT(parent_id, child_id) DO UPDATE SET edge_order = excluded.edge_order
            """,
            (parent_id, child_id, edge_order),
        )

    def add_summary_edge(
        self, *, session_id: str, parent_id: str, child_id: str, edge_order: int = 0
    ) -> None:
        def _do(conn):
            self._add_summary_edge_conn(
                conn,
                session_id=session_id,
                parent_id=parent_id,
                child_id=child_id,
                edge_order=edge_order,
            )

        self.db._execute_write(_do)

    def write_active_projection(
        self,
        *,
        session_id: str,
        engine_version: str,
        projection: List[Dict[str, Any]],
        fresh_tail_start_message_id: Optional[int] = None,
        latest_raw_message_id: Optional[int] = None,
        token_estimate: Optional[int] = None,
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Projection:
        now = time.time()
        projection_json = _json_dumps(projection)
        metadata_json = _json_dumps(metadata or {})

        def _do(conn):
            self._validate_session_message_ids(
                conn,
                session_id,
                (fresh_tail_start_message_id, latest_raw_message_id),
                "projection cursor",
            )
            conn.execute(
                """
                INSERT INTO context_projection (
                    session_id, engine_version, status, projection_json,
                    fresh_tail_start_message_id, latest_raw_message_id,
                    token_estimate, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, engine_version, status) DO UPDATE SET
                    projection_json = excluded.projection_json,
                    fresh_tail_start_message_id = excluded.fresh_tail_start_message_id,
                    latest_raw_message_id = excluded.latest_raw_message_id,
                    token_estimate = excluded.token_estimate,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session_id,
                    engine_version,
                    status,
                    projection_json,
                    fresh_tail_start_message_id,
                    latest_raw_message_id,
                    token_estimate,
                    now,
                    now,
                    metadata_json,
                ),
            )
            return conn.execute(
                """
                SELECT * FROM context_projection
                WHERE session_id = ? AND engine_version = ? AND status = ?
                """,
                (session_id, engine_version, status),
            ).fetchone()

        return self._row_to_projection(self.db._execute_write(_do))

    def read_active_projection(
        self, session_id: str, engine_version: str, status: str = "active"
    ) -> Optional[Projection]:
        with self.db._lock:
            row = self.db._conn.execute(
                """
                SELECT * FROM context_projection
                WHERE session_id = ? AND engine_version = ? AND status = ?
                """,
                (session_id, engine_version, status),
            ).fetchone()
        return self._row_to_projection(row) if row else None

    def write_checkpoint(
        self,
        *,
        session_id: str,
        last_ingested_message_id: Optional[int] = None,
        last_projection_message_id: Optional[int] = None,
        last_anchor_message_id: Optional[int] = None,
        anchor_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        now = time.time()
        metadata_json = _json_dumps(metadata or {})

        def _do(conn):
            self._validate_session_message_ids(
                conn,
                session_id,
                (last_ingested_message_id, last_projection_message_id, last_anchor_message_id),
                "checkpoint cursor",
            )
            conn.execute(
                """
                INSERT INTO context_checkpoints (
                    session_id, last_ingested_message_id, last_projection_message_id,
                    last_anchor_message_id, anchor_hash, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_ingested_message_id = excluded.last_ingested_message_id,
                    last_projection_message_id = excluded.last_projection_message_id,
                    last_anchor_message_id = excluded.last_anchor_message_id,
                    anchor_hash = excluded.anchor_hash,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session_id,
                    last_ingested_message_id,
                    last_projection_message_id,
                    last_anchor_message_id,
                    anchor_hash,
                    now,
                    metadata_json,
                ),
            )
            return conn.execute(
                "SELECT * FROM context_checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        return self._row_to_checkpoint(self.db._execute_write(_do))

    def read_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT * FROM context_checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def append_mutation_log(
        self,
        *,
        session_id: str,
        operation: str,
        status: str,
        idempotency_key: str,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MutationLogEntry:
        """Append a mutation record idempotently by ``idempotency_key``."""

        now = time.time()
        payload_json = _json_dumps(payload or {})
        metadata_json = _json_dumps(metadata or {})

        def _do(conn):
            conn.execute(
                """
                INSERT INTO context_mutation_log (
                    session_id, operation, status, idempotency_key, payload_json,
                    error, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, idempotency_key) DO UPDATE SET
                    status = excluded.status,
                    error = excluded.error,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    session_id,
                    operation,
                    status,
                    idempotency_key,
                    payload_json,
                    error,
                    now,
                    now,
                    metadata_json,
                ),
            )
            return conn.execute(
                "SELECT * FROM context_mutation_log WHERE session_id = ? AND idempotency_key = ?",
                (session_id, idempotency_key),
            ).fetchone()

        return self._row_to_mutation(self.db._execute_write(_do))
