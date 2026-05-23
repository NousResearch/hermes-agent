"""Sidecar helpers for large DAG context/tool-output projections.

Sidecar rows are stored in the additive ``context_message_parts`` table and
reference canonical raw ``messages.id`` rows.  They never delete or rewrite the
raw transcript; projections receive a bounded preview plus a ref/hash that can be
expanded later.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from agent.context_dag_store import ContextDAGStore, _json_dumps, _json_loads

DEFAULT_SIDECAR_THRESHOLD_BYTES = 16_384
DEFAULT_PREVIEW_CHARS = 2_000
SIDECAR_REF_PREFIX = "sidecar://message/"


def _line_count(text: str) -> int:
    if text == "":
        return 0
    return text.count("\n") + 1


class SidecarStore:
    """Session-scoped read/write helpers for message sidecar parts."""

    def __init__(self, store: ContextDAGStore):
        self.store = store
        self.db = store.db

    def _validate_message_owner(self, conn, session_id: str, message_id: int) -> None:
        row = conn.execute("SELECT session_id FROM messages WHERE id = ?", (int(message_id),)).fetchone()
        if row is None or row["session_id"] != session_id:
            raise ValueError(f"message id {message_id!r} must exist and belong to session {session_id!r}")

    def _part_ref(self, message_id: int, part_index: int) -> str:
        return f"{SIDECAR_REF_PREFIX}{int(message_id)}/part/{int(part_index)}"

    def write_message_part(
        self,
        session_id: str,
        message_id: int,
        content: str,
        *,
        part_index: int = 0,
        part_type: str = "tool_output",
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Upsert a sidecar row and return its projection descriptor."""

        text = "" if content is None else str(content)
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        size_bytes = len(encoded)
        preview = text[: max(0, int(preview_chars))]
        line_count = _line_count(text)
        ref = self._part_ref(message_id, part_index)
        now = time.time()
        descriptor = {
            "ref": ref,
            "message_id": int(message_id),
            "part_index": int(part_index),
            "part_type": part_type,
            "sha256": digest,
            "size_bytes": size_bytes,
            "line_count": line_count,
            "preview": preview,
            "preview_chars": len(preview),
            "truncated": len(preview) < len(text),
        }
        effective_metadata = dict(metadata or {})
        effective_metadata.update(
            {
                "sidecar": True,
                "line_count": line_count,
                "preview_chars": len(preview),
                "truncated": len(preview) < len(text),
            }
        )

        def _do(conn):
            self._validate_message_owner(conn, session_id, int(message_id))
            conn.execute(
                """
                INSERT INTO context_message_parts (
                    message_id, part_index, part_type, content_inline, content_ref,
                    sha256, size_bytes, token_estimate, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id, part_index) DO UPDATE SET
                    part_type = excluded.part_type,
                    content_inline = excluded.content_inline,
                    content_ref = excluded.content_ref,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    token_estimate = excluded.token_estimate,
                    metadata_json = excluded.metadata_json
                """,
                (
                    int(message_id),
                    int(part_index),
                    part_type,
                    text,
                    ref,
                    digest,
                    size_bytes,
                    max(1, (len(text) + 3) // 4) if text else 0,
                    _json_dumps(effective_metadata),
                    now,
                ),
            )
            return conn.execute(
                """
                SELECT * FROM context_message_parts
                WHERE message_id = ? AND part_index = ?
                """,
                (int(message_id), int(part_index)),
            ).fetchone()

        row = self.db._execute_write(_do)
        # Return stable descriptor values rather than raw DB rows, so repeated
        # writes are equality/idempotency friendly even if created_at differs.
        return descriptor

    def list_message_parts(self, session_id: str, message_id: int) -> List[Dict[str, Any]]:
        with self.db._lock:
            self._validate_message_owner(self.db._conn, session_id, int(message_id))
            rows = self.db._conn.execute(
                """
                SELECT * FROM context_message_parts
                WHERE message_id = ?
                ORDER BY part_index
                """,
                (int(message_id),),
            ).fetchall()
        return [self._row_to_part(row) for row in rows]

    def read_part_page(
        self,
        session_id: str,
        message_id: int,
        *,
        part_index: int = 0,
        offset: int = 0,
        max_chars: int = DEFAULT_PREVIEW_CHARS,
    ) -> Dict[str, Any]:
        parts = self.list_message_parts(session_id, message_id)
        selected = next((part for part in parts if part["part_index"] == int(part_index)), None)
        if selected is None:
            raise ValueError(f"sidecar part {part_index} for message {message_id} was not found")
        text = selected.get("content") or ""
        start = max(0, int(offset or 0))
        end = start + max(1, int(max_chars or DEFAULT_PREVIEW_CHARS))
        page = text[start:end]
        selected.update(
            {
                "content": page,
                "offset": start,
                "returned_chars": len(page),
                "truncated": end < len(text),
                "omitted_chars": max(0, len(text) - end),
            }
        )
        return selected

    def _row_to_part(self, row) -> Dict[str, Any]:
        metadata = _json_loads(row["metadata_json"], {})
        text = row["content_inline"] or ""
        return {
            "message_id": row["message_id"],
            "part_index": row["part_index"],
            "part_type": row["part_type"],
            "content": text,
            "ref": row["content_ref"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "token_estimate": row["token_estimate"],
            "line_count": metadata.get("line_count", _line_count(text)),
            "metadata": metadata,
        }


def project_tool_output_with_sidecar(
    store: ContextDAGStore,
    *,
    session_id: str,
    message_id: int,
    content: str,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
    threshold_bytes: int = DEFAULT_SIDECAR_THRESHOLD_BYTES,
) -> Dict[str, Any]:
    """Return a projection item for a tool output, writing sidecar if large."""

    text = "" if content is None else str(content)
    projection: Dict[str, Any] = {"role": "tool", "content": text}
    if len(text.encode("utf-8")) <= int(threshold_bytes):
        return projection

    descriptor = SidecarStore(store).write_message_part(
        session_id,
        int(message_id),
        text,
        part_index=0,
        part_type="tool_output",
        preview_chars=preview_chars,
    )
    projection["content"] = descriptor["preview"]
    projection["sidecar"] = {key: descriptor[key] for key in descriptor if key != "preview"}
    return projection
