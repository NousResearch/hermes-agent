"""Display-index projections for SessionDB: dedupe key, generation folding, drift reconcile
and the legacy read-only display page. Mixin bound via the MRO, built on SessionDB's
_read_ctx/_execute_write/_read_* primitives and the content decoders owned by SessionMessagesMixin."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from agent.context_compressor import MODEL_ONLY_DISPLAY_METADATA_KEY, split_user_originated_turn
from hermes_state_common import _placeholders
from hermes_state_identity import _stable_tool_key
from hermes_state_messages import _DISPLAY_INDEX_MISSING_SQL, _DISPLAY_ORDER_DRIFT_SQL


class SessionDisplayMixin:
    def _display_dedupe_key(self, row) -> tuple[Any, ...]:
        """Historical display identity, including normalized live content from user handoff carriers."""
        dedupe_content = row["content"]
        if row["role"] == "user":
            handoff, live_view = split_user_originated_turn({
                "role": "user", "content": self._decode_content(row["content"]),
                "display_kind": row["display_kind"],
                "display_metadata": self._decode_display_metadata(row["display_metadata"])})
            if handoff is not None and live_view is not None:
                dedupe_content = self._encode_content(live_view.get("content"))
        return _stable_tool_key(row) or (
            row["role"], dedupe_content, row["timestamp"], row["tool_call_id"], row["tool_calls"], row["tool_name"])

    def _is_model_only_row(self, row) -> bool:
        """Python twin of :data:`DISPLAY_VISIBLE_SQL`."""
        return bool((self._decode_display_metadata(row["display_metadata"]) or {}).get(MODEL_ONLY_DISPLAY_METADATA_KEY))

    @staticmethod
    def _display_identity(key: tuple[Any, ...]) -> bytes:
        """Fixed-width durable identity for indexed display-generation lookup."""
        return hashlib.sha256(repr(key).encode("utf-8", "surrogatepass")).digest()

    def _dedupe_display_generations(self, rows):
        """Collapse compaction generations so each logical message appears once (the protected tail is copied
        into each generation: same role/content/timestamp, different ``active``/id); prefer the live row, then
        the newest. The ONE definition every display projection shares. *rows* must be ordered by ``id``."""
        seen: dict[tuple[Any, ...], Any] = {}
        first_id: dict[tuple[Any, ...], int] = {}
        for row in rows:
            if self._is_model_only_row(row):
                continue
            key = self._display_dedupe_key(row)
            cur = seen.get(key)
            if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
                seen[key] = row
            first_id[key] = min(first_id.get(key, row["id"]), row["id"])
        # Order by the logical message's FIRST row, not the chosen representative's: a protected-tail
        # copy in a newer generation has a higher id than messages emitted after the original.
        return [seen[key] for key in sorted(seen, key=first_id.__getitem__)]

    def _reconcile_display_orders(self, conn, session_id: str) -> None:
        """Re-fold split display generations by the recomputed display key.

        A compaction generation that computed the display identity differently (or cloned
        without inheriting it) leaves one logical message in two ``display_order`` groups,
        and ``GROUP BY display_order`` then projects it twice (#122167). Folding by the same
        recomputed key :meth:`_dedupe_display_generations` uses keeps every display projection
        on one definition of a logical message; the live copy wins its group via the read
        path's ``ORDER BY candidate.active DESC, candidate.id DESC``. Writes only on drift."""
        first_id: dict[bytes, int] = {}
        last_id = 0
        while True:
            rows = conn.execute(
                "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name, "
                "display_kind, display_metadata, display_order, display_identity "
                "FROM messages INDEXED BY idx_messages_session_id "
                "WHERE session_id = ? AND id > ? AND (active = 1 OR compacted = 1) "
                "ORDER BY id LIMIT 1000",
                (session_id, last_id))
            batch_start = last_id
            updates = []
            for row in rows:
                last_id = row["id"]
                identity = self._display_identity(self._display_dedupe_key(row))
                order = first_id.setdefault(identity, last_id)
                if order != row["display_order"] or identity != row["display_identity"]:
                    updates.append((order, identity, last_id))
            rows.close()
            if last_id == batch_start:
                break
            conn.executemany(
                "UPDATE messages SET display_order = ?, display_identity = ? WHERE id = ?", updates)

    def _ensure_display_order(self, session_id: str) -> bool:
        """Ensure indexed display generations are folded before a display read.

        Besides legacy rows missing the index, an interrupted or older compaction can leave
        every row indexed but split across multiple generations. Reconcile both cases before
        projecting ``GROUP BY display_order``; the reconciler only writes rows whose durable
        identity actually drifted.
        """
        with self._read_ctx() as conn:
            columns = set(self._message_column_names(conn))
        if not {"display_order", "display_identity"} <= columns:
            return False
        if getattr(self, "read_only", False):
            # The read-only fallback deduplicates by recomputed identity in Python.
            return False
        if (self._read_one(_DISPLAY_INDEX_MISSING_SQL, (session_id,)) is None
                and self._read_one(_DISPLAY_ORDER_DRIFT_SQL, (session_id,)) is None):
            return True

        def _do(conn):
            self._reconcile_display_orders(conn, session_id)
            return True

        return bool(self._execute_write(_do))

    def _legacy_display_page(self, session_id: str, *, active_clause: str, limit: Optional[int], offset: int,
                             latest: bool) -> list[Any]:
        """Project a legacy read-only display page without retaining transcript payloads."""
        representatives: dict[bytes, tuple[int, int]] = {}
        with self._read_ctx() as conn:
            conn.execute("BEGIN")
            try:
                has_session_index = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                    ("idx_messages_session_id",),
                ).fetchone() is not None
                index_hint = "INDEXED BY idx_messages_session_id" if has_session_index else "NOT INDEXED"
                rows = conn.execute(
                    "SELECT id, role, content, timestamp, tool_call_id, tool_calls, tool_name, active, "
                    f"display_kind, display_metadata FROM messages {index_hint} "
                    f"WHERE session_id = ?{active_clause} ORDER BY id ASC",
                    (session_id,))
                for row in rows:
                    if self._is_model_only_row(row):
                        continue
                    identity = self._display_identity(self._display_dedupe_key(row))
                    current = representatives.get(identity)
                    candidate = (row["active"], row["id"])
                    if current is None or candidate > current:
                        representatives[identity] = candidate
                rows.close()

                identities = list(representatives)
                identities = identities[::-1][offset:][:limit][::-1] if latest else identities[offset:][:limit]
                selected_ids = [representatives[identity][1] for identity in identities]
                selected = {}
                for start in range(0, len(selected_ids), 900):
                    chunk = selected_ids[start:start + 900]
                    selected.update({row["id"]: row for row in conn.execute(
                        f"SELECT * FROM messages WHERE session_id = ?{active_clause} "
                        f"AND id IN ({_placeholders(chunk)})",
                        (session_id, *chunk))})
                return [selected[row_id] for row_id in selected_ids if row_id in selected]
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
