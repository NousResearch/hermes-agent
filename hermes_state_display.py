"""Authoritative persisted transcript pages and their display identity."""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from hermes_state_common import (
    MESSAGE_CLONE_LINEAGE_LEGACY_CEILING_KEY,
    _NOT_BRANCH_MARKER_CHILD_SQL,
    _NOT_DELEGATE_MARKER_CHILD_SQL,
    _RESET_END_REASONS_SQL,
    _legacy_reset_child_sql,
    _sql_session_last_active,
    escape_like as _escape_like,
)


logger = logging.getLogger("hermes_state")


class SessionDisplayMixin:
    """Display lineage, revision, paging, and clone-provenance operations."""

    def _resolve_session_id_from_conn(self, conn, session_id_or_prefix: str) -> Optional[str]:
        exact = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id_or_prefix,)
        ).fetchone()
        if exact:
            return exact["id"]
        escaped = _escape_like(session_id_or_prefix)
        matches = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                "ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            ).fetchall()
        ]
        return matches[0] if len(matches) == 1 else None

    def _display_lineage_root_from_conn(self, conn, session_id: str) -> str:
        """Resolve a session's backward compression-only display root."""
        current = session_id
        seen = set()
        for _ in range(100):
            if not current or current in seen:
                break
            seen.add(current)
            row = conn.execute(
                """
                SELECT child.id, child.parent_session_id, parent.end_reason,
                       json_extract(COALESCE(child.model_config, '{}'), '$._branched_from') AS branched_from,
                       json_extract(COALESCE(child.model_config, '{}'), '$._delegate_from') AS delegated_from,
                       child.source
                FROM sessions child
                LEFT JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE child.id = ?
                """,
                (current,),
            ).fetchone()
            if row is None:
                break
            if (
                not row["parent_session_id"]
                or row["end_reason"] != "compression"
                or row["branched_from"] == row["parent_session_id"]
                or row["delegated_from"] == row["parent_session_id"]
                or (row["source"] or "") == "tool"
            ):
                break
            current = row["parent_session_id"]
        return current or session_id

    @staticmethod
    def _display_revision_from_conn(conn, lineage_root_id: str) -> int:
        row = conn.execute(
            "SELECT revision FROM conversation_display_revisions WHERE lineage_root_id = ?",
            (lineage_root_id,),
        ).fetchone()
        return int(row["revision"]) if row else 0

    def _display_roots_from_conn(self, conn, session_ids) -> set[str]:
        return {
            self._display_lineage_root_from_conn(conn, session_id)
            for session_id in session_ids
            if session_id
        }

    def _invalidate_display_topology(self, conn, roots_before: set[str], affected_session_ids) -> None:
        """Advance old and new roots to one shared value after a topology write."""
        roots = set(roots_before)
        roots.update(self._display_roots_from_conn(conn, affected_session_ids))
        roots.discard("")
        if not roots:
            return
        next_revision = max(self._display_revision_from_conn(conn, root) for root in roots) + 1
        now = time.time()
        for root_id in roots:
            conn.execute(
                """
                INSERT INTO conversation_display_revisions (lineage_root_id, revision, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(lineage_root_id) DO UPDATE SET
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (root_id, next_revision, now),
            )

    def _capture_display_ancestor_roots(self, conn, session_ids) -> set[str]:
        roots: set[str] = set()
        for session_id in session_ids:
            if session_id:
                roots.update(self._display_ancestor_roots_for_session_from_conn(conn, session_id))
        return roots

    @staticmethod
    def _surviving_children_for_deleted_sessions(conn, deleted_session_ids) -> set[str]:
        deleted_ids = {session_id for session_id in deleted_session_ids if session_id}
        if not deleted_ids:
            return set()
        ids = list(deleted_ids)
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id FROM sessions WHERE parent_session_id IN ({placeholders}) "
            f"AND id NOT IN ({placeholders})",
            [*ids, *ids],
        ).fetchall()
        return {row["id"] for row in rows}

    def _bump_display_revision(
        self, conn, session_id: str, roots_before: Optional[set[str]] = None
    ) -> int:
        own_root_id = self._display_lineage_root_from_conn(conn, session_id)
        roots = set(roots_before or ())
        roots.update(self._display_ancestor_roots_for_session_from_conn(conn, session_id))
        revisions = {root: self._bump_display_revision_root(conn, root) for root in roots}
        return revisions[own_root_id]

    def _display_ancestor_roots_for_session_from_conn(self, conn, session_id: str) -> set[str]:
        roots = {self._display_lineage_root_from_conn(conn, session_id)}
        checked_roots = set(roots)
        current = session_id
        seen = {current}
        for _ in range(100):
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?", (current,)
            ).fetchone()
            if row is None or not row["parent_session_id"]:
                break
            parent_id = row["parent_session_id"]
            if parent_id in seen:
                break
            seen.add(parent_id)
            root_id = self._display_lineage_root_from_conn(conn, parent_id)
            if root_id not in checked_roots:
                checked_roots.add(root_id)
                _tip_id, lineage_ids = self._resolve_resume_lineage_from_conn(conn, root_id)
                if session_id not in lineage_ids:
                    break
                roots.add(root_id)
            current = parent_id
        return roots

    def _bump_display_revision_root(self, conn, lineage_root_id: str) -> int:
        conn.execute(
            """
            INSERT INTO conversation_display_revisions (lineage_root_id, revision, updated_at)
            VALUES (?, 1, ?)
            ON CONFLICT(lineage_root_id) DO UPDATE SET
                revision = conversation_display_revisions.revision + 1,
                updated_at = excluded.updated_at
            """,
            (lineage_root_id, time.time()),
        )
        return self._display_revision_from_conn(conn, lineage_root_id)

    def get_display_revision(self, session_id: str) -> int:
        with self._read_ctx() as conn:
            root_id = self._display_lineage_root_from_conn(conn, session_id)
            return self._display_revision_from_conn(conn, root_id)

    def get_display_lineage_identity(self, session_id: str) -> tuple[str, str]:
        with self._read_ctx() as conn:
            root_id = self._display_lineage_root_from_conn(conn, session_id)
            return root_id, self._get_compression_tip_from_conn(conn, root_id)

    def get_display_revisions(self, lineage_root_ids: list[str]) -> dict[str, int]:
        root_ids = list(dict.fromkeys(root for root in lineage_root_ids if root))
        revisions = {root: 0 for root in root_ids}
        if not root_ids:
            return revisions
        with self._read_ctx() as conn:
            for offset in range(0, len(root_ids), 900):
                chunk = root_ids[offset : offset + 900]
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT lineage_root_id, revision FROM conversation_display_revisions "
                    f"WHERE lineage_root_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                revisions.update({row["lineage_root_id"]: int(row["revision"]) for row in rows})
        return revisions

    def _get_compression_chain_from_conn(self, conn, session_id: str) -> List[str]:
        current = session_id
        chain = [current] if current else []
        seen = {current} if current else set()
        for _ in range(100):
            row = conn.execute(
                f"""
                SELECT child.id
                FROM sessions parent
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.id = ?
                  AND parent.end_reason = 'compression'
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="child.")
                + f"""
                ORDER BY
                  CASE WHEN child.end_reason = 'compression' THEN 0
                       WHEN child.ended_at IS NULL THEN 1 ELSE 2 END,
                  {_sql_session_last_active('child')} DESC,
                  child.started_at DESC,
                  child.id DESC
                LIMIT 1
                """,
                (current, current, current),
            ).fetchone()
            if row is None:
                return chain
            child_id = row["id"]
            if not child_id or child_id in seen:
                return chain
            seen.add(child_id)
            current = child_id
            chain.append(child_id)
        return chain

    def _get_compression_tip_from_conn(self, conn, session_id: str) -> Optional[str]:
        chain = self._get_compression_chain_from_conn(conn, session_id)
        return chain[-1] if chain else session_id

    @staticmethod
    def _path_between_sessions_from_conn(conn, root_id: str, tip_id: str) -> list[str]:
        reverse_path = []
        current = tip_id
        seen = set()
        for _ in range(100):
            if not current or current in seen:
                return [root_id]
            seen.add(current)
            reverse_path.append(current)
            if current == root_id:
                return list(reversed(reverse_path))
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                return [root_id]
            current = row["parent_session_id"]
        return [root_id]

    def _resolve_resume_lineage_from_conn(self, conn, session_id: str) -> tuple[str, list[str]]:
        compression_tip = self._get_compression_tip_from_conn(conn, session_id)
        fallback_tip = compression_tip or session_id
        fallback_path = self._path_between_sessions_from_conn(conn, session_id, fallback_tip)
        current = fallback_tip
        current_path = list(fallback_path)
        seen = set(current_path)
        best_id = None
        best_path = None
        for _ in range(32):
            if conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (current,)
            ).fetchone() is not None:
                best_id, best_path = current, list(current_path)
            child_row = conn.execute(
                "SELECT id FROM sessions AS child WHERE child.parent_session_id = ? "
                f"AND {_NOT_BRANCH_MARKER_CHILD_SQL.format(a='child')} "
                f"AND {_NOT_DELEGATE_MARKER_CHILD_SQL.format(a='child')} "
                "AND json_extract(COALESCE(child.model_config, '{}'), '$._reset_from') IS NULL "
                f"AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
                "AND COALESCE(child.source, '') != 'tool' "
                "ORDER BY child.started_at DESC, child.id DESC LIMIT 1",
                (current,),
            ).fetchone()
            if child_row is None:
                break
            child_id = child_row["id"]
            if not child_id or child_id in seen:
                break
            seen.add(child_id)
            current = child_id
            current_path.append(child_id)
        if best_id is not None and best_path is not None:
            return best_id, best_path
        return fallback_tip, fallback_path

    def _record_message_clone_lineage(
        self, conn, source_ids: list[int], clone_session_id: str, clone_floor: int
    ) -> None:
        clone_ids = [
            int(row["id"])
            for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND id > ? ORDER BY id",
                (clone_session_id, clone_floor),
            ).fetchall()
        ]
        if len(clone_ids) != len(source_ids):
            raise RuntimeError("message clone provenance count did not match copied tail")
        conn.executemany(
            "INSERT INTO message_clone_lineage (source_message_id, clone_message_id) VALUES (?, ?)",
            list(zip(source_ids, clone_ids)),
        )

    def _clone_message_rows_with_lineage(
        self, conn, source_ids: list[int], clone_session_id: str
    ) -> int:
        source_ids = [int(source_id) for source_id in source_ids]
        if not source_ids:
            return 0
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("message clone source ids must be unique")
        source_ids.sort()
        clone_cols = [
            column
            for column in self._message_column_names(conn)
            if column not in ("id", "session_id", "active", "compacted")
        ]
        col_list = ", ".join(clone_cols)
        for start in range(0, len(source_ids), 800):
            chunk = source_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            clone_floor = int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0])
            conn.execute(
                f"INSERT INTO messages ({col_list}, session_id, active, compacted) "
                f"SELECT {col_list}, ?, 1, 0 FROM messages "
                f"WHERE id IN ({placeholders}) ORDER BY id",
                [clone_session_id, *chunk],
            )
            self._record_message_clone_lineage(conn, chunk, clone_session_id, clone_floor)
        return len(source_ids)

    def _remove_proven_message_clone_sources(self, conn, rows, selected_session_ids) -> list:
        rows = list(rows)
        visible_ids = {int(row["id"]) for row in rows}
        selected_session_ids = list(dict.fromkeys(selected_session_ids))
        if not visible_ids or not selected_session_ids:
            return rows
        hidden_source_ids: set[int] = set()
        ids = list(visible_ids)
        chunk_size = max(1, 900 - len(selected_session_ids))
        session_placeholders = ",".join("?" for _ in selected_session_ids)
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            source_placeholders = ",".join("?" for _ in chunk)
            mappings = conn.execute(
                "SELECT lineage.source_message_id FROM message_clone_lineage AS lineage "
                "JOIN messages AS clone ON clone.id = lineage.clone_message_id "
                f"WHERE lineage.source_message_id IN ({source_placeholders}) "
                f"AND clone.session_id IN ({session_placeholders})",
                [*chunk, *selected_session_ids],
            ).fetchall()
            hidden_source_ids.update(int(row["source_message_id"]) for row in mappings)
        return [row for row in rows if int(row["id"]) not in hidden_source_ids]

    def _dedupe_legacy_message_clone_rows(self, conn, rows) -> list:
        rows = list(rows)
        boundary_row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ? LIMIT 1",
            (MESSAGE_CLONE_LINEAGE_LEGACY_CEILING_KEY,),
        ).fetchone()
        if boundary_row is None:
            return rows
        try:
            legacy_ceiling = int(boundary_row[0])
        except (TypeError, ValueError):
            return rows
        if legacy_ceiling <= 0:
            return rows
        seen: dict = {}
        for row in rows:
            if int(row["id"]) > legacy_ceiling:
                continue
            dedupe_content = row["content"]
            if row["role"] == "user":
                from agent.context_compressor import split_user_originated_turn

                candidate = {
                    "role": "user",
                    "content": self._decode_content(row["content"]),
                    "display_kind": row["display_kind"],
                    "display_metadata": self._decode_display_metadata(row["display_metadata"]),
                }
                handoff, live_view = split_user_originated_turn(candidate)
                if handoff is not None and live_view is not None:
                    dedupe_content = self._encode_content(live_view.get("content"))
            key = (
                row["role"], dedupe_content, row["timestamp"], row["tool_call_id"],
                row["tool_calls"], row["tool_name"],
            )
            current = seen.get(key)
            if current is None or (row["active"], row["id"]) > (current["active"], current["id"]):
                seen[key] = row
        retained = {int(row["id"]) for row in seen.values()}
        return [row for row in rows if int(row["id"]) > legacy_ceiling or int(row["id"]) in retained]

    @staticmethod
    def _page_message_rows(rows, *, limit, offset, latest) -> list:
        rows = list(rows)
        if latest:
            rows.reverse()
        rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        if latest:
            rows.reverse()
        return rows

    def _get_message_rows_from_conn(
        self, conn, session_id: str, include_inactive: bool = False,
        include_compacted: bool = False, limit: Optional[int] = None,
        offset: int = 0, latest: bool = False, after_id: Optional[int] = None,
    ) -> list:
        if after_id is not None and (latest or offset):
            raise ValueError("after_id is incompatible with latest/offset paging")
        if after_id is not None and include_compacted:
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        if include_inactive:
            active_clause = ""
        elif include_compacted:
            active_clause = " AND (active = 1 OR compacted = 1)"
        else:
            active_clause = " AND active = 1"
        keyset_clause = " AND id > ?" if after_id is not None else ""
        sql = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause}{keyset_clause} ORDER BY id {'DESC' if latest else 'ASC'}"
        )
        params: list = [session_id]
        if after_id is not None:
            params.append(after_id)
        if include_compacted:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ?" + active_clause + " ORDER BY id ASC",
                [session_id],
            ).fetchall()
            visible_rows = self._remove_proven_message_clone_sources(conn, rows, [session_id])
            return self._page_message_rows(
                self._dedupe_legacy_message_clone_rows(conn, visible_rows),
                limit=limit, offset=offset, latest=latest,
            )
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params.extend([-1 if limit is None else limit, offset])
        rows = conn.execute(sql, params).fetchall()
        if latest:
            rows.reverse()
        return rows

    def _get_display_lineage_message_rows_from_conn(
        self, conn, session_ids: list[str], *, include_compacted: bool,
        limit: Optional[int], offset: int, latest: bool,
    ) -> list:
        if len(session_ids) == 1:
            return self._get_message_rows_from_conn(
                conn, session_ids[0], include_compacted=include_compacted,
                limit=limit, offset=offset, latest=latest,
            )
        rows = []
        for session_id in session_ids:
            rows.extend(
                self._get_message_rows_from_conn(
                    conn, session_id, include_compacted=include_compacted, latest=False
                )
            )
        visible_rows = self._remove_proven_message_clone_sources(conn, rows, session_ids)
        return self._page_message_rows(
            self._dedupe_legacy_message_clone_rows(conn, visible_rows),
            limit=limit, offset=offset, latest=latest,
        )

    def _decode_message_rows(self, rows) -> List[Dict[str, Any]]:
        result = []
        for row in rows:
            msg = dict(row)
            if msg.pop("_compressed_summary", 0):
                msg["_compressed_summary"] = True
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            if msg.get("display_metadata") is not None:
                msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
            result.append(msg)
        return result

    def get_display_message_page(
        self, requested_session_id: str, *, limit: int, offset: int = 0,
        latest: bool = True, include_compacted: bool = True,
        known_display_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Load one conditional transcript page from one SQLite snapshot."""
        rows = None
        with self._read_transaction_ctx() as conn:
            resolved_id = self._resolve_session_id_from_conn(conn, requested_session_id)
            if resolved_id is None:
                raise KeyError(f"Session not found: {requested_session_id}")
            root_id = self._display_lineage_root_from_conn(conn, resolved_id)
            tip_id, lineage_ids = self._resolve_resume_lineage_from_conn(conn, root_id)
            revision = self._display_revision_from_conn(conn, root_id)
            unchanged = (
                isinstance(known_display_revision, int)
                and not isinstance(known_display_revision, bool)
                and known_display_revision >= 0
                and known_display_revision == revision
            )
            if not unchanged:
                rows = self._get_display_lineage_message_rows_from_conn(
                    conn, lineage_ids, include_compacted=include_compacted,
                    limit=limit, offset=offset, latest=latest,
                )
        messages = [] if unchanged else self._decode_message_rows(rows)
        return {
            "session_id": tip_id,
            "lineage_root_id": root_id,
            "resolved_tip_id": tip_id,
            "display_revision": revision,
            "unchanged": unchanged,
            "messages": messages,
            "pagination": {
                "limit": limit, "offset": offset,
                "order": "latest" if latest else "oldest",
                "returned": len(messages),
            },
        }

    def _history_lineage_root_to_tip(self, session_id: str) -> List[str]:
        """Return history segments, stopping at a parent-bound branch edge."""
        lineage = self._session_lineage_root_to_tip(session_id)
        if len(lineage) <= 1:
            return lineage
        placeholders = ",".join("?" for _ in lineage)
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT id, parent_session_id, model_config FROM sessions "
                f"WHERE id IN ({placeholders})",
                tuple(lineage),
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        for index, session_id in enumerate(lineage):
            row = by_id.get(session_id)
            if row is None or not row["parent_session_id"]:
                continue
            raw_config = row["model_config"]
            try:
                config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(config, dict) and config.get("_branched_from") == row["parent_session_id"]:
                return lineage[index:]
        return lineage
