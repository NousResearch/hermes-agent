"""Native PostgreSQL session/state operations for later SessionDB composition.

This module deliberately contains authored PostgreSQL SQL.  It does not
translate SQLite statements or expose driver cursors beyond ``_run`` scopes.
``PostgresSessionOperations`` is a mixin for ``PostgresSessionDBBase``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

from state_store.session_api import APISessionMutationAbort, APISessionMutationResult


class PostgresSessionOperations:
    """PostgreSQL implementation of the durable SessionDB lifecycle surface."""

    MAX_TITLE_LENGTH = 100

    def _row(self, connection: Any, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        cursor = connection.execute(query, params)
        return self._normalize_cursor_row(cursor)

    def _normalize_cursor_row(self, cursor: Any) -> Optional[Dict[str, Any]]:
        from state_store.session_api import normalize_row

        return normalize_row(cursor.fetchone(), columns=self._cursor_columns(cursor))

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        if not title:
            return None
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", title)
        cleaned = re.sub(
            r"[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]",
            "",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return None
        if len(cleaned) > PostgresSessionOperations.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max "
                f"{PostgresSessionOperations.MAX_TITLE_LENGTH})"
            )
        return cleaned

    # ── Session lifecycle and API atomic operations ──────────────────────

    def _insert_session(
        self, connection: Any, session_id: str, source: str, **kwargs: Any
    ) -> None:
        model_config = kwargs.get("model_config")
        connection.execute(
            """
            INSERT INTO sessions (
              id, source, user_id, session_key, chat_id, chat_type, thread_id,
              model, model_config, system_prompt, parent_session_id, cwd,
              profile_name, started_at
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
              EXTRACT(EPOCH FROM clock_timestamp())
            )
            ON CONFLICT (id) DO UPDATE SET
              model = COALESCE(sessions.model, EXCLUDED.model),
              model_config = COALESCE(sessions.model_config, EXCLUDED.model_config),
              system_prompt = COALESCE(sessions.system_prompt, EXCLUDED.system_prompt),
              session_key = COALESCE(sessions.session_key, EXCLUDED.session_key),
              chat_id = COALESCE(sessions.chat_id, EXCLUDED.chat_id),
              chat_type = COALESCE(sessions.chat_type, EXCLUDED.chat_type),
              thread_id = COALESCE(sessions.thread_id, EXCLUDED.thread_id),
              parent_session_id = COALESCE(
                sessions.parent_session_id, EXCLUDED.parent_session_id
              ),
              cwd = COALESCE(sessions.cwd, EXCLUDED.cwd),
              profile_name = COALESCE(sessions.profile_name, EXCLUDED.profile_name)
            """,
            (
                session_id, source, kwargs.get("user_id"), kwargs.get("session_key"),
                kwargs.get("chat_id"), kwargs.get("chat_type"), kwargs.get("thread_id"),
                kwargs.get("model"),
                self._json_dumps(model_config) if model_config else None,
                kwargs.get("system_prompt"), kwargs.get("parent_session_id"),
                kwargs.get("cwd"), kwargs.get("profile_name"),
            ),
        )

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        self._run(lambda connection: self._insert_session(connection, session_id, source, **kwargs))
        return session_id

    def ensure_session(
        self, session_id: str, source: str = "unknown", model: str = None, **kwargs
    ) -> str:
        return self.create_session(session_id, source, model=model, **kwargs)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_one("SELECT * FROM sessions WHERE id = %s", (session_id,))

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return str(exact["id"])
        escaped = (
            session_id_or_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        rows = self._read_many(
            "SELECT id FROM sessions WHERE id LIKE %s ESCAPE '\\' "
            "ORDER BY started_at DESC LIMIT 2",
            (f"{escaped}%",),
            limit=2,
        )
        return str(rows[0]["id"]) if len(rows) == 1 else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        exact = self.get_session_by_title(title)
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        numbered = self._read_many(
            """
            SELECT id FROM sessions WHERE title LIKE %s ESCAPE '\'
            ORDER BY started_at DESC LIMIT 1
            """,
            (f"{escaped} #%",),
            limit=1,
        )
        if numbered:
            return str(numbered[0]["id"])
        return None if exact is None else str(exact["id"])

    def end_session(self, session_id: str, end_reason: str) -> None:
        self._write(
            """
            UPDATE sessions SET ended_at = EXTRACT(EPOCH FROM clock_timestamp()),
              end_reason = %s
            WHERE id = %s AND ended_at IS NULL
            """,
            (end_reason, session_id),
        )

    def reopen_session(self, session_id: str) -> None:
        self._write(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = %s",
            (session_id,),
        )

    def _title_in_tx(
        self, connection: Any, session_id: str, title: Optional[str]
    ) -> int:
        if title:
            # Row locks cannot protect a title that does not exist yet.
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (title,),
            )
            conflict = self._row(
                connection,
                "SELECT id FROM sessions WHERE title = %s AND id != %s FOR UPDATE",
                (title, session_id),
            )
            if conflict:
                conflict_id = str(conflict["id"])
                if self._is_compression_ancestor_in_tx(
                    connection,
                    ancestor_id=conflict_id,
                    descendant_id=session_id,
                ):
                    connection.execute(
                        "UPDATE sessions SET title = NULL WHERE id = %s",
                        (conflict_id,),
                    )
                else:
                    raise ValueError(
                        f"Title '{title}' is already in use by session {conflict_id}"
                    )
        cursor = connection.execute(
            "UPDATE sessions SET title = %s WHERE id = %s", (title, session_id)
        )
        return int(getattr(cursor, "rowcount", 0) or 0)

    def _is_compression_ancestor_in_tx(
        self,
        connection: Any,
        *,
        ancestor_id: str,
        descendant_id: str,
    ) -> bool:
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        row = self._row(
            connection,
            """
            WITH RECURSIVE ancestors(id) AS (
              SELECT %s::text
              UNION
              SELECT parent.id
              FROM ancestors a
              JOIN sessions child ON child.id = a.id
              JOIN sessions parent ON parent.id = child.parent_session_id
              WHERE parent.end_reason = 'compression'
                AND child.started_at >= parent.ended_at
            )
            SELECT 1 AS is_ancestor
            FROM ancestors
            WHERE id = %s AND id != %s
            LIMIT 1
            """,
            (descendant_id, ancestor_id, descendant_id),
        )
        return row is not None

    def create_api_session_with_title(
        self,
        session_id: str,
        *,
        model: Optional[str],
        system_prompt: Optional[str],
        title: Optional[str] = None,
    ) -> APISessionMutationResult:
        try:
            sanitized = self.sanitize_title(title) if title is not None else None
        except ValueError as exc:
            return APISessionMutationResult("invalid_title", error=str(exc))

        def operation(connection: Any) -> APISessionMutationResult:
            if self._row(
                connection, "SELECT id FROM sessions WHERE id = %s FOR UPDATE", (session_id,)
            ):
                return APISessionMutationResult("destination_exists")
            self._insert_session(
                connection, session_id, "api_server", model=model, system_prompt=system_prompt
            )
            try:
                self._title_in_tx(connection, session_id, sanitized)
            except ValueError as exc:
                raise APISessionMutationAbort(
                    APISessionMutationResult("invalid_title", error=str(exc))
                )
            row = self._row(connection, "SELECT * FROM sessions WHERE id = %s", (session_id,))
            return APISessionMutationResult("created", session=row)

        try:
            return self._run(operation)
        except APISessionMutationAbort as exc:
            return exc.result

    def fork_api_session(
        self, source_session_id: str, fork_session_id: str, *, title: Optional[str] = None
    ) -> APISessionMutationResult:
        explicit_title = title is not None
        try:
            requested_title = self.sanitize_title(title) if explicit_title else None
        except ValueError as exc:
            return APISessionMutationResult("invalid_title", error=str(exc))

        def operation(connection: Any) -> APISessionMutationResult:
            source = self._row(
                connection, "SELECT * FROM sessions WHERE id = %s FOR UPDATE", (source_session_id,)
            )
            if source is None:
                return APISessionMutationResult("source_missing")
            if self._row(
                connection, "SELECT id FROM sessions WHERE id = %s FOR UPDATE", (fork_session_id,)
            ):
                return APISessionMutationResult("destination_exists")
            fork_title = requested_title
            if not explicit_title:
                base = str(source.get("title") or "fork")
                fork_title = self._next_title_in_tx(connection, base)
                try:
                    fork_title = self.sanitize_title(fork_title)
                except ValueError as exc:
                    return APISessionMutationResult("invalid_title", error=str(exc))
            connection.execute(
                """
                UPDATE sessions SET ended_at = EXTRACT(EPOCH FROM clock_timestamp()),
                  end_reason = 'branched'
                WHERE id = %s AND ended_at IS NULL
                """,
                (source_session_id,),
            )
            self._insert_session(
                connection,
                fork_session_id,
                "api_server",
                model=source.get("model"),
                system_prompt=source.get("system_prompt"),
                parent_session_id=source_session_id,
            )
            connection.execute(
                """
                INSERT INTO messages (
                  session_id, role, content, tool_call_id, tool_calls, tool_name,
                  effect_disposition, timestamp, token_count, finish_reason,
                  reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                  codex_message_items, platform_message_id, observed, active, compacted
                )
                SELECT %s, role, content, tool_call_id, tool_calls, tool_name,
                  effect_disposition, timestamp, token_count, finish_reason,
                  reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                  codex_message_items, platform_message_id, observed, active, compacted
                FROM messages WHERE session_id = %s AND active = 1 ORDER BY id
                """,
                (fork_session_id, source_session_id),
            )
            connection.execute(
                """
                UPDATE sessions SET
                  message_count = (SELECT COUNT(*) FROM messages WHERE session_id = %s AND active = 1),
                  tool_call_count = (
                    SELECT COALESCE(SUM(
                      CASE WHEN tool_calls IS NULL THEN 0
                        WHEN jsonb_typeof(tool_calls::jsonb) = 'array'
                          THEN jsonb_array_length(tool_calls::jsonb)
                        ELSE 1 END
                    ), 0) FROM messages WHERE session_id = %s AND active = 1
                  )
                WHERE id = %s
                """,
                (fork_session_id, fork_session_id, fork_session_id),
            )
            try:
                self._title_in_tx(connection, fork_session_id, fork_title)
            except ValueError as exc:
                raise APISessionMutationAbort(
                    APISessionMutationResult("invalid_title", error=str(exc))
                )
            return APISessionMutationResult(
                "created",
                session=self._row(
                    connection, "SELECT * FROM sessions WHERE id = %s", (fork_session_id,)
                ),
            )

        try:
            return self._run(operation)
        except APISessionMutationAbort as exc:
            return exc.result

    # ── Gateway routing and peer operations ──────────────────────────────

    def record_gateway_session_peer(
        self, session_id: str, *, source: str, user_id: str = None, session_key: str = None,
        chat_id: str = None, chat_type: str = None, thread_id: str = None,
        display_name: str = None, origin_json: str = None,
    ) -> None:
        if not session_id or not session_key:
            return
        self._write(
            """
            UPDATE sessions SET session_key = %s, source = %s, user_id = %s,
              chat_id = %s, chat_type = %s, thread_id = %s,
              display_name = COALESCE(%s, display_name),
              origin_json = COALESCE(%s, origin_json)
            WHERE id = %s
            """,
            (session_key, source, user_id, chat_id, chat_type, thread_id,
             display_name, origin_json, session_id),
        )

    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        if session_id:
            self._write(
                "UPDATE sessions SET expiry_finalized = %s WHERE id = %s",
                (1 if finalized else 0, session_id),
            )

    def save_gateway_routing_entry(self, session_key: str, entry_json: str, *, scope: str = "") -> None:
        if not session_key or not entry_json:
            return
        self._write(
            """
            INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at)
            VALUES (%s, %s, %s, EXTRACT(EPOCH FROM clock_timestamp()))
            ON CONFLICT(scope, session_key) DO UPDATE SET
              entry_json = EXCLUDED.entry_json, updated_at = EXCLUDED.updated_at
            """,
            (scope, session_key, entry_json),
        )

    def replace_gateway_routing_entries(self, entries: Dict[str, str], *, scope: str = "") -> None:
        def operation(connection: Any) -> None:
            connection.execute("DELETE FROM gateway_routing WHERE scope = %s", (scope,))
            for session_key, entry_json in entries.items():
                if session_key and entry_json:
                    connection.execute(
                        """
                        INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at)
                        VALUES (%s, %s, %s, EXTRACT(EPOCH FROM clock_timestamp()))
                        """,
                        (scope, session_key, entry_json),
                    )
        self._run(operation)

    def load_gateway_routing_entries(self, *, scope: str = "") -> Dict[str, str]:
        rows = self._read_many(
            "SELECT session_key, entry_json FROM gateway_routing WHERE scope = %s "
            "ORDER BY session_key LIMIT %s",
            (scope, self._MAX_READ_ROWS),
            limit=self._MAX_READ_ROWS,
        )
        return {str(row["session_key"]): str(row["entry_json"]) for row in rows}

    def delete_gateway_routing_entries(self, session_keys: List[str], *, scope: str = "") -> None:
        if not session_keys:
            return
        self._write(
            "DELETE FROM gateway_routing WHERE scope = %s AND session_key = ANY(%s)",
            (scope, session_keys),
        )

    def list_gateway_sessions(
        self, *, platform: Optional[str] = None, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        clauses = ["s.session_key IS NOT NULL"]
        params: List[Any] = []
        if platform:
            clauses.append("LOWER(s.source) = LOWER(%s)")
            params.append(platform)
        if active_only:
            clauses.append("s.ended_at IS NULL")
        params.append(self._MAX_READ_ROWS)
        return self._read_many(
            f"""
            SELECT s.*, COALESCE((SELECT MAX(m.timestamp) FROM messages m
              WHERE m.session_id = s.id), s.started_at) AS last_active
            FROM sessions s WHERE {' AND '.join(clauses)}
            AND s.started_at = (SELECT MAX(s2.started_at) FROM sessions s2
              WHERE s2.session_key = s.session_key)
            ORDER BY last_active DESC LIMIT %s
            """,
            tuple(params),
            limit=self._MAX_READ_ROWS,
        )

    def find_session_by_origin(
        self, *, platform: str, chat_id: str, thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        if not platform or chat_id in (None, ""):
            return None
        clauses = [
            "LOWER(source) = LOWER(%s)", "session_key IS NOT NULL",
            "chat_id = %s", "ended_at IS NULL",
        ]
        params: List[Any] = [platform, str(chat_id)]
        if thread_id is not None:
            clauses.append("COALESCE(thread_id, '') = %s")
            params.append(str(thread_id))
        rows = self._read_many(
            f"SELECT id, user_id FROM sessions WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC LIMIT 2",
            tuple(params),
            limit=2,
        )
        if not rows:
            return None
        if user_id:
            exact = [row for row in rows if str(row.get("user_id") or "") == str(user_id)]
            return str(exact[0]["id"]) if exact else (None if len(rows) > 1 else str(rows[0]["id"]))
        users = {str(row.get("user_id") or "") for row in rows if row.get("user_id")}
        return None if len(users) > 1 else str(rows[0]["id"])

    def find_latest_gateway_session_for_peer(
        self, *, source: str, user_id: Optional[str] = None, session_key: Optional[str] = None,
        chat_id: Optional[str] = None, chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not session_key:
            return None
        direct = self._read_one(
            """
            SELECT * FROM sessions WHERE session_key = %s AND source = %s
              AND (ended_at IS NULL OR end_reason IN ('agent_close', 'ws_orphan_reap'))
              AND (COALESCE(message_count, 0) > 0 OR EXISTS (
                SELECT 1 FROM messages WHERE messages.session_id = sessions.id
              ))
            ORDER BY started_at DESC LIMIT 1
            """,
            (session_key, source),
        )
        if direct or chat_id is None or chat_type is None:
            return direct
        return self._read_one(
            """
            SELECT * FROM sessions WHERE source = %s
              AND COALESCE(user_id, '') = COALESCE(%s, '')
              AND COALESCE(chat_id, '') = COALESCE(%s, '')
              AND COALESCE(chat_type, '') = COALESCE(%s, '')
              AND COALESCE(thread_id, '') = COALESCE(%s, '')
              AND (ended_at IS NULL OR end_reason IN ('agent_close', 'ws_orphan_reap'))
              AND (COALESCE(message_count, 0) > 0 OR EXISTS (
                SELECT 1 FROM messages WHERE messages.session_id = sessions.id
              ))
            ORDER BY started_at DESC LIMIT 1
            """,
            (source, user_id, chat_id, chat_type, thread_id),
        )

    # ── Session metadata, titles, listing, and cron ──────────────────────

    def update_session_cwd(self, session_id: str, cwd: str, git_branch: str = None, git_repo_root: str = None) -> None:
        if not session_id or not cwd:
            return
        self._write(
            """
            UPDATE sessions SET cwd = %s,
              git_branch = COALESCE(NULLIF(%s, ''), git_branch),
              git_repo_root = COALESCE(NULLIF(%s, ''), git_repo_root)
            WHERE id = %s
            """,
            (cwd, git_branch or "", git_repo_root or "", session_id),
        )

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if cwd and root]
        if not pairs:
            return
        def operation(connection: Any) -> None:
            for root, cwd in pairs:
                connection.execute(
                    "UPDATE sessions SET git_repo_root = %s WHERE cwd = %s "
                    "AND COALESCE(git_repo_root, '') = ''",
                    (root, cwd),
                )
        self._run(operation)

    def update_session_meta(self, session_id: str, model_config_json: str, model: Optional[str] = None) -> None:
        self._write(
            "UPDATE sessions SET model_config = %s, model = COALESCE(%s, model) WHERE id = %s",
            (model_config_json, model, session_id),
        )

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        self._write("UPDATE sessions SET system_prompt = %s WHERE id = %s", (system_prompt, session_id))

    def update_session_model(self, session_id: str, model: str) -> None:
        self._write("UPDATE sessions SET model = %s WHERE id = %s", (model, session_id))

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str] = None
    ) -> None:
        self._write(
            """
            UPDATE sessions SET billing_provider = %s, billing_base_url = %s,
              billing_mode = COALESCE(%s, billing_mode), system_prompt = NULL
            WHERE id = %s
            """,
            (provider, base_url, billing_mode, session_id),
        )

    def set_session_title(self, session_id: str, title: str) -> bool:
        sanitized = self.sanitize_title(title)
        return bool(self._run(lambda connection: self._title_in_tx(connection, session_id, sanitized)))

    def get_session_title(self, session_id: str) -> Optional[str]:
        row = self._read_one("SELECT title FROM sessions WHERE id = %s", (session_id,))
        return None if row is None else row.get("title")

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        return self._read_one("SELECT * FROM sessions WHERE title = %s", (title,))

    @staticmethod
    def _next_title_value(base_title: str, titles: Sequence[str]) -> str:
        match = re.match(r"^(.*?) #(\d+)$", base_title)
        base = match.group(1) if match else base_title
        if not titles:
            return base
        maximum = 1
        for title in titles:
            suffix = re.match(r"^.* #(\d+)$", title)
            if suffix:
                maximum = max(maximum, int(suffix.group(1)))
        return f"{base} #{maximum + 1}"

    def _next_title_in_tx(self, connection: Any, base_title: str) -> str:
        match = re.match(r"^(.*?) #(\d+)$", base_title)
        base = match.group(1) if match else base_title
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        # Lock the lineage name even if this is the first title in the lineage.
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (base,),
        )
        row = self._row(
            connection,
            """
            WITH locked AS MATERIALIZED (
              SELECT title FROM sessions
              WHERE title = %s OR title LIKE %s ESCAPE '\'
              FOR UPDATE
            )
            SELECT COUNT(*)::int AS title_count,
              COALESCE(MAX(
                CASE WHEN title ~ ' #[0-9]+$'
                  THEN substring(title FROM ' #([0-9]+)$')::int
                  ELSE 0
                END
              ), 0)::int AS max_suffix
            FROM locked
            """,
            (base, f"{escaped} #%"),
        )
        if not row or not int(row.get("title_count") or 0):
            return base
        maximum = max(1, int(row.get("max_suffix") or 0))
        return f"{base} #{maximum + 1}"

    def get_next_title_in_lineage(self, base_title: str) -> str:
        return self._run(lambda connection: self._next_title_in_tx(connection, base_title))

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        if not session_id:
            return session_id
        row = self._read_one(
            """
            WITH RECURSIVE chain(id, depth) AS (
              SELECT %s::text, 0
              UNION ALL
              SELECT child.id, chain.depth + 1
              FROM chain
              JOIN sessions parent ON parent.id = chain.id
              JOIN LATERAL (
                SELECT c.id
                FROM sessions c
                WHERE c.parent_session_id = parent.id
                  AND parent.end_reason = 'compression'
                  AND c.started_at >= parent.ended_at
                  AND c.model_config NOT LIKE '%%"_branched_from"%%'
                  AND c.model_config NOT LIKE '%%"_delegate_from"%%'
                  AND COALESCE(c.source, '') != 'tool'
                ORDER BY
                  CASE
                    WHEN c.end_reason = 'compression' THEN 0
                    WHEN c.ended_at IS NULL THEN 1
                    ELSE 2
                  END,
                  COALESCE((
                    SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = c.id
                  ), c.started_at) DESC,
                  c.started_at DESC,
                  c.id DESC
                LIMIT 1
              ) child ON TRUE
              WHERE chain.depth < 100
            )
            SELECT id FROM chain ORDER BY depth DESC LIMIT 1
            """,
            (session_id,),
        )
        return session_id if row is None else str(row["id"])

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        row = self._write_returning(
            """
            WITH RECURSIVE ancestors(id) AS (
              SELECT %s::text UNION
              SELECT parent.id FROM ancestors a JOIN sessions child ON child.id = a.id
              JOIN sessions parent ON parent.id = child.parent_session_id
              WHERE parent.end_reason = 'compression'
            ), descendants(id) AS (
              SELECT %s::text UNION
              SELECT child.id FROM descendants d JOIN sessions parent ON parent.id = d.id
              JOIN sessions child ON child.parent_session_id = parent.id
              WHERE parent.end_reason = 'compression'
            ), lineage AS (SELECT id FROM ancestors UNION SELECT id FROM descendants),
            changed AS (
              UPDATE sessions SET archived = %s WHERE id IN (SELECT id FROM lineage)
              RETURNING id
            ) SELECT COUNT(*)::int AS count FROM changed
            """,
            (session_id, session_id, 1 if archived else 0),
        )
        return bool(row and int(row.get("count") or 0))

    def distinct_session_cwds(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Return bounded workspace usage aggregates for repository discovery."""
        where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
        if not include_archived:
            where += " AND archived = 0"
        rows = self._read_many(
            f"""
            SELECT cwd, COUNT(*)::bigint AS sessions,
              MAX(COALESCE(ended_at, started_at, 0)) AS last_active
            FROM sessions
            WHERE {where}
            GROUP BY cwd
            LIMIT %s
            """,
            (self._MAX_READ_ROWS,),
            limit=self._MAX_READ_ROWS,
        )
        return [
            {
                "cwd": row["cwd"],
                "sessions": int(row.get("sessions") or 0),
                "last_active": float(row.get("last_active") or 0),
            }
            for row in rows
        ]

    def list_sessions_rich(
        self, source: str = None, exclude_sources: List[str] = None, cwd_prefix: str = None,
        limit: int = 20, offset: int = 0, include_children: bool = False,
        min_message_count: int = 0, project_compression_tips: bool = True,
        order_by_last_active: bool = False, include_archived: bool = False,
        archived_only: bool = False, id_query: str = None, search_query: str = None,
        compact_rows: bool = False,
    ) -> List[Dict[str, Any]]:
        del project_compression_tips, compact_rows
        limit = max(1, min(int(limit), self._MAX_READ_ROWS))
        clauses, params = ["1 = 1"], []
        if not include_children:
            clauses.append("(s.parent_session_id IS NULL OR EXISTS (SELECT 1 FROM sessions p "
                           "WHERE p.id = s.parent_session_id AND p.end_reason = 'branched'))")
        if source:
            clauses.append("s.source = %s"); params.append(source)
        if exclude_sources:
            clauses.append("NOT (s.source = ANY(%s))"); params.append(exclude_sources)
        if cwd_prefix:
            prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
            clauses.append("(s.cwd = %s OR s.cwd LIKE %s OR s.cwd LIKE %s)")
            params.extend((prefix, f"{prefix}/%", f"{prefix}\\%"))
        if min_message_count:
            clauses.append("s.message_count >= %s"); params.append(min_message_count)
        if archived_only:
            clauses.append("s.archived = 1")
        elif not include_archived:
            clauses.append("s.archived = 0")
        if id_query:
            clauses.append("LOWER(s.id) LIKE %s"); params.append(f"%{id_query.lower()}%")
        if search_query:
            clauses.append("(LOWER(COALESCE(s.title, '')) LIKE %s OR LOWER(s.id) LIKE %s)")
            params.extend((f"%{search_query.lower()}%", f"%{search_query.lower()}%"))
        order = "last_active DESC, s.started_at DESC, s.id DESC" if order_by_last_active else "s.started_at DESC"
        params.extend((limit, max(0, int(offset))))
        rows = self._read_many(
            f"""
            SELECT s.*, COALESCE((SELECT LEFT(REPLACE(REPLACE(m.content, E'\\n', ' '), E'\\r', ' '), 63)
              FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
              AND m.content IS NOT NULL ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw,
              COALESCE((SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                s.started_at) AS last_active
            FROM sessions s WHERE {' AND '.join(clauses)}
            ORDER BY {order} LIMIT %s OFFSET %s
            """,
            tuple(params),
            limit=limit,
        )
        for row in rows:
            raw = str(row.pop("_preview_raw", "") or "").strip()
            row["preview"] = raw[:60] + ("..." if len(raw) > 60 else "") if raw else ""
        return rows

    def list_cron_job_runs(self, job_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        prefix = f"cron_{job_id}_"
        high = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        rows = self._read_many(
            """
            SELECT s.*, COALESCE((SELECT LEFT(m.content, 63) FROM messages m
              WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
              ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw,
              COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id),
                s.started_at) AS last_active
            FROM sessions s WHERE s.source = 'cron' AND s.id >= %s AND s.id < %s
            ORDER BY s.started_at DESC, s.id DESC LIMIT %s OFFSET %s
            """,
            (prefix, high, min(max(1, int(limit)), self._MAX_READ_ROWS), max(0, int(offset))),
            limit=min(max(1, int(limit)), self._MAX_READ_ROWS),
        )
        for row in rows:
            raw = str(row.pop("_preview_raw", "") or "").strip()
            row["preview"] = raw[:60] + ("..." if len(raw) > 60 else "") if raw else ""
        return rows

    def session_count(
        self,
        source: str = None,
        cwd_prefix: str = None,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
        exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions using the same filtering contract as rich listing."""
        clauses: List[str] = []
        params: List[Any] = []
        if exclude_children:
            clauses.append(
                """
                (
                  s.parent_session_id IS NULL
                  OR COALESCE(s.model_config, '{}')::jsonb -> '_branched_from' IS NOT NULL
                  OR EXISTS (
                    SELECT 1 FROM sessions p
                    WHERE p.id = s.parent_session_id
                      AND p.end_reason = 'branched'
                      AND s.started_at >= p.ended_at
                  )
                )
                """
            )
            clauses.append(
                "COALESCE(s.model_config, '{}')::jsonb -> '_delegate_from' IS NULL"
            )
        if source:
            clauses.append("s.source = %s")
            params.append(source)
        if exclude_sources:
            clauses.append("NOT (s.source = ANY(%s))")
            params.append(exclude_sources)
        if cwd_prefix:
            prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
            clauses.append("(s.cwd = %s OR s.cwd LIKE %s OR s.cwd LIKE %s)")
            params.extend((prefix, f"{prefix}/%", f"{prefix}\\%"))
        if min_message_count > 0:
            clauses.append("s.message_count >= %s")
            params.append(min_message_count)
        if archived_only:
            clauses.append("s.archived = 1")
        elif not include_archived:
            clauses.append("s.archived = 0")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._read_one(
            f"SELECT COUNT(*)::bigint AS count FROM sessions s{where}",
            tuple(params),
        )
        return int((row or {}).get("count") or 0)

    def finalize_orphaned_compression_sessions(self) -> int:
        return self._write(
            """
            UPDATE sessions SET ended_at = EXTRACT(EPOCH FROM clock_timestamp()),
              end_reason = 'orphaned_compression'
            WHERE api_call_count = 0 AND ended_at IS NULL AND end_reason IS NULL
              AND started_at < EXTRACT(EPOCH FROM clock_timestamp()) - 604800
              AND parent_session_id IS NOT NULL
              AND EXISTS (SELECT 1 FROM sessions p WHERE p.id = sessions.parent_session_id
                AND p.end_reason = 'compression' AND p.ended_at IS NOT NULL)
              AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = sessions.id)
            """
        )

    # ── Compression cooldowns and server-clock leases ───────────────────

    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: Optional[str] = None) -> None:
        if session_id:
            self._write(
                "UPDATE sessions SET compression_failure_cooldown_until = %s, "
                "compression_failure_error = %s WHERE id = %s",
                (cooldown_until, error, session_id),
            )

    def get_compression_failure_cooldown(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        row = self._read_one(
            """
            SELECT compression_failure_cooldown_until AS cooldown_until,
              compression_failure_error AS error,
              compression_failure_cooldown_until - EXTRACT(EPOCH FROM clock_timestamp())
                AS remaining_seconds
            FROM sessions WHERE id = %s
              AND compression_failure_cooldown_until > EXTRACT(EPOCH FROM clock_timestamp())
            """,
            (session_id,),
        )
        return row

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        if session_id:
            self._write(
                "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                "compression_failure_error = NULL WHERE id = %s",
                (session_id,),
            )

    def get_compression_fallback_streak(self, session_id: str) -> int:
        row = self._read_one(
            "SELECT compression_fallback_streak FROM sessions WHERE id = %s", (session_id,)
        )
        try:
            return max(0, int((row or {}).get("compression_fallback_streak") or 0))
        except (TypeError, ValueError):
            return 0

    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        if session_id:
            self._write(
                "UPDATE sessions SET compression_fallback_streak = %s WHERE id = %s",
                (max(0, int(streak)), session_id),
            )

    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id or not holder:
            return False
        row = self._write_returning(
            """
            INSERT INTO compression_locks(session_id, holder, acquired_at, expires_at)
            VALUES (%s, %s, EXTRACT(EPOCH FROM clock_timestamp()),
              EXTRACT(EPOCH FROM clock_timestamp()) + %s)
            ON CONFLICT(session_id) DO UPDATE SET holder = EXCLUDED.holder,
              acquired_at = EXCLUDED.acquired_at, expires_at = EXCLUDED.expires_at
            WHERE compression_locks.expires_at < EXTRACT(EPOCH FROM clock_timestamp())
            RETURNING holder
            """,
            (session_id, holder, float(ttl_seconds)),
        )
        return bool(row and row.get("holder") == holder)

    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        row = self._write_returning(
            """
            UPDATE compression_locks SET expires_at = EXTRACT(EPOCH FROM clock_timestamp()) + %s
            WHERE session_id = %s AND holder = %s
              AND expires_at >= EXTRACT(EPOCH FROM clock_timestamp()) RETURNING holder
            """,
            (float(ttl_seconds), session_id, holder),
        )
        return bool(row and row.get("holder") == holder)

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        if session_id:
            self._write(
                "DELETE FROM compression_locks WHERE session_id = %s AND holder = %s",
                (session_id, holder),
            )

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        row = self._read_one(
            "SELECT holder FROM compression_locks WHERE session_id = %s "
            "AND expires_at >= EXTRACT(EPOCH FROM clock_timestamp())",
            (session_id,),
        )
        return None if row is None else row.get("holder")

    # ── Usage accounting and insights ────────────────────────────────────

    def _record_model_usage(self, connection: Any, session_id: str, *, model: Optional[str],
                            billing_provider: Optional[str], billing_base_url: Optional[str],
                            billing_mode: Optional[str], input_tokens: int, output_tokens: int,
                            cache_read_tokens: int, cache_write_tokens: int, reasoning_tokens: int,
                            estimated_cost_usd: Optional[float], actual_cost_usd: Optional[float],
                            cost_status: Optional[str], cost_source: Optional[str],
                            api_call_count: int, task: str = "") -> None:
        session = self._row(
            connection,
            """
            SELECT model, billing_provider, billing_base_url, billing_mode
            FROM sessions WHERE id = %s
            """,
            (session_id,),
        ) or {}
        if task:
            effective_model = model or "unknown"
            effective_provider = billing_provider or ""
            effective_base_url = billing_base_url or ""
            effective_billing_mode = billing_mode or ""
        else:
            effective_model = model or session.get("model") or "unknown"
            effective_provider = billing_provider or session.get("billing_provider") or ""
            effective_base_url = billing_base_url or session.get("billing_base_url") or ""
            effective_billing_mode = billing_mode or session.get("billing_mode") or ""
        connection.execute(
            """
            INSERT INTO session_model_usage(
              session_id, model, billing_provider, billing_base_url, billing_mode, task,
              api_call_count, input_tokens, output_tokens, cache_read_tokens,
              cache_write_tokens, reasoning_tokens, estimated_cost_usd, actual_cost_usd,
              cost_status, cost_source, first_seen, last_seen
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
              %s, %s, EXTRACT(EPOCH FROM clock_timestamp()), EXTRACT(EPOCH FROM clock_timestamp()))
            ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
            DO UPDATE SET api_call_count = session_model_usage.api_call_count + EXCLUDED.api_call_count,
              input_tokens = session_model_usage.input_tokens + EXCLUDED.input_tokens,
              output_tokens = session_model_usage.output_tokens + EXCLUDED.output_tokens,
              cache_read_tokens = session_model_usage.cache_read_tokens + EXCLUDED.cache_read_tokens,
              cache_write_tokens = session_model_usage.cache_write_tokens + EXCLUDED.cache_write_tokens,
              reasoning_tokens = session_model_usage.reasoning_tokens + EXCLUDED.reasoning_tokens,
              estimated_cost_usd = session_model_usage.estimated_cost_usd + EXCLUDED.estimated_cost_usd,
              actual_cost_usd = session_model_usage.actual_cost_usd + EXCLUDED.actual_cost_usd,
              cost_status = COALESCE(EXCLUDED.cost_status, session_model_usage.cost_status),
              cost_source = COALESCE(EXCLUDED.cost_source, session_model_usage.cost_source),
              last_seen = EXCLUDED.last_seen
            """,
            (session_id, effective_model, effective_provider, effective_base_url,
             effective_billing_mode, task, api_call_count or 0, input_tokens or 0,
             output_tokens or 0, cache_read_tokens or 0, cache_write_tokens or 0,
             reasoning_tokens or 0, float(estimated_cost_usd or 0), float(actual_cost_usd or 0),
             cost_status, cost_source),
        )

    def update_token_counts(
        self, session_id: str, input_tokens: int = 0, output_tokens: int = 0, model: str = None,
        cache_read_tokens: int = 0, cache_write_tokens: int = 0, reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None, actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None, cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None, billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None, billing_mode: Optional[str] = None,
        api_call_count: int = 0, absolute: bool = False,
    ) -> None:
        has_accounted_usage = bool(
            input_tokens
            or output_tokens
            or cache_read_tokens
            or cache_write_tokens
            or reasoning_tokens
            or api_call_count
            or estimated_cost_usd
            or actual_cost_usd
        )

        def operation(connection: Any) -> None:
            self._insert_session(connection, session_id, "unknown", model=model)
            current = self._row(
                connection,
                """
                SELECT model, billing_provider, api_call_count
                FROM sessions WHERE id = %s FOR UPDATE
                """,
                (session_id,),
            ) or {}
            first_accounted_route = (
                int(current.get("api_call_count") or 0) == 0
                and has_accounted_usage
                and bool(model)
                and bool(billing_provider)
                and (
                    current.get("model") != model
                    or current.get("billing_provider") != billing_provider
                )
            )
            if first_accounted_route:
                connection.execute(
                    """
                    UPDATE sessions SET model = %s, billing_provider = %s,
                      billing_base_url = %s, billing_mode = %s
                    WHERE id = %s
                    """,
                    (model, billing_provider, billing_base_url, billing_mode, session_id),
                )
            if absolute:
                connection.execute(
                    """
                    UPDATE sessions SET input_tokens = %s, output_tokens = %s,
                      cache_read_tokens = %s, cache_write_tokens = %s, reasoning_tokens = %s,
                      estimated_cost_usd = COALESCE(%s, 0),
                      actual_cost_usd = CASE WHEN %s IS NULL THEN actual_cost_usd ELSE %s END,
                      cost_status = COALESCE(%s, cost_status), cost_source = COALESCE(%s, cost_source),
                      pricing_version = COALESCE(%s, pricing_version),
                      billing_provider = COALESCE(billing_provider, %s),
                      billing_base_url = COALESCE(billing_base_url, %s),
                      billing_mode = COALESCE(billing_mode, %s), model = COALESCE(model, %s),
                      api_call_count = %s WHERE id = %s
                    """,
                    (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
                     estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source,
                     pricing_version,
                     billing_provider if has_accounted_usage else None,
                     billing_base_url if has_accounted_usage else None,
                     billing_mode if has_accounted_usage else None,
                     model if has_accounted_usage else None,
                     api_call_count, session_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE sessions SET input_tokens = input_tokens + %s, output_tokens = output_tokens + %s,
                      cache_read_tokens = cache_read_tokens + %s, cache_write_tokens = cache_write_tokens + %s,
                      reasoning_tokens = reasoning_tokens + %s,
                      estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(%s, 0),
                      actual_cost_usd = CASE WHEN %s IS NULL THEN actual_cost_usd
                        ELSE COALESCE(actual_cost_usd, 0) + %s END,
                      cost_status = COALESCE(%s, cost_status), cost_source = COALESCE(%s, cost_source),
                      pricing_version = COALESCE(%s, pricing_version),
                      billing_provider = COALESCE(billing_provider, %s),
                      billing_base_url = COALESCE(billing_base_url, %s),
                      billing_mode = COALESCE(billing_mode, %s), model = COALESCE(model, %s),
                      api_call_count = COALESCE(api_call_count, 0) + %s WHERE id = %s
                    """,
                    (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
                     estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source,
                     pricing_version,
                     billing_provider if has_accounted_usage else None,
                     billing_base_url if has_accounted_usage else None,
                     billing_mode if has_accounted_usage else None,
                     model if has_accounted_usage else None,
                     api_call_count, session_id),
                )
                if has_accounted_usage:
                    self._record_model_usage(
                        connection, session_id, model=model, billing_provider=billing_provider,
                        billing_base_url=billing_base_url, billing_mode=billing_mode,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
                        reasoning_tokens=reasoning_tokens, estimated_cost_usd=estimated_cost_usd,
                        actual_cost_usd=actual_cost_usd, cost_status=cost_status,
                        cost_source=cost_source, api_call_count=api_call_count,
                    )
        self._run(operation)

    def record_auxiliary_usage(
        self, session_id: str, task: str, *, model: Optional[str] = None,
        billing_provider: Optional[str] = None, billing_base_url: Optional[str] = None,
        input_tokens: int = 0, output_tokens: int = 0, cache_read_tokens: int = 0,
        cache_write_tokens: int = 0, reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        if not session_id or not task:
            return
        def operation(connection: Any) -> None:
            self._insert_session(connection, session_id, "unknown")
            self._record_model_usage(
                connection, session_id, model=model, billing_provider=billing_provider,
                billing_base_url=billing_base_url, billing_mode=None, input_tokens=input_tokens,
                output_tokens=output_tokens, cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens, reasoning_tokens=reasoning_tokens,
                estimated_cost_usd=estimated_cost_usd, actual_cost_usd=None, cost_status=None,
                cost_source=None, api_call_count=1, task=task,
            )
        self._run(operation)

    def get_insights_sessions(self, cutoff: float, source: Optional[str] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT id, source, model, started_at, ended_at, message_count, tool_call_count, "
            "input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
            "billing_provider, billing_base_url, billing_mode, estimated_cost_usd, "
            "actual_cost_usd, cost_status, cost_source, api_call_count FROM sessions "
            "WHERE started_at >= %s"
            + (" AND source = %s" if source else "")
            + " ORDER BY started_at DESC LIMIT %s"
        )
        params: Sequence[Any] = (cutoff, source, self._MAX_READ_ROWS) if source else (cutoff, self._MAX_READ_ROWS)
        return self._read_many(query, params, limit=self._MAX_READ_ROWS)

    def get_insights_tool_name_counts(self, cutoff: float, source: Optional[str] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT m.tool_name, COUNT(*)::int AS count FROM messages m JOIN sessions s "
            "ON s.id = m.session_id WHERE s.started_at >= %s"
            + (" AND s.source = %s" if source else "")
            + " AND m.role = 'tool' AND m.tool_name IS NOT NULL "
            "GROUP BY m.tool_name ORDER BY count DESC LIMIT %s"
        )
        params: Sequence[Any] = (cutoff, source, self._MAX_READ_ROWS) if source else (cutoff, self._MAX_READ_ROWS)
        return self._read_many(query, params, limit=self._MAX_READ_ROWS)

    def get_insights_assistant_tool_calls_page(
        self, cutoff: float, source: Optional[str] = None, *, after_message_id: int = 0,
        limit: int = 200, include_timestamp: bool = False,
    ) -> List[Dict[str, Any]]:
        page_limit = max(1, min(int(limit), 1000))
        columns = "m.id, m.tool_calls, m.timestamp" if include_timestamp else "m.id, m.tool_calls"
        query = (
            f"SELECT {columns} FROM messages m JOIN sessions s ON s.id = m.session_id "
            "WHERE s.started_at >= %s AND m.id > %s"
            + (" AND s.source = %s" if source else "")
            + " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL "
            "ORDER BY m.id ASC LIMIT %s"
        )
        params: Sequence[Any] = (cutoff, after_message_id, source, page_limit) if source else (cutoff, after_message_id, page_limit)
        return self._read_many(query, params, limit=page_limit)

    def get_insights_message_stats(self, cutoff: float, source: Optional[str] = None) -> Dict[str, Any]:
        query = (
            "SELECT COUNT(*)::int AS total_messages, "
            "COUNT(*) FILTER (WHERE m.role = 'user')::int AS user_messages, "
            "COUNT(*) FILTER (WHERE m.role = 'assistant')::int AS assistant_messages, "
            "COUNT(*) FILTER (WHERE m.role = 'tool')::int AS tool_messages "
            "FROM messages m JOIN sessions s ON s.id = m.session_id WHERE s.started_at >= %s"
            + (" AND s.source = %s" if source else "")
        )
        row = self._read_one(query, (cutoff, source) if source else (cutoff,))
        return row or {"total_messages": 0, "user_messages": 0, "assistant_messages": 0, "tool_messages": 0}

    def get_insights_model_usage(self, cutoff: float, source: Optional[str] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url, u.api_call_count, "
            "u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, "
            "u.reasoning_tokens, u.estimated_cost_usd, u.actual_cost_usd, u.cost_status, "
            "u.cost_source, u.billing_mode FROM session_model_usage u JOIN sessions s "
            "ON s.id = u.session_id WHERE s.started_at >= %s"
            + (" AND s.source = %s" if source else "")
            + " LIMIT %s"
        )
        params: Sequence[Any] = (cutoff, source, self._MAX_READ_ROWS) if source else (cutoff, self._MAX_READ_ROWS)
        return self._read_many(query, params, limit=self._MAX_READ_ROWS)

    # ── Async delegation durability ──────────────────────────────────────

    def persist_async_delegation(self, record: Mapping[str, Any], *, owner_pid: Optional[int],
                                 owner_started_at: Optional[int], updated_at: Optional[float] = None) -> None:
        now = time.time() if updated_at is None else updated_at
        payload = {key: record.get(key) for key in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch") if key in record}
        self._write(
            """
            INSERT INTO async_delegations(delegation_id, origin_session, origin_ui_session_id,
              parent_session_id, state, dispatched_at, updated_at, delivery_state,
              delivery_attempts, owner_pid, owner_started_at, task_json)
            VALUES (%s, %s, %s, %s, 'running', %s, %s, 'pending', 0, %s, %s, %s)
            ON CONFLICT(delegation_id) DO UPDATE SET origin_session = EXCLUDED.origin_session,
              origin_ui_session_id = EXCLUDED.origin_ui_session_id,
              parent_session_id = EXCLUDED.parent_session_id, state = 'running',
              dispatched_at = EXCLUDED.dispatched_at, updated_at = EXCLUDED.updated_at,
              delivery_state = 'pending', delivery_attempts = 0, owner_pid = EXCLUDED.owner_pid,
              owner_started_at = EXCLUDED.owner_started_at, task_json = EXCLUDED.task_json
            """,
            (record["delegation_id"], record.get("session_key", ""), record.get("origin_ui_session_id", ""),
             record.get("parent_session_id"), record["dispatched_at"], now, owner_pid,
             owner_started_at, self._json_dumps(payload)),
        )

    def delete_async_delegation(self, delegation_id: str) -> None:
        self._write("DELETE FROM async_delegations WHERE delegation_id = %s", (delegation_id,))

    def prune_async_delegations(self, *, retention_seconds: float, max_retained_completed: int,
                                max_pending: int, now: Optional[float] = None) -> None:
        timestamp = time.time() if now is None else now
        def operation(connection: Any) -> None:
            connection.execute(
                "DELETE FROM async_delegations WHERE delivery_state = 'delivered' "
                "AND updated_at < %s", (timestamp - retention_seconds,)
            )
            connection.execute(
                """
                DELETE FROM async_delegations WHERE delegation_id IN (
                  SELECT delegation_id FROM async_delegations
                  WHERE state NOT IN ('running', 'finalizing')
                  ORDER BY CASE WHEN delivery_state = 'delivered' THEN 0 ELSE 1 END,
                    updated_at ASC OFFSET %s
                )
                """,
                (max(0, int(max_retained_completed)),),
            )
            connection.execute(
                """
                DELETE FROM async_delegations WHERE delegation_id IN (
                  SELECT delegation_id FROM async_delegations
                  WHERE state NOT IN ('running', 'finalizing') AND delivery_state = 'pending'
                  ORDER BY updated_at ASC OFFSET %s
                )
                """,
                (max(0, int(max_pending)),),
            )
        self._run(operation)

    def complete_async_delegation(self, event: Mapping[str, Any], result: Mapping[str, Any],
                                  *, updated_at: Optional[float] = None) -> None:
        now = time.time() if updated_at is None else updated_at
        self._write(
            """
            UPDATE async_delegations SET state = %s, completed_at = %s, updated_at = %s,
              event_json = %s, result_json = %s, delivery_state = 'pending'
            WHERE delegation_id = %s
            """,
            (event.get("status", "completed"), event.get("completed_at", now), now,
             self._json_dumps(dict(event)), self._json_dumps(dict(result)), event["delegation_id"]),
        )

    def note_async_delegation_delivery_attempt(self, delegation_id: str, *, updated_at: Optional[float] = None) -> None:
        self._write(
            "UPDATE async_delegations SET delivery_attempts = delivery_attempts + 1, updated_at = %s "
            "WHERE delegation_id = %s",
            (time.time() if updated_at is None else updated_at, delegation_id),
        )

    def list_recoverable_async_delegations(self) -> List[Dict[str, Any]]:
        rows = self._read_many(
            """
            SELECT delegation_id, origin_session, origin_ui_session_id, parent_session_id,
              dispatched_at, owner_pid, owner_started_at, task_json
            FROM async_delegations WHERE state IN ('running', 'finalizing')
            ORDER BY dispatched_at ASC LIMIT %s
            """,
            (self._MAX_READ_ROWS,),
            limit=self._MAX_READ_ROWS,
        )
        for row in rows:
            row["task"] = self._json_loads(row.pop("task_json", None), default={})
        return rows

    def mark_async_delegation_unknown(self, delegation_id: str, event: Mapping[str, Any],
                                      result: Mapping[str, Any], *, updated_at: Optional[float] = None) -> bool:
        row = self._write_returning(
            """
            UPDATE async_delegations SET state = 'unknown', completed_at = %s, updated_at = %s,
              event_json = %s, result_json = %s, delivery_state = 'pending'
            WHERE delegation_id = %s AND state IN ('running', 'finalizing') RETURNING delegation_id
            """,
            (event.get("completed_at", time.time()), time.time() if updated_at is None else updated_at,
             self._json_dumps(dict(event)), self._json_dumps(dict(result)), delegation_id),
        )
        return row is not None

    def list_pending_async_delegation_events(self) -> List[Dict[str, Any]]:
        rows = self._read_many(
            "SELECT event_json FROM async_delegations WHERE state != 'running' "
            "AND delivery_state = 'pending' AND event_json IS NOT NULL "
            "ORDER BY completed_at, delegation_id LIMIT %s",
            (self._MAX_READ_ROWS,),
            limit=self._MAX_READ_ROWS,
        )
        return [self._json_loads(row["event_json"], default={}) for row in rows]

    def mark_async_delegation_delivered(self, delegation_id: str, *, updated_at: Optional[float] = None) -> bool:
        return bool(self._write_returning(
            """
            UPDATE async_delegations SET delivery_state = 'delivered', delivered_at = %s,
              updated_at = %s WHERE delegation_id = %s AND delivery_state != 'delivered'
            RETURNING delegation_id
            """,
            (time.time() if updated_at is None else updated_at,
             time.time() if updated_at is None else updated_at, delegation_id),
        ))

    def claim_async_delegation_delivery(self, delegation_id: str, claim_id: str,
                                        *, claim_timeout_seconds: float = 300,
                                        updated_at: Optional[float] = None) -> bool:
        now = time.time() if updated_at is None else updated_at
        def operation(connection: Any) -> bool:
            existing = self._row(
                connection, "SELECT delivery_state FROM async_delegations WHERE delegation_id = %s FOR UPDATE",
                (delegation_id,),
            )
            if existing is None:
                return True
            row = self._row(
                connection,
                """
                UPDATE async_delegations SET delivery_claim = %s, delivery_claimed_at = %s,
                  delivery_attempts = delivery_attempts + 1, updated_at = %s
                WHERE delegation_id = %s AND delivery_state = 'pending'
                  AND (delivery_claim IS NULL OR delivery_claimed_at < %s)
                RETURNING delegation_id
                """,
                (claim_id, now, now, delegation_id, now - claim_timeout_seconds),
            )
            return row is not None
        return self._run(operation)

    def release_async_delegation_delivery(self, delegation_id: str, claim_id: str, *, updated_at: Optional[float] = None) -> bool:
        return bool(self._write_returning(
            """
            UPDATE async_delegations SET delivery_claim = NULL, delivery_claimed_at = NULL,
              updated_at = %s WHERE delegation_id = %s AND delivery_state = 'pending'
              AND delivery_claim = %s RETURNING delegation_id
            """,
            (time.time() if updated_at is None else updated_at, delegation_id, claim_id),
        ))

    def complete_async_delegation_delivery(self, delegation_id: str, claim_id: str, *, updated_at: Optional[float] = None) -> bool:
        now = time.time() if updated_at is None else updated_at
        return bool(self._write_returning(
            """
            UPDATE async_delegations SET delivery_state = 'delivered', delivered_at = %s,
              updated_at = %s, delivery_claim = NULL, delivery_claimed_at = NULL
            WHERE delegation_id = %s AND delivery_state = 'pending' AND delivery_claim = %s
            RETURNING delegation_id
            """,
            (now, now, delegation_id, claim_id),
        ))

    def get_async_delegation(self, delegation_id: str) -> Optional[Dict[str, Any]]:
        row = self._read_one(
            """
            SELECT origin_session, state, dispatched_at, completed_at, result_json,
              delivery_state, delivery_attempts FROM async_delegations WHERE delegation_id = %s
            """,
            (delegation_id,),
        )
        if row is not None:
            row["delegation_id"] = delegation_id
            row["result"] = self._json_loads(row.pop("result_json", None), default=None)
        return row

    # ── State metadata, Telegram, and handoff ────────────────────────────

    def get_meta(self, key: str) -> Optional[str]:
        row = self._read_one("SELECT value FROM state_meta WHERE key = %s", (key,))
        return None if row is None else row.get("value")

    def set_meta(self, key: str, value: str) -> None:
        self._write(
            "INSERT INTO state_meta(key, value) VALUES (%s, %s) "
            "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def apply_telegram_topic_migration(self) -> None:
        self._state_store.ensure_telegram_schema()

    def enable_telegram_topic_mode(self, *, chat_id: str, user_id: str,
                                   has_topics_enabled: Optional[bool] = None,
                                   allows_users_to_create_topics: Optional[bool] = None) -> None:
        self.apply_telegram_topic_migration()
        self._write(
            """
            INSERT INTO telegram_dm_topic_mode(chat_id, user_id, enabled, activated_at, updated_at,
              has_topics_enabled, allows_users_to_create_topics, capability_checked_at)
            VALUES (%s, %s, 1, EXTRACT(EPOCH FROM clock_timestamp()), EXTRACT(EPOCH FROM clock_timestamp()),
              %s, %s, EXTRACT(EPOCH FROM clock_timestamp()))
            ON CONFLICT(chat_id) DO UPDATE SET user_id = EXCLUDED.user_id, enabled = 1,
              updated_at = EXCLUDED.updated_at, has_topics_enabled = EXCLUDED.has_topics_enabled,
              allows_users_to_create_topics = EXCLUDED.allows_users_to_create_topics,
              capability_checked_at = EXCLUDED.capability_checked_at
            """,
            (str(chat_id), str(user_id),
             None if has_topics_enabled is None else int(has_topics_enabled),
             None if allows_users_to_create_topics is None else int(allows_users_to_create_topics)),
        )

    def disable_telegram_topic_mode(self, *, chat_id: str, clear_bindings: bool = True) -> None:
        def operation(connection: Any) -> None:
            connection.execute(
                "UPDATE telegram_dm_topic_mode SET enabled = 0, "
                "updated_at = EXTRACT(EPOCH FROM clock_timestamp()) WHERE chat_id = %s",
                (str(chat_id),),
            )
            if clear_bindings:
                connection.execute("DELETE FROM telegram_dm_topic_bindings WHERE chat_id = %s", (str(chat_id),))
        self._run(operation)

    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str) -> bool:
        row = self._read_one(
            "SELECT enabled FROM telegram_dm_topic_mode WHERE chat_id = %s AND user_id = %s",
            (str(chat_id), str(user_id)),
        )
        return bool(row and row.get("enabled"))

    def get_telegram_topic_binding(self, *, chat_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        return self._read_one(
            "SELECT * FROM telegram_dm_topic_bindings WHERE chat_id = %s AND thread_id = %s",
            (str(chat_id), str(thread_id)),
        )

    def list_telegram_topic_bindings_for_chat(self, *, chat_id: str) -> List[Dict[str, Any]]:
        return self._read_many(
            "SELECT * FROM telegram_dm_topic_bindings WHERE chat_id = %s "
            "ORDER BY updated_at DESC LIMIT %s",
            (str(chat_id), self._MAX_READ_ROWS),
            limit=self._MAX_READ_ROWS,
        )

    def get_telegram_topic_binding_by_session(self, *, session_id: str) -> Optional[Dict[str, Any]]:
        return self._read_one(
            "SELECT * FROM telegram_dm_topic_bindings WHERE session_id = %s", (str(session_id),)
        )

    def delete_telegram_topic_binding(self, *, chat_id: str, thread_id: str) -> int:
        def operation(connection: Any) -> int:
            cursor = connection.execute(
                "DELETE FROM telegram_dm_topic_bindings WHERE chat_id = %s AND thread_id = %s",
                (str(chat_id), str(thread_id)),
            )
            count = int(getattr(cursor, "rowcount", 0) or 0)
            if count and not self._row(
                connection,
                "SELECT 1 FROM telegram_dm_topic_bindings WHERE chat_id = %s LIMIT 1",
                (str(chat_id),),
            ):
                connection.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, "
                    "updated_at = EXTRACT(EPOCH FROM clock_timestamp()) WHERE chat_id = %s",
                    (str(chat_id),),
                )
            return count
        return self._run(operation)

    def bind_telegram_topic(self, *, chat_id: str, thread_id: str, user_id: str,
                            session_key: str, session_id: str, managed_mode: str = "auto") -> None:
        self.apply_telegram_topic_migration()
        def operation(connection: Any) -> None:
            existing = self._row(
                connection, "SELECT chat_id, thread_id FROM telegram_dm_topic_bindings "
                "WHERE session_id = %s FOR UPDATE", (str(session_id),)
            )
            if existing and (str(existing["chat_id"]) != str(chat_id) or str(existing["thread_id"]) != str(thread_id)):
                raise ValueError("session is already linked to another Telegram topic")
            connection.execute(
                """
                INSERT INTO telegram_dm_topic_bindings(chat_id, thread_id, user_id, session_key,
                  session_id, managed_mode, linked_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, EXTRACT(EPOCH FROM clock_timestamp()),
                  EXTRACT(EPOCH FROM clock_timestamp()))
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET user_id = EXCLUDED.user_id,
                  session_key = EXCLUDED.session_key, session_id = EXCLUDED.session_id,
                  managed_mode = EXCLUDED.managed_mode, updated_at = EXCLUDED.updated_at
                """,
                (str(chat_id), str(thread_id), str(user_id), str(session_key), str(session_id), managed_mode),
            )
        self._run(operation)

    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        return self._read_one(
            "SELECT 1 AS linked FROM telegram_dm_topic_bindings WHERE session_id = %s LIMIT 1",
            (str(session_id),),
        ) is not None

    def list_unlinked_telegram_sessions_for_user(self, *, chat_id: str, user_id: str,
                                                 limit: int = 10) -> List[Dict[str, Any]]:
        del chat_id
        page = max(1, min(int(limit), self._MAX_READ_ROWS))
        rows = self._read_many(
            """
            SELECT s.*, COALESCE((SELECT LEFT(m.content, 63) FROM messages m
              WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
              ORDER BY m.timestamp, m.id LIMIT 1), '') AS _preview_raw,
              COALESCE((SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id),
                s.started_at) AS last_active
            FROM sessions s WHERE s.source = 'telegram' AND s.user_id = %s
              AND NOT EXISTS (SELECT 1 FROM telegram_dm_topic_bindings b WHERE b.session_id = s.id)
            ORDER BY last_active DESC, s.started_at DESC LIMIT %s
            """,
            (str(user_id), page),
            limit=page,
        )
        for row in rows:
            raw = str(row.pop("_preview_raw", "") or "").strip()
            row["preview"] = raw[:60] + ("..." if len(raw) > 60 else "") if raw else ""
        return rows

    def request_handoff(self, session_id: str, platform: str) -> bool:
        return bool(self._write_returning(
            """
            UPDATE sessions SET handoff_state = 'pending', handoff_platform = %s,
              handoff_error = NULL WHERE id = %s AND (
                handoff_state IS NULL OR handoff_state IN ('completed', 'failed')
              ) RETURNING id
            """,
            (platform, session_id),
        ))

    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = self._read_one(
            "SELECT handoff_state, handoff_platform, handoff_error FROM sessions WHERE id = %s",
            (session_id,),
        )
        if row is None:
            return None
        return {"state": row.get("handoff_state"), "platform": row.get("handoff_platform"), "error": row.get("handoff_error")}

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        return self._read_many(
            "SELECT * FROM sessions WHERE handoff_state = 'pending' ORDER BY started_at ASC LIMIT %s",
            (self._MAX_READ_ROWS,),
            limit=self._MAX_READ_ROWS,
        )

    def claim_handoff(self, session_id: str) -> bool:
        return bool(self._write_returning(
            "UPDATE sessions SET handoff_state = 'running' WHERE id = %s "
            "AND handoff_state = 'pending' RETURNING id",
            (session_id,),
        ))

    def complete_handoff(self, session_id: str) -> None:
        self._write(
            "UPDATE sessions SET handoff_state = 'completed', handoff_error = NULL WHERE id = %s",
            (session_id,),
        )

    def fail_handoff(self, session_id: str, error: str) -> None:
        self._write(
            "UPDATE sessions SET handoff_state = 'failed', handoff_error = %s WHERE id = %s",
            (error[:500], session_id),
        )
