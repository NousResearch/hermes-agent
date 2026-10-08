"""Gateway route CRUD, CAS transcript switching, and routed session listings."""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from state_store import MessageRecord
from state_store_postgresql_sessions import _SESSION_METADATA_COLUMNS


class PostgreSQLGatewayRoutesMixin:
    @staticmethod
    def _route_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _gateway_route_row(self, cursor: Any) -> dict[str, Any] | None:
        row = cursor.fetchone()
        return None if row is None else dict(row)

    def get_or_create_gateway_session_route(self, tenant_namespace: str, session_key: str, session_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if not tenant_namespace or not session_key or not session_id:
            raise ValueError("gateway route requires tenant namespace, session key, and session id")
        session_metadata = {key: value for key, value in dict(metadata).items() if key in _SESSION_METADATA_COLUMNS}
        self.ensure_session(session_id, source=str(metadata.get("source") or "gateway"), metadata=session_metadata)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            clock_row = cursor.fetchone()
            now = float(clock_row["extract"] if isinstance(clock_row, Mapping) else clock_row[0])
            cursor.execute(
                "INSERT INTO gateway_session_routes (tenant_namespace, session_key, session_id, flags, metadata, created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (tenant_namespace, session_key) DO UPDATE SET updated_at=gateway_session_routes.updated_at "
                "RETURNING tenant_namespace,session_key,session_id,generation,flags,metadata,created_at,updated_at",
                (tenant_namespace, session_key, session_id, self._psycopg.types.json.Jsonb({}), self._psycopg.types.json.Jsonb(self._route_payload(metadata)), now, now),
            )
            return dict(cursor.fetchone())

    def get_gateway_session_route_by_key(self, tenant_namespace: str, session_key: str) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT tenant_namespace,session_key,session_id,generation,flags,metadata,created_at,updated_at FROM gateway_session_routes WHERE tenant_namespace=%s AND session_key=%s", (tenant_namespace, session_key))
            return self._gateway_route_row(cursor)

    def get_gateway_session_route_by_session_id(self, tenant_namespace: str, session_id: str) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT tenant_namespace,session_key,session_id,generation,flags,metadata,created_at,updated_at FROM gateway_session_routes WHERE tenant_namespace=%s AND session_id=%s", (tenant_namespace, session_id))
            return self._gateway_route_row(cursor)

    def switch_gateway_session_route(self, tenant_namespace: str, session_key: str, session_id: str, expected_session_id: str, expected_generation: int, metadata: Mapping[str, Any], flags: Mapping[str, Any]) -> dict[str, Any] | None:
        session_metadata = {key: value for key, value in dict(metadata).items() if key in _SESSION_METADATA_COLUMNS}
        self.ensure_session(session_id, source=str(metadata.get("source") or "gateway"), metadata=session_metadata)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            clock_row = cursor.fetchone()
            now = float(clock_row["extract"] if isinstance(clock_row, Mapping) else clock_row[0])
            cursor.execute("UPDATE gateway_session_routes SET session_id=%s,generation=generation+1,flags=%s,metadata=%s,updated_at=%s WHERE tenant_namespace=%s AND session_key=%s AND session_id=%s AND generation=%s RETURNING tenant_namespace,session_key,session_id,generation,flags,metadata,created_at,updated_at", (session_id, self._psycopg.types.json.Jsonb(self._route_payload(flags)), self._psycopg.types.json.Jsonb(self._route_payload(metadata)), now, tenant_namespace, session_key, expected_session_id, expected_generation))
            return self._gateway_route_row(cursor)

    @property
    def tenant_schema(self) -> str:
        """Trusted tenant namespace this store is bound to (route/transcript authority)."""
        return self._schema

    def append_message_and_switch_route(
        self, tenant_namespace: str, session_key: str, record: MessageRecord, *,
        expected_session_id: str, expected_generation: int, new_session_id: str,
        metadata: Mapping[str, Any], flags: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically append one message to ``new_session_id``'s transcript and
        CAS-switch the gateway route from ``expected_session_id`` to
        ``new_session_id`` in one server transaction.

        The route row is locked ``FOR UPDATE`` before anything else runs, so the
        target-session upsert, the transcript append, the session-activity bump,
        and the generation increment either all commit or all roll back.  A lost
        CAS (the route moved after the caller's snapshot) raises
        :class:`GatewayRouteContentionError` **before any mutation**: a contended
        route's transcript is never appended and its generation never advances.
        """
        from state_store_postgresql import (
            GatewayRouteContentionError, _MESSAGE_RECORD_WRITE_COLUMNS, _MESSAGE_INSERT_PLACEHOLDERS,
        )

        if not tenant_namespace or not session_key:
            raise ValueError("gateway route switch requires tenant namespace and session key")
        if not new_session_id or new_session_id == expected_session_id:
            raise ValueError("route switch requires a distinct new session id")
        session_metadata = {key: value for key, value in dict(metadata).items() if key in _SESSION_METADATA_COLUMNS}
        source = str(metadata.get("source") or "gateway")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                "SELECT session_id, generation FROM gateway_session_routes "
                "WHERE tenant_namespace=%s AND session_key=%s FOR UPDATE",
                (tenant_namespace, session_key),
            )
            current = cursor.fetchone()
            if current is None:
                raise GatewayRouteContentionError(
                    f"gateway route {session_key!r} does not exist; refusing transcript append")
            if current["session_id"] != expected_session_id or int(current["generation"]) != expected_generation:
                raise GatewayRouteContentionError(
                    f"gateway route {session_key!r} CAS lost "
                    f"(expected {expected_session_id}@{expected_generation}, "
                    f"found {current['session_id']}@{current['generation']})")
            self._ensure_session_on_cursor(cursor, new_session_id, source, session_metadata)
            self._transcript_write_guards(cursor, new_session_id)
            cursor.execute(
                f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) "
                f"VALUES ({_MESSAGE_INSERT_PLACEHOLDERS}) RETURNING id, created_at",
                self._record_params(new_session_id, record, cursor=cursor),
            )
            row = cursor.fetchone()
            created_at = row["created_at"]
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET last_activity_at = GREATEST("
                "COALESCE(last_activity_at, started_at), %s) WHERE id = %s",
                (created_at, new_session_id),
            )
            cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            clock_row = cursor.fetchone()
            now = float(clock_row["extract"] if isinstance(clock_row, Mapping) else clock_row[0])
            cursor.execute(
                "UPDATE gateway_session_routes SET session_id=%s,generation=generation+1,flags=%s,metadata=%s,updated_at=%s "
                "WHERE tenant_namespace=%s AND session_key=%s AND session_id=%s AND generation=%s "
                "RETURNING tenant_namespace,session_key,session_id,generation,flags,metadata,created_at,updated_at",
                (new_session_id, self._psycopg.types.json.Jsonb(self._route_payload(flags)),
                 self._psycopg.types.json.Jsonb(self._route_payload(metadata)), now,
                 tenant_namespace, session_key, expected_session_id, expected_generation))
            switched = self._gateway_route_row(cursor)
            if switched is None:
                # The FOR UPDATE lock makes this unreachable against this store's
                # own writers; fail closed rather than return an unswitched commit.
                raise GatewayRouteContentionError(
                    f"gateway route {session_key!r} disappeared under the route switch transaction")
            return switched

    def _ensure_session_on_cursor(
        self, cursor: Any, session_id: str, source: str, metadata: Mapping[str, Any],
    ) -> None:
        """In-transaction twin of ``ensure_session`` for callers already holding a
        connection (the route-switch transaction must not acquire a second pooled
        connection while the first is held)."""
        metadata = dict(metadata or {})
        metadata_columns = tuple(column for column in _SESSION_METADATA_COLUMNS if column in metadata)
        columns = ("id", "source", "started_at", *metadata_columns)
        placeholders = ", ".join("%s" for _ in columns)
        values: list[Any] = [session_id, source, time.time()]
        for column in metadata_columns:
            value = metadata[column]
            if column == "model_config":
                value = self._psycopg.types.json.Jsonb(value) if value else None
            values.append(value)
        cursor.execute(
            f"INSERT INTO {self._schema}.sessions ({', '.join(columns)}) VALUES ({placeholders}) "
            "ON CONFLICT (id) DO UPDATE SET "
            "model = COALESCE(sessions.model, EXCLUDED.model), "
            "model_config = COALESCE(sessions.model_config, EXCLUDED.model_config), "
            "session_key = COALESCE(sessions.session_key, EXCLUDED.session_key), "
            "chat_id = COALESCE(sessions.chat_id, EXCLUDED.chat_id), "
            "chat_type = COALESCE(sessions.chat_type, EXCLUDED.chat_type), "
            "thread_id = COALESCE(sessions.thread_id, EXCLUDED.thread_id), "
            "parent_session_id = COALESCE(sessions.parent_session_id, EXCLUDED.parent_session_id), "
            "cwd = COALESCE(sessions.cwd, EXCLUDED.cwd), "
            "profile_name = COALESCE(sessions.profile_name, EXCLUDED.profile_name), "
            "git_repo_root = COALESCE(sessions.git_repo_root, EXCLUDED.git_repo_root), "
            "origin_json = COALESCE(sessions.origin_json, EXCLUDED.origin_json), "
            "display_name = COALESCE(sessions.display_name, EXCLUDED.display_name)",
            values,
        )

    def delete_or_repair_gateway_session_route(self, tenant_namespace: str, session_key: str, expected_session_id: str, expected_generation: int, replacement_session_id: str | None, metadata: Mapping[str, Any]) -> dict[str, Any] | bool | None:
        if replacement_session_id:
            return self.switch_gateway_session_route(tenant_namespace, session_key, replacement_session_id, expected_session_id, expected_generation, metadata, {})
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM gateway_session_routes WHERE tenant_namespace=%s AND session_key=%s AND session_id=%s AND generation=%s", (tenant_namespace, session_key, expected_session_id, expected_generation))
            return cursor.rowcount == 1

    def list_gateway_sessions(
        self, *, platform: str | None = None, active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """One live mapping per gateway routing key, mirroring SQLite's listing surface.

        ``gateway_session_routes`` is the key→current-session authority on PostgreSQL, so
        each currently-routed key contributes exactly its pointed-at session row (the PK
        on ``(tenant_namespace, session_key)`` keeps the join one-row-per-key). The
        ``last_active`` projection matches SQLite's freshest-of expression: activity
        timestamp, latest message timestamp, else ``started_at``. Read-only status
        consumers call this; it never creates sessions and never touches SQLite.
        """
        clauses, params = ["route.tenant_namespace = %s"], [self._schema]
        if platform:
            clauses.append("LOWER(s.source) = LOWER(%s)")
            params.append(platform)
        if active_only:
            clauses.append("s.ended_at IS NULL")
        last_active = (
            "COALESCE(GREATEST(s.last_activity_at, "
            f"(SELECT MAX(m.created_at) FROM {self._schema}.messages AS m WHERE m.session_id = s.id), "
            "s.started_at))"
        )
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT s.id, s.source, s.session_key, s.chat_id, s.chat_type, s.thread_id, "
                f"s.display_name, s.started_at, s.ended_at, s.end_reason, {last_active} AS last_active "
                f"FROM {self._schema}.gateway_session_routes AS route "
                f"JOIN {self._schema}.sessions AS s ON s.id = route.session_id "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY last_active DESC, s.started_at DESC, s.id DESC",
                params,
            )
            return [dict(row) for row in cursor.fetchall()]
