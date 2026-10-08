"""PostgreSQL implementation of the first State Store session/message slice.

This deliberately owns only the narrow compatibility contract in ``state_store``.
It uses a fixed internal schema name, never interpolates caller data into SQL, and
is not yet a replacement for the full SessionDB state surface.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
import math
import queue
import re
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any, Collection

from hermes_cli.timefmt import coerce_epoch
from agent.message_metadata import TOOL_CALL_UID, TOOL_CALL_UIDS, mint_uid, message_uid_or_none, index_tool_call_uids, resolve_tool_call_uid
from agent.message_sanitization import coalesce_tool_call_id, tool_result_id_variants
from agent.session_activity import bound_activity_description, normalize_activity_provenance
from hermes_state_identity import _absorbed_uids_json, _tool_call_uids_json, _tool_call_uid_or_none, _tool_call_uid_map, _restore_identity_columns
from hermes_state_common import TITLE_SOURCE_LLM as _TITLE_SOURCE_LLM
from hermes_state_ids import new_session_id
from hermes_state_runtime_ownership import RuntimeOwner, RuntimeOwnershipReceipt, SessionRuntimeOwnershipMixin, TurnState
from state_store import MessageRecord, PostgreSQLStateStoreConfig, StateStoreConfigurationError
from state_store_postgresql_search import compile_postgresql_search_expression
from state_store_postgresql_gateway_routes import PostgreSQLGatewayRoutesMixin
from state_store_postgresql_sessions import (
    PostgreSQLSessionsMixin, _SESSION_METADATA_COLUMNS, _USAGE_ROUTE_FIELDS,
    _USAGE_SUM_FIELDS, _USAGE_COUNTERS, _sanitize_title,
)
from state_store_alembic.migration_helpers import TrustedTenantSchema, require_trusted_tenant_schema
from token_usage_transport import TokenUsageTransport

logger = logging.getLogger(__name__)

_SCHEMA_PATTERN = re.compile(r"^hermes_state_store_tenant_[0-9a-f]{32}$")
_SESSION_METADATA_SCHEMA_VERSION = 2
_PARENT_SESSION_FOREIGN_KEY_SCHEMA_VERSION = 3
# Version 4 is a deliberately recorded compatibility checkpoint. It has no DDL
# because it only establishes a durable, validated ledger boundary for the v1-v3
# contract after releases that wrote the parent key outside the migration ledger.
_COMPATIBILITY_CHECKPOINT_SCHEMA_VERSION = 4
_SCHEMA_VERSION = 5
_VISIBILITY_SCHEMA_VERSION = 6
_MESSAGE_RECORD_SCHEMA_VERSION = 7
_RESUME_PROJECTION_SCHEMA_VERSION = 8
_SYSTEM_PROMPT_SCHEMA_VERSION = 9
_MODEL_USAGE_SCHEMA_VERSION = 10
_CONVERSATION_GENERATION_SCHEMA_VERSION = 11
_MODEL_CONFIG_LIFECYCLE_SCHEMA_VERSION = 12
_GIT_METADATA_GENERATION_SCHEMA_VERSION = 13
_SEARCH_DOCUMENT_SCHEMA_VERSION = 14
_SEARCH_INDEX_SCHEMA_VERSION = 15
_BOUNDED_BROWSE_SCHEMA_VERSION = 16
_SEARCH_HEALTH_SCHEMA_VERSION = 17
_SESSION_RUNTIME_OWNERSHIP_SCHEMA_VERSION = 18
_COMPRESSION_COORDINATION_SCHEMA_VERSION = 19
_COMPRESSION_ROTATION_SCHEMA_VERSION = 20
_SESSION_CONTROL_STATE_SCHEMA_VERSION = 21
_TRANSCRIPT_REWIND_SCHEMA_VERSION = 22
_FOREIGN_IMPORT_RECEIPT_SCHEMA_VERSION = 23
_GATEWAY_SESSION_ROUTE_SCHEMA_VERSION = 24
_GATEWAY_TRANSCRIPT_SCHEMA_VERSION = 25
_SEARCH_INDEX_NAME = "messages_search_document_gin"
_USAGE_SESSION_COLUMNS = (*_USAGE_COUNTERS, "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source", "pricing_version", "billing_provider", "billing_base_url", "billing_mode", "api_call_count")
_MESSAGE_RECORD_COLUMNS = (
    "tool_call_id", "tool_calls", "tool_name", "effect_disposition", "token_count", "finish_reason",
    "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
    "platform_message_id", "observed", "_compressed_summary", "active", "compacted", "api_content",
    "display_kind", "display_metadata", "display_identity",
    "message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid",
)
_MESSAGE_RECORD_WRITE_COLUMNS = tuple(
    column for column in _MESSAGE_RECORD_COLUMNS if column not in {"active", "compacted", "display_identity"}
)
_MESSAGE_INSERT_PLACEHOLDERS = ", ".join("%s" for _ in range(4 + len(_MESSAGE_RECORD_WRITE_COLUMNS)))
_SEARCH_RESULT_FIELDS = (
    "id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model", "session_started", "context",
)
_SESSION_METADATA_TYPES = {
    "user_id": "text", "session_key": "text", "chat_id": "text", "chat_type": "text", "thread_id": "text",
    "display_name": "text", "origin_json": "text", "model": "text", "model_config": "jsonb",
    "parent_session_id": "text", "cwd": "text", "profile_name": "text", "git_repo_root": "text",
}


class GatewayRouteContentionError(RuntimeError):
    """A CAS-guarded gateway route transaction refused to run against a moved route.

    Raised strictly before any transcript or route mutation: the caller must
    re-read the route and decide whether to retry with the fresh snapshot; the
    losing writer never leaves a partial append or generation bump behind.
    """


class PostgreSQLStateStore(PostgreSQLGatewayRoutesMixin, PostgreSQLSessionsMixin, SessionRuntimeOwnershipMixin):
    """Thread-safe bounded psycopg connection pool for session/message persistence."""

    # StateStoreInterface conformance: the automatic-title source a titler writes with.
    TITLE_SOURCE_LLM = _TITLE_SOURCE_LLM

    @staticmethod
    def sanitize_title(title: str | None) -> str | None:
        """Same normalization as the SQLite store: strip control/zero-width/bidi
        chars, collapse whitespace, normalize empty to None, ValueError past the cap."""
        return _sanitize_title(title)

    def __init__(self, settings: PostgreSQLStateStoreConfig, dsn: str, *, schema: TrustedTenantSchema) -> None:
        try:
            schema = require_trusted_tenant_schema(schema)
        except Exception as exc:
            raise StateStoreConfigurationError(
                "PostgreSQL State Store requires a runtime-derived trusted tenant schema capability"
            ) from exc
        try:
            self._psycopg = importlib.import_module("psycopg")
        except ImportError as exc:
            raise StateStoreConfigurationError(
                "PostgreSQL State Store requires the optional dependency: pip install 'hermes-agent[state-store]'"
            ) from exc
        self._settings = settings
        self._dsn = dsn
        self._tenant_schema = schema
        self._schema = schema.name
        self._idle: queue.LifoQueue[Any] = queue.LifoQueue(maxsize=settings.pool_max_size)
        self._created = 1
        self._lock = threading.Lock()
        self._closed = False
        self._token_usage_transport = TokenUsageTransport(
            self._persist_token_usage_delta, sum_fields=_USAGE_SUM_FIELDS,
            cost_fields=("estimated_cost_usd", "actual_cost_usd"), route_fields=_USAGE_ROUTE_FIELDS,
            idle_seconds=lambda: 1.0,
        )
        connection = self._new_connection()
        try:
            self._probe_and_migrate(connection)
            self._set_connection_search_path(connection)
        except Exception:
            connection.close()
            raise
        self._idle.put(connection)

    def _new_connection(self) -> Any:
        return self._psycopg.connect(
            self._dsn,
            connect_timeout=self._settings.connect_timeout_seconds,
            autocommit=False,
        )

    def _set_connection_search_path(self, connection: Any) -> None:
        """Reset the pooled connection to this store's trusted tenant schema."""
        with connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{self._schema}", pg_catalog')
        connection.commit()

    def _probe_and_migrate(self, connection: Any) -> None:
        """Bootstrap the sole Alembic head, then validate every v1-v25 core contract.

        Alembic owns schema evolution.  The legacy numeric ledger is rejected by
        the bootstrap before mutation; validators remain separate proof that the
        tenant catalog still satisfies every historical core invariant.
        """
        try:
            from state_store_alembic.runner import upgrade_new_tenant_to_current

            upgrade_new_tenant_to_current(connection, self._tenant_schema)
            with connection.cursor() as cursor:
                self._validate_core_v1_v25_catalog(cursor)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _validate_core_v1_v25_catalog(self, cursor: Any) -> None:
        """Read-only proof that Alembic's current catalog is exact."""
        from state_store_alembic.semantic_catalog import validate_current_catalog_cursor

        validate_current_catalog_cursor(cursor, self._schema)

    @staticmethod
    def _control_kind(control_kind: str) -> str:
        if control_kind not in {"goal", "heartbeat", "loop"}:
            raise ValueError(f"unsupported session control kind: {control_kind!r}")
        return control_kind

    def get_session_control_state(self, session_id: str, control_kind: str) -> dict[str, Any] | None:
        control_kind = self._control_kind(control_kind)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT status, payload, revision, updated_at FROM session_control_state WHERE session_id=%s AND control_kind=%s", (session_id, control_kind))
            row = cursor.fetchone()
        if row is None:
            return None
        status, payload, revision, updated_at = row
        return {"status": status, "payload": payload, "revision": int(revision), "updated_at": float(updated_at)}

    def put_session_control_state(self, session_id: str, control_kind: str, status: str, payload: Mapping[str, Any], *, expected_revision: int | None = None) -> int | None:
        control_kind = self._control_kind(control_kind)
        if not session_id or not isinstance(payload, Mapping):
            raise ValueError("session control state requires session_id and object payload")
        encoded = self._psycopg.types.json.Jsonb(dict(payload))
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            now = float(cursor.fetchone()[0])
            if expected_revision is None:
                cursor.execute(
                    "INSERT INTO session_control_state (session_id, control_kind, status, payload, revision, updated_at) VALUES (%s, %s, %s, %s, 1, %s) "
                    "ON CONFLICT (session_id, control_kind) DO UPDATE SET status=EXCLUDED.status, payload=EXCLUDED.payload, revision=session_control_state.revision + 1, updated_at=EXCLUDED.updated_at RETURNING revision",
                    (session_id, control_kind, status, encoded, now),
                )
            else:
                cursor.execute(
                    "UPDATE session_control_state SET status=%s, payload=%s, revision=revision + 1, updated_at=%s WHERE session_id=%s AND control_kind=%s AND revision=%s RETURNING revision",
                    (status, encoded, now, session_id, control_kind, expected_revision),
                )
            row = cursor.fetchone()
            return None if row is None else int(row[0])

    def list_session_control_states(self, control_kind: str, *, status: str | None = None) -> list[dict[str, Any]]:
        control_kind = self._control_kind(control_kind)
        with self._connection() as connection, connection.cursor() as cursor:
            if status is None:
                cursor.execute("SELECT session_id, status, payload, revision, updated_at FROM session_control_state WHERE control_kind=%s", (control_kind,))
            else:
                cursor.execute("SELECT session_id, status, payload, revision, updated_at FROM session_control_state WHERE control_kind=%s AND status=%s", (control_kind, status))
            rows = cursor.fetchall()
        return [{"session_id": row[0], "status": row[1], "payload": row[2], "revision": int(row[3]), "updated_at": float(row[4])} for row in rows]

    def transfer_session_control_states(self, parent_session_id: str, child_session_id: str) -> bool:
        """Atomically move all durable controls for compression without duplicate actives."""
        if not parent_session_id or not child_session_id or parent_session_id == child_session_id:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            # Advisory locks serialize absent-row checks across concurrent rotations.
            for session_id in sorted((parent_session_id, child_session_id)):
                cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{self._schema}:session-control:{session_id}",))
            cursor.execute("SELECT 1 FROM sessions WHERE id=%s FOR KEY SHARE", (parent_session_id,))
            if cursor.fetchone() is None:
                return False
            cursor.execute("SELECT 1 FROM sessions WHERE id=%s FOR KEY SHARE", (child_session_id,))
            if cursor.fetchone() is None:
                return False
            cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
            now = float(cursor.fetchone()[0])
            return self._transfer_session_control_states_in_transaction(
                cursor, parent_session_id, child_session_id, now, require_empty_child=True,
            )

    def _transfer_session_control_states_in_transaction(
        self, cursor: Any, parent_session_id: str, child_session_id: str, now: float, *, require_empty_child: bool,
    ) -> bool:
        """Move active controls while the caller owns the parent/child publication transaction."""
        cursor.execute("SELECT 1 FROM session_control_state WHERE session_id=%s FOR UPDATE", (child_session_id,))
        if require_empty_child and cursor.fetchone() is not None:
            raise RuntimeError("compression child unexpectedly already has session controls")
        cursor.execute("SELECT control_kind, status, payload FROM session_control_state WHERE session_id=%s FOR UPDATE", (parent_session_id,))
        rows = cursor.fetchall()
        normalized_rows = [
            (row["control_kind"], row["status"], row["payload"])
            if isinstance(row, Mapping) else (row[0], row[1], row[2])
            for row in rows
        ]
        active = [(kind, status, payload) for kind, status, payload in normalized_rows if status not in {"cleared", "done"}]
        for kind, status, payload in active:
            cursor.execute("INSERT INTO session_control_state (session_id, control_kind, status, payload, revision, updated_at) VALUES (%s, %s, %s, %s, 1, %s)", (child_session_id, kind, status, self._psycopg.types.json.Jsonb(payload), now))
            archived = dict(payload)
            archived["status"] = "cleared"
            cursor.execute("UPDATE session_control_state SET status='cleared', payload=%s, revision=revision+1, updated_at=%s WHERE session_id=%s AND control_kind=%s", (self._psycopg.types.json.Jsonb(archived), now, parent_session_id, kind))
        return bool(active)

    @staticmethod
    def _runtime_namespace(namespace: str | None) -> str:
        return (namespace or "").strip()

    @staticmethod
    def _runtime_ttl_seconds(ttl_seconds: float) -> float:
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl):
            raise ValueError("runtime ownership ttl_seconds must be finite")
        return max(0.1, ttl)

    def _ownership_lock_and_clock(self, cursor: Any, namespace: str, session_id: str) -> float:
        """Serialize one owner key, including absent-row claims, then read server time."""
        cursor.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"{self._schema}:runtime-owner:{namespace}:{session_id}",),
        )
        cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp())")
        row = cursor.fetchone()
        return float(next(iter(row.values())) if isinstance(row, Mapping) else row[0])

    @staticmethod
    def _receipt_matches(row: Mapping[Any, Any], receipt: RuntimeOwnershipReceipt, now: float) -> bool:
        owner = receipt.owner
        return (
            int(row["fence"]) == receipt.fence and float(row["expires_at"]) > now
            and row["installation_id"] == owner.installation_id and row["host"] == owner.host
            and row["process_generation"] == owner.process_generation
        )

    def acquire_session_runtime_ownership(
        self, session_id: str, owner: RuntimeOwner, *, ttl_seconds: float = 300.0, namespace: str | None = None,
    ) -> RuntimeOwnershipReceipt | None:
        if not session_id:
            return None
        installation_id, host, process_generation = self._runtime_owner_columns(owner)
        namespace = self._runtime_namespace(namespace)
        ttl = self._runtime_ttl_seconds(ttl_seconds)
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._ownership_lock_and_clock(cursor, namespace, session_id)
            expires_at = now + ttl
            cursor.execute(
                "SELECT installation_id, host, process_generation, fence, expires_at "
                "FROM session_runtime_owners WHERE namespace=%s AND session_id=%s FOR UPDATE",
                (namespace, session_id),
            )
            raw_row = cursor.fetchone()
            row = None if raw_row is None else dict(zip(
                ("installation_id", "host", "process_generation", "fence", "expires_at"), raw_row))
            if row is None:
                fence = 1
                cursor.execute(
                    "INSERT INTO session_runtime_owners (namespace, session_id, installation_id, host, process_generation, fence, expires_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (namespace, session_id, installation_id, host, process_generation, fence, expires_at, now),
                )
            elif (row["installation_id"], row["host"], row["process_generation"]) == (installation_id, host, process_generation):
                fence = int(row["fence"])
                cursor.execute(
                    "UPDATE session_runtime_owners SET expires_at=%s, updated_at=%s WHERE namespace=%s AND session_id=%s AND fence=%s",
                    (expires_at, now, namespace, session_id, fence),
                )
            elif float(row["expires_at"]) > now:
                return None
            else:
                fence = int(row["fence"]) + 1
                cursor.execute(
                    "UPDATE session_runtime_owners SET installation_id=%s, host=%s, process_generation=%s, fence=%s, expires_at=%s, updated_at=%s "
                    "WHERE namespace=%s AND session_id=%s AND fence=%s",
                    (installation_id, host, process_generation, fence, expires_at, now, namespace, session_id, int(row["fence"])),
                )
                cursor.execute(
                    "UPDATE session_runtime_turns SET state='indeterminate', updated_at=%s "
                    "WHERE namespace=%s AND session_id=%s AND state='running' AND owner_fence < %s",
                    (now, namespace, session_id, fence),
                )
        return RuntimeOwnershipReceipt(namespace, session_id, owner, fence, expires_at)

    def renew_session_runtime_ownership(self, receipt: RuntimeOwnershipReceipt, *, ttl_seconds: float = 300.0) -> RuntimeOwnershipReceipt | None:
        installation_id, host, process_generation = self._runtime_owner_columns(receipt.owner)
        ttl = self._runtime_ttl_seconds(ttl_seconds)
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._ownership_lock_and_clock(cursor, receipt.namespace, receipt.session_id)
            expires_at = now + ttl
            cursor.execute(
                "UPDATE session_runtime_owners SET expires_at=%s, updated_at=%s WHERE namespace=%s AND session_id=%s "
                "AND installation_id=%s AND host=%s AND process_generation=%s AND fence=%s AND expires_at > %s",
                (expires_at, now, receipt.namespace, receipt.session_id, installation_id, host, process_generation, receipt.fence, now),
            )
            if cursor.rowcount != 1:
                return None
        return RuntimeOwnershipReceipt(receipt.namespace, receipt.session_id, receipt.owner, receipt.fence, expires_at)

    def release_session_runtime_ownership(self, receipt: RuntimeOwnershipReceipt) -> bool:
        installation_id, host, process_generation = self._runtime_owner_columns(receipt.owner)
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._ownership_lock_and_clock(cursor, receipt.namespace, receipt.session_id)
            # Keep the row as a durable fence tombstone; a later owner can never reuse a fence.
            cursor.execute(
                "UPDATE session_runtime_owners SET expires_at=%s, updated_at=%s WHERE namespace=%s AND session_id=%s "
                "AND installation_id=%s AND host=%s AND process_generation=%s AND fence=%s",
                (now, now, receipt.namespace, receipt.session_id, installation_id, host, process_generation, receipt.fence),
            )
            return cursor.rowcount == 1

    def begin_session_runtime_turn(self, receipt: RuntimeOwnershipReceipt, turn_id: str) -> bool:
        if not turn_id:
            return False
        self._runtime_owner_columns(receipt.owner)
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._ownership_lock_and_clock(cursor, receipt.namespace, receipt.session_id)
            cursor.execute("SELECT installation_id, host, process_generation, fence, expires_at FROM session_runtime_owners "
                           "WHERE namespace=%s AND session_id=%s FOR UPDATE", (receipt.namespace, receipt.session_id))
            raw_owner = cursor.fetchone()
            owner = None if raw_owner is None else dict(zip(
                ("installation_id", "host", "process_generation", "fence", "expires_at"), raw_owner))
            if owner is None or not self._receipt_matches(owner, receipt, now):
                return False
            cursor.execute("SELECT state, owner_fence FROM session_runtime_turns WHERE namespace=%s AND session_id=%s AND turn_id=%s FOR UPDATE",
                           (receipt.namespace, receipt.session_id, turn_id))
            raw_row = cursor.fetchone()
            row = None if raw_row is None else dict(zip(("state", "owner_fence"), raw_row))
            if row is not None:
                return row["state"] == "running" and int(row["owner_fence"]) == receipt.fence
            cursor.execute(
                "INSERT INTO session_runtime_turns (namespace, session_id, turn_id, state, owner_fence, receipt_json, created_at, updated_at) "
                "VALUES (%s, %s, %s, 'running', %s, NULL, %s, %s)",
                (receipt.namespace, receipt.session_id, turn_id, receipt.fence, now, now),
            )
            return True

    def resolve_session_runtime_turn(self, receipt: RuntimeOwnershipReceipt, turn_id: str, *, state: TurnState, receipt_data: dict | None = None) -> bool:
        if state not in {"settled", "indeterminate"} or not turn_id:
            return False
        if state == "settled" and receipt_data is None:
            raise ValueError("settled turn requires a verified receipt")
        self._runtime_owner_columns(receipt.owner)
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._ownership_lock_and_clock(cursor, receipt.namespace, receipt.session_id)
            cursor.execute("SELECT installation_id, host, process_generation, fence, expires_at FROM session_runtime_owners "
                           "WHERE namespace=%s AND session_id=%s FOR UPDATE", (receipt.namespace, receipt.session_id))
            raw_owner = cursor.fetchone()
            owner = None if raw_owner is None else dict(zip(
                ("installation_id", "host", "process_generation", "fence", "expires_at"), raw_owner))
            if owner is None or not self._receipt_matches(owner, receipt, now):
                return False
            cursor.execute("SELECT state, receipt_json FROM session_runtime_turns WHERE namespace=%s AND session_id=%s AND turn_id=%s FOR UPDATE",
                           (receipt.namespace, receipt.session_id, turn_id))
            raw_row = cursor.fetchone()
            row = None if raw_row is None else dict(zip(("state", "receipt_json"), raw_row))
            if row is None or row["state"] == "settled":
                return False
            payload = json.dumps(receipt_data, sort_keys=True) if receipt_data is not None else None
            cursor.execute("UPDATE session_runtime_turns SET state=%s, receipt_json=%s::jsonb, updated_at=%s "
                           "WHERE namespace=%s AND session_id=%s AND turn_id=%s AND state IN ('running', 'indeterminate')",
                           (state, payload, now, receipt.namespace, receipt.session_id, turn_id))
            return cursor.rowcount == 1

    @contextlib.contextmanager
    def _connection(self) -> Iterator[Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL State Store is closed")
            try:
                connection = self._idle.get_nowait()
            except queue.Empty:
                if self._created < self._settings.pool_max_size:
                    self._created += 1
                    connection = self._new_connection()
                else:
                    connection = None
        if connection is None:
            connection = self._idle.get()
        try:
            self._set_connection_search_path(connection)
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            with self._lock:
                if self._closed:
                    connection.close()
                else:
                    self._idle.put(connection)

    def _search_maintenance_lock_name(self) -> str:
        return f"{self._schema}:search-index-maintenance"

    def _search_index_catalog(self, cursor: Any) -> tuple[str, str]:
        """Return generated-document and GIN catalog health without trusting an index name alone."""
        self._validate_core_v1_v25_catalog(cursor)
        cursor.execute(
            "SELECT i.indisvalid, i.indisready, i.indislive, am.amname, "
            "array_agg(a.attname ORDER BY key.ordinality) "
            "FROM pg_class AS c JOIN pg_index AS i ON i.indexrelid = c.oid "
            "JOIN pg_am AS am ON am.oid = c.relam "
            "JOIN unnest(i.indkey) WITH ORDINALITY AS key(attnum, ordinality) ON true "
            "JOIN pg_attribute AS a ON a.attrelid = i.indrelid AND a.attnum = key.attnum "
            "WHERE c.relnamespace = %s::regnamespace AND c.relname = %s "
            "GROUP BY i.indisvalid, i.indisready, i.indislive, am.amname",
            (self._schema, _SEARCH_INDEX_NAME),
        )
        row = cursor.fetchone()
        if row is None:
            return "valid", "missing"
        valid, ready, live, access_method, columns = row
        if bool(valid) and bool(ready) and bool(live) and access_method == "gin" and list(columns) == ["search_document"]:
            return "valid", "valid"
        return "valid", "invalid"

    def search_index_status(self) -> dict[str, Any]:
        """PostgreSQL generated-search health, deliberately unlike SQLite FTS rebuild progress.

        Canonical rows synchronously derive ``search_document``; PostgreSQL therefore has no
        detached-corruption fallback, deferred high-water backfill, retry, or quarantine state.
        A missing/invalid GIN catalog entry makes contextual routing unavailable rather than
        silently claiming SQLite's canonical-LIKE fallback semantics.
        """
        with self._connection() as connection, connection.cursor() as cursor:
            document, gin_index = self._search_index_catalog(cursor)
            cursor.execute(
                f"SELECT last_success_at, last_error FROM {self._schema}.search_index_maintenance WHERE singleton"
            )
            maintenance = cursor.fetchone() or (None, None)
            cursor.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (self._search_maintenance_lock_name(),))
            acquired = bool(cursor.fetchone()[0])
            if acquired:
                cursor.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (self._search_maintenance_lock_name(),))
        available = document == "valid" and gin_index == "valid"
        return {
            "backend": "postgresql",
            "available": available,
            "query_path_available": available,
            "generated_document": document,
            "gin_index": gin_index,
            "rebuild": {"supported": False, "operation": "alembic_only", "in_progress": not acquired},
            "last_successful_rebuild_at": maintenance[0],
            "last_error": maintenance[1],
            "sqlite_fts_semantics": {
                "corruption_detach": False, "canonical_like_fallback": False,
                "deferred_backfill": False, "high_water": False, "retry_quarantine": False,
            },
        }

    def rebuild_search_index(self) -> dict[str, Any]:
        """Refuse runtime search-index repair; Alembic owns this catalog object.

        This compatibility method performs a read-only health observation only.
        A healthy tenant receives an explicit unsupported result; a drifted
        tenant is rejected by the same catalog validation path used at startup.
        It never mutates a search index at runtime.
        """
        try:
            result = self.search_index_status()
        except Exception as exc:
            raise StateStoreConfigurationError(
                "PostgreSQL State Store search catalog drift requires Alembic-managed reinitialization; "
                "runtime index repair is disabled"
            ) from exc
        if not result["available"]:
            raise StateStoreConfigurationError(
                "PostgreSQL State Store search catalog drift requires Alembic-managed reinitialization; "
                "runtime index repair is disabled"
            )
        result["rebuild"] = {**result["rebuild"], "operation": "unsupported_alembic_only"}
        return result

    def ensure_session(
        self, session_id: str, source: str = "unknown", *, metadata: Mapping[str, Any] | None = None,
    ) -> str:
        metadata = dict(metadata or {})
        unknown = set(metadata) - set(_SESSION_METADATA_COLUMNS)
        if unknown:
            raise ValueError(f"Unsupported session metadata fields: {', '.join(sorted(unknown))}")
        metadata_columns = tuple(column for column in _SESSION_METADATA_COLUMNS if column in metadata)
        columns = ("id", "source", "started_at", *metadata_columns)
        placeholders = ", ".join("%s" for _ in columns)
        values = [session_id, source, time.time()]
        for column in metadata_columns:
            value = metadata[column]
            if column == "model_config":
                value = self._psycopg.types.json.Jsonb(value) if value else None
            values.append(value)
        with self._connection() as connection, connection.cursor() as cursor:
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
            cursor.execute(
                f"UPDATE {self._schema}.sessions AS child SET "
                "cwd = COALESCE(child.cwd, parent.cwd), "
                "git_branch = COALESCE(child.git_branch, parent.git_branch), "
                "git_repo_root = COALESCE(child.git_repo_root, parent.git_repo_root) "
                f"FROM {self._schema}.sessions AS parent "
                "WHERE child.id = %s AND child.parent_session_id IS NOT NULL AND parent.id = child.parent_session_id",
                (session_id,),
            )
        return session_id

    @staticmethod
    def _foreign_import_fingerprint(origin: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        tool, path, foreign_id = origin.get("tool"), origin.get("path"), origin.get("foreign_session_id")
        if not isinstance(tool, str) or not tool.strip() or not isinstance(path, str) or not path.strip():
            raise ValueError("foreign import origin requires non-empty tool and path")
        if foreign_id is not None and (not isinstance(foreign_id, str) or not foreign_id.strip()):
            raise ValueError("foreign import foreign_session_id must be a non-empty string when present")
        canonical = {"tool": tool.strip(), "path": path, "foreign_session_id": foreign_id.strip() if isinstance(foreign_id, str) else None}
        identity = ({"tool": canonical["tool"], "foreign_session_id": canonical["foreign_session_id"]}
                    if canonical["foreign_session_id"] else {"tool": canonical["tool"], "path": canonical["path"]})
        return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(), canonical

    def import_foreign_history(self, origin: Mapping[str, Any], messages: list[Mapping[str, Any]], *, title: str,
                               cwd: str | None, profile: str | None) -> dict[str, Any]:
        """Atomically import one normalized foreign transcript into this tenant.

        Validation completes before a connection opens. The receipt makes a lost
        acknowledgement retry-safe; imported rows deliberately carry no live
        routing, activity, lease, or ownership state.
        """
        if not isinstance(origin, Mapping) or not isinstance(messages, list):
            raise ValueError("foreign import requires an origin and a messages list")
        fingerprint, canonical_origin = self._foreign_import_fingerprint(origin)
        if not isinstance(title, str) or not (clean_title := _sanitize_title(title)):
            raise ValueError("foreign import requires a non-empty title")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError("foreign import cwd must be a string or null")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("foreign import profile must be a string or null")
        normalized: list[tuple[str, str]] = []
        previous_role = None
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("foreign import messages must be objects")
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError("foreign import messages require non-empty user or assistant content")
            if role == previous_role:
                raise ValueError("foreign import messages must alternate roles")
            normalized.append((role, content))
            previous_role = role
        if not normalized or normalized[0][0] != "user":
            raise ValueError("foreign import must begin with a user message")
        now = time.time()
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{self._schema}:foreign-import:{fingerprint}",))
            cursor.execute(f"SELECT session_id FROM {self._schema}.foreign_import_receipts WHERE origin_fingerprint=%s", (fingerprint,))
            receipt = cursor.fetchone()
            if receipt is not None:
                return {"session_id": str(receipt["session_id"]), "already_imported": True}
            session_id = new_session_id(hex_len=12)
            stored_title = clean_title
            cursor.execute(f"SELECT 1 FROM {self._schema}.sessions WHERE title=%s FOR UPDATE", (stored_title,))
            if cursor.fetchone() is not None:
                stored_title = _sanitize_title(f"{clean_title[:84]} ({session_id[-12:]})")
            cursor.execute(
                f"INSERT INTO {self._schema}.sessions (id, source, started_at, title, title_source, cwd, profile_name, origin_json, last_activity_at) "
                "VALUES (%s, %s, %s, %s, 'user', %s, %s, %s, NULL)",
                (session_id, canonical_origin["tool"], now, stored_title, cwd, profile, json.dumps({"imported_from": canonical_origin})),
            )
            for role, content in normalized:
                cursor.execute(f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)", (session_id, role, content, now))
            cursor.execute(f"INSERT INTO {self._schema}.foreign_import_receipts (origin_fingerprint, session_id, origin_json, committed_at) VALUES (%s, %s, %s, %s)",
                           (fingerprint, session_id, self._psycopg.types.json.Jsonb(canonical_origin), now))
            return {"session_id": session_id, "already_imported": False}

    @staticmethod
    def _encode_content(content: Any) -> Any:
        if isinstance(content, str) or content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            return "__hermes_state_json__:" + json.dumps(content)
        except (TypeError, ValueError):
            return str(content)

    @staticmethod
    def _record_json(value: Any, *, object_only: bool = False) -> Any:
        if not value:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return None if object_only else []
        if object_only and not isinstance(value, dict):
            return None
        return value

    @staticmethod
    def _record_json_text(value: Any) -> str | None:
        return None if not value else (value if isinstance(value, str) else json.dumps(value))

    def _paired_tool_call_uid(self, cursor: Any, session_id: str, tool_call_id: str | None) -> str | None:
        """Pair only with a named, active assistant call after the last user turn.

        A nearer assistant naming the same provider id but lacking a durable UID
        shadows earlier occurrences. Never guess an identity from the provider id.
        """
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None
        variants = set(tool_result_id_variants(tool_call_id))
        cursor.execute(
            f"SELECT tool_calls, tool_call_uids FROM {self._schema}.messages "
            "WHERE session_id=%s AND active AND role='assistant' AND id > COALESCE(("
            f"SELECT MAX(id) FROM {self._schema}.messages WHERE session_id=%s AND active AND role='user'"
            "), 0) ORDER BY id DESC",
            (session_id, session_id),
        )
        for row in cursor.fetchall():
            calls = row["tool_calls"] if isinstance(row, Mapping) else row[0]
            stored_uids = row["tool_call_uids"] if isinstance(row, Mapping) else row[1]
            if not isinstance(calls, list):
                continue
            index: dict[str, str] = {}
            named = index_tool_call_uids(index, {"tool_calls": calls, TOOL_CALL_UIDS: _tool_call_uid_map({"tool_call_uids": stored_uids})})
            if not variants.isdisjoint(named):
                return resolve_tool_call_uid(index, tool_call_id)
        return None

    def _record_params(self, session_id: str, record: MessageRecord, *, cursor: Any) -> tuple[Any, ...]:
        timestamp = coerce_epoch(record.timestamp, field="message timestamp")
        tool_calls = self._record_json(record.tool_calls)
        display_metadata = self._record_json(record.display_metadata, object_only=True)
        identity = {name: getattr(record, name) for name in (
            "message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid")}
        uid = message_uid_or_none(identity) or mint_uid()
        if record.role == "assistant" and isinstance(tool_calls, list):
            uids = _tool_call_uid_map(identity)
            for call in tool_calls:
                call_id = coalesce_tool_call_id(call)
                if call_id and call_id not in uids:
                    uids[call_id] = mint_uid()
            identity["tool_call_uids"] = uids
        if record.role == "tool" and _tool_call_uid_or_none(identity) is None:
            identity["tool_call_uid"] = self._paired_tool_call_uid(cursor, session_id, record.tool_call_id)
        return (
            session_id, record.role, self._encode_content(record.content),
            timestamp if timestamp is not None else time.time(), record.tool_call_id,
            self._psycopg.types.json.Jsonb(tool_calls) if tool_calls else None, record.tool_name,
            record.effect_disposition, record.token_count,
            record.finish_reason, record.reasoning, record.reasoning_content,
            self._record_json_text(record.reasoning_details), self._record_json_text(record.codex_reasoning_items),
            self._record_json_text(record.codex_message_items), record.platform_message_id, bool(record.observed),
            bool(record._compressed_summary), record.api_content, record.display_kind,
            self._psycopg.types.json.Jsonb(display_metadata) if display_metadata else None,
            uid, _absorbed_uids_json(identity), _tool_call_uids_json(identity), _tool_call_uid_or_none(identity),
        )

    def append_message(self, session_id: str, *, role: str, content: str | None = None) -> int:
        return self.append_message_record(session_id, MessageRecord(role=role, content=content))

    def append_message_record(self, session_id: str, record: MessageRecord, *,
                              turn_lease_holder: str | None = None,
                              turn_lease_ttl_seconds: float = 300.0,
                              reject_active_turn_lease: bool = False) -> int:
        """Append one record row, optionally fenced by the turn-lease guards.

        Mirrors the SQLite oracle's ``append_message`` guard kwargs: a writer
        holding a turn lease passes ``turn_lease_holder`` (renewed on expiry,
        refused when lost); a destructive mutation passes
        ``reject_active_turn_lease`` to refuse while any active lease exists.
        """
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            self._transcript_write_guards(cursor, session_id,
                turn_lease_holder=turn_lease_holder,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
                reject_active_turn_lease=reject_active_turn_lease)
            cursor.execute(
                f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) "
                f"VALUES ({_MESSAGE_INSERT_PLACEHOLDERS}) RETURNING id, created_at",
                self._record_params(session_id, record, cursor=cursor),
            )
            row = cursor.fetchone()
            message_id = row["id"] if isinstance(row, Mapping) else row[0]
            created_at = row["created_at"] if isinstance(row, Mapping) else row[1]
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET last_activity_at = GREATEST("
                "COALESCE(last_activity_at, started_at), %s) WHERE id = %s",
                (created_at, session_id),
            )
            return int(message_id)

    def append_message_records(self, session_id: str, records: list[MessageRecord]) -> int:
        if not records:
            return 0
        with self._connection() as connection, connection.cursor() as cursor:
            for record in records:
                cursor.execute(
                    f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) "
                    f"VALUES ({_MESSAGE_INSERT_PLACEHOLDERS}) RETURNING created_at",
                    self._record_params(session_id, record, cursor=cursor),
                )
                created_at = cursor.fetchone()[0]
                cursor.execute(
                    f"UPDATE {self._schema}.sessions SET last_activity_at = GREATEST("
                    "COALESCE(last_activity_at, started_at), %s) WHERE id = %s",
                    (created_at, session_id),
                )
        return len(records)

    def branch_session(
        self, *, parent_session_id: str, child_session_id: str, source: str,
        model: str | None, model_config: Mapping[str, Any], title: str,
    ) -> str:
        """Atomically publish an interactive branch from one live parent.

        This is deliberately distinct from compression rotation: a branch owns a
        complete physical copy of the parent's active transcript and remains a
        separately visible session.  Holding the parent row lock fences appenders
        through the message FK while the watermark and copy are taken, so a
        failure cannot leave either a closed parent or a partial child.
        """
        if not parent_session_id or not child_session_id:
            raise ValueError("branch requires parent and child session ids")
        if parent_session_id == child_session_id:
            raise ValueError("branch child must differ from parent")
        if not title:
            raise ValueError("branch requires a title")
        metadata = dict(model_config)
        metadata["_branched_from"] = parent_session_id
        message_columns = (
            "session_id", "role", "content", "created_at", *_MESSAGE_RECORD_WRITE_COLUMNS,
            "active", "compacted",
        )
        source_columns = ("role", "content", "created_at", *_MESSAGE_RECORD_WRITE_COLUMNS, "active", "compacted")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            now = self._coordination_clock_and_lock(cursor, "interactive_branch", parent_session_id)
            cursor.execute(
                f"SELECT id FROM {self._schema}.sessions WHERE id=%s AND ended_at IS NULL FOR UPDATE",
                (parent_session_id,),
            )
            if cursor.fetchone() is None:
                raise ValueError("branch parent is missing or no longer active")
            cursor.execute(
                f"SELECT holder FROM {self._schema}.session_turn_leases "
                "WHERE conversation_id=%s AND expires_at>%s FOR UPDATE",
                (parent_session_id, now),
            )
            lease = cursor.fetchone()
            if lease is not None:
                raise RuntimeError("branch parent has an active turn lease")
            cursor.execute(
                f"SELECT COALESCE(MAX(id), 0) AS watermark FROM {self._schema}.messages "
                "WHERE session_id=%s AND active",
                (parent_session_id,),
            )
            watermark = int(cursor.fetchone()["watermark"])
            cursor.execute(
                f"INSERT INTO {self._schema}.sessions "
                "(id, source, started_at, model, model_config, parent_session_id, title, title_source, last_activity_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'user', %s)",
                (child_session_id, source, now, model, self._psycopg.types.json.Jsonb(metadata),
                 parent_session_id, _sanitize_title(title), now),
            )
            cursor.execute(
                f"INSERT INTO {self._schema}.messages ({', '.join(message_columns)}) "
                f"SELECT %s, {', '.join(source_columns)} FROM {self._schema}.messages "
                "WHERE session_id=%s AND active AND id<=%s ORDER BY id",
                (child_session_id, parent_session_id, watermark),
            )
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET ended_at=%s, end_reason='branched' "
                "WHERE id=%s AND ended_at IS NULL",
                (now, parent_session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("branch parent changed while publishing")
        return child_session_id

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Explicitly advertised durable capabilities; never infer from methods."""
        return ("atomic-compression-rotation-v1", "transcript-rewind-v1")

    def get_compression_publication_receipt(self, request_id: str) -> dict[str, Any] | None:
        """Return the committed publication for one caller-owned request id.

        This is the only recovery read for an indeterminate publish acknowledgement:
        an absent row means the caller must not guess whether to replay, while a
        present row is the durable parent/child fact that may be safely adopted.
        """
        if not request_id:
            raise ValueError("compression publication receipt requires a request id")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT request_id, parent_session_id, child_session_id, holder, fence, committed_at "
                f"FROM {self._schema}.compression_rotation_receipts WHERE request_id=%s",
                (request_id,),
            )
            receipt = cursor.fetchone()
            return None if receipt is None else dict(receipt)

    def get_active_message_watermark(self, session_id: str) -> int:
        """Tenant-local high-water mark used to preserve a concurrent parent tail."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT COALESCE(MAX(id), 0) FROM {self._schema}.messages WHERE session_id=%s AND active", (session_id,))
            return int(cursor.fetchone()[0])

    def publish_compression_child(
        self, *, parent_session_id: str, child_session_id: str, source: str,
        messages: list[dict[str, Any]], model: str | None = None, model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None, cwd: str | None = None, profile_name: str | None = None,
        compression_lock_holder: str | None = None, require_compression_lease: bool = True,
        require_lease_refresh: bool = False, lease_ttl_seconds: float = 300.0,
        watermark: int | None = None, watermark_ceiling: int | None = None,
        request_id: str | None = None,
    ) -> str:
        """Commit one fenced parent→child handoff or leave no partial mutation.

        Receipt lookup happens under the same parent advisory lock as lease
        validation.  A caller may safely inspect a returned receipt after a lost
        acknowledgement, but this method deliberately never retries an unknown
        request on its own.
        """
        if not parent_session_id or not child_session_id or not messages:
            raise ValueError("compression publication requires parent, child, and non-empty handoff")
        receipt_id = request_id or child_session_id
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            now = self._coordination_clock_and_lock(cursor, "compression_locks", parent_session_id)
            cursor.execute(f"SELECT parent_session_id, child_session_id FROM {self._schema}.compression_rotation_receipts WHERE request_id=%s FOR UPDATE", (receipt_id,))
            receipt = cursor.fetchone()
            if receipt is not None:
                if receipt["parent_session_id"] != parent_session_id or receipt["child_session_id"] != child_session_id:
                    raise RuntimeError("compression receipt request id was reused for another publication")
                return str(receipt["child_session_id"])
            cursor.execute(f"SELECT holder, fence, expires_at FROM {self._schema}.compression_locks WHERE session_id=%s FOR UPDATE", (parent_session_id,))
            lease = cursor.fetchone()
            if require_compression_lease:
                if not compression_lock_holder or lease is None or lease["holder"] != compression_lock_holder:
                    raise RuntimeError(f"Compression lease lost before publication: {parent_session_id}")
                if require_lease_refresh:
                    cursor.execute(f"UPDATE {self._schema}.compression_locks SET expires_at=%s, updated_at=%s WHERE session_id=%s AND holder=%s AND fence=%s", (now + self._coordination_ttl(lease_ttl_seconds), now, parent_session_id, compression_lock_holder, lease["fence"]))
                    if cursor.rowcount != 1:
                        raise RuntimeError(f"Compression lease refresh lost before publication: {parent_session_id}")
                elif float(lease["expires_at"]) <= now:
                    raise RuntimeError(f"Compression lease expired before publication: {parent_session_id}")
            fence = int(lease["fence"]) if lease is not None else 1
            cursor.execute(f"SELECT * FROM {self._schema}.sessions WHERE id=%s FOR UPDATE", (parent_session_id,))
            parent = cursor.fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                raise RuntimeError(f"Compression parent already ended: {parent_session_id}")
            # ``sessions`` keeps only the prompt hash; callers publish the live
            # cached prompt.  A missing prompt remains intentionally absent.
            prompt = system_prompt
            prompt_hash = None if prompt is None else hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_hash is not None:
                cursor.execute(f"INSERT INTO {self._schema}.system_prompts (hash, prompt) VALUES (%s, %s) ON CONFLICT (hash) DO NOTHING", (prompt_hash, prompt))
            # The title index is intentionally unique. Move the title with the
            # lineage boundary rather than duplicating it on the closed parent.
            cursor.execute(f"UPDATE {self._schema}.sessions SET ended_at=%s, end_reason='compression', title=NULL, title_source=NULL WHERE id=%s AND ended_at IS NULL", (now, parent_session_id))
            if cursor.rowcount != 1:
                raise RuntimeError(f"Compression parent changed during publication: {parent_session_id}")
            columns = ("id", "source", "started_at", "model", "model_config", "system_prompt_hash", "parent_session_id", "cwd", "git_branch", "git_repo_root", "profile_name", "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json", "title", "title_source", "hidden", "archived", "pinned", "last_activity_at")
            values = (child_session_id, source or parent["source"], now, model or parent["model"], self._psycopg.types.json.Jsonb(model_config) if model_config else parent["model_config"], prompt_hash, parent_session_id, cwd or parent["cwd"], parent["git_branch"], parent["git_repo_root"], profile_name or parent["profile_name"], parent["user_id"], parent["session_key"], parent["chat_id"], parent["chat_type"], parent["thread_id"], parent["display_name"], parent["origin_json"], parent["title"], parent["title_source"], parent["hidden"], parent["archived"], parent["pinned"], now)
            cursor.execute(f"INSERT INTO {self._schema}.sessions ({', '.join(columns)}) VALUES ({', '.join('%s' for _ in columns)})", values)
            for message in messages:
                record = self._replace_record_from_dict(message)
                cursor.execute(f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) VALUES ({_MESSAGE_INSERT_PLACEHOLDERS})", self._record_params(child_session_id, record, cursor=cursor))
            if watermark is not None:
                upper = int(watermark_ceiling) if watermark_ceiling is not None else 9223372036854775807
                cursor.execute(f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_COLUMNS)}) SELECT %s, role, content, created_at, {', '.join(_MESSAGE_RECORD_COLUMNS)} FROM {self._schema}.messages WHERE session_id=%s AND active AND id > %s AND id <= %s ORDER BY id", (child_session_id, parent_session_id, int(watermark), upper))
            # The child is being published here, so control state must cross the
            # lineage boundary in this same transaction.  Moving it afterwards
            # could leave a durable child without its active /goal,/heartbeat or
            # /loop controls if the process dies between commits.
            self._transfer_session_control_states_in_transaction(
                cursor, parent_session_id, child_session_id, now, require_empty_child=True,
            )
            cursor.execute(f"INSERT INTO {self._schema}.compression_rotation_receipts (request_id, parent_session_id, child_session_id, holder, fence, committed_at) VALUES (%s, %s, %s, %s, %s, %s)", (receipt_id, parent_session_id, child_session_id, compression_lock_holder or "unleased", fence, now))
        return child_session_id

    @staticmethod
    def _coordination_ttl(ttl_seconds: float) -> float:
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl):
            raise ValueError("compression coordination ttl_seconds must be finite")
        return max(0.1, ttl)

    def _coordination_clock_and_lock(self, cursor: Any, kind: str, key: str) -> float:
        """Serialize an extant or absent lease key and use PostgreSQL's clock.

        Row locks alone cannot protect a missing lease row.  The transaction
        advisory lock is scoped by the trusted tenant schema so two profiles
        never coordinate accidentally.
        """
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{self._schema}:{kind}:{key}",))
        cursor.execute("SELECT EXTRACT(EPOCH FROM clock_timestamp()) AS server_now")
        row = cursor.fetchone()
        return float(row["server_now"] if isinstance(row, Mapping) else row[0])

    def touch_session_activity(self, session_id: str, ts: float | None = None, *, description: str | None = None,
                               provenance: Any = None) -> None:
        """Monotonically publish the current activity observation and its labels."""
        if not session_id:
            return
        when = float(ts if ts is not None else time.time())
        label = bound_activity_description(description)
        source = normalize_activity_provenance(provenance).value
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET "
                "last_activity_at = GREATEST(COALESCE(last_activity_at, started_at), %s), "
                "last_activity_description = CASE WHEN last_activity_at IS NULL OR last_activity_at <= %s THEN %s ELSE last_activity_description END, "
                "last_activity_provenance = CASE WHEN last_activity_at IS NULL OR last_activity_at <= %s THEN %s ELSE last_activity_provenance END "
                "WHERE id = %s",
                (when, when, label, when, source, session_id),
            )

    def clear_session_activity_labels(self, session_id: str) -> None:
        """Clear only transient labels; keep the durable last-activity clock."""
        if not session_id:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET last_activity_description='', last_activity_provenance='unknown' "
                "WHERE id=%s AND (last_activity_description <> '' OR last_activity_provenance <> 'unknown')",
                (session_id,),
            )

    def record_compression_failure_cooldown(self, session_id: str, cooldown_until: float, error: str | None = None) -> None:
        """Merge-max a durable retry deadline; a later short failure cannot reopen it."""
        if not session_id:
            return
        deadline = float(cooldown_until)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.sessions SET compression_failure_cooldown_until=GREATEST("
                "COALESCE(compression_failure_cooldown_until, '-Infinity'::float8), %s), "
                "compression_failure_error=%s WHERE id=%s",
                (deadline, error, session_id),
            )

    def get_compression_failure_cooldown(self, session_id: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT compression_failure_cooldown_until, compression_failure_error, "
                "EXTRACT(EPOCH FROM clock_timestamp()) FROM " + f"{self._schema}.sessions WHERE id=%s",
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None or row[0] is None or float(row[0]) <= float(row[2]):
            return None
        return {"cooldown_until": float(row[0]), "remaining_seconds": float(row[0]) - float(row[2]), "error": row[1]}

    def get_compression_failure_cooldown_row(self, session_id: str) -> dict[str, Any]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT compression_failure_cooldown_until, compression_failure_error FROM {self._schema}.sessions WHERE id=%s", (session_id,))
            row = cursor.fetchone()
        return {"session_exists": row is not None, "cooldown_until": None if row is None else row[0], "error": None if row is None else row[1]}

    def restore_compression_failure_cooldown_row(self, session_id: str, snapshot: Mapping[str, Any]) -> None:
        if not snapshot.get("session_exists", False):
            if self.get_compression_failure_cooldown_row(session_id)["session_exists"]:
                raise RuntimeError("cannot restore absent compression cooldown row: session now exists")
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"UPDATE {self._schema}.sessions SET compression_failure_cooldown_until=%s, compression_failure_error=%s WHERE id=%s", (snapshot.get("cooldown_until"), snapshot.get("error"), session_id))
            if cursor.rowcount != 1:
                return
        if self.get_compression_failure_cooldown_row(session_id) != {"session_exists": True, "cooldown_until": snapshot.get("cooldown_until"), "error": snapshot.get("error")}:
            raise RuntimeError("compression cooldown rollback verification failed")

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        if not session_id:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"UPDATE {self._schema}.sessions SET compression_failure_cooldown_until=NULL, compression_failure_error=NULL WHERE id=%s", (session_id,))

    def _compression_counter(self, session_id: str, column: str, *, decimal: bool = False) -> int | float:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT {column} FROM {self._schema}.sessions WHERE id=%s", (session_id,))
            row = cursor.fetchone()
        value = 0.0 if row is None or row[0] is None else float(row[0])
        return max(0.0, value) if decimal else max(0, int(value))

    def _set_compression_counter(self, session_id: str, column: str, value: int | float, *, decimal: bool = False) -> None:
        if not session_id:
            return
        normalized = max(0.0, float(value or 0)) if decimal else max(0, int(value or 0))
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"UPDATE {self._schema}.sessions SET {column}=%s WHERE id=%s", (normalized or None if decimal else normalized, session_id))

    def get_compression_fallback_streak(self, session_id: str) -> int: return int(self._compression_counter(session_id, "compression_fallback_streak"))
    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None: self._set_compression_counter(session_id, "compression_fallback_streak", streak)
    def get_compression_ineffective_count(self, session_id: str) -> int: return int(self._compression_counter(session_id, "compression_ineffective_count"))
    def set_compression_ineffective_count(self, session_id: str, count: int) -> None: self._set_compression_counter(session_id, "compression_ineffective_count", count)
    def get_compression_recovery_deadline(self, session_id: str) -> float: return float(self._compression_counter(session_id, "compression_recovery_deadline", decimal=True))
    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None: self._set_compression_counter(session_id, "compression_recovery_deadline", deadline, decimal=True)

    def _try_coordination_lease(self, cursor: Any, *, table: str, key_column: str, key: str,
                                holder: str, ttl_seconds: float) -> bool:
        if not key or not holder:
            return False
        now = self._coordination_clock_and_lock(cursor, table, key)
        cursor.execute(f"SELECT holder, fence, expires_at FROM {self._schema}.{table} WHERE {key_column}=%s FOR UPDATE", (key,))
        row = cursor.fetchone()
        expires_at = now + self._coordination_ttl(ttl_seconds)
        if row is None:
            cursor.execute(f"INSERT INTO {self._schema}.{table} ({key_column}, holder, fence, expires_at, updated_at) VALUES (%s, %s, 1, %s, %s)", (key, holder, expires_at, now))
            return True
        current_holder, fence, old_expiry = str(row[0]), int(row[1]), float(row[2])
        if current_holder != holder and old_expiry > now:
            return False
        next_fence = fence if current_holder == holder else fence + 1
        cursor.execute(f"UPDATE {self._schema}.{table} SET holder=%s, fence=%s, expires_at=%s, updated_at=%s WHERE {key_column}=%s AND fence=%s", (holder, next_fence, expires_at, now, key, fence))
        return cursor.rowcount == 1

    def _compression_turn_lease_key_on_cursor(self, cursor: Any, session_id: str) -> str:
        """Resolve a compression lineage root in the same lease transaction."""
        current, seen = session_id, {session_id}
        while current:
            cursor.execute(f"SELECT parent_session_id, end_reason, model_config FROM {self._schema}.sessions WHERE id=%s", (current,))
            row = cursor.fetchone()
            if row is None or (row["parent_session_id"] if isinstance(row, Mapping) else row[0]) is None:
                return current
            parent_id = str(row["parent_session_id"] if isinstance(row, Mapping) else row[0])
            config = row["model_config"] if isinstance(row, Mapping) else row[2]
            if parent_id in seen:
                return current
            cursor.execute(f"SELECT end_reason FROM {self._schema}.sessions WHERE id=%s", (parent_id,))
            parent = cursor.fetchone()
            if parent is None or (parent["end_reason"] if isinstance(parent, Mapping) else parent[0]) != "compression" or self._is_explicit_branch({"model_config": config}):
                return current
            seen.add(parent_id)
            current = parent_id
        return session_id

    def _refuse_closed_compression_parent(self, cursor: Any, session_id: str) -> None:
        """Refuse a transcript write against a parent already closed by compression."""
        from hermes_state_errors import CompressionSessionClosedError
        cursor.execute(f"SELECT ended_at, end_reason FROM {self._schema}.sessions WHERE id=%s", (session_id,))
        row = cursor.fetchone()
        if row is not None:
            ended_at = row["ended_at"] if isinstance(row, Mapping) else row[0]
            end_reason = row["end_reason"] if isinstance(row, Mapping) else row[1]
            if ended_at is not None and end_reason == "compression":
                raise CompressionSessionClosedError(session_id)

    def _transcript_write_guards(self, cursor: Any, session_id: str, *, turn_lease_holder: str | None = None,
                                 turn_lease_ttl_seconds: float = 300.0,
                                 reject_active_turn_lease: bool = False,
                                 reject_active_compression_lock: bool = False,
                                 compression_lock_holder: str | None = None) -> None:
        """PostgreSQL twin of the SQLite oracle's ``_check_transcript_write_guards``.

        Ordinary appends only refuse a compression-closed parent (the advisory
        lease machinery is untouched, exactly like SQLite, where plain appends
        never consult compression_locks). Destructive mutations opt in via the
        ``reject_active_*`` flags; a turn-lease holder renews on expiry and is
        refused the moment it no longer owns the lease. Stale-lease liveness is
        expiry-only here: cross-process PID forensics stay SQLite-local.
        """
        from hermes_state_errors import SessionCompressionInProgressError, SessionTurnLeaseLostError
        if not (turn_lease_holder or reject_active_turn_lease or reject_active_compression_lock):
            self._refuse_closed_compression_parent(cursor, session_id)
            return
        now = self._coordination_clock_and_lock(cursor, "transcript-write", session_id)
        if reject_active_compression_lock:
            cursor.execute(
                f"SELECT holder FROM {self._schema}.compression_locks WHERE session_id=%s AND expires_at>%s FOR UPDATE",
                (session_id, now),
            )
            lock = cursor.fetchone()
            if lock is not None:
                holder = lock["holder"] if isinstance(lock, Mapping) else lock[0]
                if holder != compression_lock_holder:
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer")
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._compression_turn_lease_key_on_cursor(cursor, session_id)
            cursor.execute(
                f"SELECT holder, expires_at FROM {self._schema}.session_turn_leases WHERE conversation_id=%s FOR UPDATE",
                (conversation_id,),
            )
            lease = cursor.fetchone()
            holder = None if lease is None else (lease["holder"] if isinstance(lease, Mapping) else lease[0])
            expires_at = None if lease is None else float(lease["expires_at"] if isinstance(lease, Mapping) else lease[1])
            if turn_lease_holder:
                if lease is None or holder != turn_lease_holder:
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write for {session_id!r}")
                assert expires_at is not None
                if expires_at <= now:
                    # Expiry makes the row reclaimable, not lost: the advisory lock
                    # serializes this renewal with acquisition, so a starved owner
                    # that still holds the lease recovers (same rule as SQLite).
                    cursor.execute(
                        f"UPDATE {self._schema}.session_turn_leases SET expires_at=%s, updated_at=%s "
                        "WHERE conversation_id=%s AND holder=%s",
                        (now + self._coordination_ttl(turn_lease_ttl_seconds), now, conversation_id, turn_lease_holder))
            elif lease is not None:
                assert expires_at is not None
                if expires_at > now:
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript mutation for {session_id!r}")
                cursor.execute(
                    f"DELETE FROM {self._schema}.session_turn_leases WHERE conversation_id=%s AND holder=%s",
                    (conversation_id, holder))
        self._refuse_closed_compression_parent(cursor, session_id)

    @staticmethod
    def _replace_record_from_dict(msg: Mapping[str, Any]) -> MessageRecord:
        """Bind one replacement message dict to a :class:`MessageRecord`.

        Mirrors the oracle's ``_message_row_params`` binding rules:
        ``platform_message_id`` falls back to ``message_id`` (yuanbao's
        message-dict convention) and reasoning columns are kept only on
        assistant rows (``keep_reasoning`` semantics).
        """
        keep_reasoning = msg.get("role") == "assistant"
        fields = MessageRecord.__dataclass_fields__
        values = {name: msg[name] for name in fields if name in msg}
        for column, live_key in (("absorbed_message_uids", "_absorbed_message_uids"),
                                 ("tool_call_uids", TOOL_CALL_UIDS), ("tool_call_uid", TOOL_CALL_UID)):
            if live_key in msg:
                values[column] = msg[live_key]
        values.setdefault("role", "unknown")
        values["platform_message_id"] = msg.get("platform_message_id") or msg.get("message_id")
        values["observed"] = bool(msg.get("observed"))
        values["_compressed_summary"] = bool(msg.get("_compressed_summary"))
        if not keep_reasoning:
            for column in ("reasoning", "reasoning_content", "reasoning_details",
                           "codex_reasoning_items", "codex_message_items"):
                values[column] = None
        return MessageRecord(**values)

    def replace_messages(self, session_id: str, messages: list[dict[str, Any]], active_only: bool = False,
                         archive_dropped: bool = False, reject_active_turn_lease: bool = False) -> None:
        """Atomically replace a session's messages (/retry, /undo, /compress).

        Mirrors the SQLite oracle's ``replace_messages``: destructive DELETE by
        default (``active_only`` spares soft-archived rows); ``archive_dropped``
        soft-archives the live rows rewind-style instead; ``reject_active_turn_lease``
        refuses the rewrite while another writer holds an active lease (and also
        refuses an active compression lock, matching the oracle's guard pairing).
        """
        from hermes_state_errors import CompressionSessionClosedError
        records = [self._replace_record_from_dict(msg) for msg in messages]
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            if reject_active_turn_lease:
                self._transcript_write_guards(cursor, session_id,
                    reject_active_turn_lease=True, reject_active_compression_lock=True)
            else:
                self._refuse_closed_compression_parent(cursor, session_id)
            # Serialize concurrent rewriters on the session row before the drop.
            cursor.execute(f"SELECT id FROM {self._schema}.sessions WHERE id=%s FOR UPDATE", (session_id,))
            if cursor.fetchone() is None:
                raise CompressionSessionClosedError(session_id)
            if archive_dropped:
                cursor.execute(
                    f"UPDATE {self._schema}.messages SET active=false WHERE session_id=%s AND active",
                    (session_id,))
            else:
                cursor.execute(
                    f"DELETE FROM {self._schema}.messages WHERE session_id=%s{' AND active' if active_only else ''}",
                    (session_id,))
            for record in records:
                cursor.execute(
                    f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) "
                    f"VALUES ({_MESSAGE_INSERT_PLACEHOLDERS})",
                    self._record_params(session_id, record, cursor=cursor),
                )

    def get_messages_as_conversation(self, session_id: str, include_inactive: bool = False,
                                     repair_alternation: bool = False,
                                     include_row_ids: bool = False) -> list[dict[str, Any]]:
        """Load messages in OpenAI format, mirroring the SQLite oracle projection.

        Covers the gateway transcript read path: born-durable persistence marker,
        sanitized user/assistant text, verbatim ``api_content``, platform id
        exposed as ``message_id``, reasoning restored on assistant rows only.
        Ancestor lineage, display-generation dedupe, and alternation repair
        remain SessionDB-only and are deliberately not ported in this slice.
        """
        from hermes_state import _strip_background_review_harness, _strip_stale_tool_call_markers
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from agent.memory_manager import sanitize_context
        columns = (
            "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
            "finish_reason, reasoning, reasoning_content, reasoning_details, "
            "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
            "_compressed_summary, created_at AS timestamp, api_content, display_kind, display_metadata, "
            "message_uid, absorbed_message_uids, tool_call_uids, tool_call_uid"
        )
        active_clause = "" if include_inactive else " AND active"
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT {columns} FROM {self._schema}.messages WHERE session_id=%s{active_clause} ORDER BY id",
                (session_id,))
            rows = list(cursor.fetchall())
        messages: list[dict[str, Any]] = []
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg: dict[str, Any] = {"role": row["role"], "content": content, _DB_PERSISTED_MARKER: True}
            _restore_identity_columns(row, msg)
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = int(row["id"])
            msg.update((column, row[column]) for column in ("api_content", "display_kind") if row[column])
            if row["display_metadata"]:
                metadata = self._record_json(row["display_metadata"], object_only=True)
                if metadata is not None:
                    msg["display_metadata"] = metadata
            msg.update(
                (column, row[column]) for column in ("timestamp", "tool_call_id", "tool_name", "effect_disposition") if row[column])
            if row["tool_calls"]:
                tool_calls = self._record_json(row["tool_calls"])
                msg["tool_calls"] = tool_calls if isinstance(tool_calls, list) else []
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            if row["role"] == "assistant":
                msg.update((column, row[column]) for column in ("finish_reason", "reasoning") if row[column])
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                for column in ("reasoning_details", "codex_reasoning_items", "codex_message_items"):
                    if row[column]:
                        value = self._record_json(row[column])
                        if value is not None:
                            msg[column] = value
            messages.append(msg)
        messages = _strip_stale_tool_call_markers(_strip_background_review_harness(messages))
        if repair_alternation and messages:
            from agent.agent_runtime_helpers import repair_message_sequence
            repair_message_sequence(None, messages)
        return messages

    def latest_message_row_id(self, session_id: str, *, role: str = "user", offset: int = 0,
                              require_text: bool = True) -> int | None:
        """Row id of the most recent active *role* message, or ``None`` (oracle parity)."""
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            return None
        text_filter = "AND content IS NOT NULL AND btrim(content) != '' " if require_text else ""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id FROM {self._schema}.messages WHERE session_id=%s AND role=%s AND active "
                f"{text_filter}ORDER BY id DESC LIMIT 1 OFFSET %s",
                (session_id, role, int(offset)))
            row = cursor.fetchone()
            return int(row[0]) if row is not None else None

    def latest_conversation_role(self, session_id: str) -> str | None:
        """Role of the newest active non-bookkeeping row, or ``None`` (oracle parity)."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT role FROM {self._schema}.messages WHERE session_id=%s AND active "
                "AND role NOT IN ('session_meta', 'system') ORDER BY id DESC LIMIT 1",
                (session_id,))
            row = cursor.fetchone()
            return row[0] if row is not None else None

    def has_gateway_input_owner(self, session_id: str, owner: str) -> bool:
        """Probe the accepted-input marker without allocating message bodies."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT 1 FROM {self._schema}.messages WHERE session_id=%s AND role='user' "
                "AND observed=false AND (active OR compacted) "
                "AND display_metadata->>'gateway_input_owner' = %s LIMIT 1",
                (session_id, owner))
            return cursor.fetchone() is not None

    def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        """Partial-index probe for the gateway's transient-failure dedupe guard."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT 1 FROM {self._schema}.messages WHERE session_id=%s AND platform_message_id=%s LIMIT 1",
                (session_id, platform_message_id))
            return cursor.fetchone() is not None

    def set_message_api_content(self, session_id: str, row_id: int, content: Any, api_content: str) -> int:
        """Backfill the ``api_content`` sidecar onto ONE known durable user row."""
        if not session_id or isinstance(row_id, bool) or not isinstance(row_id, int) or row_id <= 0:
            return 0
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.messages SET api_content=%s WHERE id=%s AND session_id=%s "
                "AND role='user' AND active AND content IS NOT DISTINCT FROM %s",
                (api_content, row_id, session_id, self._encode_content(content)))
            return int(cursor.rowcount)

    def set_latest_user_api_content(self, session_id: str, content: Any, api_content: str) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row (0/1 rows)."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {self._schema}.messages SET api_content=%s WHERE id=(SELECT id FROM {self._schema}.messages "
                "WHERE session_id=%s AND role='user' AND active ORDER BY id DESC LIMIT 1) "
                "AND content IS NOT DISTINCT FROM %s",
                (api_content, session_id, self._encode_content(content)))
            return int(cursor.rowcount)

    def try_acquire_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        with self._connection() as connection, connection.cursor() as cursor:
            return self._try_coordination_lease(cursor, table="compression_locks", key_column="session_id", key=session_id, holder=holder, ttl_seconds=ttl_seconds)

    def refresh_compression_lock(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        if not session_id or not holder:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._coordination_clock_and_lock(cursor, "compression_locks", session_id)
            cursor.execute(f"UPDATE {self._schema}.compression_locks SET expires_at=%s, updated_at=%s WHERE session_id=%s AND holder=%s", (now + self._coordination_ttl(ttl_seconds), now, session_id, holder))
            return cursor.rowcount == 1

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        if not session_id or not holder:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            self._coordination_clock_and_lock(cursor, "compression_locks", session_id)
            cursor.execute(f"DELETE FROM {self._schema}.compression_locks WHERE session_id=%s AND holder=%s", (session_id, holder))

    def get_compression_lock_holder(self, session_id: str) -> str | None:
        if not session_id:
            return None
        with self._connection() as connection, connection.cursor() as cursor:
            now = self._coordination_clock_and_lock(cursor, "compression_locks", session_id)
            cursor.execute(f"SELECT holder FROM {self._schema}.compression_locks WHERE session_id=%s AND expires_at > %s", (session_id, now))
            row = cursor.fetchone()
        return None if row is None else str(row[0])

    def try_acquire_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0, **_ignored: Any) -> bool:
        if not session_id or not holder:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            key = self._compression_turn_lease_key_on_cursor(cursor, session_id)
            return self._try_coordination_lease(cursor, table="session_turn_leases", key_column="conversation_id", key=key, holder=holder, ttl_seconds=ttl_seconds)

    def refresh_session_turn_lease(self, session_id: str, holder: str, *, ttl_seconds: float = 300.0) -> bool:
        if not session_id or not holder:
            return False
        with self._connection() as connection, connection.cursor() as cursor:
            key = self._compression_turn_lease_key_on_cursor(cursor, session_id)
            now = self._coordination_clock_and_lock(cursor, "session_turn_leases", key)
            cursor.execute(f"UPDATE {self._schema}.session_turn_leases SET expires_at=%s, updated_at=%s WHERE conversation_id=%s AND holder=%s", (now + self._coordination_ttl(ttl_seconds), now, key, holder))
            return cursor.rowcount == 1

    def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        if not session_id or not holder:
            return
        with self._connection() as connection, connection.cursor() as cursor:
            key = self._compression_turn_lease_key_on_cursor(cursor, session_id)
            self._coordination_clock_and_lock(cursor, "session_turn_leases", key)
            cursor.execute(f"DELETE FROM {self._schema}.session_turn_leases WHERE conversation_id=%s AND holder=%s", (key, holder))

    @staticmethod
    def _decode_content(content: Any) -> Any:
        prefix = "__hermes_state_json__:"
        if isinstance(content, str) and content.startswith(prefix):
            try:
                return json.loads(content[len(prefix):])
            except json.JSONDecodeError:
                return content
        return content

    def get_active_message_ids(self, session_id: str) -> list[int]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT id FROM {self._schema}.messages WHERE session_id=%s AND active ORDER BY id", (session_id,))
            return [int(row[0]) for row in cursor.fetchall()]

    def get_rewind_receipt(self, request_id: str) -> dict[str, Any] | None:
        if not request_id:
            raise ValueError("rewind receipt requires a request id")
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT * FROM {self._schema}.rewind_receipts WHERE request_id=%s", (request_id,))
            row = cursor.fetchone()
        return None if row is None else dict(row)

    def rewind_to_message(self, session_id: str, target_message_id: int, *, preserve_compaction_handoff: bool = False,
                          expected_active_ids: Collection[int] | None = None, expected_target_content: Any = None,
                          request_id: str | None = None) -> dict[str, Any]:
        """Commit one fenced, receipt-backed soft rewind or return its prior receipt.

        Every refusal precedes mutation. A caller that loses the acknowledgement retries
        only with the same ``request_id`` and adopts this durable receipt; it must not
        replay a request whose receipt is absent.
        """
        from uuid import uuid4
        from agent.context_compressor import split_user_originated_turn
        from agent.memory_manager import sanitize_context
        request_id = request_id or uuid4().hex
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            now = self._coordination_clock_and_lock(cursor, "transcript-rewind", session_id)
            cursor.execute(f"SELECT * FROM {self._schema}.rewind_receipts WHERE request_id=%s FOR UPDATE", (request_id,))
            prior = cursor.fetchone()
            if prior is not None:
                if prior["session_id"] != session_id or int(prior["target_message_id"]) != target_message_id:
                    raise ValueError("rewind request id is already bound to another mutation")
                return {"request_id": request_id, "rewound_count": int(prior["retired_count"]),
                        "new_head_id": prior["replacement_message_id"], "replacement_message_id": prior["replacement_message_id"],
                        "active_prefix_ids": list(prior["active_prefix_ids"])}
            cursor.execute(f"SELECT id, ended_at FROM {self._schema}.sessions WHERE id=%s FOR UPDATE", (session_id,))
            session = cursor.fetchone()
            if session is None or session["ended_at"] is not None:
                raise ValueError("rewind session is missing or no longer active")
            root = self._compression_turn_lease_key_on_cursor(cursor, session_id)
            cursor.execute(f"SELECT holder, fence FROM {self._schema}.session_turn_leases WHERE conversation_id=%s AND expires_at>%s FOR UPDATE", (root, now))
            turn_lease = cursor.fetchone()
            if turn_lease is not None:
                raise RuntimeError("session has an active turn lease; refusing transcript mutation")
            cursor.execute(f"SELECT holder, fence FROM {self._schema}.compression_locks WHERE session_id=%s AND expires_at>%s FOR UPDATE", (session_id, now))
            compression_lease = cursor.fetchone()
            if compression_lease is not None:
                raise RuntimeError("session is being compressed by another writer")
            cursor.execute(f"SELECT * FROM {self._schema}.messages WHERE session_id=%s AND active ORDER BY id FOR UPDATE", (session_id,))
            rows = list(cursor.fetchall())
            active_ids = [int(row["id"]) for row in rows]
            if expected_active_ids is not None and active_ids != [int(item) for item in expected_active_ids]:
                raise RuntimeError("active transcript changed before the rewind could be persisted")
            target = next((row for row in rows if int(row["id"]) == target_message_id), None)
            if target is None or target["role"] != "user":
                raise ValueError("rewind target is not an active user message")
            content = self._decode_content(target["content"])
            handoff, live = split_user_originated_turn({"role": "user", "content": content, "display_kind": target["display_kind"], "display_metadata": target["display_metadata"]})
            if live is None:
                raise ValueError("rewind target is not a user-originated turn")
            from agent.session_persistence import _durable_content
            actual = _durable_content(live.get("content"))
            if isinstance(actual, str):
                actual = sanitize_context(actual).strip()
            if expected_target_content is not None and actual != expected_target_content:
                raise RuntimeError("rewind target changed before it could be persisted")
            if preserve_compaction_handoff and handoff is None:
                raise ValueError("preserve_compaction_handoff requires an active composite carrier")
            replacement_id = None
            if preserve_compaction_handoff:
                if handoff is None:  # guarded above; keeps the transactional path type-safe.
                    raise ValueError("preserve_compaction_handoff requires an active composite carrier")
                record = self._replace_record_from_dict(handoff)
                cursor.execute(f"INSERT INTO {self._schema}.messages (session_id, role, content, created_at, {', '.join(_MESSAGE_RECORD_WRITE_COLUMNS)}) VALUES ({_MESSAGE_INSERT_PLACEHOLDERS}) RETURNING id", self._record_params(session_id, record, cursor=cursor))
                inserted = cursor.fetchone()
                replacement_id = int(inserted["id"] if isinstance(inserted, Mapping) else inserted[0])
            if replacement_id is None:
                cursor.execute(f"UPDATE {self._schema}.messages SET active=false WHERE session_id=%s AND active AND id >= %s", (session_id, target_message_id))
            else:
                cursor.execute(f"UPDATE {self._schema}.messages SET active=false WHERE session_id=%s AND active AND id >= %s AND id <> %s", (session_id, target_message_id, replacement_id))
            retired = sum(1 for row in rows if int(row["id"]) >= target_message_id)
            prefix_ids = [row_id for row_id in active_ids if row_id < target_message_id]
            if replacement_id is not None:
                prefix_ids.append(replacement_id)
            cursor.execute(f"UPDATE {self._schema}.sessions SET rewind_count=rewind_count+1, last_activity_at=GREATEST(COALESCE(last_activity_at, started_at), %s) WHERE id=%s", (now, session_id))
            cursor.execute(f"INSERT INTO {self._schema}.rewind_receipts (request_id, session_id, conversation_root_id, target_message_id, turn_holder, turn_fence, compression_holder, compression_fence, replacement_message_id, retired_count, active_prefix_ids, committed_at) VALUES (%s, %s, %s, %s, NULL, NULL, NULL, NULL, %s, %s, %s, %s)", (request_id, session_id, root, target_message_id, replacement_id, retired, self._psycopg.types.json.Jsonb(prefix_ids), now))
            return {"request_id": request_id, "rewound_count": retired, "target_message": dict(target), "new_head_id": replacement_id, "replacement_message_id": replacement_id, "active_prefix_ids": prefix_ids}

    def get_message_records(
        self, session_id: str, *, include_compacted: bool = False,
    ) -> list[dict[str, Any]]:
        """Return live rows, or the user-visible compacted history for export.

        Rewind-retired rows have both flags false and remain excluded.  PostgreSQL
        compression rotation normally publishes a child instead of compacting in
        place, but imported or future in-place rows still need the same export
        contract as ``SessionDB.get_messages(include_compacted=True)``.
        """
        columns = (
            "id, session_id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
            "created_at AS timestamp, token_count, finish_reason, reasoning, reasoning_content, reasoning_details, "
            "codex_reasoning_items, codex_message_items, platform_message_id, observed, _compressed_summary, "
            "active, compacted, api_content, display_kind, display_metadata, "
            "message_uid, absorbed_message_uids, tool_call_uids, tool_call_uid"
        )
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            visibility = "(active OR compacted)" if include_compacted else "active"
            cursor.execute(
                f"SELECT {columns} FROM {self._schema}.messages "
                f"WHERE session_id = %s AND {visibility} ORDER BY id",
                (session_id,),
            )
            records = list(cursor.fetchall())
        for record in records:
            record["content"] = self._decode_content(record["content"])
        return records

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT id, session_id, role, content, created_at FROM {self._schema}.messages WHERE session_id = %s ORDER BY id",
                (session_id,),
            )
            return list(cursor.fetchall())

    def _contextual_message_rows(self, cursor: Any, session_id: str, ids: list[int]) -> list[dict[str, Any]]:
        """Hydrate bounded contextual rows in the same snapshot as their seek.

        Contextual browsing is physical transcript order: ids, rather than wall
        timestamps, preserve tool-call adjacency when clocks tie or regress.
        """
        if not ids:
            return []
        columns = (
            "id, session_id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
            "created_at AS timestamp, token_count, finish_reason, reasoning, reasoning_content, reasoning_details, "
            "codex_reasoning_items, codex_message_items, platform_message_id, observed, _compressed_summary, "
            "active, compacted, api_content, display_kind, display_metadata, "
            "message_uid, absorbed_message_uids, tool_call_uids, tool_call_uid"
        )
        cursor.execute(
            f"SELECT {columns} FROM {self._schema}.messages "
            "WHERE session_id = %s AND id = ANY(%s) ORDER BY id",
            (session_id, ids),
        )
        rows = list(cursor.fetchall())
        for row in rows:
            row["content"] = self._decode_content(row["content"])
        return rows

    def get_messages_around(self, session_id: str, around_message_id: int, *, window: int = 5) -> dict[str, Any]:
        """Return SQLite-compatible physical transcript neighbours around one anchor.

        A foreign anchor deliberately yields an empty view.  The anchor is included
        in the backward seek so re-anchoring at a page boundary repeats it.
        """
        window = max(int(window), 0)
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(
                f"SELECT id FROM {self._schema}.messages WHERE id = %s AND session_id = %s",
                (around_message_id, session_id),
            )
            if cursor.fetchone() is None:
                return {"window": [], "messages_before": 0, "messages_after": 0}
            cursor.execute(
                f"SELECT id FROM {self._schema}.messages WHERE session_id = %s AND id <= %s "
                "ORDER BY id DESC LIMIT %s",
                (session_id, around_message_id, window + 1),
            )
            before_ids = [row["id"] for row in cursor.fetchall()]
            cursor.execute(
                f"SELECT id FROM {self._schema}.messages WHERE session_id = %s AND id > %s "
                "ORDER BY id ASC LIMIT %s",
                (session_id, around_message_id, window),
            )
            after_ids = [row["id"] for row in cursor.fetchall()]
            rows = self._contextual_message_rows(cursor, session_id, list(reversed(before_ids)) + after_ids)
        return {"window": rows, "messages_before": max(0, len(before_ids) - 1), "messages_after": len(after_ids)}

    def get_anchored_view(
        self, session_id: str, around_message_id: int, *, window: int = 5, bookend: int = 3,
        keep_roles: tuple[str, ...] | None = ("user", "assistant"),
    ) -> dict[str, Any]:
        """Return the filtered anchor view and non-overlapping same-session bookends."""
        bookend = max(int(bookend), 0)
        primitive = self.get_messages_around(session_id, around_message_id, window=window)
        physical_window = primitive["window"]
        if not physical_window:
            return {"window": [], "messages_before": 0, "messages_after": 0, "bookend_start": [], "bookend_end": []}
        filtered_window = physical_window
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [row for row in physical_window if row["id"] == around_message_id or row["role"] in keep_set]
        start_rows: list[dict[str, Any]] = []
        end_rows: list[dict[str, Any]] = []
        if bookend:
            role_clause, role_params = "", []
            if keep_roles is not None:
                role_clause, role_params = " AND role = ANY(%s)", [list(keep_roles)]
            with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
                cursor.execute(
                    f"SELECT id FROM {self._schema}.messages WHERE session_id = %s AND id < %s{role_clause} "
                    "AND length(content) > 0 ORDER BY id ASC LIMIT %s",
                    [session_id, physical_window[0]["id"], *role_params, bookend],
                )
                start_rows = self._contextual_message_rows(cursor, session_id, [row["id"] for row in cursor.fetchall()])
                cursor.execute(
                    f"SELECT id FROM {self._schema}.messages WHERE session_id = %s AND id > %s{role_clause} "
                    "AND length(content) > 0 ORDER BY id DESC LIMIT %s",
                    [session_id, physical_window[-1]["id"], *role_params, bookend],
                )
                end_rows = self._contextual_message_rows(cursor, session_id, list(reversed([row["id"] for row in cursor.fetchall()])))
        return {
            "window": filtered_window, "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"], "bookend_start": start_rows, "bookend_end": end_rows,
        }

    @staticmethod
    def _flatten_search_context(content: Any) -> str:
        """Match SessionDB's contextual text projection for decoded content."""
        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
            return " ".join(part for part in parts if part).strip() or "[multimodal content]"
        return content if isinstance(content, str) else ""

    def _search_contexts(self, message_ids: Collection[int]) -> dict[int, list[dict[str, str]]]:
        """Batch same-session timestamp/identity neighbours for canonical candidates."""
        contexts = {int(message_id): [] for message_id in message_ids}
        if not contexts:
            return contexts
        sql = f"""
            WITH target AS (
                SELECT id, session_id, created_at FROM {self._schema}.messages WHERE id = ANY(%s)
            )
            SELECT t.id AS match_id, m.role, m.content
            FROM target AS t JOIN {self._schema}.messages AS m ON m.id IN (
                t.id,
                (SELECT p.id FROM {self._schema}.messages AS p
                 WHERE p.session_id = t.session_id AND (p.created_at, p.id) < (t.created_at, t.id)
                 ORDER BY p.created_at DESC, p.id DESC LIMIT 1),
                (SELECT n.id FROM {self._schema}.messages AS n
                 WHERE n.session_id = t.session_id AND (n.created_at, n.id) > (t.created_at, t.id)
                 ORDER BY n.created_at, n.id LIMIT 1)
            )
            ORDER BY t.id, m.created_at, m.id
        """
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(sql, (list(contexts),))
            for row in cursor.fetchall():
                contexts[int(row["match_id"])].append({
                    "role": row["role"],
                    "content": self._flatten_search_context(self._decode_content(row["content"]))[:200],
                })
        return contexts

    def search_messages(
        self, query: str, source_filter: list[str] | None = None, exclude_sources: list[str] | None = None,
        role_filter: list[str] | None = None, limit: int = 20, offset: int = 0, sort: str | None = None,
        include_inactive: bool = False, fields: Collection[str] | None = None,
        after_ts: int | None = None, before_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Bounded lexical search, deliberately narrower than SQLite FTS5.

        Latin terms use PostgreSQL's built-in ``simple`` tsvector/tsquery. CJK
        uses a parameterized canonical-row substring fallback because PostgreSQL
        ships no CJK tokenizer here. Neither route uses pgvector or pg_trgm.
        """
        if not isinstance(query, str) or not query.strip() or limit <= 0 or offset < 0:
            return []
        result_fields: tuple[str, ...] | None = None
        if fields is not None:
            if isinstance(fields, str):
                raise TypeError("search fields must be a collection of field names, not a string")
            unknown = set(fields).difference(_SEARCH_RESULT_FIELDS)
            if unknown:
                raise ValueError(f"unsupported PostgreSQL search result field(s): {', '.join(sorted(unknown))}")
            result_fields = tuple(field for field in _SEARCH_RESULT_FIELDS if field in fields)
        predicates = ["COALESCE(m.display_kind, '') <> 'hidden'"]
        params: list[Any] = []
        if not include_inactive:
            predicates.append("(m.active OR m.compacted)")
        if source_filter is not None:
            if not source_filter:
                return []
            predicates.append("s.source = ANY(%s)"); params.append(source_filter)
        if exclude_sources:
            predicates.append("NOT (s.source = ANY(%s))"); params.append(exclude_sources)
        if role_filter:
            predicates.append("m.role = ANY(%s)"); params.append(role_filter)
        if after_ts is not None:
            predicates.append("s.started_at >= %s"); params.append(int(after_ts))
        if before_ts is not None:
            predicates.append("s.started_at < %s"); params.append(int(before_ts))
        expression = compile_postgresql_search_expression(query)
        if expression.is_cjk_literal:
            searchable = "(coalesce(m.content, '') || ' ' || coalesce(m.tool_name, '') || ' ' || coalesce(m.tool_calls::text, ''))"
            predicates.extend(f"position(%s in {searchable}) > 0" for _ in expression.params)
            params.extend(expression.params)
            rank, snippet, order = "0.0", "left(coalesce(m.content, m.tool_name, ''), 240)", "m.created_at DESC, m.id DESC"
        else:
            query_sql, query_params = expression.sql, list(expression.params)
            predicates.append(f"m.search_document @@ {query_sql}")
            rank = f"ts_rank_cd(m.search_document, {query_sql})"
            snippet = f"ts_headline('simple', coalesce(m.content, m.tool_name, ''), {query_sql}, 'StartSel=>>>, StopSel=<<<, MaxWords=40, MinWords=1')"
            # SELECT placeholders precede filters/WHERE. The expression is static SQL plus
            # separately-bound literals, including the parser-generated prefix marker.
            params = [*query_params, *query_params, *params, *query_params]
            order = "m.created_at DESC, m.id DESC" if sort == "newest" else "m.created_at ASC, m.id ASC" if sort == "oldest" else "rank DESC, m.id DESC"
        params.extend([limit, offset])
        sql = (
            f"SELECT m.id, m.session_id, m.role, {snippet} AS snippet, m.created_at AS timestamp, m.tool_name, "
            f"s.source, s.model, s.started_at AS session_started, {rank} AS rank "
            f"FROM {self._schema}.messages m JOIN {self._schema}.sessions s ON s.id = m.session_id "
            f"WHERE {' AND '.join(predicates)} ORDER BY {order} LIMIT %s OFFSET %s"
        )
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(sql, params)
            rows = list(cursor.fetchall())
        for row in rows:
            row.pop("rank", None)
        if result_fields is None or "context" in result_fields:
            try:
                contexts = self._search_contexts([int(row["id"]) for row in rows])
            except Exception:  # Context is best-effort; lexical candidates remain usable on enrichment failure.
                logger.exception("Failed to enrich PostgreSQL search result context")
                contexts = {}
            for row in rows:
                row["context"] = contexts.get(row["id"], [])
        if result_fields is not None:
            rows = [{field: row[field] for field in result_fields if field in row} for row in rows]
        return rows

    @staticmethod
    def _is_explicit_branch(session: Mapping[str, Any]) -> bool:
        config = session.get("model_config") or {}
        if isinstance(config, str):
            try:
                config = json.loads(config)
            except json.JSONDecodeError:
                config = {}
        return isinstance(config, Mapping) and "_branched_from" in config

    def get_compression_lineage(self, session_id: str) -> list[str]:
        session = self.get_session(session_id)
        if session is None or self._is_explicit_branch(session):
            return [session_id] if session else []
        root, seen = session, {session_id}
        while root.get("parent_session_id"):
            parent = self.get_session(str(root["parent_session_id"]))
            if parent is None or str(parent["id"]) in seen or parent.get("end_reason") != "compression" or self._is_explicit_branch(root):
                break
            root = parent
            seen.add(str(root["id"]))
        lineage, current = [str(root["id"])], root
        seen = {str(root["id"])}
        while current.get("end_reason") == "compression":
            with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
                cursor.execute(
                    f"SELECT id, parent_session_id, end_reason, model_config FROM {self._schema}.sessions "
                    "WHERE parent_session_id = %s ORDER BY started_at ASC, id ASC", (current["id"],))
                children = cursor.fetchall()
            next_child = next((child for child in children if not self._is_explicit_branch(child)), None)
            if next_child is None or next_child["id"] in seen:
                break
            current = next_child
            lineage.append(str(current["id"]))
            seen.add(str(current["id"]))
        return lineage if session_id in lineage else [session_id]

    def get_compression_tip(self, session_id: str) -> str | None:
        lineage = self.get_compression_lineage(session_id)
        return lineage[-1] if lineage else session_id

    def get_conversation_root(self, session_id: str) -> str:
        lineage = self.get_compression_lineage(session_id)
        return lineage[0] if lineage else session_id

    def _resume_lineage_ids(self, session_id: str) -> list[str]:
        session = self.get_session(session_id)
        return [session_id] if session is None or self._is_explicit_branch(session) else self.get_compression_lineage(session_id)

    def _projection_rows(self, session_ids: list[str], *, display: bool) -> list[dict[str, Any]]:
        if not session_ids:
            return []
        predicate = "(active OR compacted)" if display else "active"
        placeholders = ", ".join("%s" for _ in session_ids)
        columns = (
            "id, session_id, role, content, created_at AS timestamp, tool_call_id, tool_calls, tool_name, "
            "effect_disposition, finish_reason, reasoning, reasoning_content, reasoning_details, codex_reasoning_items, "
            "codex_message_items, platform_message_id, observed, _compressed_summary, active, compacted, api_content, "
            "display_kind, display_metadata"
        )
        with self._connection() as connection, connection.cursor(row_factory=self._psycopg.rows.dict_row) as cursor:
            cursor.execute(f"SELECT {columns} FROM {self._schema}.messages WHERE session_id IN ({placeholders}) AND {predicate} ORDER BY id", session_ids)
            rows = list(cursor.fetchall())
        for row in rows:
            row["content"] = self._decode_content(row["content"])
        if not display:
            return rows
        chosen, first = {}, {}
        for row in rows:
            key = (row["role"], json.dumps(row["content"], sort_keys=True, default=str), row["timestamp"], row["tool_call_id"], json.dumps(row["tool_calls"], sort_keys=True, default=str), row["tool_name"])
            previous = chosen.get(key)
            if previous is None or (bool(row["active"]), row["id"]) > (bool(previous["active"]), previous["id"]):
                chosen[key] = row
            first[key] = min(first.get(key, row["id"]), row["id"])
        return [chosen[key] for key in sorted(chosen, key=first.__getitem__)]

    @staticmethod
    def _conversation(rows: list[dict[str, Any]], *, row_ids: bool = True) -> list[dict[str, Any]]:
        messages = []
        for row in rows:
            message = {"role": row["role"], "content": row["content"]}
            if row_ids:
                message["_row_id"] = row["id"]
            for key in ("timestamp", "tool_call_id", "tool_name", "effect_disposition", "api_content", "display_kind"):
                if row.get(key):
                    message[key] = row[key]
            if row.get("tool_calls"):
                message["tool_calls"] = row["tool_calls"]
            if row.get("display_metadata"):
                message["display_metadata"] = row["display_metadata"]
            if row.get("observed"):
                message["observed"] = True
            if row["role"] == "assistant":
                for key in ("finish_reason", "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items"):
                    if row.get(key) is not None:
                        message[key] = row[key]
            messages.append(message)
        return messages

    def get_resume_conversations(self, session_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        lineage = self._resume_lineage_ids(session_id)
        display_rows = self._projection_rows(lineage, display=True)
        model_rows = [row for row in self._projection_rows([session_id], display=False) if row["active"]]
        return self._conversation(model_rows), self._conversation(display_rows)

    def get_ancestor_display_prefix(self, session_id: str) -> list[dict[str, Any]]:
        lineage = self._resume_lineage_ids(session_id)
        rows = self._projection_rows(lineage, display=True)
        return [
            {key: value for key, value in message.items() if key != "_row_id"}
            for row, message in zip(rows, self._conversation(rows))
            if row["session_id"] != session_id
        ]

    def get_resume_message_count(self, session_id: str, *, tip_only: bool = False) -> int:
        return len(self._projection_rows([session_id] if tip_only else self._resume_lineage_ids(session_id), display=not tip_only))

    def assert_resume_safe(self, session_id: str, max_messages: int | None = None, *, tip_only: bool = False) -> int:
        if max_messages is None:
            from hermes_state import resolved_max_resume_messages
            max_messages = resolved_max_resume_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        count = self.get_resume_message_count(session_id, tip_only=tip_only)
        if count > max_messages:
            from hermes_state import SessionResumeTooLargeError
            raise SessionResumeTooLargeError(count, max_messages, scope="in its tip segment" if tip_only else "across its lineage")
        return count

    def close(self) -> None:
        self._token_usage_transport.close()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = []
            while True:
                try:
                    connections.append(self._idle.get_nowait())
                except queue.Empty:
                    break
        for connection in connections:
            connection.close()
