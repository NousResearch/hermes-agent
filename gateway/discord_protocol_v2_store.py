"""Durable SQLite store for Discord Native Multi-Bot Protocol v2.

Slice 0B is intentionally limited to schema, idempotent persistence helpers,
and lease/state-transition primitives.  It does not wire any runtime Discord
clients, gateway workers, routing engines, Hermes invocation, or outbox sender.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from gateway.discord_protocol_v2_events import (
    create_agent_event_envelope,
    is_valid_agent_event_id,
    new_agent_event_id,
)
from gateway.secret_refs import redact_sensitive_data


AUTHOR_KINDS = ("human", "registered_bot", "external_bot", "webhook", "system")
DISCORD_SOURCE_TYPE = "discord_message"
INTERNAL_SOURCE_TYPE = "internal_event"
SOURCE_TYPES = (DISCORD_SOURCE_TYPE, INTERNAL_SOURCE_TYPE)
INTERNAL_REQUEST_EVENT_TYPES = (
    "handoff.requested",
    "consult.requested",
    "review.requested",
)

INBOUND_STATUSES = ("pending", "leased", "completed", "failed", "retryable")
OUTBOX_STATUSES = (
    "pending",
    "leased",
    "sending",
    "sent",
    "acked",
    "uncertain",
    "reconciled",
)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS identity_registry (
    agent_id TEXT PRIMARY KEY,
    hermes_profile TEXT NOT NULL,
    discord_application_id TEXT NOT NULL,
    discord_bot_user_id TEXT NOT NULL,
    token_secret_ref TEXT NOT NULL,
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    scopes_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1)),
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    thread_id TEXT,
    parent_channel_id TEXT,
    title TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS topic_agent_sessions (
    topic_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    hermes_session_id TEXT NOT NULL,
    session_key TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'active',
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (topic_id, agent_id),
    UNIQUE (topic_id, agent_id),
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id) ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES identity_registry(agent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS message_map (
    discord_message_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    thread_id TEXT,
    parent_channel_id TEXT,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound', 'projection')),
    agent_id TEXT,
    delivery_key TEXT,
    outbox_delivery_id TEXT,
    agent_event_id TEXT,
    author_id TEXT NOT NULL,
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
    author_bot_user_id TEXT,
    source_client_agent_id TEXT,
    mentions_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_events (
    agent_event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    source_agent_id TEXT,
    target_agent_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS inbound_deliveries (
    delivery_key TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('discord_message', 'internal_event')),
    source_id TEXT NOT NULL,
    discord_message_id TEXT,
    agent_event_id TEXT,
    target_agent_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'leased', 'completed', 'failed', 'retryable')),
    lease_owner TEXT,
    lease_until TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE (source_type, source_id, target_agent_id),
    CHECK (source_type != 'discord_message' OR author_kind = 'human'),
    CHECK (
        (source_type = 'discord_message' AND discord_message_id = source_id AND agent_event_id IS NULL)
        OR
        (source_type = 'internal_event' AND agent_event_id = source_id AND discord_message_id IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS outbox_deliveries (
    outbox_delivery_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    target_agent_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    thread_id TEXT,
    source_inbound_delivery_key TEXT,
    source_agent_event_id TEXT,
    delivery_kind TEXT NOT NULL CHECK (delivery_kind IN ('response', 'projection', 'diagnostic')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'leased', 'sending', 'sent', 'acked', 'uncertain', 'reconciled')),
    lease_owner TEXT,
    lease_until TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (source_inbound_delivery_key) REFERENCES inbound_deliveries(delivery_key) ON DELETE SET NULL,
    FOREIGN KEY (source_agent_event_id) REFERENCES agent_events(agent_event_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS outbox_parts (
    outbox_delivery_id TEXT NOT NULL,
    part_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    discord_message_id TEXT,
    PRIMARY KEY (outbox_delivery_id, part_index),
    UNIQUE (outbox_delivery_id, part_index),
    FOREIGN KEY (outbox_delivery_id) REFERENCES outbox_deliveries(outbox_delivery_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS route_decisions (
    decision_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (source_type IN ('discord_message', 'internal_event')),
    source_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
    decision TEXT NOT NULL,
    target_agent_ids_json TEXT NOT NULL DEFAULT '[]',
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    source_inbound_delivery_key TEXT,
    source_agent_event_id TEXT,
    target_agent_id TEXT NOT NULL,
    agent_id TEXT,
    topic_id TEXT NOT NULL,
    requesting_event_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (source_inbound_delivery_key) REFERENCES inbound_deliveries(delivery_key) ON DELETE SET NULL,
    FOREIGN KEY (source_agent_event_id) REFERENCES agent_events(agent_event_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS approval_audit_events (
    audit_event_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor_user_id TEXT,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (approval_id) REFERENCES approvals(approval_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    agent_event_id TEXT NOT NULL UNIQUE,
    source_agent_id TEXT,
    target_agent_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (agent_event_id) REFERENCES agent_events(agent_event_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reconciliation_runs (
    reconciliation_run_id TEXT PRIMARY KEY,
    source_agent_event_id TEXT,
    outbox_delivery_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (source_agent_event_id) REFERENCES agent_events(agent_event_id) ON DELETE SET NULL,
    FOREIGN KEY (outbox_delivery_id) REFERENCES outbox_deliveries(outbox_delivery_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_inbound_status ON inbound_deliveries(status, lease_until, created_at);
CREATE INDEX IF NOT EXISTS idx_inbound_source ON inbound_deliveries(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox_deliveries(status, lease_until, created_at);
CREATE INDEX IF NOT EXISTS idx_route_decisions_source ON route_decisions(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_message_map_agent_event ON message_map(agent_event_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_approval_audit_approval ON approval_audit_events(approval_id, created_at);
"""


class DiscordProtocolV2Store:
    """Small deterministic SQLite contract for Discord protocol v2 Slice 0B."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_schema_without_indexes())
        self._migrate_schema()
        # Recreate indexes that SQLite drops when a migration rebuilds a table.
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Apply additive migrations for stores created by earlier v2 slices."""

        self._migrate_inbound_deliveries_for_internal_events()
        self._migrate_route_decisions_for_internal_events()

        approval_columns = self.table_columns("approvals")
        if "agent_id" not in approval_columns:
            self.conn.execute("ALTER TABLE approvals ADD COLUMN agent_id TEXT")
            self.conn.execute("UPDATE approvals SET agent_id = target_agent_id WHERE agent_id IS NULL")
        if "requesting_event_id" not in approval_columns:
            self.conn.execute("ALTER TABLE approvals ADD COLUMN requesting_event_id TEXT")

        outbox_columns = self.table_columns("outbox_deliveries")
        if "source_agent_event_id" not in outbox_columns:
            self.conn.execute("ALTER TABLE outbox_deliveries ADD COLUMN source_agent_event_id TEXT")

        message_map_columns = self.table_columns("message_map")
        if "outbox_delivery_id" not in message_map_columns:
            self.conn.execute("ALTER TABLE message_map ADD COLUMN outbox_delivery_id TEXT")
        if "agent_event_id" not in message_map_columns:
            self.conn.execute("ALTER TABLE message_map ADD COLUMN agent_event_id TEXT")
        if "source_client_agent_id" not in message_map_columns:
            self.conn.execute("ALTER TABLE message_map ADD COLUMN source_client_agent_id TEXT")

    def _migrate_inbound_deliveries_for_internal_events(self) -> None:
        """Rebuild older inbound tables whose CHECKs cannot admit internal events."""

        table_sql = self._table_sql("inbound_deliveries")
        columns = self.table_columns("inbound_deliveries")
        needs_rebuild = (
            "internal_event" not in table_sql
            or "source_id" not in columns
            or "agent_event_id" not in columns
        )
        if not needs_rebuild:
            return

        self._rebuild_table(
            table="inbound_deliveries",
            create_sql="""
                CREATE TABLE inbound_deliveries_new (
                    delivery_key TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL CHECK (source_type IN ('discord_message', 'internal_event')),
                    source_id TEXT NOT NULL,
                    discord_message_id TEXT,
                    agent_event_id TEXT,
                    target_agent_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    route_reason TEXT NOT NULL,
                    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'leased', 'completed', 'failed', 'retryable')),
                    lease_owner TEXT,
                    lease_until TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    state_version INTEGER NOT NULL DEFAULT 1,
                    UNIQUE (source_type, source_id, target_agent_id),
                    CHECK (source_type != 'discord_message' OR author_kind = 'human'),
                    CHECK (
                        (source_type = 'discord_message' AND discord_message_id = source_id AND agent_event_id IS NULL)
                        OR
                        (source_type = 'internal_event' AND agent_event_id = source_id AND discord_message_id IS NULL)
                    )
                )
            """,
            copy_sql=f"""
                INSERT OR IGNORE INTO inbound_deliveries_new (
                    delivery_key, source_type, source_id, discord_message_id, agent_event_id,
                    target_agent_id, topic_id, route_reason, author_kind, payload_json,
                    status, lease_owner, lease_until, attempts, created_at, updated_at, state_version
                )
                SELECT
                    delivery_key,
                    COALESCE(source_type, 'discord_message'),
                    {self._column_expr(columns, 'source_id', "COALESCE(discord_message_id, delivery_key)")},
                    discord_message_id,
                    {self._column_expr(columns, 'agent_event_id', 'NULL')},
                    target_agent_id, topic_id, route_reason, author_kind,
                    COALESCE(payload_json, '{{}}'),
                    COALESCE(status, 'pending'),
                    {self._column_expr(columns, 'lease_owner', 'NULL')},
                    {self._column_expr(columns, 'lease_until', 'NULL')},
                    {self._column_expr(columns, 'attempts', '0')},
                    created_at, updated_at,
                    {self._column_expr(columns, 'state_version', '1')}
                FROM inbound_deliveries
            """,
        )

    def _migrate_route_decisions_for_internal_events(self) -> None:
        """Rebuild older route decision CHECKs so internal_event decisions persist."""

        table_sql = self._table_sql("route_decisions")
        if "internal_event" in table_sql:
            return
        columns = self.table_columns("route_decisions")
        self._rebuild_table(
            table="route_decisions",
            create_sql="""
                CREATE TABLE route_decisions_new (
                    decision_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL CHECK (source_type IN ('discord_message', 'internal_event')),
                    source_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
                    decision TEXT NOT NULL,
                    target_agent_ids_json TEXT NOT NULL DEFAULT '[]',
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                )
            """,
            copy_sql=f"""
                INSERT OR IGNORE INTO route_decisions_new (
                    decision_id, source_type, source_id, topic_id, author_kind, decision,
                    target_agent_ids_json, reason, created_at, payload_json
                )
                SELECT
                    decision_id, source_type, source_id, topic_id, author_kind, decision,
                    {self._column_expr(columns, 'target_agent_ids_json', "'[]'")},
                    reason, created_at,
                    {self._column_expr(columns, 'payload_json', "'{}'")}
                FROM route_decisions
            """,
        )

    def _rebuild_table(self, *, table: str, create_sql: str, copy_sql: str) -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        self.conn.execute("PRAGMA foreign_keys = OFF")
        try:
            self.conn.execute(create_sql)
            self.conn.execute(copy_sql)
            self.conn.execute(f"DROP TABLE {table}")
            self.conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
        finally:
            self.conn.execute("PRAGMA foreign_keys = ON")

    def _table_sql(self, table: str) -> str:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return str(row["sql"] or "") if row else ""

    @staticmethod
    def _column_expr(columns: set[str], name: str, fallback: str) -> str:
        return name if name in columns else fallback

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DiscordProtocolV2Store":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---- identity/topic/session helpers ---------------------------------

    def upsert_identity(
        self,
        *,
        agent_id: str,
        hermes_profile: str,
        discord_application_id: str,
        discord_bot_user_id: str,
        token_secret_ref: str,
        capabilities: Any | None = None,
        scopes: Any | None = None,
        enabled: bool = True,
        version: int = 1,
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO identity_registry (
                agent_id, hermes_profile, discord_application_id, discord_bot_user_id,
                token_secret_ref, capabilities_json, scopes_json, enabled, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                hermes_profile = excluded.hermes_profile,
                discord_application_id = excluded.discord_application_id,
                discord_bot_user_id = excluded.discord_bot_user_id,
                token_secret_ref = excluded.token_secret_ref,
                capabilities_json = excluded.capabilities_json,
                scopes_json = excluded.scopes_json,
                enabled = excluded.enabled,
                version = excluded.version
            """,
            (
                agent_id,
                hermes_profile,
                discord_application_id,
                discord_bot_user_id,
                token_secret_ref,
                _json(capabilities or []),
                _json(scopes or {}),
                1 if enabled else 0,
                version,
            ),
        )
        self.conn.commit()
        return self.get_identity(agent_id)  # type: ignore[return-value]

    def get_identity(self, agent_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM identity_registry WHERE agent_id = ?", (agent_id,))

    def upsert_topic(
        self,
        *,
        topic_id: str,
        guild_id: str,
        channel_id: str,
        title: str,
        thread_id: str | None = None,
        parent_channel_id: str | None = None,
        state: Any | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO topics (
                topic_id, guild_id, channel_id, thread_id, parent_channel_id,
                title, state_json, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET
                guild_id = excluded.guild_id,
                channel_id = excluded.channel_id,
                thread_id = excluded.thread_id,
                parent_channel_id = excluded.parent_channel_id,
                title = excluded.title,
                state_json = excluded.state_json,
                version = excluded.version
            """,
            (
                topic_id,
                guild_id,
                channel_id,
                thread_id,
                parent_channel_id,
                title,
                _json(state or {}),
                version,
            ),
        )
        self.conn.commit()
        return self.get_topic(topic_id)  # type: ignore[return-value]

    def get_topic(self, topic_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM topics WHERE topic_id = ?", (topic_id,))

    def get_topic_agent_session(
        self,
        *,
        topic_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM topic_agent_sessions WHERE topic_id = ? AND agent_id = ?",
            (topic_id, agent_id),
        )

    def list_topic_agent_sessions(
        self,
        *,
        topic_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if topic_id is not None:
            clauses.append("topic_id = ?")
            params.append(topic_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._fetch_all(
            f"SELECT * FROM topic_agent_sessions{where} ORDER BY topic_id, agent_id",
            tuple(params),
        )

    def upsert_topic_agent_session(
        self,
        *,
        topic_id: str,
        agent_id: str,
        hermes_session_id: str,
        session_key: str,
        state: str = "active",
        version: int = 1,
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO topic_agent_sessions (
                topic_id, agent_id, hermes_session_id, session_key, state, version
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, agent_id) DO UPDATE SET
                hermes_session_id = excluded.hermes_session_id,
                session_key = excluded.session_key,
                state = excluded.state,
                version = excluded.version
            """,
            (topic_id, agent_id, hermes_session_id, session_key, state, version),
        )
        self.conn.commit()
        return self._fetch_one(
            "SELECT * FROM topic_agent_sessions WHERE topic_id = ? AND agent_id = ?",
            (topic_id, agent_id),
        )  # type: ignore[return-value]

    # ---- message map and route decisions --------------------------------

    def upsert_message_map(
        self,
        *,
        discord_message_id: str,
        guild_id: str,
        channel_id: str,
        direction: str,
        author_id: str,
        author_kind: str,
        thread_id: str | None = None,
        parent_channel_id: str | None = None,
        agent_id: str | None = None,
        delivery_key: str | None = None,
        outbox_delivery_id: str | None = None,
        agent_event_id: str | None = None,
        author_bot_user_id: str | None = None,
        source_client_agent_id: str | None = None,
        mentions: Any | None = None,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        _require_one_of(direction, ("inbound", "outbound", "projection"), "direction")
        _require_one_of(author_kind, AUTHOR_KINDS, "author_kind")
        now = _now()
        self.conn.execute(
            """
            INSERT INTO message_map (
                discord_message_id, guild_id, channel_id, thread_id, parent_channel_id,
                direction, agent_id, delivery_key, outbox_delivery_id, agent_event_id,
                author_id, author_kind, author_bot_user_id, source_client_agent_id,
                mentions_json, created_at, updated_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(discord_message_id) DO UPDATE SET
                guild_id = excluded.guild_id,
                channel_id = excluded.channel_id,
                thread_id = excluded.thread_id,
                parent_channel_id = excluded.parent_channel_id,
                direction = excluded.direction,
                agent_id = excluded.agent_id,
                delivery_key = excluded.delivery_key,
                outbox_delivery_id = excluded.outbox_delivery_id,
                agent_event_id = excluded.agent_event_id,
                author_id = excluded.author_id,
                author_kind = excluded.author_kind,
                author_bot_user_id = excluded.author_bot_user_id,
                source_client_agent_id = excluded.source_client_agent_id,
                mentions_json = excluded.mentions_json,
                updated_at = excluded.updated_at,
                payload_json = excluded.payload_json
            """,
            (
                discord_message_id,
                guild_id,
                channel_id,
                thread_id,
                parent_channel_id,
                direction,
                agent_id,
                delivery_key,
                outbox_delivery_id,
                agent_event_id,
                author_id,
                author_kind,
                author_bot_user_id,
                source_client_agent_id,
                _json(mentions or []),
                now,
                now,
                _json(payload or {}),
            ),
        )
        self.conn.commit()
        return self.get_message_map(discord_message_id)  # type: ignore[return-value]

    def get_message_map(self, discord_message_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM message_map WHERE discord_message_id = ?",
            (discord_message_id,),
        )

    def record_route_decision(
        self,
        *,
        source_type: str,
        source_id: str,
        topic_id: str,
        author_kind: str,
        decision: str,
        target_agent_ids: Iterable[str] | None = None,
        reason: str,
        payload: Any | None = None,
        decision_id: str | None = None,
    ) -> dict[str, Any]:
        _require_one_of(source_type, SOURCE_TYPES, "source_type")
        _require_one_of(author_kind, AUTHOR_KINDS, "author_kind")
        target_ids = list(dict.fromkeys(target_agent_ids or []))
        decision_id = decision_id or f"route:{source_type}:{source_id}:{decision}"
        now = _now()
        self.conn.execute(
            """
            INSERT INTO route_decisions (
                decision_id, source_type, source_id, topic_id, author_kind, decision,
                target_agent_ids_json, reason, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(decision_id) DO UPDATE SET
                source_type = excluded.source_type,
                source_id = excluded.source_id,
                topic_id = excluded.topic_id,
                author_kind = excluded.author_kind,
                decision = excluded.decision,
                target_agent_ids_json = excluded.target_agent_ids_json,
                reason = excluded.reason,
                payload_json = excluded.payload_json
            """,
            (
                decision_id,
                source_type,
                source_id,
                topic_id,
                author_kind,
                decision,
                _json(target_ids),
                reason,
                now,
                _json(payload or {}),
            ),
        )
        self.conn.commit()
        return self._fetch_one(
            "SELECT * FROM route_decisions WHERE decision_id = ?", (decision_id,)
        )  # type: ignore[return-value]

    # ---- inbound deliveries ---------------------------------------------

    def create_discord_inbound_deliveries(
        self,
        *,
        discord_message_id: str,
        guild_id: str,
        channel_id: str,
        topic_id: str,
        author_id: str,
        author_kind: str,
        target_agent_ids: Iterable[str],
        route_reason: str,
        thread_id: str | None = None,
        parent_channel_id: str | None = None,
        author_bot_user_id: str | None = None,
        source_client_agent_id: str | None = None,
        mentions: Any | None = None,
        payload: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Create Discord-originated deliveries only for human-authored messages."""

        targets = list(dict.fromkeys(target_agent_ids))
        self.upsert_message_map(
            discord_message_id=discord_message_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            parent_channel_id=parent_channel_id,
            direction="inbound",
            author_id=author_id,
            author_kind=author_kind,
            author_bot_user_id=author_bot_user_id,
            source_client_agent_id=source_client_agent_id,
            mentions=mentions or targets,
            payload=payload or {},
        )

        if author_kind != "human":
            self.record_route_decision(
                source_type=DISCORD_SOURCE_TYPE,
                source_id=discord_message_id,
                topic_id=topic_id,
                author_kind=author_kind,
                decision="zero_delivery",
                target_agent_ids=[],
                reason=f"non_human_author:{author_kind}",
                payload=payload or {},
            )
            return []

        deliveries = [
            self._create_inbound_delivery(
                source_type=DISCORD_SOURCE_TYPE,
                source_id=discord_message_id,
                target_agent_id=target_agent_id,
                topic_id=topic_id,
                route_reason=route_reason,
                author_kind=author_kind,
                discord_message_id=discord_message_id,
                agent_event_id=None,
                payload=payload or {},
            )
            for target_agent_id in targets
        ]
        self.record_route_decision(
            source_type=DISCORD_SOURCE_TYPE,
            source_id=discord_message_id,
            topic_id=topic_id,
            author_kind=author_kind,
            decision="delivered" if deliveries else "zero_delivery",
            target_agent_ids=[row["target_agent_id"] for row in deliveries],
            reason=route_reason,
            payload=payload or {},
        )
        return deliveries

    def create_agent_event(
        self,
        *,
        event_type: str,
        source_agent_id: str | None,
        target_agent_id: str,
        topic_id: str,
        payload: Any | None = None,
        status: str = "pending",
        agent_event_id: str | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        agent_event_id = agent_event_id or f"agent-event:{uuid.uuid4().hex}"
        safe_payload = _redacted(payload or {})
        payload_json = _json(safe_payload)
        existing = self.get_agent_event(agent_event_id)
        if existing is not None:
            expected = {
                "event_type": event_type,
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "topic_id": topic_id,
                "payload": safe_payload,
            }
            actual = {
                "event_type": existing.get("event_type"),
                "source_agent_id": existing.get("source_agent_id"),
                "target_agent_id": existing.get("target_agent_id"),
                "topic_id": existing.get("topic_id"),
                "payload": _loads_json(existing.get("payload_json"), {}),
            }
            if actual != expected:
                raise ValueError(f"agent_event_id conflict for immutable event {agent_event_id!r}")
            return existing

        now = _now()
        self.conn.execute(
            """
            INSERT INTO agent_events (
                agent_event_id, event_type, source_agent_id, target_agent_id,
                topic_id, payload_json, status, created_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_event_id,
                event_type,
                source_agent_id,
                target_agent_id,
                topic_id,
                payload_json,
                status,
                now,
                version,
            ),
        )
        self.conn.commit()
        return self.get_agent_event(agent_event_id)  # type: ignore[return-value]

    def get_agent_event(self, agent_event_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM agent_events WHERE agent_event_id = ?", (agent_event_id,)
        )

    def create_internal_event_delivery(
        self,
        *,
        agent_event_id: str,
        target_agent_id: str,
        topic_id: str,
        route_reason: str,
        payload: Any | None = None,
        author_kind: str = "registered_bot",
    ) -> dict[str, Any]:
        return self._create_inbound_delivery(
            source_type=INTERNAL_SOURCE_TYPE,
            source_id=agent_event_id,
            target_agent_id=target_agent_id,
            topic_id=topic_id,
            route_reason=route_reason,
            author_kind=author_kind,
            discord_message_id=None,
            agent_event_id=agent_event_id,
            payload=payload or {},
        )

    def create_internal_handoff(
        self,
        *,
        event_type: str,
        source_agent_id: str | None,
        target_agent_id: str,
        topic_id: str,
        payload: Any | None = None,
        agent_event_id: str | None = None,
        route_reason: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if event_type not in INTERNAL_REQUEST_EVENT_TYPES:
            raise ValueError(
                "event_type must be one of: " + ", ".join(INTERNAL_REQUEST_EVENT_TYPES)
            )
        effective_event_id = agent_event_id
        if effective_event_id is not None and not is_valid_agent_event_id(effective_event_id):
            effective_event_id = new_agent_event_id(str(effective_event_id))
        envelope = create_agent_event_envelope(
            event_type=event_type,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            topic_id=topic_id,
            payload=payload or {},
            agent_event_id=effective_event_id,
            idempotency_seed=agent_event_id,
        )
        safe_payload = envelope.payload
        event = self.create_agent_event(
            event_type=envelope.event_type,
            source_agent_id=envelope.source_agent_id,
            target_agent_id=envelope.target_agent_id,
            topic_id=envelope.topic_id,
            payload=safe_payload,
            status="requested",
            agent_event_id=envelope.agent_event_id,
        )
        if event.get("status") != "requested":
            self.conn.execute(
                "UPDATE agent_events SET status = ? WHERE agent_event_id = ?",
                ("requested", envelope.agent_event_id),
            )
            self.conn.commit()
            refreshed = self.get_agent_event(envelope.agent_event_id)
            if refreshed is None:
                raise KeyError(envelope.agent_event_id)
            event = refreshed
        delivery = self.create_internal_event_delivery(
            agent_event_id=event["agent_event_id"],
            target_agent_id=envelope.target_agent_id,
            topic_id=envelope.topic_id,
            route_reason=route_reason or envelope.event_type,
            payload={**safe_payload, "agent_event_id": event["agent_event_id"]},
        )
        if envelope.event_type == "handoff.requested":
            self.upsert_handoff(
                handoff_id=f"handoff:{event['agent_event_id']}",
                agent_event_id=event["agent_event_id"],
                source_agent_id=envelope.source_agent_id,
                target_agent_id=envelope.target_agent_id,
                topic_id=envelope.topic_id,
                status="requested",
                payload=safe_payload,
            )
        self.record_route_decision(
            source_type=INTERNAL_SOURCE_TYPE,
            source_id=event["agent_event_id"],
            topic_id=envelope.topic_id,
            author_kind="registered_bot",
            decision="delivered",
            target_agent_ids=[envelope.target_agent_id],
            reason=route_reason or envelope.event_type,
            payload=safe_payload,
        )
        topic = self.get_topic(envelope.topic_id)
        if topic is not None:
            self.create_outbox_delivery(
                idempotency_key=projection_idempotency_key(
                    event["agent_event_id"], envelope.target_agent_id
                ),
                target_agent_id=envelope.target_agent_id,
                topic_id=envelope.topic_id,
                channel_id=str(topic["channel_id"]),
                thread_id=topic.get("thread_id"),
                delivery_kind="projection",
                source_agent_event_id=event["agent_event_id"],
                payload={
                    "content": safe_payload.get("summary")
                    or safe_payload.get("task")
                    or envelope.event_type,
                    "agent_event_id": event["agent_event_id"],
                    "event_type": envelope.event_type,
                    "source_agent_id": envelope.source_agent_id,
                    "target_agent_id": envelope.target_agent_id,
                },
            )
        return event, delivery

    def record_projection_message(
        self,
        *,
        agent_event_id: str,
        discord_message_id: str,
        guild_id: str,
        channel_id: str,
        topic_id: str,
        target_agent_id: str,
        author_id: str,
        author_kind: str = "registered_bot",
        thread_id: str | None = None,
        parent_channel_id: str | None = None,
        outbox_delivery_id: str | None = None,
        source_client_agent_id: str | None = None,
        mentions: Any | None = None,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        """Persist a Discord projection of an internal event without delivery fan-out."""

        message = self.upsert_message_map(
            discord_message_id=discord_message_id,
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            parent_channel_id=parent_channel_id,
            direction="projection",
            agent_id=target_agent_id,
            author_id=author_id,
            author_kind=author_kind,
            agent_event_id=agent_event_id,
            outbox_delivery_id=outbox_delivery_id,
            source_client_agent_id=source_client_agent_id,
            mentions=mentions or [],
            payload=payload or {},
        )
        self.record_route_decision(
            source_type=INTERNAL_SOURCE_TYPE,
            source_id=agent_event_id,
            topic_id=topic_id,
            author_kind=author_kind,
            decision="projection",
            target_agent_ids=[target_agent_id],
            reason="discord_projection_no_delivery",
            payload=payload or {},
        )
        return message

    def _create_inbound_delivery(
        self,
        *,
        source_type: str,
        source_id: str,
        target_agent_id: str,
        topic_id: str,
        route_reason: str,
        author_kind: str,
        discord_message_id: str | None,
        agent_event_id: str | None,
        payload: Any,
    ) -> dict[str, Any]:
        _require_one_of(source_type, SOURCE_TYPES, "source_type")
        _require_one_of(author_kind, AUTHOR_KINDS, "author_kind")
        if source_type == DISCORD_SOURCE_TYPE and author_kind != "human":
            raise ValueError("Discord-originated inbound deliveries require author_kind=human")
        delivery_key = inbound_delivery_key(source_type, source_id, target_agent_id)
        now = _now()
        self.conn.execute(
            """
            INSERT OR IGNORE INTO inbound_deliveries (
                delivery_key, source_type, source_id, discord_message_id, agent_event_id,
                target_agent_id, topic_id, route_reason, author_kind, payload_json,
                status, lease_owner, lease_until, attempts, created_at, updated_at, state_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, 0, ?, ?, 1)
            """,
            (
                delivery_key,
                source_type,
                source_id,
                discord_message_id,
                agent_event_id,
                target_agent_id,
                topic_id,
                route_reason,
                author_kind,
                _json(payload),
                now,
                now,
            ),
        )
        self.conn.commit()
        return self.get_inbound_delivery(delivery_key)  # type: ignore[return-value]

    def get_inbound_delivery(self, delivery_key: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM inbound_deliveries WHERE delivery_key = ?", (delivery_key,)
        )

    def list_inbound_deliveries(
        self,
        *,
        source_type: str | None = None,
        source_id: str | None = None,
        target_agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if source_type is not None:
            where.append("source_type = ?")
            params.append(source_type)
        if source_id is not None:
            where.append("source_id = ?")
            params.append(source_id)
        if target_agent_id is not None:
            where.append("target_agent_id = ?")
            params.append(target_agent_id)
        sql = "SELECT * FROM inbound_deliveries"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, delivery_key"
        return self._fetch_all(sql, tuple(params))

    def lease_next_inbound(
        self,
        *,
        lease_owner: str,
        lease_seconds: int = 60,
        target_agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        now = _now()
        lease_until = _future(lease_seconds)
        where = "status IN ('pending', 'retryable') OR (status = 'leased' AND lease_until <= ?)"
        params: list[Any] = [now]
        if target_agent_id is not None:
            where = f"({where}) AND target_agent_id = ?"
            params.append(target_agent_id)
        for _attempt in range(10):
            with self.conn:
                row = self.conn.execute(
                    f"SELECT * FROM inbound_deliveries WHERE {where} ORDER BY created_at, delivery_key LIMIT 1",
                    tuple(params),
                ).fetchone()
                if row is None:
                    return None
                current = dict(row)
                if self._cas_lease_inbound(current, lease_owner, lease_until, now):
                    return self.get_inbound_delivery(current["delivery_key"])
        return None

    def _cas_lease_inbound(
        self,
        current: dict[str, Any],
        lease_owner: str,
        lease_until: str,
        now: str,
    ) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE inbound_deliveries
            SET status = 'leased', lease_owner = ?, lease_until = ?, attempts = attempts + 1,
                updated_at = ?, state_version = state_version + 1
            WHERE delivery_key = ? AND state_version = ?
            """,
            (lease_owner, lease_until, now, current["delivery_key"], current["state_version"]),
        )
        return cursor.rowcount == 1

    def complete_inbound(self, delivery_key: str) -> dict[str, Any]:
        return self.transition_inbound(delivery_key, "completed")

    def complete_inbound_if_leased_by(
        self,
        delivery_key: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        return self.transition_inbound_if_leased_by(delivery_key, "completed", lease_owner)

    def fail_inbound(self, delivery_key: str) -> dict[str, Any]:
        return self.transition_inbound(delivery_key, "failed")

    def retry_inbound(self, delivery_key: str) -> dict[str, Any]:
        return self.transition_inbound(delivery_key, "retryable")

    def retry_inbound_if_leased_by(
        self,
        delivery_key: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        return self.transition_inbound_if_leased_by(delivery_key, "retryable", lease_owner)

    def refresh_inbound_lease_if_leased_by(
        self,
        delivery_key: str,
        lease_owner: str,
        lease_seconds: int = 60,
    ) -> dict[str, Any] | None:
        """Extend an active inbound lease only for its current owner."""

        row = self.get_inbound_delivery(delivery_key)
        if row is None:
            raise KeyError(delivery_key)
        now = _now()
        lease_until = _future(lease_seconds)
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE inbound_deliveries
                SET lease_until = ?, updated_at = ?, state_version = state_version + 1
                WHERE delivery_key = ? AND status = 'leased' AND lease_owner = ?
                """,
                (lease_until, now, delivery_key, lease_owner),
            )
        if cursor.rowcount != 1:
            return None
        return self.get_inbound_delivery(delivery_key)

    def transition_inbound(self, delivery_key: str, status: str) -> dict[str, Any]:
        _require_one_of(status, INBOUND_STATUSES, "status")
        row = self.get_inbound_delivery(delivery_key)
        if row is None:
            raise KeyError(delivery_key)
        if (row["status"], status) not in {
            ("leased", "completed"),
            ("leased", "failed"),
            ("leased", "retryable"),
        }:
            raise ValueError(f"invalid inbound transition {row['status']} -> {status}")
        now = _now()
        self.conn.execute(
            """
            UPDATE inbound_deliveries
            SET status = ?, lease_owner = NULL, lease_until = NULL,
                updated_at = ?, state_version = state_version + 1
            WHERE delivery_key = ?
            """,
            (status, now, delivery_key),
        )
        self.conn.commit()
        return self.get_inbound_delivery(delivery_key)  # type: ignore[return-value]

    def transition_inbound_if_leased_by(
        self,
        delivery_key: str,
        status: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        """Transition only when the caller still owns the active lease."""

        _require_one_of(status, INBOUND_STATUSES, "status")
        if status not in {"completed", "failed", "retryable"}:
            raise ValueError(f"invalid owner-guarded inbound transition to {status}")
        row = self.get_inbound_delivery(delivery_key)
        if row is None:
            raise KeyError(delivery_key)
        now = _now()
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE inbound_deliveries
                SET status = ?, lease_owner = NULL, lease_until = NULL,
                    updated_at = ?, state_version = state_version + 1
                WHERE delivery_key = ? AND status = 'leased' AND lease_owner = ?
                """,
                (status, now, delivery_key, lease_owner),
            )
        if cursor.rowcount != 1:
            return None
        return self.get_inbound_delivery(delivery_key)

    # ---- outbox ----------------------------------------------------------

    def create_outbox_delivery(
        self,
        *,
        idempotency_key: str,
        target_agent_id: str,
        topic_id: str,
        channel_id: str,
        delivery_kind: str,
        payload: Any | None = None,
        thread_id: str | None = None,
        source_inbound_delivery_key: str | None = None,
        source_agent_event_id: str | None = None,
        outbox_delivery_id: str | None = None,
    ) -> dict[str, Any]:
        _require_one_of(delivery_kind, ("response", "projection", "diagnostic"), "delivery_kind")
        outbox_delivery_id = outbox_delivery_id or f"outbox:{uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key)}"
        now = _now()
        self.conn.execute(
            """
            INSERT OR IGNORE INTO outbox_deliveries (
                outbox_delivery_id, idempotency_key, target_agent_id, topic_id,
                channel_id, thread_id, source_inbound_delivery_key, source_agent_event_id,
                delivery_kind, payload_json, status, lease_owner, lease_until, attempts,
                created_at, updated_at, state_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, 0, ?, ?, 1)
            """,
            (
                outbox_delivery_id,
                idempotency_key,
                target_agent_id,
                topic_id,
                channel_id,
                thread_id,
                source_inbound_delivery_key,
                source_agent_event_id,
                delivery_kind,
                _json(payload or {}),
                now,
                now,
            ),
        )
        self.conn.commit()
        return self.get_outbox_delivery_by_key(idempotency_key)  # type: ignore[return-value]

    def get_outbox_delivery(self, outbox_delivery_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM outbox_deliveries WHERE outbox_delivery_id = ?",
            (outbox_delivery_id,),
        )

    def get_outbox_delivery_by_key(self, idempotency_key: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM outbox_deliveries WHERE idempotency_key = ?",
            (idempotency_key,),
        )

    def add_outbox_part(
        self,
        *,
        outbox_delivery_id: str,
        part_index: int,
        status: str = "pending",
        discord_message_id: str | None = None,
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO outbox_parts (outbox_delivery_id, part_index, status, discord_message_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(outbox_delivery_id, part_index) DO UPDATE SET
                status = excluded.status,
                discord_message_id = excluded.discord_message_id
            """,
            (outbox_delivery_id, part_index, status, discord_message_id),
        )
        self.conn.commit()
        return self._fetch_one(
            "SELECT * FROM outbox_parts WHERE outbox_delivery_id = ? AND part_index = ?",
            (outbox_delivery_id, part_index),
        )  # type: ignore[return-value]

    def has_fresh_outbox_lease(
        self,
        outbox_delivery_id: str,
        lease_owner: str,
        *,
        status: str | None = None,
    ) -> bool:
        """Return true only while ``lease_owner`` still owns a non-expired outbox lease."""

        now = _now()
        params: list[Any] = [outbox_delivery_id, lease_owner, now]
        status_sql = ""
        if status is not None:
            _require_one_of(status, OUTBOX_STATUSES, "status")
            status_sql = " AND status = ?"
            params.append(status)
        row = self.conn.execute(
            f"""
            SELECT 1 FROM outbox_deliveries
            WHERE outbox_delivery_id = ? AND lease_owner = ? AND lease_until > ?{status_sql}
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        return row is not None

    def lease_next_outbox(
        self,
        *,
        lease_owner: str,
        lease_seconds: int = 60,
    ) -> dict[str, Any] | None:
        now = _now()
        lease_until = _future(lease_seconds)
        for _attempt in range(10):
            with self.conn:
                row = self.conn.execute(
                    """
                    SELECT * FROM outbox_deliveries
                    WHERE status = 'pending'
                       OR (status = 'leased' AND lease_until <= ?)
                    ORDER BY created_at, outbox_delivery_id
                    LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if row is None:
                    return None
                current = dict(row)
                if self._cas_lease_outbox(current, lease_owner, lease_until, now):
                    return self.get_outbox_delivery(current["outbox_delivery_id"])
        return None

    def _cas_lease_outbox(
        self,
        current: dict[str, Any],
        lease_owner: str,
        lease_until: str,
        now: str,
    ) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE outbox_deliveries
            SET status = 'leased', lease_owner = ?, lease_until = ?, attempts = attempts + 1,
                updated_at = ?, state_version = state_version + 1
            WHERE outbox_delivery_id = ? AND state_version = ?
            """,
            (
                lease_owner,
                lease_until,
                now,
                current["outbox_delivery_id"],
                current["state_version"],
            ),
        )
        return cursor.rowcount == 1

    def mark_outbox_sending(self, outbox_delivery_id: str) -> dict[str, Any]:
        return self.transition_outbox(outbox_delivery_id, "sending")

    def mark_outbox_sending_if_leased_by(
        self,
        outbox_delivery_id: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        return self.transition_outbox_if_leased_by(outbox_delivery_id, "sending", lease_owner)

    def mark_outbox_sent(self, outbox_delivery_id: str) -> dict[str, Any]:
        return self.transition_outbox(outbox_delivery_id, "sent")

    def mark_outbox_sent_if_leased_by(
        self,
        outbox_delivery_id: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        return self.transition_outbox_if_leased_by(outbox_delivery_id, "sent", lease_owner)

    def mark_outbox_acked(self, outbox_delivery_id: str) -> dict[str, Any]:
        return self.transition_outbox(outbox_delivery_id, "acked")

    def mark_outbox_uncertain(self, outbox_delivery_id: str) -> dict[str, Any]:
        return self.transition_outbox(outbox_delivery_id, "uncertain")

    def mark_outbox_uncertain_if_leased_by(
        self,
        outbox_delivery_id: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        return self.transition_outbox_if_leased_by(outbox_delivery_id, "uncertain", lease_owner)

    def mark_outbox_reconciled(self, outbox_delivery_id: str) -> dict[str, Any]:
        return self.transition_outbox(outbox_delivery_id, "reconciled")

    def transition_outbox(self, outbox_delivery_id: str, status: str) -> dict[str, Any]:
        _require_one_of(status, OUTBOX_STATUSES, "status")
        row = self.get_outbox_delivery(outbox_delivery_id)
        if row is None:
            raise KeyError(outbox_delivery_id)
        allowed = {
            ("pending", "leased"),
            ("pending", "sending"),
            ("leased", "sending"),
            ("leased", "uncertain"),
            ("leased", "acked"),
            ("sending", "sent"),
            ("sending", "acked"),
            ("sending", "uncertain"),
            ("sent", "acked"),
            ("sent", "reconciled"),
            ("acked", "reconciled"),
            ("uncertain", "leased"),
            ("uncertain", "sending"),
            ("uncertain", "acked"),
            ("uncertain", "reconciled"),
        }
        if (row["status"], status) not in allowed:
            raise ValueError(f"invalid outbox transition {row['status']} -> {status}")
        now = _now()
        lease_owner = None if status != "leased" else row["lease_owner"]
        lease_until = None if status != "leased" else row["lease_until"]
        self.conn.execute(
            """
            UPDATE outbox_deliveries
            SET status = ?, lease_owner = ?, lease_until = ?,
                updated_at = ?, state_version = state_version + 1
            WHERE outbox_delivery_id = ?
            """,
            (status, lease_owner, lease_until, now, outbox_delivery_id),
        )
        self.conn.commit()
        return self.get_outbox_delivery(outbox_delivery_id)  # type: ignore[return-value]

    def transition_outbox_if_leased_by(
        self,
        outbox_delivery_id: str,
        status: str,
        lease_owner: str,
    ) -> dict[str, Any] | None:
        """Transition an outbox row only while the caller owns its lease.

        The sending state intentionally keeps the lease owner attached so all
        terminal sender writes can be guarded after the Discord side effect.
        """

        _require_one_of(status, OUTBOX_STATUSES, "status")
        if status not in {"sending", "sent", "acked", "uncertain"}:
            raise ValueError(f"invalid owner-guarded outbox transition to {status}")
        row = self.get_outbox_delivery(outbox_delivery_id)
        if row is None:
            raise KeyError(outbox_delivery_id)
        allowed = {
            ("leased", "sending"),
            ("leased", "uncertain"),
            ("leased", "acked"),
            ("sending", "sent"),
            ("sending", "acked"),
            ("sending", "uncertain"),
        }
        if (row["status"], status) not in allowed:
            raise ValueError(f"invalid outbox transition {row['status']} -> {status}")
        now = _now()
        next_owner = lease_owner if status == "sending" else None
        next_until = row.get("lease_until") if status == "sending" else None
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE outbox_deliveries
                SET status = ?, lease_owner = ?, lease_until = ?,
                    updated_at = ?, state_version = state_version + 1
                WHERE outbox_delivery_id = ? AND lease_owner = ?
                  AND status IN ('leased', 'sending')
                  AND lease_until > ?
                """,
                (status, next_owner, next_until, now, outbox_delivery_id, lease_owner, now),
            )
        if cursor.rowcount != 1:
            return None
        return self.get_outbox_delivery(outbox_delivery_id)

    # ---- restart-safe stable-ID adjuncts --------------------------------

    def upsert_approval(
        self,
        *,
        approval_id: str,
        target_agent_id: str,
        agent_id: str | None = None,
        topic_id: str,
        requesting_event_id: str | None = None,
        source_inbound_delivery_key: str | None = None,
        source_agent_event_id: str | None = None,
        status: str = "pending",
        payload: Any | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        now = _now()
        effective_agent_id = str(agent_id or target_agent_id)
        effective_requesting_event_id = requesting_event_id or source_agent_event_id or source_inbound_delivery_key
        self.conn.execute(
            """
            INSERT INTO approvals (
                approval_id, source_inbound_delivery_key, source_agent_event_id,
                target_agent_id, agent_id, topic_id, requesting_event_id, status,
                payload_json, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(approval_id) DO UPDATE SET
                source_inbound_delivery_key = excluded.source_inbound_delivery_key,
                source_agent_event_id = excluded.source_agent_event_id,
                target_agent_id = excluded.target_agent_id,
                agent_id = excluded.agent_id,
                topic_id = excluded.topic_id,
                requesting_event_id = excluded.requesting_event_id,
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at,
                version = excluded.version
            WHERE approvals.status = 'pending'
            """,
            (
                approval_id,
                source_inbound_delivery_key,
                source_agent_event_id,
                target_agent_id,
                effective_agent_id,
                topic_id,
                effective_requesting_event_id,
                status,
                _json(payload or {}),
                now,
                now,
                version,
            ),
        )
        self.conn.commit()
        return self._fetch_one("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))  # type: ignore[return-value]

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM approvals WHERE approval_id = ?", (approval_id,))

    def list_approval_audit_events(self, approval_id: str | None = None) -> list[dict[str, Any]]:
        if approval_id is None:
            return self._fetch_all("SELECT * FROM approval_audit_events ORDER BY created_at")
        return self._fetch_all(
            "SELECT * FROM approval_audit_events WHERE approval_id = ? ORDER BY created_at",
            (approval_id,),
        )

    def upsert_handoff(
        self,
        *,
        handoff_id: str,
        agent_event_id: str,
        source_agent_id: str | None,
        target_agent_id: str,
        topic_id: str,
        status: str = "pending",
        payload: Any | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        safe_payload = _redacted(payload or {})
        existing = self.get_handoff(handoff_id)
        if existing is not None:
            expected = {
                "agent_event_id": agent_event_id,
                "source_agent_id": source_agent_id,
                "target_agent_id": target_agent_id,
                "topic_id": topic_id,
            }
            actual = {
                "agent_event_id": existing.get("agent_event_id"),
                "source_agent_id": existing.get("source_agent_id"),
                "target_agent_id": existing.get("target_agent_id"),
                "topic_id": existing.get("topic_id"),
            }
            if actual != expected:
                raise ValueError(f"handoff conflict for immutable handoff {handoff_id!r}")
            if existing.get("status") not in {"pending", "requested"}:
                return existing

        now = _now()
        self.conn.execute(
            """
            INSERT INTO handoffs (
                handoff_id, agent_event_id, source_agent_id, target_agent_id, topic_id,
                status, payload_json, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(handoff_id) DO UPDATE SET
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at,
                version = excluded.version
            WHERE handoffs.status IN ('pending', 'requested')
            """,
            (
                handoff_id,
                agent_event_id,
                source_agent_id,
                target_agent_id,
                topic_id,
                status,
                _json(safe_payload),
                now,
                now,
                version,
            ),
        )
        self.conn.commit()
        return self._fetch_one("SELECT * FROM handoffs WHERE handoff_id = ?", (handoff_id,))  # type: ignore[return-value]

    def get_handoff(self, handoff_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM handoffs WHERE handoff_id = ?", (handoff_id,))

    def get_handoff_by_agent_event(self, agent_event_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM handoffs WHERE agent_event_id = ?",
            (agent_event_id,),
        )

    def update_handoff_status(
        self,
        *,
        handoff_id: str,
        status: str,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        now = _now()
        self.conn.execute(
            """
            UPDATE handoffs
            SET status = ?, payload_json = ?, updated_at = ?, version = version + 1
            WHERE handoff_id = ?
            """,
            (status, _json(payload or {}), now, handoff_id),
        )
        self.conn.commit()
        row = self.get_handoff(handoff_id)
        if row is None:
            raise KeyError(handoff_id)
        return row

    def list_handoffs(
        self,
        *,
        status: str | None = None,
        agent_event_id: str | None = None,
        target_agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if agent_event_id is not None:
            where.append("agent_event_id = ?")
            params.append(agent_event_id)
        if target_agent_id is not None:
            where.append("target_agent_id = ?")
            params.append(target_agent_id)
        sql = "SELECT * FROM handoffs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, handoff_id"
        return self._fetch_all(sql, tuple(params))

    def create_reconciliation_run(
        self,
        *,
        reconciliation_run_id: str | None = None,
        source_agent_event_id: str | None = None,
        outbox_delivery_id: str | None = None,
        status: str = "pending",
        payload: Any | None = None,
        version: int = 1,
    ) -> dict[str, Any]:
        reconciliation_run_id = reconciliation_run_id or f"reconcile:{uuid.uuid4().hex}"
        now = _now()
        self.conn.execute(
            """
            INSERT INTO reconciliation_runs (
                reconciliation_run_id, source_agent_event_id, outbox_delivery_id,
                status, payload_json, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(reconciliation_run_id) DO UPDATE SET
                source_agent_event_id = excluded.source_agent_event_id,
                outbox_delivery_id = excluded.outbox_delivery_id,
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at,
                version = excluded.version
            """,
            (
                reconciliation_run_id,
                source_agent_event_id,
                outbox_delivery_id,
                status,
                _json(payload or {}),
                now,
                now,
                version,
            ),
        )
        self.conn.commit()
        return self._fetch_one(
            "SELECT * FROM reconciliation_runs WHERE reconciliation_run_id = ?",
            (reconciliation_run_id,),
        )  # type: ignore[return-value]

    # ---- generic query helpers for tests and future slices ---------------

    def count_rows(self, table: str, where: str = "", params: tuple[Any, ...] = ()) -> int:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        sql = f"SELECT COUNT(*) AS count FROM {table}"
        if where:
            sql += f" WHERE {where}"
        row = self.conn.execute(sql, params).fetchone()
        return int(row["count"])

    def route_decisions_for(self, source_type: str, source_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM route_decisions WHERE source_type = ? AND source_id = ? ORDER BY decision_id",
            (source_type, source_id),
        )

    def table_columns(self, table: str) -> set[str]:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def _fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]


def default_db_path() -> Path:
    return get_hermes_home() / "gateway" / "discord_protocol_v2.sqlite3"


def _schema_without_indexes() -> str:
    return "\n".join(
        line for line in SCHEMA_SQL.splitlines() if not line.strip().startswith("CREATE INDEX")
    )


def inbound_delivery_key(source_type: str, source_id: str, target_agent_id: str) -> str:
    _require_one_of(source_type, SOURCE_TYPES, "source_type")
    return f"{source_type}:{source_id}:{target_agent_id}"


def response_idempotency_key(inbound_delivery_key_value: str, target_agent_id: str) -> str:
    return f"response:{inbound_delivery_key_value}:{target_agent_id}"


def projection_idempotency_key(agent_event_id: str, target_agent_id: str) -> str:
    return f"projection:{agent_event_id}:{target_agent_id}"


def _json(value: Any) -> str:
    return json.dumps(_redacted(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _redacted(value: Any) -> Any:
    return redact_sensitive_data(value)


def _loads_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _future(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _require_one_of(value: str, allowed: Iterable[str], name: str) -> None:
    allowed_tuple = tuple(allowed)
    if value not in allowed_tuple:
        raise ValueError(f"{name} must be one of: {', '.join(allowed_tuple)}")
