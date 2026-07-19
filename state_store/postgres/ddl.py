"""PostgreSQL-native v22 durable state schema.

The SQLite schema is intentionally not translated at runtime. Keeping this
manifest as authored PostgreSQL DDL makes lifecycle changes reviewable and
prevents SQLite-only syntax from leaking into the backend.
"""

from dataclasses import dataclass
from typing import Tuple


SCHEMA_VERSION = 22


@dataclass(frozen=True)
class PostgresTableDDL:
    """One durable table plus its PostgreSQL-native create statement."""

    name: str
    columns: Tuple[str, ...]
    create_sql: str


@dataclass(frozen=True)
class PostgresColumnUpgrade:
    """One idempotent column addition required by the experimental v16 schema."""

    table: str
    column: str
    add_sql: str


@dataclass(frozen=True)
class PostgresColumnContract:
    """Catalog-level requirements for one authored PostgreSQL column."""

    name: str
    type_name: str
    not_null: bool
    default: str | None
    identity: str | None = None


@dataclass(frozen=True)
class PostgresIndexContract:
    """Catalog-level requirements for one durable PostgreSQL index."""

    name: str
    table: str
    columns: Tuple[Tuple[str, bool], ...]
    unique: bool = False
    predicate: str | None = None


CORE_TABLES = (
    PostgresTableDDL(
        name="schema_version",
        columns=("version",),
        create_sql="""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """,
    ),
    PostgresTableDDL(
        name="sessions",
        columns=(
            "id", "source", "user_id", "session_key", "chat_id", "chat_type",
            "thread_id", "display_name", "origin_json", "expiry_finalized",
            "model", "model_config", "system_prompt", "parent_session_id",
            "started_at", "ended_at", "end_reason", "message_count",
            "tool_call_count", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
            "cwd", "git_branch", "git_repo_root", "billing_provider",
            "billing_base_url", "billing_mode", "estimated_cost_usd",
            "actual_cost_usd", "cost_status", "cost_source", "pricing_version",
            "title", "api_call_count", "handoff_state", "handoff_platform",
            "handoff_error", "compression_failure_cooldown_until",
            "compression_failure_error", "compression_fallback_streak",
            "profile_name", "rewind_count", "archived",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                session_key TEXT,
                chat_id TEXT,
                chat_type TEXT,
                thread_id TEXT,
                display_name TEXT,
                origin_json TEXT,
                expiry_finalized INTEGER DEFAULT 0,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT REFERENCES sessions(id),
                started_at DOUBLE PRECISION NOT NULL,
                ended_at DOUBLE PRECISION,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                cwd TEXT,
                git_branch TEXT,
                git_repo_root TEXT,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd DOUBLE PRECISION,
                actual_cost_usd DOUBLE PRECISION,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                handoff_state TEXT,
                handoff_platform TEXT,
                handoff_error TEXT,
                compression_failure_cooldown_until DOUBLE PRECISION,
                compression_failure_error TEXT,
                compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
                profile_name TEXT,
                rewind_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0
            )
        """,
    ),
    PostgresTableDDL(
        name="messages",
        columns=(
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls",
            "tool_name", "effect_disposition", "timestamp", "token_count",
            "finish_reason", "reasoning", "reasoning_content", "reasoning_details",
            "codex_reasoning_items", "codex_message_items", "platform_message_id",
            "observed", "active", "compacted",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                effect_disposition TEXT,
                timestamp DOUBLE PRECISION NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                platform_message_id TEXT,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            )
        """,
    ),
    PostgresTableDDL(
        name="session_model_usage",
        columns=(
            "session_id", "model", "billing_provider", "billing_base_url",
            "billing_mode", "task", "api_call_count", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd",
            "cost_status", "cost_source", "first_seen", "last_seen",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS session_model_usage (
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                billing_provider TEXT NOT NULL DEFAULT '',
                billing_base_url TEXT NOT NULL DEFAULT '',
                billing_mode TEXT NOT NULL DEFAULT '',
                task TEXT NOT NULL DEFAULT '',
                api_call_count INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
                actual_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
                cost_status TEXT,
                cost_source TEXT,
                first_seen DOUBLE PRECISION,
                last_seen DOUBLE PRECISION,
                PRIMARY KEY (
                    session_id, model, billing_provider, billing_base_url,
                    billing_mode, task
                )
            )
        """,
    ),
    PostgresTableDDL(
        name="state_meta",
        columns=("key", "value"),
        create_sql="""
            CREATE TABLE IF NOT EXISTS state_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """,
    ),
    PostgresTableDDL(
        name="gateway_routing",
        columns=("scope", "session_key", "entry_json", "updated_at"),
        create_sql="""
            CREATE TABLE IF NOT EXISTS gateway_routing (
                scope TEXT NOT NULL DEFAULT '',
                session_key TEXT NOT NULL,
                entry_json TEXT NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (scope, session_key)
            )
        """,
    ),
    PostgresTableDDL(
        name="compression_locks",
        columns=("session_id", "holder", "acquired_at", "expires_at"),
        create_sql="""
            CREATE TABLE IF NOT EXISTS compression_locks (
                session_id TEXT PRIMARY KEY,
                holder TEXT NOT NULL,
                acquired_at DOUBLE PRECISION NOT NULL,
                expires_at DOUBLE PRECISION NOT NULL
            )
        """,
    ),
    PostgresTableDDL(
        name="async_delegations",
        columns=(
            "delegation_id", "origin_session", "origin_ui_session_id",
            "parent_session_id", "state", "dispatched_at", "completed_at",
            "updated_at", "event_json", "result_json", "delivery_state",
            "delivery_attempts", "delivered_at", "owner_pid", "owner_started_at",
            "task_json", "delivery_claim", "delivery_claimed_at",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS async_delegations (
                delegation_id TEXT PRIMARY KEY,
                origin_session TEXT NOT NULL,
                origin_ui_session_id TEXT NOT NULL DEFAULT '',
                parent_session_id TEXT,
                state TEXT NOT NULL,
                dispatched_at DOUBLE PRECISION NOT NULL,
                completed_at DOUBLE PRECISION,
                updated_at DOUBLE PRECISION NOT NULL,
                event_json TEXT,
                result_json TEXT,
                delivery_state TEXT NOT NULL DEFAULT 'pending',
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                delivered_at DOUBLE PRECISION,
                owner_pid INTEGER,
                owner_started_at INTEGER,
                task_json TEXT,
                delivery_claim TEXT,
                delivery_claimed_at DOUBLE PRECISION
            )
        """,
    ),
)

TELEGRAM_TABLES = (
    PostgresTableDDL(
        name="telegram_dm_topic_mode",
        columns=(
            "chat_id", "user_id", "enabled", "activated_at", "updated_at",
            "has_topics_enabled", "allows_users_to_create_topics",
            "capability_checked_at", "intro_message_id", "pinned_message_id",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                activated_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                has_topics_enabled INTEGER,
                allows_users_to_create_topics INTEGER,
                capability_checked_at DOUBLE PRECISION,
                intro_message_id TEXT,
                pinned_message_id TEXT
            )
        """,
    ),
    PostgresTableDDL(
        name="telegram_dm_topic_bindings",
        columns=(
            "chat_id", "thread_id", "user_id", "session_key", "session_id",
            "managed_mode", "linked_at", "updated_at",
        ),
        create_sql="""
            CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
                chat_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                managed_mode TEXT NOT NULL DEFAULT 'auto',
                linked_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (chat_id, thread_id)
            )
        """,
    ),
)

_INTEGER_COLUMNS = {
    "schema_version": {"version"},
    "sessions": {
        "expiry_finalized",
        "message_count",
        "tool_call_count",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "api_call_count",
        "compression_fallback_streak",
        "rewind_count",
        "archived",
    },
    "messages": {"token_count", "observed", "active", "compacted"},
    "session_model_usage": {
        "api_call_count",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    },
    "async_delegations": {"delivery_attempts", "owner_pid", "owner_started_at"},
    "telegram_dm_topic_mode": {
        "enabled",
        "has_topics_enabled",
        "allows_users_to_create_topics",
    },
}
_DOUBLE_COLUMNS = {
    "sessions": {
        "started_at",
        "ended_at",
        "estimated_cost_usd",
        "actual_cost_usd",
        "compression_failure_cooldown_until",
    },
    "messages": {"timestamp"},
    "session_model_usage": {
        "estimated_cost_usd",
        "actual_cost_usd",
        "first_seen",
        "last_seen",
    },
    "gateway_routing": {"updated_at"},
    "compression_locks": {"acquired_at", "expires_at"},
    "async_delegations": {
        "dispatched_at",
        "completed_at",
        "updated_at",
        "delivered_at",
        "delivery_claimed_at",
    },
    "telegram_dm_topic_mode": {
        "activated_at",
        "updated_at",
        "capability_checked_at",
    },
    "telegram_dm_topic_bindings": {"linked_at", "updated_at"},
}
_BIGINT_COLUMNS = {"messages": {"id"}}
_NOT_NULL_COLUMNS = {
    "schema_version": {"version"},
    "sessions": {
        "id",
        "source",
        "started_at",
        "compression_fallback_streak",
        "rewind_count",
        "archived",
    },
    "messages": {"id", "session_id", "role", "timestamp", "active", "compacted"},
    "session_model_usage": {
        "session_id",
        "model",
        "billing_provider",
        "billing_base_url",
        "billing_mode",
        "task",
        "api_call_count",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "estimated_cost_usd",
        "actual_cost_usd",
    },
    "state_meta": {"key"},
    "gateway_routing": {"scope", "session_key", "entry_json", "updated_at"},
    "compression_locks": {"session_id", "holder", "acquired_at", "expires_at"},
    "async_delegations": {
        "delegation_id",
        "origin_session",
        "origin_ui_session_id",
        "state",
        "dispatched_at",
        "updated_at",
        "delivery_state",
        "delivery_attempts",
    },
    "telegram_dm_topic_mode": {
        "chat_id",
        "user_id",
        "enabled",
        "activated_at",
        "updated_at",
    },
    "telegram_dm_topic_bindings": {
        "chat_id",
        "thread_id",
        "user_id",
        "session_key",
        "session_id",
        "managed_mode",
        "linked_at",
        "updated_at",
    },
}
_COLUMN_DEFAULTS = {
    "sessions": {
        "expiry_finalized": "0",
        "message_count": "0",
        "tool_call_count": "0",
        "input_tokens": "0",
        "output_tokens": "0",
        "cache_read_tokens": "0",
        "cache_write_tokens": "0",
        "reasoning_tokens": "0",
        "api_call_count": "0",
        "compression_fallback_streak": "0",
        "rewind_count": "0",
        "archived": "0",
    },
    "messages": {"observed": "0", "active": "1", "compacted": "0"},
    "session_model_usage": {
        "billing_provider": "''",
        "billing_base_url": "''",
        "billing_mode": "''",
        "task": "''",
        "api_call_count": "0",
        "input_tokens": "0",
        "output_tokens": "0",
        "cache_read_tokens": "0",
        "cache_write_tokens": "0",
        "reasoning_tokens": "0",
        "estimated_cost_usd": "0",
        "actual_cost_usd": "0",
    },
    "gateway_routing": {"scope": "''"},
    "async_delegations": {
        "origin_ui_session_id": "''",
        "delivery_state": "'pending'",
        "delivery_attempts": "0",
    },
    "telegram_dm_topic_mode": {"enabled": "1"},
    "telegram_dm_topic_bindings": {"managed_mode": "'auto'"},
}
_IDENTITY_COLUMNS = {"messages": {"id": "BY DEFAULT"}}


def _column_type(table_name: str, column_name: str) -> str:
    if column_name in _BIGINT_COLUMNS.get(table_name, set()):
        return "bigint"
    if column_name in _DOUBLE_COLUMNS.get(table_name, set()):
        return "double precision"
    if column_name in _INTEGER_COLUMNS.get(table_name, set()):
        return "integer"
    return "text"


def _column_contracts(table: PostgresTableDDL) -> Tuple[PostgresColumnContract, ...]:
    defaults = _COLUMN_DEFAULTS.get(table.name, {})
    identities = _IDENTITY_COLUMNS.get(table.name, {})
    not_null = _NOT_NULL_COLUMNS.get(table.name, set())
    return tuple(
        PostgresColumnContract(
            name=column_name,
            type_name=_column_type(table.name, column_name),
            not_null=column_name in not_null,
            default=defaults.get(column_name),
            identity=identities.get(column_name),
        )
        for column_name in table.columns
    )


POSTGRES_COLUMN_CONTRACTS = {
    table.name: _column_contracts(table) for table in CORE_TABLES + TELEGRAM_TABLES
}

V16_TO_V22_COLUMN_UPGRADES = (
    PostgresColumnUpgrade(
        "sessions",
        "session_key",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_key TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "chat_id",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS chat_id TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "chat_type",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS chat_type TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "thread_id",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS thread_id TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "display_name",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS display_name TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "origin_json",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS origin_json TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "expiry_finalized",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS expiry_finalized INTEGER DEFAULT 0",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "git_branch",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS git_branch TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "git_repo_root",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS git_repo_root TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "compression_failure_cooldown_until",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "
        "compression_failure_cooldown_until DOUBLE PRECISION",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "compression_failure_error",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS compression_failure_error TEXT",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "compression_fallback_streak",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS "
        "compression_fallback_streak INTEGER NOT NULL DEFAULT 0",
    ),
    PostgresColumnUpgrade(
        "sessions",
        "profile_name",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS profile_name TEXT",
    ),
    PostgresColumnUpgrade(
        "messages",
        "effect_disposition",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS effect_disposition TEXT",
    ),
    PostgresColumnUpgrade(
        "messages",
        "compacted",
        "ALTER TABLE messages ADD COLUMN IF NOT EXISTS "
        "compacted INTEGER NOT NULL DEFAULT 0",
    ),
)

CORE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model)",
    "CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery "
    "ON async_delegations(delivery_state, completed_at)",
    "CREATE INDEX IF NOT EXISTS idx_messages_session_active "
    "ON messages(session_id, active, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_messages_active_null ON messages(active) "
    "WHERE active IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_sessions_session_key "
    "ON sessions(session_key, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer "
    "ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state "
    "ON sessions(handoff_state, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
    "ON messages(session_id, platform_message_id) "
    "WHERE platform_message_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
    "ON sessions(title) WHERE title IS NOT NULL",
)

TELEGRAM_INDEXES = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session "
    "ON telegram_dm_topic_bindings(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user "
    "ON telegram_dm_topic_bindings(user_id, chat_id)",
)

CORE_INDEX_CONTRACTS = (
    PostgresIndexContract("idx_sessions_source", "sessions", (("source", False),)),
    PostgresIndexContract(
        "idx_sessions_source_id", "sessions", (("source", False), ("id", False))
    ),
    PostgresIndexContract(
        "idx_sessions_parent", "sessions", (("parent_session_id", False),)
    ),
    PostgresIndexContract(
        "idx_sessions_started", "sessions", (("started_at", True),)
    ),
    PostgresIndexContract(
        "idx_messages_session",
        "messages",
        (("session_id", False), ("timestamp", False)),
    ),
    PostgresIndexContract(
        "idx_compression_locks_expires",
        "compression_locks",
        (("expires_at", False),),
    ),
    PostgresIndexContract(
        "idx_session_model_usage_session",
        "session_model_usage",
        (("session_id", False),),
    ),
    PostgresIndexContract(
        "idx_session_model_usage_model",
        "session_model_usage",
        (("model", False),),
    ),
    PostgresIndexContract(
        "idx_async_delegations_delivery",
        "async_delegations",
        (("delivery_state", False), ("completed_at", False)),
    ),
    PostgresIndexContract(
        "idx_messages_session_active",
        "messages",
        (("session_id", False), ("active", False), ("timestamp", False)),
    ),
    PostgresIndexContract(
        "idx_messages_active_null",
        "messages",
        (("active", False),),
        predicate="active IS NULL",
    ),
    PostgresIndexContract(
        "idx_sessions_session_key",
        "sessions",
        (("session_key", False), ("started_at", True)),
    ),
    PostgresIndexContract(
        "idx_sessions_gateway_peer",
        "sessions",
        (
            ("source", False),
            ("user_id", False),
            ("chat_id", False),
            ("chat_type", False),
            ("thread_id", False),
            ("started_at", True),
        ),
    ),
    PostgresIndexContract(
        "idx_sessions_handoff_state",
        "sessions",
        (("handoff_state", False), ("started_at", False)),
    ),
    PostgresIndexContract(
        "idx_messages_platform_msg_id",
        "messages",
        (("session_id", False), ("platform_message_id", False)),
        predicate="platform_message_id IS NOT NULL",
    ),
    PostgresIndexContract(
        "idx_sessions_title_unique",
        "sessions",
        (("title", False),),
        unique=True,
        predicate="title IS NOT NULL",
    ),
)
TELEGRAM_INDEX_CONTRACTS = (
    PostgresIndexContract(
        "idx_telegram_dm_topic_bindings_session",
        "telegram_dm_topic_bindings",
        (("session_id", False),),
        unique=True,
    ),
    PostgresIndexContract(
        "idx_telegram_dm_topic_bindings_user",
        "telegram_dm_topic_bindings",
        (("user_id", False), ("chat_id", False)),
    ),
)


def schema_statements(*, include_telegram: bool = False) -> Tuple[str, ...]:
    """Return idempotent PostgreSQL DDL for the requested durable tables."""

    tables = CORE_TABLES + (TELEGRAM_TABLES if include_telegram else ())
    indexes = CORE_INDEXES + (TELEGRAM_INDEXES if include_telegram else ())
    return tuple(table.create_sql for table in tables) + indexes


def schema_table_statements(*, include_telegram: bool = False) -> Tuple[str, ...]:
    """Return create-table statements separately from index creation."""

    tables = CORE_TABLES + (TELEGRAM_TABLES if include_telegram else ())
    return tuple(table.create_sql for table in tables)
