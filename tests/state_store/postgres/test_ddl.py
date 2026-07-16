from state_store.postgres import (
    CORE_INDEXES,
    CORE_TABLES,
    TELEGRAM_INDEXES,
    TELEGRAM_TABLES,
    ddl_table_names,
    schema_manifest_matches_postgres_ddl,
    schema_statements,
)
from state_store.schema import SCHEMA_V22_MANIFEST


def _index_names(statements: tuple[str, ...]) -> set[str]:
    return {
        statement.split("INDEX IF NOT EXISTS ", 1)[1].split(" ", 1)[0]
        for statement in statements
    }


def test_v22_ddl_has_exact_manifest_table_and_column_contract() -> None:
    # Literal v22 contract from SCHEMA_SQL and the opt-in Telegram migration.
    expected_columns = {
        "schema_version": ("version",),
        "sessions": (
            "id",
            "source",
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "thread_id",
            "display_name",
            "origin_json",
            "expiry_finalized",
            "model",
            "model_config",
            "system_prompt",
            "parent_session_id",
            "started_at",
            "ended_at",
            "end_reason",
            "message_count",
            "tool_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "cwd",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
            "api_call_count",
            "handoff_state",
            "handoff_platform",
            "handoff_error",
            "compression_failure_cooldown_until",
            "compression_failure_error",
            "compression_fallback_streak",
            "profile_name",
            "rewind_count",
            "archived",
        ),
        "messages": (
            "id",
            "session_id",
            "role",
            "content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "effect_disposition",
            "timestamp",
            "token_count",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "platform_message_id",
            "observed",
            "active",
            "compacted",
        ),
        "session_model_usage": (
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
            "cost_status",
            "cost_source",
            "first_seen",
            "last_seen",
        ),
        "state_meta": ("key", "value"),
        "gateway_routing": ("scope", "session_key", "entry_json", "updated_at"),
        "compression_locks": ("session_id", "holder", "acquired_at", "expires_at"),
        "async_delegations": (
            "delegation_id",
            "origin_session",
            "origin_ui_session_id",
            "parent_session_id",
            "state",
            "dispatched_at",
            "completed_at",
            "updated_at",
            "event_json",
            "result_json",
            "delivery_state",
            "delivery_attempts",
            "delivered_at",
            "owner_pid",
            "owner_started_at",
            "task_json",
            "delivery_claim",
            "delivery_claimed_at",
        ),
        "telegram_dm_topic_mode": (
            "chat_id",
            "user_id",
            "enabled",
            "activated_at",
            "updated_at",
            "has_topics_enabled",
            "allows_users_to_create_topics",
            "capability_checked_at",
            "intro_message_id",
            "pinned_message_id",
        ),
        "telegram_dm_topic_bindings": (
            "chat_id",
            "thread_id",
            "user_id",
            "session_key",
            "session_id",
            "managed_mode",
            "linked_at",
            "updated_at",
        ),
    }

    assert {
        table.name: table.columns for table in CORE_TABLES + TELEGRAM_TABLES
    } == expected_columns
    assert ddl_table_names() == SCHEMA_V22_MANIFEST.core_tables
    assert ddl_table_names(include_telegram=True) == (
        SCHEMA_V22_MANIFEST.core_tables | SCHEMA_V22_MANIFEST.telegram_tables
    )
    assert schema_manifest_matches_postgres_ddl()


def test_v22_ddl_has_all_indexes_and_no_sqlite_features() -> None:
    assert _index_names(CORE_INDEXES + TELEGRAM_INDEXES) == {
        "idx_sessions_source",
        "idx_sessions_source_id",
        "idx_sessions_parent",
        "idx_sessions_started",
        "idx_messages_session",
        "idx_compression_locks_expires",
        "idx_session_model_usage_session",
        "idx_session_model_usage_model",
        "idx_async_delegations_delivery",
        "idx_messages_session_active",
        "idx_messages_active_null",
        "idx_sessions_session_key",
        "idx_sessions_gateway_peer",
        "idx_sessions_handoff_state",
        "idx_messages_platform_msg_id",
        "idx_sessions_title_unique",
        "idx_telegram_dm_topic_bindings_session",
        "idx_telegram_dm_topic_bindings_user",
    }

    core_statements = schema_statements()
    all_statements = schema_statements(include_telegram=True)
    assert len(all_statements) == len(core_statements) + len(TELEGRAM_TABLES) + len(
        TELEGRAM_INDEXES
    )
    assert all(table.create_sql not in core_statements for table in TELEGRAM_TABLES)
    assert all(index not in core_statements for index in TELEGRAM_INDEXES)

    ddl = "\n".join(all_statements).lower()
    for forbidden in (
        "sqlite",
        "pragma",
        "autoincrement",
        "rowid",
        "fts5",
        "virtual table",
        "begin immediate",
        "insert or ignore",
        "insert or replace",
    ):
        assert forbidden not in ddl
    assert "generated by default as identity" in ddl
