"""Real PostgreSQL contract for the first session/message State Store slice."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, cast

import pytest

from state_store import (
    MessageRecord, StateStoreConfigurationError, contextual_session_search_store, open_state_store,
    resolve_contextual_session_search_store,
)
from hermes_state import SessionDB
from tools.session_search_tool import session_search
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget


_DSN_ENV = "HERMES_STATE_STORE_TEST_DSN"
_SCHEMA = ""
_TARGET: OwnedPostgreSQLTestTarget | None = None


def _config() -> dict[str, object]:
    return {
        "state_store": {
            "backend": "postgresql",
            "postgresql": {
                "dsn_env": _DSN_ENV,
                "connect_timeout_seconds": 5,
                "pool_max_size": 2,
            },
        },
    }


def _psycopg():
    return importlib.import_module("psycopg")


@pytest.fixture(autouse=True)
def owned_postgresql_schema(monkeypatch, postgresql_test_target):
    """Route every StateStore factory acquisition to one marked disposable schema."""
    global _SCHEMA, _TARGET
    _SCHEMA, _TARGET = postgresql_test_target.schema, postgresql_test_target
    monkeypatch.setenv(_DSN_ENV, postgresql_test_target.dsn)
    import state_store
    original_schema = state_store.postgresql_tenant_schema
    targets: dict[str, OwnedPostgreSQLTestTarget] = {}

    def owned_schema(*args: object, **kwargs: object) -> str:
        trusted_schema = original_schema(*args, **kwargs)
        if not targets:
            targets[trusted_schema] = postgresql_test_target
        elif trusted_schema not in targets:
            targets[trusted_schema] = OwnedPostgreSQLTestTarget(postgresql_test_target.dsn).allocate()
        return targets[trusted_schema].schema

    monkeypatch.setattr(state_store, "postgresql_tenant_schema", owned_schema)
    yield
    for target in targets.values():
        if target is not postgresql_test_target:
            target.drop()
    _TARGET = None


def _target() -> OwnedPostgreSQLTestTarget:
    assert _TARGET is not None
    return _TARGET


def _reset_schema(_dsn: str) -> None:
    """Per-test fixture allocation owns schema lifecycle; never drop a shared schema here."""
    _target().reset()


def _seed_v2_schema(dsn: str, ledger_versions: tuple[int, ...]) -> None:
    """Seed the actual v1/v2 shape, optionally with historical ledger rows."""
    for statement, parameters in (
        (f"CREATE TABLE {_SCHEMA}.schema_migrations (version integer PRIMARY KEY, applied_at double precision NOT NULL)", ()),
        (f"CREATE TABLE {_SCHEMA}.sessions (id text PRIMARY KEY, source text NOT NULL, started_at double precision NOT NULL, ended_at double precision, end_reason text)", ()),
        (f"CREATE TABLE {_SCHEMA}.messages (id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, session_id text NOT NULL REFERENCES {_SCHEMA}.sessions(id), role text NOT NULL, content text, created_at double precision NOT NULL)", ()),
        (f"CREATE INDEX messages_session_id_id ON {_SCHEMA}.messages (session_id, id)", ()),
    ):
        _target().execute(statement, parameters)
    for column, type_name in (("user_id", "text"), ("session_key", "text"), ("chat_id", "text"), ("chat_type", "text"), ("thread_id", "text"), ("display_name", "text"), ("origin_json", "text"), ("model", "text"), ("model_config", "jsonb"), ("parent_session_id", "text"), ("cwd", "text"), ("profile_name", "text"), ("git_repo_root", "text")):
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN {column} {type_name}")
    _target().execute(f"CREATE INDEX sessions_source_session_key ON {_SCHEMA}.sessions (source, session_key)")
    _target().execute(f"CREATE INDEX sessions_parent_session_id ON {_SCHEMA}.sessions (parent_session_id)")
    for version in ledger_versions:
        _target().execute(f"INSERT INTO {_SCHEMA}.schema_migrations (version, applied_at) VALUES (%s, 1)", (version,))


def _migration_versions(dsn: str) -> list[int]:
    with _psycopg().connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT version FROM {_SCHEMA}.schema_migrations ORDER BY version")
        return [int(row[0]) for row in cursor.fetchall()]


def test_postgresql_store_persists_session_messages_and_closes_pool(monkeypatch):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    store = open_state_store(_config())
    session_id = f"state-store-slice-{uuid.uuid4()}"
    try:
        assert store.ensure_session(session_id, source="integration") == session_id
        first_id = store.append_message(session_id, role="user", content="first")
        second_id = store.append_message(session_id, role="assistant", content="second")
        assert second_id > first_id
        assert [(message["role"], message["content"]) for message in store.get_messages(session_id)] == [
            ("user", "first"),
            ("assistant", "second"),
        ]
        store.end_session(session_id, "integration_complete")
        assert store.get_session(session_id)["end_reason"] == "integration_complete"
    finally:
        store.close()


def test_sqlite_and_postgresql_preserve_session_creation_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    metadata = {
        "user_id": "user", "session_key": "key", "chat_id": "chat", "chat_type": "direct",
        "thread_id": "thread", "display_name": "Display", "origin_json": "{\"platform\": \"test\"}",
        "model": "model", "model_config": {"provider": "test"}, "parent_session_id": None,
        "cwd": "/work", "profile_name": "test", "git_repo_root": "/repo",
    }
    stores = (
        open_state_store({}, db_path=tmp_path / "state.db"),
        open_state_store(_config()),
    )
    sessions = []
    try:
        for store in stores:
            session_id = f"state-store-metadata-{uuid.uuid4()}"
            parent_id = f"state-store-parent-{uuid.uuid4()}"
            store.ensure_session(parent_id, source="integration")
            store.ensure_session(session_id, source="integration", metadata={"user_id": "user"})
            store.ensure_session(session_id, source="changed", metadata={**metadata, "parent_session_id": parent_id})
            session = store.get_session(session_id)
            assert session is not None
            sessions.append(session)
        assert [session["source"] for session in sessions] == ["integration", "integration"]
        expected_without_model_config = {key: value for key, value in metadata.items() if key != "model_config"}
        assert [{key: session[key] for key in expected_without_model_config} for session in sessions] == [
            {**expected_without_model_config, "parent_session_id": session["parent_session_id"]}
            for session in sessions
        ]
        assert [json.loads(session["model_config"]) if isinstance(session["model_config"], str) else session["model_config"]
                for session in sessions] == [metadata["model_config"]] * 2
        for store in stores:
            with pytest.raises(Exception):
                store.ensure_session(
                    f"state-store-orphan-{uuid.uuid4()}", source="integration",
                    metadata={"parent_session_id": f"missing-parent-{uuid.uuid4()}"},
                )
    finally:
        for store in stores:
            store.close()


def test_postgresql_store_rejects_missing_driver_before_open(monkeypatch):
    monkeypatch.delenv(_DSN_ENV, raising=False)
    # A missing secret is rejected before a connection attempt and never falls back to SQLite.
    try:
        open_state_store(_config())
    except Exception as exc:
        assert _DSN_ENV in str(exc)
    else:
        raise AssertionError("PostgreSQL selection unexpectedly opened a fallback store")


def test_sqlite_and_postgresql_title_contract_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    try:
        for store in stores:
            root, tip = f"title-root-{uuid.uuid4()}", f"title-tip-{uuid.uuid4()}"
            base_title = f"Project Plan {uuid.uuid4()}"
            store.ensure_session(root, source="integration")
            store.end_session(root, "compression")
            store.ensure_session(tip, source="integration", metadata={"parent_session_id": root})
            assert store.set_session_title(root, base_title)
            assert store.set_session_title(tip, f"{base_title} #2")
            assert store.set_session_title(tip, base_title)
            assert store.get_session_title(root) is None
            assert store.get_session_title(tip) == base_title
            assert store.get_session_by_title(base_title)["id"] == tip
            assert store.resolve_session_by_title(base_title) == tip
            assert store.get_next_title_in_lineage(base_title) == f"{base_title} #2"
            derived_title, llm_title = f"Derived {uuid.uuid4()}", f"LLM {uuid.uuid4()}"
            assert store.set_auto_title(tip, derived_title, source="derived") is False
            assert store.set_session_title_source(tip, "derived")
            assert store.set_auto_title(tip, llm_title, source="llm")
            assert store.get_session_title_source(tip) == "llm"
            canonical = store.get_session_by_title("Bot Chat")
            if canonical is None:
                ordinary = f"title-ordinary-{uuid.uuid4()}"
                store.ensure_session(ordinary, source="integration")
                assert store.set_session_title(ordinary, "Bot Chat")
                assert store.set_session_hidden(ordinary, True)
            else:
                ordinary = canonical["id"]
                assert bool(canonical["hidden"]) is True
            with pytest.raises(ValueError, match="canonical Bot Chat"):
                store.set_session_title(ordinary, "renamed")
            surrogate_id = f"title-surrogate-{uuid.uuid4()}"
            store.ensure_session(surrogate_id, source="integration")
            assert store.set_session_title(surrogate_id, "\ud800 Clean\u200b Title")
            assert store.get_session_title(surrogate_id) == "� Clean Title"
            observations.append({
                "tip_source": store.get_session_title_source(tip),
                "canonical_title": store.get_session_title(ordinary),
                "canonical_hidden": bool(store.get_session(ordinary)["hidden"]),
            })
        assert observations == [{"tip_source": "llm", "canonical_title": "Bot Chat", "canonical_hidden": True}] * 2
    finally:
        for store in stores:
            store.close()


def test_sqlite_and_postgresql_visibility_summary_contract_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    root, tip = f"visibility-root-{uuid.uuid4()}", f"visibility-tip-{uuid.uuid4()}"
    hidden, archived, excluded, pinned = (f"visibility-{name}-{uuid.uuid4()}" for name in ("hidden", "archived", "excluded", "pinned"))
    try:
        for store in stores:
            store.ensure_session(root, source="visible")
            store.append_message(root, role="user", content="root")
            store.end_session(root, "compression")
            store.ensure_session(tip, source="visible", metadata={"parent_session_id": root})
            store.append_message(tip, role="assistant", content="tip")
            store.ensure_session(hidden, source="visible")
            store.ensure_session(archived, source="visible")
            store.ensure_session(excluded, source="excluded")
            store.ensure_session(pinned, source="visible")
            assert store.set_session_hidden(hidden, True)
            assert store.set_session_archived(root, True)
            assert store.set_session_pinned(pinned, True)
            assert store.set_session_pinned(hidden, True)
            assert store.get_session(root)["archived"] is True
            assert store.get_session(tip)["archived"] is True
            assert store.get_session(hidden)["hidden"] is False
            assert store.set_session_archived("missing", True) is False
            assert store.set_session_pinned("missing", True) is False
            normal = store.list_session_summaries(source="visible", limit=2)
            with_pins = store.list_session_summaries(source="visible", limit=2, include_pinned=True)
            archived_rows = store.list_session_summaries(source="visible", archived_only=True, include_hidden=True)
            visible_rows = store.list_session_summaries(source="visible", exclude_sources=("excluded",), include_hidden=True)
            all_rows = store.list_session_summaries(source="visible", include_archived=True, include_hidden=True)
            observations.append({
                "normal_ids": [row["id"] for row in normal],
                "with_pins_ids": [row["id"] for row in with_pins],
                "archived_ids": sorted(row["id"] for row in archived_rows),
                "visible_ids": sorted(row["id"] for row in visible_rows),
                "tip_message_count": next(row["message_count"] for row in all_rows if row["id"] == tip),
                "boolean_types": [type(row[key]) is bool for row in with_pins for key in ("hidden", "archived", "pinned")],
            })
        assert observations[0] == observations[1]
        assert observations[0]["normal_ids"] == observations[0]["with_pins_ids"][:2]
        assert pinned in observations[0]["with_pins_ids"]
        assert observations[0]["archived_ids"] == sorted([root, tip])
        assert excluded not in observations[0]["visible_ids"]
        assert observations[0]["tip_message_count"] == 1
        assert all(observations[0]["boolean_types"])
    finally:
        for store in stores:
            store.close()


def test_sqlite_and_postgresql_active_message_record_contract_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    expected_records = [
        MessageRecord(
            role="assistant", content=[{"type": "text", "text": "structured"}],
            tool_call_id="call-1", tool_calls=[{"id": "call-1", "type": "function"}], tool_name="tool",
            effect_disposition="applied", timestamp=1234.5, token_count=17, finish_reason="stop",
            reasoning="because", reasoning_content="details", reasoning_details={"trace": [1]},
            codex_reasoning_items=[{"kind": "reasoning"}], codex_message_items=[{"kind": "message"}],
            platform_message_id="platform-1", observed=True, _compressed_summary=True, api_content="exact api",
            display_kind="tool_result", display_metadata={"task_count": 1},
        ),
        MessageRecord(role="tool", content=None, timestamp=1235.5),
    ]
    observations = []
    try:
        for store in stores:
            session_id = f"message-record-{uuid.uuid4()}"
            store.ensure_session(session_id, source="integration")
            assert store.append_message_records(session_id, expected_records) == 2
            records = store.get_message_records(session_id)
            assert [record["id"] for record in records] == sorted(record["id"] for record in records)
            observations.append([{key: value for key, value in record.items() if key not in {"id", "session_id"}} for record in records])
        assert observations[0] == observations[1]
        assert observations[0][0]["content"] == expected_records[0].content
        assert observations[0][1]["content"] is None
    finally:
        for store in stores:
            store.close()


def test_message_record_batch_failure_is_atomic_for_both_backends(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    try:
        for store in stores:
            session_id = f"message-record-atomic-{uuid.uuid4()}"
            with pytest.raises(Exception):
                store.append_message_records(session_id, [MessageRecord(role="user", content="must rollback")])
            assert store.get_message_records(session_id) == []
    finally:
        for store in stores:
            store.close()


def test_sqlite_and_postgresql_content_addressed_system_prompt_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    prompt = "System policy\\n" + ("cache-safe instruction\\n" * 4)
    observations = []
    try:
        for store in stores:
            first, second = f"prompt-first-{uuid.uuid4()}", f"prompt-second-{uuid.uuid4()}"
            store.ensure_session(first, source="integration")
            store.ensure_session(second, source="integration")
            store.set_system_prompt(first, prompt)
            store.set_system_prompt(second, prompt)
            assert store.get_system_prompt(first) == prompt
            assert store.get_session(first)["system_prompt"] == prompt
            store.set_system_prompt(first, None)
            store.set_system_prompt(f"missing-{uuid.uuid4()}", "orphan must roll back")
            assert store.get_system_prompt(f"missing-{uuid.uuid4()}") is None
            observations.append((store.get_system_prompt(first), store.get_system_prompt(second)))
        assert observations == [(None, prompt), (None, prompt)]
    finally:
        for store in stores:
            store.close()


def test_postgresql_fresh_migration_contract_is_linear_idempotent_and_catalog_complete(monkeypatch):
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    try:
        store = open_state_store(_config())
        store.close()
        assert _migration_versions(dsn) == list(range(1, 26))
        with _psycopg().connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{_SCHEMA}' AND table_name = 'sessions'")
            columns = {row[0] for row in cursor.fetchall()}
            assert {"id", "source", "started_at", "last_activity_at", "parent_session_id", "system_prompt_hash", "title", "title_source", "hidden", "archived", "pinned", "git_branch", "git_metadata_generation"} <= columns
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{_SCHEMA}' AND table_name = 'system_prompts'")
            assert {"hash", "prompt"} <= {row[0] for row in cursor.fetchall()}
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema = '{_SCHEMA}' AND table_name = 'messages'")
            message_columns = {row[0] for row in cursor.fetchall()}
            assert {"tool_calls", "reasoning_details", "display_metadata", "active", "compacted", "search_document"} <= message_columns
            cursor.execute(f"SELECT indexname FROM pg_indexes WHERE schemaname = '{_SCHEMA}'")
            assert {"messages_session_id_id", "messages_resume_projection", "messages_search_document_gin", "sessions_source_session_key", "sessions_parent_session_id", "sessions_title_unique", "sessions_visibility_started_at", "sessions_pinned_started_at", "sessions_effective_activity"} <= {row[0] for row in cursor.fetchall()}
            cursor.execute(f"SELECT conname, convalidated FROM pg_constraint WHERE conrelid = '{_SCHEMA}.sessions'::regclass AND contype = 'f' ORDER BY conname")
            assert cursor.fetchall() == [
                ("sessions_parent_session_id_fkey", False),
                ("sessions_system_prompt_hash_fkey", True),
            ]
            cursor.execute(
                "SELECT conname FROM pg_constraint "
                f"WHERE conrelid = '{_SCHEMA}.conversation_generations'::regclass "
                "AND contype = 'p'"
            )
            assert cursor.fetchall() == [("conversation_generations_pkey",)]
        store = open_state_store(_config())
        store.close()
        assert _migration_versions(dsn) == list(range(1, 26))
        _target().execute(f"INSERT INTO {_SCHEMA}.schema_migrations (version, applied_at) VALUES (26, 0)")
        with pytest.raises(
            StateStoreConfigurationError,
            match=r"Unsupported PostgreSQL State Store schema migration versions: \[26\]",
        ):
            open_state_store(_config())
        _target().execute(f"DELETE FROM {_SCHEMA}.schema_migrations WHERE version=26")
        _target().execute(f"ALTER TABLE {_SCHEMA}.conversation_generations DROP CONSTRAINT conversation_generations_pkey")
        _target().execute(f"ALTER TABLE {_SCHEMA}.conversation_generations ADD CONSTRAINT conversation_generations_pkey PRIMARY KEY (session_key, source)")
        with pytest.raises(StateStoreConfigurationError, match="primary key must be"):
            open_state_store(_config())
    finally:
        _reset_schema(dsn)


def test_postgresql_upgrade_migrations_accept_v2_and_legacy_v5_ledgers(monkeypatch):
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    try:
        _seed_v2_schema(dsn, (1, 2))
        _target().execute(f"INSERT INTO {_SCHEMA}.sessions (id, source, started_at) VALUES ('survives-v2', 'fixture', 1)")
        store = open_state_store(_config())
        session = store.get_session("survives-v2")
        assert session is not None
        assert session["source"] == "fixture"
        store.close()
        assert _migration_versions(dsn) == list(range(1, 26))

        _reset_schema(dsn)
        _seed_v2_schema(dsn, (1, 2, 3, 4))
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD CONSTRAINT sessions_parent_session_id_fkey FOREIGN KEY (parent_session_id) REFERENCES {_SCHEMA}.sessions(id) NOT VALID")
        store = open_state_store(_config())
        store.close()
        assert _migration_versions(dsn) == list(range(1, 26))

        _reset_schema(dsn)
        _seed_v2_schema(dsn, (1, 2, 5))
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN title text")
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN title_source text")
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD COLUMN hidden boolean NOT NULL DEFAULT false")
        _target().execute(f"CREATE UNIQUE INDEX sessions_title_unique ON {_SCHEMA}.sessions (title) WHERE title IS NOT NULL")
        _target().execute(f"ALTER TABLE {_SCHEMA}.sessions ADD CONSTRAINT sessions_parent_session_id_fkey FOREIGN KEY (parent_session_id) REFERENCES {_SCHEMA}.sessions(id) NOT VALID")
        store = open_state_store(_config())
        store.close()
        assert _migration_versions(dsn) == list(range(1, 26))
        _target().execute(f"DROP INDEX {_SCHEMA}.sessions_title_unique")
        with pytest.raises(StateStoreConfigurationError, match="sessions_title_unique"):
            open_state_store(_config())
    finally:
        _reset_schema(dsn)


def test_sqlite_and_postgresql_generation_lifecycle_parity_and_aba_survival(monkeypatch, tmp_path):
    """Reset generation is source/key scoped, first-stamp-wins, and outlives sessions."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    try:
        for store in stores:
            raw_store = cast(Any, store)
            source, key = f"generation-source-{uuid.uuid4()}", f"generation-key-{uuid.uuid4()}"
            session = f"generation-session-{uuid.uuid4()}"
            store.ensure_session(session, source=source, metadata={"session_key": key})
            assert store.latest_conversation_boundary(key, source) is None
            store.end_session(session, "compression")
            assert store.latest_conversation_boundary(key, source) is None
            assert store.promote_to_session_reset(session) is False

            promoted = f"generation-promoted-{uuid.uuid4()}"
            store.ensure_session(promoted, source=source, metadata={"session_key": key})
            store.end_session(promoted, "agent_close")
            assert store.promote_to_session_reset(promoted)
            assert store.promote_to_session_reset(promoted) is False
            store.end_session(promoted, "idle")
            ended = store.get_session(promoted)
            assert ended is not None and ended["end_reason"] == "session_reset"
            assert store.latest_conversation_boundary(key, source) == 1

            unkeyed = f"generation-unkeyed-{uuid.uuid4()}"
            store.ensure_session(unkeyed, source=source)
            store.end_session(unkeyed, "session_reset")
            assert store.latest_conversation_boundary(key, source) == 1
            assert store.latest_conversation_boundary(key, f"other-{source}") is None

            if hasattr(store, "_session_db"):
                assert raw_store._session_db.delete_session(promoted)
            else:
                _target().execute(f"DELETE FROM {_SCHEMA}.sessions WHERE id = %s", (promoted,))
            successor = f"generation-successor-{uuid.uuid4()}"
            store.ensure_session(successor, source=source, metadata={"session_key": key})
            store.end_session(successor, "session_reset")
            successor_row = store.get_session(successor)
            assert successor_row is not None
            observations.append((store.latest_conversation_boundary(key, source), successor_row["end_reason"]))
        assert observations == [(2, "session_reset"), (2, "session_reset")]
    finally:
        for store in stores:
            store.close()


def test_postgresql_generation_end_and_promotion_are_atomic_under_concurrency(monkeypatch):
    """Concurrent close/promotion commits one reset and one generation, never a half-boundary."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    store = open_state_store(_config())
    competing_store = open_state_store(_config())
    source, key = f"concurrent-source-{uuid.uuid4()}", f"concurrent-key-{uuid.uuid4()}"
    session = f"concurrent-session-{uuid.uuid4()}"
    try:
        store.ensure_session(session, source=source, metadata={"session_key": key})
        start = Barrier(2)

        def close() -> None:
            start.wait()
            store.end_session(session, "agent_close")

        def promote() -> bool:
            start.wait()
            return competing_store.promote_to_session_reset(session)

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda operation: operation(), (close, promote)))
        session_row = store.get_session(session)
        assert session_row is not None and session_row["end_reason"] == "session_reset"
        assert store.latest_conversation_boundary(key, source) == 1

        duplicate = f"duplicate-session-{uuid.uuid4()}"
        store.ensure_session(duplicate, source=source, metadata={"session_key": key})
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda _: store.end_session(duplicate, "session_reset"), range(2)))
        duplicate_row = store.get_session(duplicate)
        assert duplicate_row is not None and duplicate_row["end_reason"] == "session_reset"
        assert store.latest_conversation_boundary(key, source) == 2
    finally:
        store.close()
        competing_store.close()


def test_postgresql_generation_update_rolls_back_with_failed_boundary(monkeypatch):
    """A failed transaction cannot leave a generation whose end stamp vanished."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    store = open_state_store(_config())
    source, key = f"rollback-source-{uuid.uuid4()}", f"rollback-key-{uuid.uuid4()}"
    session = f"rollback-session-{uuid.uuid4()}"
    raw_store = cast(Any, store)
    original = raw_store._bump_conversation_generation
    try:
        store.ensure_session(session, source=source, metadata={"session_key": key})
        raw_store._bump_conversation_generation = lambda *_args: (_ for _ in ()).throw(RuntimeError("inject rollback"))
        with pytest.raises(RuntimeError, match="inject rollback"):
            store.end_session(session, "session_reset")
        session_row = store.get_session(session)
        assert session_row is not None and session_row["end_reason"] is None
        assert store.latest_conversation_boundary(key, source) is None
        promotion = f"rollback-promotion-{uuid.uuid4()}"
        store.ensure_session(promotion, source=source, metadata={"session_key": key})
        assert store.promote_to_session_reset(promotion) is False
        promotion_row = store.get_session(promotion)
        assert promotion_row is not None and promotion_row["end_reason"] is None
        assert store.latest_conversation_boundary(key, source) is None
    finally:
        raw_store._bump_conversation_generation = original
        store.close()


def test_sqlite_and_postgresql_resume_projection_and_lineage_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    root, tip = f"resume-root-{uuid.uuid4()}", f"resume-tip-{uuid.uuid4()}"
    try:
        for store in stores:
            store.ensure_session(root, source="integration")
            store.append_message_record(root, MessageRecord(role="user", content="before compression", timestamp=1))
            store.end_session(root, "compression")
            store.ensure_session(tip, source="integration", metadata={"parent_session_id": root})
            store.append_message_record(tip, MessageRecord(role="assistant", content="after compression", timestamp=2))
            model, display = store.get_resume_conversations(tip)
            observations.append({
                "tip": store.get_compression_tip(root),
                "lineage": store.get_compression_lineage(tip),
                "root": store.get_conversation_root(tip),
                "model": [(row["role"], row["content"]) for row in model],
                "display": [(row["role"], row["content"]) for row in display],
                "ancestor": [(row["role"], row["content"]) for row in store.get_ancestor_display_prefix(tip)],
                "all_count": store.get_resume_message_count(tip),
                "tip_count": store.get_resume_message_count(tip, tip_only=True),
                "guard": store.assert_resume_safe(tip, 2),
            })
        assert observations[0] == observations[1]
        assert observations[0]["tip"] == observations[0]["lineage"][-1]
        assert observations[0]["root"] == observations[0]["lineage"][0]
        assert observations[0]["model"] == [("assistant", "after compression")]
        assert observations[0]["display"] == [("user", "before compression"), ("assistant", "after compression")]
        assert observations[0]["ancestor"] == [("user", "before compression")]
        assert observations[0]["all_count"] == observations[0]["guard"] == 2
        assert observations[0]["tip_count"] == 1
        for store, observation in zip(stores, observations):
            with pytest.raises(Exception):
                store.assert_resume_safe(observation["lineage"][-1], 1)
    finally:
        for store in stores:
            store.close()


def test_sqlite_and_postgresql_token_usage_transport_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    try:
        for store in stores:
            session_id = f"usage-{uuid.uuid4()}"
            route = dict(model="m1", billing_provider="p1", billing_base_url="url", billing_mode="mode")
            store.queue_token_counts(session_id, input_tokens=2, api_call_count=1, **route)
            store.queue_token_counts(session_id, input_tokens=3, api_call_count=1, **route)
            store.queue_token_counts(session_id, input_tokens=50, api_call_count=3, absolute=True, **route)
            assert store.flush_token_counts()
            store.record_auxiliary_usage(session_id, "vision", model="aux", billing_provider="auxp", input_tokens=7)
            if hasattr(store, "_connection"):
                with store._connection() as connection, connection.cursor() as cursor:
                    cursor.execute(f"SELECT input_tokens, api_call_count FROM {_SCHEMA}.sessions WHERE id=%s", (session_id,))
                    observations.append(cursor.fetchone())
                    cursor.execute(f"SELECT task, model, input_tokens FROM {_SCHEMA}.session_model_usage WHERE session_id=%s ORDER BY task", (session_id,))
                    assert cursor.fetchall() == [("", "m1", 5), ("vision", "aux", 7)]
            else:
                rows = store._session_db._read_all("SELECT task, model, input_tokens FROM session_model_usage WHERE session_id=? ORDER BY task", (session_id,))
                assert [(row["task"], row["model"], row["input_tokens"]) for row in rows] == [("", "m1", 5), ("vision", "aux", 7)]
                summary = store._session_db._read_one("SELECT input_tokens, api_call_count FROM sessions WHERE id=?", (session_id,))
                observations.append((summary["input_tokens"], summary["api_call_count"]))
        assert observations == [(50, 3), (50, 3)]
    finally:
        for store in stores:
            store.close()


def test_postgresql_token_usage_transport_uses_source_for_first_durable_session(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    store = open_state_store(_config())
    session_id = f"usage-source-{uuid.uuid4()}"
    try:
        store.queue_token_counts(
            session_id, source="cli", input_tokens=17, output_tokens=5, api_call_count=1,
            model="source-model", billing_provider="source-provider", billing_base_url="source-url",
            billing_mode="source-mode",
        )
        assert store.flush_token_counts()
        assert not (tmp_path / "state.db").exists()
        raw_store = cast(Any, store)
        with raw_store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"SELECT source, model, input_tokens, output_tokens, api_call_count "
                f"FROM {_SCHEMA}.sessions WHERE id=%s", (session_id,),
            )
            assert cursor.fetchone() == ("cli", "source-model", 17, 5, 1)
            cursor.execute(
                f"SELECT model, billing_provider, billing_base_url, billing_mode, input_tokens, output_tokens, api_call_count "
                f"FROM {_SCHEMA}.session_model_usage WHERE session_id=%s", (session_id,),
            )
            assert cursor.fetchone() == ("source-model", "source-provider", "source-url", "source-mode", 17, 5, 1)
    finally:
        store.close()


def test_postgresql_token_usage_delta_rolls_back_summary_when_attribution_fails(monkeypatch):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    store = open_state_store(_config())
    session_id = f"usage-rollback-{uuid.uuid4()}"
    try:
        store.ensure_session(session_id)
        monkeypatch.setattr(store, "_record_model_usage", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected attribution failure")))
        with pytest.raises(RuntimeError, match="injected attribution failure"):
            store.update_token_counts(session_id, input_tokens=9, model="m", billing_provider="p", api_call_count=1)
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT input_tokens, api_call_count FROM {_SCHEMA}.sessions WHERE id=%s", (session_id,))
            assert cursor.fetchone() == (0, 0)
            cursor.execute(f"SELECT COUNT(*) FROM {_SCHEMA}.session_model_usage WHERE session_id=%s", (session_id,))
            assert cursor.fetchone() == (0,)
    finally:
        store.close()


def test_sqlite_and_postgresql_model_config_lifecycle_parity(monkeypatch, tmp_path):
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    try:
        for store in stores:
            session_id = f"model-config-{uuid.uuid4()}"
            store.ensure_session(session_id, source="integration", metadata={"model": "initial", "model_config": {"keep": 1, "drop": "x"}})
            store.set_system_prompt(session_id, "cached footer")
            store.queue_token_counts(session_id, input_tokens=4, api_call_count=1, model="before", billing_provider="old", billing_base_url="old-url")
            store.update_session_model(
                session_id, "after", "provider-after",
                base_url="https://provider.example/v1", api_mode="chat",
            )
            store.patch_session_model_config(session_id, {"keep": None, "nested": {"json": None}, "new": [1, 2]})
            store.update_session_billing_route(session_id, provider="new-provider", base_url="new-url", billing_mode="metered")
            assert store.get_session_model_config_value(session_id, "missing", "fallback") == "fallback"
            before_rollback = store.get_session_model_config_value(session_id, "new")
            with pytest.raises(TypeError):
                store.patch_session_model_config(session_id, {"bad": object()})
            assert store.get_session_model_config_value(session_id, "new") == before_rollback
            session = store.get_session(session_id)
            assert session is not None
            config = json.loads(session["model_config"]) if isinstance(session["model_config"], str) else session["model_config"]
            if hasattr(store, "_connection"):
                raw_store = cast(Any, store)
                with raw_store._connection() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT billing_provider, billing_base_url, billing_mode, input_tokens, api_call_count "
                        f"FROM {_SCHEMA}.sessions WHERE id=%s", (session_id,))
                    row = cursor.fetchone()
                    route, usage = row[:3], row[3:]
            else:
                raw_store = cast(Any, store)
                raw_session = raw_store._session_db._read_one(
                    "SELECT billing_provider, billing_base_url, billing_mode, input_tokens, api_call_count FROM sessions WHERE id=?", (session_id,))
                route = tuple(raw_session[key] for key in ("billing_provider", "billing_base_url", "billing_mode"))
                usage = tuple(raw_session[key] for key in ("input_tokens", "api_call_count"))
            observations.append({
                "model": session["model"], "config": config, "prompt": session["system_prompt"],
                "route": route, "usage": usage,
            })
        assert observations == [{
            "model": "after", "config": {
                "drop": "x", "model": "after", "provider": "provider-after",
                "base_url": "https://provider.example/v1", "api_mode": "chat",
                "gateway_runtime": {
                    "provider": "provider-after", "base_url": "https://provider.example/v1",
                    "api_mode": "chat",
                },
                "nested": {"json": None}, "new": [1, 2],
            },
            "prompt": None, "route": ("new-provider", "new-url", "metered"), "usage": (4, 1),
        }] * 2
        postgresql = cast(Any, stores[1])
        with postgresql._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT version FROM {_SCHEMA}.schema_migrations ORDER BY version")
            assert [row[0] for row in cursor.fetchall()][-5:] == [21, 22, 23, 24, 25]
    finally:
        for store in stores:
            store.close()


def test_sqlite_and_postgresql_git_metadata_claim_publish_parity(monkeypatch, tmp_path):
    """Moves fence stale probes, while failed replacement probes preserve known metadata."""
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    stores = (open_state_store({}, db_path=tmp_path / "state.db"), open_state_store(_config()))
    observations = []
    try:
        for store in stores:
            parent, child = f"git-parent-{uuid.uuid4()}", f"git-child-{uuid.uuid4()}"
            store.ensure_session(parent, source="integration", metadata={"cwd": "/parent"})
            parent_generation = store.update_session_cwd(parent, "/parent")
            assert isinstance(parent_generation, int)
            assert store.publish_session_git_metadata(parent, "/parent", parent_generation, "main", "/parent")
            store.ensure_session(child, source="integration", metadata={"parent_session_id": parent})
            inherited = store.get_session(child)
            assert inherited is not None
            assert (inherited["cwd"], inherited["git_branch"], inherited["git_repo_root"]) == ("/parent", "main", "/parent")
            session = f"git-metadata-{uuid.uuid4()}"
            store.ensure_session(session, source="integration", metadata={"cwd": "/repo/A"})
            first = store.update_session_cwd(session, "/repo/A")
            assert isinstance(first, int) and not isinstance(first, bool)
            assert store.publish_session_git_metadata(session, "/repo/A", first, "baseline", "/repo/A")
            failed = store.update_session_cwd(session, "/repo/A")
            assert isinstance(failed, int)
            assert not store.publish_session_git_metadata(session, "/repo/A", failed)
            baseline = store.get_session(session)
            assert baseline is not None
            assert (baseline["git_branch"], baseline["git_repo_root"]) == ("baseline", "/repo/A")
            stale = store.update_session_cwd(session, "/repo/A")
            moved = store.update_session_cwd(session, "/repo/B")
            current = store.update_session_cwd(session, "/repo/A")
            assert isinstance(stale, int) and isinstance(moved, int) and isinstance(current, int)
            assert current > moved > stale > first
            assert store.publish_session_git_metadata(session, "/repo/A", current, "current", "/repo/current")
            assert not store.publish_session_git_metadata(session, "/repo/A", stale, "stale", "/repo/stale")
            row = store.get_session(session)
            assert row is not None
            observations.append((row["cwd"], row["git_branch"], row["git_repo_root"], row["git_metadata_generation"]))
        assert [(cwd, branch, root) for cwd, branch, root, _ in observations] == [
            ("/repo/A", "current", "/repo/current"),
        ] * 2
        assert all(generation >= 4 for _, _, _, generation in observations)
    finally:
        for store in stores:
            store.close()


def test_postgresql_git_metadata_claims_are_fenced_under_concurrency(monkeypatch):
    """Two writers may both claim, but only the current `(cwd, generation)` may publish."""
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    first_store, second_store = open_state_store(_config()), open_state_store(_config())
    session = f"git-metadata-concurrent-{uuid.uuid4()}"
    try:
        first_store.ensure_session(session, source="integration", metadata={"cwd": "/initial"})
        barrier = Barrier(2)

        def claim(store, cwd: str) -> tuple[str, int | None]:
            barrier.wait()
            return cwd, store.update_session_cwd(session, cwd)

        with ThreadPoolExecutor(max_workers=2) as executor:
            claims = list(executor.map(lambda args: claim(*args), ((first_store, "/repo/A"), (second_store, "/repo/B"))))
        assert all(isinstance(generation, int) for _, generation in claims)
        claimed = [(cwd, cast(int, generation)) for cwd, generation in claims]
        outcomes = [
            store.publish_session_git_metadata(session, cwd, generation, cwd.rsplit("/", 1)[-1], cwd)
            for store, (cwd, generation) in zip((first_store, second_store), claimed)
        ]
        assert outcomes.count(True) == 1
        row = first_store.get_session(session)
        assert row is not None
        assert row["git_metadata_generation"] == max(generation for _, generation in claimed)
        assert row["git_branch"] == row["cwd"].rsplit("/", 1)[-1]
        assert row["git_repo_root"] == row["cwd"]
    finally:
        first_store.close()
        second_store.close()


def test_postgresql_git_metadata_generation_catalog_drift_fails_closed(monkeypatch):
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    try:
        for statement in (
            f"ALTER TABLE {_SCHEMA}.sessions ALTER COLUMN git_metadata_generation SET DEFAULT 10",
            f"ALTER TABLE {_SCHEMA}.sessions ALTER COLUMN git_branch TYPE bigint USING NULL",
            f"ALTER TABLE {_SCHEMA}.sessions ALTER COLUMN git_branch SET NOT NULL",
            f"ALTER TABLE {_SCHEMA}.sessions ALTER COLUMN git_branch SET DEFAULT 'main'",
        ):
            _reset_schema(dsn)
            store = open_state_store(_config())
            store.close()
            _target().execute(statement)
            with pytest.raises(StateStoreConfigurationError, match="Git metadata columns"):
                open_state_store(_config())
    finally:
        _reset_schema(dsn)


def test_postgresql_tenant_acquisition_isolates_root_named_profiles_and_pool_search_path(monkeypatch, tmp_path):
    """Resolved homes, not metadata, choose independently migrated tenant schemas."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    root = tmp_path / "hermes-root"
    alice = root / "profiles" / "alice"
    bob = root / "profiles" / "bob"
    for home in (root, alice, bob):
        home.mkdir(parents=True, exist_ok=True)

    stores = []
    try:
        monkeypatch.setenv("HERMES_HOME", str(root))
        root_store = open_state_store(_config())
        root_store.ensure_session("tenant-shared", metadata={"profile_name": "bob"})
        root_store.append_message("tenant-shared", role="user", content="root browse only")
        stores.append(root_store)

        monkeypatch.setenv("HERMES_HOME", str(alice))
        alice_store = open_state_store(_config())
        alice_store.ensure_session("tenant-shared", metadata={"profile_name": "root"})
        alice_store.append_message("tenant-shared", role="user", content="alice browse only")
        stores.append(alice_store)

        monkeypatch.setenv("HERMES_HOME", str(bob))
        bob_store = open_state_store(_config())
        assert bob_store.get_session("tenant-shared") is None
        bob_store.ensure_session("tenant-shared")
        bob_store.append_message("tenant-shared", role="user", content="bob browse only")
        stores.append(bob_store)

        assert len({store._schema for store in stores}) == 3
        assert root_store.get_session("tenant-shared") is not None
        assert alice_store.get_session("tenant-shared") is not None
        assert bob_store.get_session("tenant-shared") is not None
        for store, preview in ((root_store, "root browse only"), (alice_store, "alice browse only"), (bob_store, "bob browse only")):
            row = next(item for item in cast(Any, store).list_recent_sessions_bounded(limit=128, exclude_sources=[], timeout_seconds=3)
                       if item["id"] == "tenant-shared")
            assert row["preview"] == preview
            assert store.search_messages("browse", fields=("session_id", "context")) == [{
                "session_id": "tenant-shared", "context": [{"role": "user", "content": preview}],
            }]
        for store in stores:
            with store._connection() as connection, connection.cursor() as cursor:
                cursor.execute("SHOW search_path")
                assert cursor.fetchone()[0].split(",")[0].strip(' "') == store._schema
                # Simulate a hostile borrower; checkout must reset before reuse.
                cursor.execute("SET search_path TO public")
            with store._connection() as connection, connection.cursor() as cursor:
                cursor.execute("SHOW search_path")
                assert cursor.fetchone()[0].split(",")[0].strip(' "') == store._schema

        # Tenant initialization is advisory-lock serialized and all openers see one head.
        monkeypatch.setenv("HERMES_HOME", str(alice))
        with ThreadPoolExecutor(max_workers=3) as executor:
            opened = list(executor.map(lambda _: open_state_store(_config()), range(3)))
        try:
            assert {store._schema for store in opened} == {alice_store._schema}
        finally:
            for store in opened:
                store.close()
    finally:
        for store in stores:
            store.close()


def test_contextual_profile_resolver_mixes_root_sqlite_named_postgresql_and_named_sqlite(monkeypatch, tmp_path):
    """Explicit target profiles route only to their own backend, tenant, and secret scope."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    root = tmp_path / ".hermes"
    pg_home = root / "profiles" / "pg-reader"
    sqlite_home = root / "profiles" / "sqlite-reader"
    for home in (root, pg_home, sqlite_home):
        home.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text("state_store:\n  backend: sqlite\n", encoding="utf-8")
    (pg_home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: PROFILE_CONTEXTUAL_DSN\n",
        encoding="utf-8")
    (pg_home / ".env").write_text(f"PROFILE_CONTEXTUAL_DSN={dsn}\n", encoding="utf-8")
    (sqlite_home / "config.yaml").write_text("state_store:\n  backend: sqlite\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    root_db = SessionDB(root / "state.db")
    sqlite_db = SessionDB(sqlite_home / "state.db")
    pg_store = None
    schema = _SCHEMA
    try:
        root_db.create_session("root-context", source="integration")
        root_db.append_message("root-context", role="user", content="rootonlycontext")
        sqlite_db.create_session("sqlite-context", source="integration")
        sqlite_db.append_message("sqlite-context", role="user", content="sqliteonlycontext")
        pg_store = cast(Any, resolve_contextual_session_search_store(profile="pg-reader", read_only=True))
        pg_store.ensure_session("pg-context", source="integration")
        pg_store.append_message("pg-context", role="user", content="pgonlycontext")
        assert pg_store._schema == schema
        assert pg_store.search_index_status()["available"] is True
        with pg_store._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
        with pg_store._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SHOW search_path")
            assert cursor.fetchone()[0].split(",")[0].strip(' \"') == schema
        pg_store.close()
        pg_store = None

        root_store = resolve_contextual_session_search_store(profile="root", read_only=True)
        named_pg_store = resolve_contextual_session_search_store(profile="pg-reader", read_only=True)
        named_sqlite_store = resolve_contextual_session_search_store(profile="sqlite-reader", read_only=True)
        try:
            assert root_store.search_messages("rootonlycontext", fields=("session_id",)) == [{"session_id": "root-context"}]
            assert named_pg_store.search_messages("pgonlycontext", fields=("session_id",)) == [{"session_id": "pg-context"}]
            public_named_pg = json.loads(session_search(query="pgonlycontext", profile="pg-reader"))
            assert public_named_pg["success"] is True
            assert [row["session_id"] for row in public_named_pg["results"]] == ["pg-context"]
            assert public_named_pg["search_index"]["backend"] == "postgresql"
            assert named_sqlite_store.search_messages("sqliteonlycontext", fields=("session_id",)) == [{"session_id": "sqlite-context"}]
            assert named_pg_store.search_messages("rootonlycontext", fields=("session_id",)) == []
            assert named_pg_store.search_messages("sqliteonlycontext", fields=("session_id",)) == []
            assert not (pg_home / "state.db").exists()
        finally:
            root_store.close()
            named_pg_store.close()
            named_sqlite_store.close()
    finally:
        root_db.close()
        sqlite_db.close()
        if pg_store is not None:
            pg_store.close()



def test_postgresql_rejects_injection_looking_schema_before_connecting():
    from state_store import PostgreSQLStateStoreConfig
    from state_store_postgresql import PostgreSQLStateStore

    with pytest.raises(StateStoreConfigurationError, match="invalid trusted tenant schema"):
        PostgreSQLStateStore(
            PostgreSQLStateStoreConfig(dsn_env="TEST_DSN", connect_timeout_seconds=1, pool_max_size=1),
            "postgresql://invalid",
            schema='tenant"; DROP SCHEMA public; --',
        )


def test_sqlite_and_postgresql_bounded_recent_compression_projection_parity(monkeypatch, tmp_path):
    """Direct bounded browse parity; contextual routing remains unavailable."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    sqlite, postgres = SessionDB(tmp_path / "state.db"), cast(Any, open_state_store(_config()))
    prefix, now = f"bounded-{uuid.uuid4()}", time.time()
    root, tip, reset, hidden, delegated, excluded = (f"{prefix}-{name}" for name in ("root", "tip", "reset", "hidden", "delegated", "excluded"))

    def seed_sqlite() -> None:
        sqlite.create_session(root, source="visible")
        sqlite.append_message(root, role="user", content="root preview")
        sqlite.end_session(root, "compression")
        sqlite.create_session(tip, source="visible", parent_session_id=root)
        sqlite.append_message(tip, role="user", content="tip preview")
        for session_id, source, config in ((reset, "visible", None), (hidden, "visible", None), (delegated, "visible", {"_delegate_from": "parent"}), (excluded, "excluded", None)):
            sqlite.create_session(session_id, source=source, model_config=config)
            sqlite.append_message(session_id, role="user", content=f"{session_id} preview")
        sqlite.set_session_hidden(hidden, True)
        for session_id, activity in ((root, now - 100), (tip, now), (reset, now - 10), (hidden, now + 10), (delegated, now + 20), (excluded, now + 30)):
            sqlite._conn.execute("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (activity, session_id))
            sqlite._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (activity, session_id))
        sqlite._conn.commit()

    def seed_postgres() -> None:
        postgres.ensure_session(root, source="visible")
        postgres.append_message(root, role="user", content="root preview")
        postgres.end_session(root, "compression")
        postgres.ensure_session(tip, source="visible", metadata={"parent_session_id": root})
        postgres.append_message(tip, role="user", content="tip preview")
        for session_id, source, config in ((reset, "visible", None), (hidden, "visible", None), (delegated, "visible", {"_delegate_from": "parent"}), (excluded, "excluded", None)):
            postgres.ensure_session(session_id, source=source, metadata={"model_config": config} if config else None)
            postgres.append_message(session_id, role="user", content=f"{session_id} preview")
        postgres.set_session_hidden(hidden, True)
        for session_id, activity in ((root, now - 100), (tip, now), (reset, now - 10), (hidden, now + 10), (delegated, now + 20), (excluded, now + 30)):
            _target().execute(f"UPDATE {_SCHEMA}.sessions SET last_activity_at = %s WHERE id = %s", (activity, session_id))
            _target().execute(f"UPDATE {_SCHEMA}.messages SET created_at = %s WHERE session_id = %s", (activity, session_id))

    try:
        seed_sqlite()
        seed_postgres()
        expected = sqlite.list_recent_sessions_bounded(limit=2, exclude_sources=["excluded"], timeout_seconds=3)
        actual = postgres.list_recent_sessions_bounded(limit=2, exclude_sources=["excluded"], timeout_seconds=3)
        assert [(row["id"], row["preview"], row.get("_lineage_root_id")) for row in actual] == [
            (row["id"], row["preview"], row.get("_lineage_root_id")) for row in expected
        ] == [(tip, "tip preview", root), (reset, f"{reset} preview", None)]
        assert [row["id"] for row in postgres.list_recent_sessions_bounded(limit=1, exclude_sources=["excluded"], timeout_seconds=3)] == [tip]
    finally:
        sqlite.close()
        postgres.close()
        _reset_schema(dsn)


def test_postgresql_search_contract_is_tenant_local_and_does_not_require_optional_extensions(monkeypatch, tmp_path):
    """PG18 differential for the deliberately bounded lexical/CJK contract."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    sqlite = open_state_store({}, db_path=tmp_path / "state.db")
    postgresql = open_state_store(_config())
    try:
        for store in (sqlite, postgresql):
            store.ensure_session("search-alpha", source="alpha")
            store.ensure_session("search-beta", source="beta")
            store.append_message("search-alpha", role="user", content="lexical needle first")
            store.append_message("search-alpha", role="assistant", content="lexical needle second")
            store.append_message("search-beta", role="assistant", content="needle private beta")
            store.append_message("search-beta", role="user", content="中文记忆断裂 substring")

        supported_cases: tuple[tuple[str, dict[str, Any]], ...] = (
            ("needle", {}),
            ("lexical needle", {"source_filter": ["alpha"]}),
            ('"lexical needle"', {"role_filter": ["assistant"]}),
            ("need*", {"exclude_sources": ["beta"]}),
            ("lexical AND needle", {}),
            ("lexical OR private", {}),
            ("needle NOT private", {"include_inactive": True}),
            ("中文记忆", {}),
        )
        for query, kwargs in supported_cases:
            sqlite_rows = sqlite.search_messages(query, fields=("id", "session_id", "role", "source", "snippet"), **kwargs)
            postgresql_rows = postgresql.search_messages(query, fields=("id", "session_id", "role", "source", "snippet"), **kwargs)
            assert sorted((row["session_id"], row["role"], row["source"]) for row in postgresql_rows) == sorted(
                (row["session_id"], row["role"], row["source"]) for row in sqlite_rows
            )
            assert all(row["snippet"] for row in postgresql_rows)

        for sort in ("newest", "oldest"):
            sqlite_page = sqlite.search_messages("needle", sort=sort, limit=1, offset=1, fields=("session_id", "role", "source"))
            postgresql_page = postgresql.search_messages("needle", sort=sort, limit=1, offset=1, fields=("session_id", "role", "source"))
            assert postgresql_page == sqlite_page

        from state_store_postgresql_search import PostgreSQLSearchQueryError
        for unsupported in ("needle OR OR private", "(needle OR private)", '"needle', "need:private", "中文 AND memory"):
            with pytest.raises(PostgreSQLSearchQueryError):
                postgresql.search_messages(unsupported)

        # Generated tsvector maintenance survives canonical update/delete and an index rebuild.
        _target().execute(f"UPDATE {postgresql._schema}.messages SET content = 'replacement token' WHERE session_id = %s AND role = 'user'", ("search-alpha",))
        _target().execute(f"DELETE FROM {postgresql._schema}.messages WHERE session_id = %s AND content LIKE %s", ("search-beta", "needle private%"))
        _target().execute(f"REINDEX INDEX {postgresql._schema}.messages_search_document_gin")
        assert [row["session_id"] for row in postgresql.search_messages("replacement", fields=("session_id",))] == ["search-alpha"]
        assert all(row["session_id"] != "search-beta" for row in postgresql.search_messages("needle", fields=("session_id",)))

        # This schema-local contract does not mutate database-wide optional extensions.
        postgresql.close()
        postgresql = None
        degraded = open_state_store(_config())
        try:
            assert degraded.search_messages("replacement", fields=("session_id",)) == [{"session_id": "search-alpha"}]
            assert degraded.search_messages("中文记忆", fields=("session_id",)) == [{"session_id": "search-beta"}]
        finally:
            degraded.close()
    finally:
        sqlite.close()
        if postgresql is not None:
            postgresql.close()
        _reset_schema(dsn)


def test_postgresql_candidate_context_projection_matches_sqlite_contract(monkeypatch, tmp_path):
    """Candidate rows retain model/metadata and timestamp-identity neighbours without routing tools."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    sqlite, postgresql = open_state_store({}, db_path=tmp_path / "context.db"), open_state_store(_config())
    try:
        structured_neighbour = "structured " + ("x" * 240)
        records = (
            MessageRecord(role="user", content="before candidate", timestamp=10),
            MessageRecord(role="user", content="candidate needle first", timestamp=20),
            MessageRecord(role="assistant", content=[{"type": "text", "text": structured_neighbour}], timestamp=20),
            MessageRecord(role="user", content="candidate needle second", timestamp=20),
            MessageRecord(role="tool", content="candidate needle tool body", tool_name="lookup", timestamp=30),
            MessageRecord(role="assistant", content="candidate needle hidden", display_kind="hidden", timestamp=40),
        )
        for store in (sqlite, postgresql):
            store.ensure_session("candidate-main", source="alpha", metadata={"model": "candidate-model"})
            store.ensure_session("candidate-other", source="beta", metadata={"model": "other-model"})
            store.append_message_records("candidate-main", list(records))
            store.append_message_record("candidate-other", MessageRecord(role="user", content="candidate needle foreign", timestamp=20))

        def projection(store, **kwargs):
            return [{key: row[key] for key in (
                "session_id", "role", "tool_name", "source", "model", "context",
            )} for row in store.search_messages("candidate needle", sort="oldest", **kwargs)]

        sqlite_rows, pg_rows = projection(sqlite), projection(postgresql)
        assert pg_rows == sqlite_rows
        assert all(isinstance(row["session_started"], float) for row in postgresql.search_messages(
            "candidate needle", sort="oldest"))
        assert [tuple(row) for row in postgresql.search_messages("candidate needle", sort="oldest", limit=1)] == [
            ("id", "session_id", "role", "snippet", "timestamp", "tool_name", "source", "model", "session_started", "context"),
        ]
        main_rows = [row for row in pg_rows if row["session_id"] == "candidate-main" and row["role"] == "user"]
        assert main_rows[0]["context"] == [
            {"role": "user", "content": "before candidate"},
            {"role": "user", "content": "candidate needle first"},
            {"role": "assistant", "content": structured_neighbour[:200]},
        ]
        assert all("foreign" not in item["content"] for row in main_rows for item in row["context"])
        first_candidate_id = next(row["id"] for row in postgresql.search_messages(
            "candidate needle", sort="oldest", fields=("id", "session_id")) if row["session_id"] == "candidate-main")
        assert cast(Any, postgresql)._search_contexts([first_candidate_id, first_candidate_id])[first_candidate_id] == main_rows[0]["context"]
        assert projection(sqlite, source_filter=["alpha"], role_filter=["user"]) == projection(
            postgresql, source_filter=["alpha"], role_filter=["user"])
        assert projection(sqlite, exclude_sources=["beta"]) == projection(postgresql, exclude_sources=["beta"])
        assert sqlite.search_messages("candidate needle", fields=("id", "model"), sort="oldest") == postgresql.search_messages(
            "candidate needle", fields=("id", "model"), sort="oldest")
        assert all("context" not in row for row in postgresql.search_messages("candidate needle", fields=("id",), sort="oldest"))
        assert sqlite.search_messages("candidate needle", role_filter=["tool"], fields=("role", "context")) == postgresql.search_messages(
            "candidate needle", role_filter=["tool"], fields=("role", "context"))
        assert sqlite.search_messages("candidate needle", sort="oldest", limit=2, offset=1, fields=("session_id", "role")) == postgresql.search_messages(
            "candidate needle", sort="oldest", limit=2, offset=1, fields=("session_id", "role"))
        monkeypatch.setattr(cast(Any, postgresql), "_search_contexts", lambda *_args: (_ for _ in ()).throw(RuntimeError("injected context failure")))
        assert all(row["context"] == [] for row in postgresql.search_messages(
            "candidate needle", sort="oldest", fields=("id", "context")))
    finally:
        sqlite.close()
        postgresql.close()
        _reset_schema(dsn)


def test_postgresql_anchored_and_scroll_views_match_sqlite_physical_boundaries(monkeypatch, tmp_path):
    """The partial PostgreSQL store may expose primitives without enabling tool routing."""
    monkeypatch.setenv(_DSN_ENV, "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test")
    dsn = os.environ[_DSN_ENV]
    _reset_schema(dsn)
    sqlite = SessionDB(tmp_path / "contextual.db")
    postgresql = open_state_store(_config())
    try:
        ids: dict[str, list[int]] = {}
        sqlite.create_session("contextual", source="integration")
        sqlite.create_session("other", source="integration")
        sqlite_records = (
            ("user", "opening", None, 30.0), ("tool", "tool body", "lookup", 10.0),
            ("assistant", "anchor", None, 20.0), ("user", "", None, 40.0),
            ("assistant", "resolution", None, 50.0),
        )
        ids["sqlite"] = [sqlite.append_message("contextual", role=role, content=content, tool_name=tool_name, timestamp=timestamp)
                         for role, content, tool_name, timestamp in sqlite_records]
        sqlite.append_message("other", role="user", content="foreign")
        postgresql.ensure_session("contextual", source="integration")
        postgresql.ensure_session("other", source="integration")
        records = (
            MessageRecord(role="user", content="opening", timestamp=30.0),
            MessageRecord(role="tool", content="tool body", tool_name="lookup", timestamp=10.0),
            MessageRecord(role="assistant", content="anchor", timestamp=20.0),
            MessageRecord(role="user", content="", timestamp=40.0),
            MessageRecord(role="assistant", content="resolution", timestamp=50.0),
        )
        ids["postgresql"] = [postgresql.append_message_record("contextual", record) for record in records]
        postgresql.append_message("other", role="user", content="foreign")

        def projection(view):
            return {
                key: ([ (row["role"], row["content"]) for row in value ] if isinstance(value, list) else value)
                for key, value in view.items()
            }

        sqlite_contextual, pg_contextual = cast(Any, sqlite), cast(Any, postgresql)
        sqlite_ids, pg_ids = ids["sqlite"], ids["postgresql"]
        for window in (-1, 0, 1, 20):
            assert projection(pg_contextual.get_messages_around("contextual", pg_ids[1], window=window)) == projection(
                sqlite_contextual.get_messages_around("contextual", sqlite_ids[1], window=window)
            )
        assert pg_contextual.get_messages_around("other", pg_ids[1], window=5) == {
            "window": [], "messages_before": 0, "messages_after": 0,
        }

        sqlite_view = sqlite_contextual.get_anchored_view("contextual", sqlite_ids[1], window=1, bookend=3)
        pg_view = pg_contextual.get_anchored_view("contextual", pg_ids[1], window=1, bookend=3)
        assert projection(pg_view) == projection(sqlite_view)
        # The filtered detail retains the tool anchor, while the physical scroll page repeats it.
        assert [row["role"] for row in pg_view["window"]] == ["user", "tool", "assistant"]
        page = pg_contextual.get_messages_around("contextual", pg_ids[1], window=1)
        next_page = pg_contextual.get_messages_around("contextual", page["window"][-1]["id"], window=1)
        assert page["window"][-1]["id"] in [row["id"] for row in next_page["window"]]
    finally:
        sqlite.close()
        postgresql.close()
        _reset_schema(dsn)


def test_postgresql_generated_search_health_rebuild_and_contextual_admission(monkeypatch, tmp_path):
    """PG18 catalog drift is tenant-local, repairable, and never SQLite FTS progress."""
    dsn = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
    monkeypatch.setenv(_DSN_ENV, dsn)
    _reset_schema(dsn)
    sqlite = SessionDB(tmp_path / "sqlite-state.db")
    postgresql = cast(Any, open_state_store(_config()))
    try:
        postgresql.ensure_session("health", source="integration")
        message_id = postgresql.append_message("health", role="user", content="health needle")
        assert sqlite.fts_rebuild_status() is None
        healthy = postgresql.search_index_status()
        assert healthy["backend"] == "postgresql"
        assert healthy["available"] is True and healthy["query_path_available"] is True
        assert healthy["generated_document"] == "valid" and healthy["gin_index"] == "valid"
        assert healthy["rebuild"] == {"supported": True, "operation": "reindex_or_create", "in_progress": False}
        assert healthy["sqlite_fts_semantics"] == {
            "corruption_detach": False, "canonical_like_fallback": False,
            "deferred_backfill": False, "high_water": False, "retry_quarantine": False,
        }
        contextual = contextual_session_search_store(postgresql, backend="postgresql")
        assert contextual.get_message_storage_state(message_id) == {"session_id": "health", "active": 1, "compacted": 0}
        assert contextual.search_messages("health", fields=("session_id",)) == [{"session_id": "health"}]

        _target().execute(f"DROP INDEX {_SCHEMA}.messages_search_document_gin")
        missing = postgresql.search_index_status()
        assert missing["available"] is False and missing["gin_index"] == "missing"
        # The predicate can sequential-scan, but contextual admission is explicitly fail-closed.
        assert postgresql.search_messages("health", fields=("session_id",)) == [{"session_id": "health"}]
        with pytest.raises(Exception, match="generated-search health"):
            contextual_session_search_store(postgresql, backend="postgresql")
        repaired = postgresql.rebuild_search_index()
        assert repaired["available"] is True and repaired["rebuild"]["operation"] == "create"
        assert repaired["last_successful_rebuild_at"] is not None

        _target().execute(f"DROP INDEX {_SCHEMA}.messages_search_document_gin")
        _target().execute(f"CREATE INDEX messages_search_document_gin ON {_SCHEMA}.messages (search_document)")
        assert postgresql.search_index_status()["gin_index"] == "invalid"
        assert postgresql.rebuild_search_index()["rebuild"]["operation"] == "replace_invalid"
        assert postgresql.search_index_status()["gin_index"] == "valid"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: postgresql.rebuild_search_index(), range(2)))
        assert all(outcome["available"] for outcome in outcomes)
        assert {outcome["rebuild"]["operation"] for outcome in outcomes} <= {"reindex", "already_running"}
    finally:
        sqlite.close()
        postgresql.close()
        _reset_schema(dsn)
