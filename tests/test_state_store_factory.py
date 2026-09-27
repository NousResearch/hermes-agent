"""Backend-neutral narrow State Store factory contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest

from state_store import open_state_store


def test_tenant_schema_capability_is_not_a_public_state_store_factory_api():
    import state_store

    assert not hasattr(state_store, "postgresql_tenant_schema")
    assert callable(state_store._resolve_postgresql_tenant_schema)


def test_sqlite_factory_preserves_session_message_and_end_contract(tmp_path):
    store = open_state_store({}, db_path=tmp_path / "state.db")
    try:
        assert store.ensure_session("slice", source="test") == "slice"
        message_id = store.append_message("slice", role="user", content="hello")
        assert message_id > 0
        assert [(message["role"], message["content"]) for message in store.get_messages("slice")] == [("user", "hello")]
        store.end_session("slice", "done")
        assert store.get_session("slice")["end_reason"] == "done"
    finally:
        store.close()


def test_sqlite_factory_preserves_session_creation_metadata(tmp_path):
    store = open_state_store({}, db_path=tmp_path / "state.db")
    try:
        store.ensure_session(
            "metadata", source="test", metadata={"user_id": "user", "chat_id": "chat", "model": "model", "cwd": "/work"},
        )
        session = store.get_session("metadata")
        assert {key: session[key] for key in ("user_id", "chat_id", "model", "cwd")} == {
            "user_id": "user", "chat_id": "chat", "model": "model", "cwd": "/work",
        }
    finally:
        store.close()


def test_postgresql_factory_refuses_historic_default_schema_before_hashed_tenant_bootstrap(monkeypatch):
    import state_store

    config = {
        "state_store": {
            "backend": "postgresql",
            "postgresql": {"dsn_env": "TEST_DSN", "connect_timeout_seconds": 1, "pool_max_size": 1},
        },
    }
    monkeypatch.setattr(state_store, "_historic_default_postgresql_state_exists", lambda *_args, **_kwargs: True)
    tenant_schema = state_store._resolve_postgresql_tenant_schema()

    monkeypatch.setattr(
        state_store,
        "_resolve_postgresql_tenant_schema",
        lambda: tenant_schema,
    )

    with pytest.raises(state_store.StateStoreConfigurationError, match="formal reinitialization/cutover"):
        open_state_store(config, secret_lookup=lambda _name: "postgresql://unused")


def test_owned_test_resolver_bypasses_only_historic_default_binding_guard(monkeypatch):
    """A test-owned resolver is not a production factory escape hatch."""
    import state_store
    from state_store_alembic.migration_helpers import _issue_owned_target_tenant_schema

    config = {
        "state_store": {
            "backend": "postgresql",
            "postgresql": {"dsn_env": "TEST_DSN", "connect_timeout_seconds": 1, "pool_max_size": 1},
        },
    }
    owned_schema = _issue_owned_target_tenant_schema("hermes_state_store_tenant_" + "a" * 32)
    monkeypatch.setattr(state_store, "_is_default_state_store_profile", lambda: True)
    monkeypatch.setattr(state_store, "_resolve_postgresql_tenant_schema", lambda: owned_schema)
    historic_calls = []

    def historic_guard(*_args, **kwargs):
        historic_calls.append(kwargs)
        return False

    monkeypatch.setattr(state_store, "_historic_default_postgresql_state_exists", historic_guard)

    captured = {}

    class _Store:
        def __init__(self, _settings, _dsn, *, schema):
            captured["schema"] = schema

    import state_store_postgresql
    monkeypatch.setattr(state_store_postgresql, "PostgreSQLStateStore", _Store)

    assert open_state_store(config, secret_lookup=lambda _name: "postgresql://unused")
    assert historic_calls == [{"canonical_default_binding": False}]
    assert captured["schema"] is owned_schema
    with pytest.raises(TypeError):
        cast(Any, open_state_store)(
            config, secret_lookup=lambda _name: "postgresql://unused", tenant_schema=owned_schema,
        )
