"""Contract tests for backend-neutral state-store configuration."""

import builtins

import pytest

from agent.secret_scope import reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope
from state_store import StateStoreConfigurationError, _scoped_secret, open_state_store, resolve_state_store_config


_PG_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {"dsn_env": "STATE_STORE_TEST_DSN"}}}


def test_default_configuration_keeps_sqlite_backend_without_postgresql_secret():
    resolved = resolve_state_store_config({})

    assert resolved.backend == "sqlite"
    assert resolved.postgresql is None


def test_postgresql_uses_named_secret_without_returning_secret_in_configuration():
    resolved = resolve_state_store_config(
        {
            "state_store": {
                "backend": "postgresql",
                "postgresql": {
                    "dsn_env": "HERMES_STATE_STORE_POSTGRES_DSN",
                    "connect_timeout_seconds": 12,
                    "pool_max_size": 4,
                },
            },
        },
        secret_lookup=lambda name: "postgresql://fixture/only" if name == "HERMES_STATE_STORE_POSTGRES_DSN" else None,
    )

    assert resolved.backend == "postgresql"
    assert resolved.postgresql.dsn_env == "HERMES_STATE_STORE_POSTGRES_DSN"
    assert resolved.postgresql.connect_timeout_seconds == 12
    assert resolved.postgresql.pool_max_size == 4
    assert "fixture/only" not in repr(resolved)


@pytest.mark.parametrize("state_store", [
    {"backend": "postgresql", "postgresql": {"dsn_env": "INVALID-NAME"}},
    {"backend": "postgresql", "postgresql": {"dsn_env": "HERMES_STATE_STORE_POSTGRES_DSN", "pool_max_size": 0}},
    {"backend": "unknown"},
])
def test_invalid_state_store_configuration_fails_closed(state_store):
    with pytest.raises(StateStoreConfigurationError):
        resolve_state_store_config({"state_store": state_store}, secret_lookup=lambda _name: None)


def test_postgresql_without_its_configured_secret_fails_closed():
    with pytest.raises(StateStoreConfigurationError, match="HERMES_STATE_STORE_POSTGRES_DSN"):
        resolve_state_store_config(
            {"state_store": {"backend": "postgresql", "postgresql": {"dsn_env": "HERMES_STATE_STORE_POSTGRES_DSN"}}},
            secret_lookup=lambda _name: None,
        )


def test_postgresql_default_lookup_keeps_unscoped_process_env(monkeypatch):
    monkeypatch.setenv("STATE_STORE_TEST_DSN", "unscoped-fixture-value")

    assert _scoped_secret("STATE_STORE_TEST_DSN") == "unscoped-fixture-value"
    assert resolve_state_store_config(_PG_CONFIG).backend == "postgresql"


def test_postgresql_default_lookup_uses_active_profile_scope(monkeypatch):
    monkeypatch.setenv("STATE_STORE_TEST_DSN", "foreign-profile-fixture-value")
    multiplex_token = set_multiplex_context(True)
    scope_token = set_secret_scope({"STATE_STORE_TEST_DSN": "active-profile-fixture-value"})
    try:
        assert _scoped_secret("STATE_STORE_TEST_DSN") == "active-profile-fixture-value"
        assert resolve_state_store_config(_PG_CONFIG).backend == "postgresql"
        reset_secret_scope(scope_token)
        scope_token = set_secret_scope({})
        assert _scoped_secret("STATE_STORE_TEST_DSN") is None
        with pytest.raises(StateStoreConfigurationError, match="STATE_STORE_TEST_DSN"):
            resolve_state_store_config(_PG_CONFIG)
    finally:
        reset_secret_scope(scope_token)
        reset_multiplex_context(multiplex_token)


def test_postgresql_scope_lookup_import_failure_never_uses_process_env_or_opens_store(monkeypatch):
    original_import = builtins.__import__

    def fail_scoped_lookup_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hermes_cli.config" and "_env_ref_lookup" in fromlist:
            raise ImportError("scoped lookup unavailable")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as isolated:
        isolated.setenv("STATE_STORE_TEST_DSN", "foreign-profile-fixture-value")
        isolated.setattr(builtins, "__import__", fail_scoped_lookup_import)
        # If the foreign process secret is accepted, opening the store would reach this seam.
        isolated.setattr("state_store._resolve_postgresql_tenant_schema", lambda: pytest.fail("store reached"))

        for lookup in (
            lambda: _scoped_secret("STATE_STORE_TEST_DSN"),
            lambda: resolve_state_store_config(_PG_CONFIG),
            lambda: open_state_store(_PG_CONFIG),
        ):
            with pytest.raises(ImportError, match="scoped lookup unavailable"):
                lookup()


def test_config_structure_reports_invalid_state_store_shape():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({"state_store": {"backend": "postgresql", "postgresql": "not-a-map"}})

    assert any(issue.severity == "error" and "state_store.postgresql" in issue.message for issue in issues)
