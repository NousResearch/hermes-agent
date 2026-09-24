"""Contract tests for backend-neutral state-store configuration."""

import pytest

from state_store import StateStoreConfigurationError, resolve_state_store_config


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


def test_config_structure_reports_invalid_state_store_shape():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({"state_store": {"backend": "postgresql", "postgresql": "not-a-map"}})

    assert any(issue.severity == "error" and "state_store.postgresql" in issue.message for issue in issues)
