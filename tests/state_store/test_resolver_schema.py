"""Focused contracts for state-store backend resolution and activation."""

from dataclasses import FrozenInstanceError

import pytest

from hermes_cli.config import (
    DEFAULT_CONFIG,
    ExplicitHomeConfigError,
    load_config_for_home,
)
from hermes_state import SCHEMA_SQL, SessionDB
from state_store import (
    SCHEMA_V22_MANIFEST,
    StateStoreConfigurationError,
    resolve_state_store,
    schema_v22_manifest_parity,
    sqlite_relational_table_names,
)
from state_store.postgres import PostgresConfigurationError


def _postgres_config() -> dict:
    return {
        "sessions": {
            "state": {
                "backend": "postgres",
                "postgres": {
                    "dsn_env": "HERMES_STATE_POSTGRES_DSN",
                    "schema": "profile_state",
                },
            }
        }
    }


def test_default_config_resolves_and_opens_the_existing_sqlite_store(tmp_path):
    spec = resolve_state_store(tmp_path, DEFAULT_CONFIG)

    assert spec.backend == "sqlite"
    assert spec.sqlite_path == tmp_path / "state.db"
    assert spec.profile == "default"
    assert DEFAULT_CONFIG["sessions"]["state"]["postgres"]["dsn_env"] == (
        "HERMES_STATE_POSTGRES_DSN"
    )
    assert DEFAULT_CONFIG["sessions"]["state"]["postgres"]["schema"] is None
    assert spec.postgres_schema is None

    db = SessionDB.for_home(tmp_path, config=DEFAULT_CONFIG)
    try:
        assert db.db_path == tmp_path / "state.db"
    finally:
        db.close()


def test_for_home_loads_target_config_without_ambient_profile_state(tmp_path, monkeypatch):
    ambient_home = tmp_path / "ambient"
    target_home = tmp_path / "target"
    ambient_home.mkdir()
    target_home.mkdir()
    (ambient_home / "config.yaml").write_text(
        "sessions:\n  state:\n    backend: sqlite\n",
        encoding="utf-8",
    )
    (target_home / "config.yaml").write_text(
        "sessions:\n"
        "  state:\n"
        "    backend: postgres\n"
        "    postgres:\n"
        "      schema: target_state\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))

    with pytest.raises(PostgresConfigurationError):
        SessionDB.for_home(target_home)

    assert not (target_home / "state.db").exists()


def test_load_config_for_home_applies_normalization_expansion_and_managed_overlay(
    tmp_path, monkeypatch
):
    target_home = tmp_path / "target"
    managed_home = tmp_path / "managed"
    target_home.mkdir()
    managed_home.mkdir()
    (target_home / "config.yaml").write_text(
        "max_turns: 7\n"
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: ${STATE_DB_DIR}/user.db\n",
        encoding="utf-8",
    )
    (managed_home / "config.yaml").write_text(
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: ${STATE_DB_DIR}/managed.db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STATE_DB_DIR", str(tmp_path / "expanded"))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "ambient"))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    config = load_config_for_home(target_home)

    assert config["agent"]["max_turns"] == 7
    assert config["sessions"]["state"]["sqlite_path"] == str(
        tmp_path / "expanded" / "managed.db"
    )


def test_explicit_home_expansion_binds_hermes_home_to_target(tmp_path, monkeypatch):
    ambient_home = tmp_path / "ambient"
    target_home = tmp_path / "target"
    target_home.mkdir()
    (target_home / "config.yaml").write_text(
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: ${HERMES_HOME}/state.db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))

    config = load_config_for_home(target_home)
    assert config["sessions"]["state"]["sqlite_path"] == str(
        target_home / "state.db"
    )

    db = SessionDB.for_home(target_home)
    try:
        assert db.db_path == target_home / "state.db"
    finally:
        db.close()

    assert not (ambient_home / "state.db").exists()


def test_managed_explicit_home_expansion_binds_hermes_home_to_target(
    tmp_path, monkeypatch
):
    ambient_home = tmp_path / "ambient"
    target_home = tmp_path / "target"
    managed_home = tmp_path / "managed"
    target_home.mkdir()
    managed_home.mkdir()
    (managed_home / "config.yaml").write_text(
        "sessions:\n"
        "  state:\n"
        "    sqlite_path: ${HERMES_HOME}/state.db\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(ambient_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    config = load_config_for_home(target_home)
    assert config["sessions"]["state"]["sqlite_path"] == str(
        target_home / "state.db"
    )

    db = SessionDB.for_home(target_home)
    try:
        assert db.db_path == target_home / "state.db"
    finally:
        db.close()

    assert not (ambient_home / "state.db").exists()


def test_for_home_fails_closed_without_exposing_malformed_config_values(tmp_path):
    target_home = tmp_path / "target"
    target_home.mkdir()
    secret_dsn = "postgresql://hermes:super-secret@db.example/state"
    (target_home / "config.yaml").write_text(
        f"sessions: [{secret_dsn}",
        encoding="utf-8",
    )

    with pytest.raises(ExplicitHomeConfigError) as exc_info:
        SessionDB.for_home(target_home)

    assert secret_dsn not in str(exc_info.value)
    assert not (target_home / "state.db").exists()


def test_explicit_homes_keep_sqlite_store_specs_isolated(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    coder_home = default_home / "profiles" / "coder"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "wrong-profile"))

    default_spec = resolve_state_store(default_home, DEFAULT_CONFIG)
    coder_spec = resolve_state_store(coder_home, DEFAULT_CONFIG)

    assert default_spec.profile == "default"
    assert coder_spec.profile == "coder"
    assert default_spec.sqlite_path != coder_spec.sqlite_path
    assert default_spec.store_key != coder_spec.store_key


def test_postgres_requires_explicit_schema_and_survives_profile_rename(tmp_path):
    config = {"sessions": {"state": {"backend": "postgres"}}}

    with pytest.raises(StateStoreConfigurationError, match="explicitly configured"):
        resolve_state_store(tmp_path / ".hermes" / "profiles" / "before-rename", config)

    config["sessions"]["state"]["postgres"] = {"schema": "owned_state"}
    before = resolve_state_store(
        tmp_path / ".hermes" / "profiles" / "before-rename", config
    )
    after = resolve_state_store(
        tmp_path / "relocated" / "profiles" / "after-rename", config
    )

    assert before.postgres_schema == after.postgres_schema == "owned_state"


@pytest.mark.parametrize("schema", ["tenant_state", "_tenant_state", "x" * 63])
def test_explicit_postgres_schema_accepts_safe_identifiers(tmp_path, schema):
    config = _postgres_config()
    config["sessions"]["state"]["postgres"]["schema"] = schema

    assert resolve_state_store(tmp_path, config).postgres_schema == schema


@pytest.mark.parametrize(
    "schema",
    ["tenant-state", "TenantState", "1tenant", "public;drop_schema", "x" * 64],
)
def test_explicit_postgres_schema_rejects_unsafe_identifiers(tmp_path, schema):
    config = _postgres_config()
    config["sessions"]["state"]["postgres"]["schema"] = schema

    with pytest.raises(StateStoreConfigurationError) as exc_info:
        resolve_state_store(tmp_path, config)

    assert schema not in str(exc_info.value)


@pytest.mark.parametrize(
    "config",
    [
        {"sessions": []},
        {"sessions": {"state": None}},
        {"sessions": {"state": {"postgres": "invalid"}}},
    ],
)
def test_malformed_state_maps_fail_closed_before_sqlite_open(tmp_path, config):
    with pytest.raises(StateStoreConfigurationError):
        SessionDB.for_home(tmp_path, config=config)

    assert not (tmp_path / "state.db").exists()


def test_sqlite_path_expands_user_home_explicitly(tmp_path, monkeypatch):
    home_dir = tmp_path / "os-home"
    monkeypatch.setenv("HOME", str(home_dir))
    config = {
        "sessions": {
            "state": {
                "sqlite_path": "~/state/custom.db",
            }
        }
    }

    spec = resolve_state_store(tmp_path / "target", config)

    assert spec.sqlite_path == home_dir / "state" / "custom.db"


def test_postgres_spec_is_immutable_and_never_reads_or_exposes_dsn(tmp_path):
    secret_dsn = "postgresql://hermes:super-secret@db.example/state"
    spec = resolve_state_store(
        tmp_path / "profiles" / "ops",
        _postgres_config(),
        environ={"HERMES_STATE_POSTGRES_DSN": secret_dsn},
    )

    assert spec.backend == "postgres"
    assert spec.postgres_dsn_env == "HERMES_STATE_POSTGRES_DSN"
    assert spec.postgres_schema == "profile_state"
    assert secret_dsn not in repr(spec)
    assert secret_dsn not in spec.store_key
    assert spec.store_key == resolve_state_store(
        tmp_path / "profiles" / "ops",
        _postgres_config(),
    ).store_key
    with pytest.raises(FrozenInstanceError):
        spec.backend = "sqlite"


def test_invalid_dsn_reference_redacts_plaintext_values(tmp_path):
    secret_dsn = "postgresql://hermes:super-secret@db.example/state"
    config = _postgres_config()
    config["sessions"]["state"]["postgres"]["dsn_env"] = secret_dsn

    with pytest.raises(StateStoreConfigurationError) as exc_info:
        resolve_state_store(tmp_path, config)

    assert secret_dsn not in str(exc_info.value)


def test_read_only_spec_and_factory_use_the_explicit_home(tmp_path):
    writable = SessionDB.for_home(tmp_path, config=DEFAULT_CONFIG)
    writable.close()

    spec = resolve_state_store(tmp_path, DEFAULT_CONFIG, read_only=True)
    assert spec.read_only is True

    readonly = SessionDB.for_home(tmp_path, read_only=True, config=DEFAULT_CONFIG)
    try:
        assert readonly.read_only is True
        assert readonly.db_path == tmp_path / "state.db"
    finally:
        readonly.close()


def test_postgres_configuration_never_implicitly_falls_back_to_sqlite(tmp_path):
    spec = resolve_state_store(tmp_path, _postgres_config())
    assert spec.backend == "postgres"
    assert spec.sqlite_path == tmp_path / "state.db"

    with pytest.raises(PostgresConfigurationError) as exc_info:
        SessionDB.for_home(tmp_path, config=_postgres_config())

    assert "postgresql://" not in str(exc_info.value).lower()
    assert not (tmp_path / "state.db").exists()


def test_schema_v22_manifest_matches_sqlite_core_and_opt_in_telegram_tables(tmp_path):
    core_tables = sqlite_relational_table_names(SCHEMA_SQL)
    assert core_tables == SCHEMA_V22_MANIFEST.core_tables

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.apply_telegram_topic_migration()
        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        parity = schema_v22_manifest_parity(
            core_tables=core_tables,
            telegram_tables={
                table for table in tables if table.startswith("telegram_dm_topic_")
            },
        )
    finally:
        db.close()

    assert parity.matches
