"""Profile isolation and explicit selection for session database configuration."""

import os
from dataclasses import FrozenInstanceError

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state_backend import DatabaseSettings, resolve_database_settings


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    monkeypatch.delenv("HERMES_DATABASE_URL", raising=False)
    token = set_hermes_home_override(None)
    yield home
    reset_hermes_home_override(token)


def configure_postgres(home, *, schema=None, database_url="postgresql://hermes:secret@localhost/hermes"):
    home.mkdir(parents=True, exist_ok=True)
    config = "database:\n  backend: postgres\n"
    if schema is not None:
        config += f"  schema: {schema}\n"
    (home / "config.yaml").write_text(config, encoding="utf-8")
    if database_url is not None:
        (home / ".env").write_text(f"HERMES_DATABASE_URL={database_url}\n", encoding="utf-8")


def test_sqlite_default_does_not_need_credentials(profile_home, monkeypatch):
    monkeypatch.setenv("HERMES_DATABASE_URL", "postgresql://ignored/ignored")
    assert resolve_database_settings() == DatabaseSettings(backend="sqlite")


def test_postgres_settings_are_immutable_and_hide_credentials(profile_home):
    configure_postgres(profile_home, schema="hermes_shared")
    settings = resolve_database_settings()
    assert settings.backend == "postgres"
    assert settings.schema == "hermes_shared"
    assert settings.database_url == "postgresql://hermes:secret@localhost/hermes"
    assert "secret" not in repr(settings)
    with pytest.raises(FrozenInstanceError):
        settings.backend = "sqlite"


def test_memory_database_stays_sqlite_with_invalid_profile_config(profile_home):
    (profile_home / "config.yaml").write_text("database: [", encoding="utf-8")
    assert resolve_database_settings(":memory:") == DatabaseSettings()


def test_explicit_database_path_uses_its_own_profile(profile_home):
    other = profile_home / "profiles" / "work"
    configure_postgres(other)
    assert resolve_database_settings(other / "archive.db").backend == "postgres"
    assert resolve_database_settings().backend == "sqlite"


def test_default_schema_is_stable_and_isolates_database_paths(profile_home):
    configure_postgres(profile_home)
    first = resolve_database_settings(profile_home / "state.db")
    assert first.schema == resolve_database_settings(profile_home / "." / "state.db").schema
    assert first.schema != resolve_database_settings(profile_home / "other.db").schema


def test_explicit_schema_can_identify_shared_store_across_homes(profile_home, tmp_path):
    other = tmp_path / "other-machine"
    configure_postgres(profile_home, schema="shared_profile")
    configure_postgres(other, schema="shared_profile")
    assert resolve_database_settings().schema == resolve_database_settings(other / "state.db").schema


def test_process_url_overrides_process_home_dotenv(profile_home, monkeypatch):
    configure_postgres(profile_home)
    monkeypatch.setenv("HERMES_DATABASE_URL", "postgresql://process/hermes")
    assert resolve_database_settings().database_url == "postgresql://process/hermes"


def test_named_profile_never_inherits_process_credentials(profile_home, monkeypatch):
    other = profile_home / "profiles" / "work"
    configure_postgres(other, database_url=None)
    monkeypatch.setenv("HERMES_DATABASE_URL", "postgresql://default-secret/hermes")
    token = set_hermes_home_override(other)
    try:
        with pytest.raises(ValueError, match="requires HERMES_DATABASE_URL"):
            resolve_database_settings()
    finally:
        reset_hermes_home_override(token)


def test_named_profile_reads_own_dotenv_without_environment_mutation(profile_home, monkeypatch):
    other = profile_home / "profiles" / "work"
    configure_postgres(other, database_url="postgresql://work:work-secret@localhost/work")
    monkeypatch.setenv("HERMES_DATABASE_URL", "postgresql://default-secret/hermes")
    before = dict(os.environ)
    settings = resolve_database_settings(other / "state.db")
    assert settings.database_url == "postgresql://work:work-secret@localhost/work"
    assert dict(os.environ) == before


def test_profile_dotenv_does_not_expand_process_secrets(profile_home, monkeypatch):
    other = profile_home / "profiles" / "work"
    configure_postgres(other, database_url="postgresql://work:${OTHER_PASSWORD}@localhost/work")
    monkeypatch.setenv("OTHER_PASSWORD", "private-default-secret")
    settings = resolve_database_settings(other / "state.db")
    assert "private-default-secret" not in settings.database_url


@pytest.mark.parametrize("body", ["database: [", "- postgres", "postgres", "database: []", "database: null"])
def test_invalid_configuration_never_defaults_to_sqlite(profile_home, body):
    (profile_home / "config.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_database_settings()


def test_unreadable_configuration_does_not_default_to_sqlite(profile_home):
    (profile_home / "config.yaml").mkdir()
    with pytest.raises(ValueError, match="Cannot read database configuration"):
        resolve_database_settings()


@pytest.mark.parametrize("backend", ["postgress", "null", "[]"])
def test_unknown_backend_is_rejected(profile_home, backend):
    (profile_home / "config.yaml").write_text(f"database:\n  backend: {backend}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="database.backend"):
        resolve_database_settings()


def test_postgres_requires_its_credential(profile_home):
    configure_postgres(profile_home, database_url=None)
    with pytest.raises(ValueError, match="requires HERMES_DATABASE_URL"):
        resolve_database_settings()


@pytest.mark.parametrize("schema", ["MixedCase", "bad-name", "pg_system", "[]", "x" * 64])
def test_invalid_schema_is_rejected(profile_home, schema):
    configure_postgres(profile_home, schema=schema)
    with pytest.raises(ValueError, match="database.schema"):
        resolve_database_settings()


def test_managed_database_settings_follow_existing_overlay_rules(profile_home, tmp_path):
    configure_postgres(profile_home, schema="profile_schema")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("database:\n  schema: managed_schema\n", encoding="utf-8")
    assert resolve_database_settings().schema == "managed_schema"
