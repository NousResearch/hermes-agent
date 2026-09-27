"""Behavior contracts for the interactive PostgreSQL state-store onboarding."""

import argparse
import os
from unittest.mock import patch

import yaml


def test_sqlite_choice_is_default_and_never_requests_or_bootstraps_a_dsn():
    from hermes_cli.setup_state_store import configure_state_store

    config = {}
    with patch("hermes_cli.setup.prompt_choice", return_value=2) as choice, \
         patch("hermes_cli.setup.save_env_value") as save_secret, \
         patch("hermes_cli.setup_state_store.bootstrap_postgresql_state_store") as bootstrap:
        configure_state_store(config)

    assert choice.call_args.args[2] == 2
    assert config["state_store"]["backend"] == "sqlite"
    save_secret.assert_not_called()
    bootstrap.assert_not_called()


def test_existing_dsn_is_saved_as_a_profile_secret_and_only_its_name_reaches_config():
    from hermes_cli.setup_state_store import POSTGRES_DSN_ENV, configure_state_store

    config = {}
    dsn = "postgresql://user:secret@db.example.test:5432/hermes"
    with patch("hermes_cli.setup.prompt_choice", return_value=0), \
         patch("hermes_cli.setup.prompt", return_value=dsn), \
         patch("hermes_cli.setup.save_env_value") as save_secret:
        configure_state_store(config)

    save_secret.assert_called_once_with(POSTGRES_DSN_ENV, dsn)
    assert config["state_store"]["backend"] == "postgresql"
    assert config["state_store"]["postgresql"]["dsn_env"] == POSTGRES_DSN_ENV
    assert dsn not in repr(config)


def test_docker_choice_bootstraps_then_saves_returned_dsn_as_secret():
    from hermes_cli.setup_state_store import POSTGRES_DSN_ENV, configure_state_store

    config = {}
    dsn = "postgresql://hermes:generated@127.0.0.1:55432/hermes"
    with patch("hermes_cli.setup.prompt_choice", return_value=1), \
         patch("hermes_cli.setup.save_env_value") as save_secret, \
         patch("hermes_cli.setup_state_store.bootstrap_postgresql_state_store", return_value=dsn) as bootstrap:
        configure_state_store(config)

    bootstrap.assert_called_once()
    save_secret.assert_called_once_with(POSTGRES_DSN_ENV, dsn)
    assert config["state_store"]["backend"] == "postgresql"
    assert config["state_store"]["postgresql"]["dsn_env"] == POSTGRES_DSN_ENV


def test_config_state_store_command_round_trips_sqlite_selection_without_secret(tmp_path, monkeypatch):
    from hermes_cli.config import config_command, load_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with patch("hermes_cli.setup.prompt_choice", return_value=2):
        config_command(argparse.Namespace(config_command="state-store"))

    resolved = load_config()
    assert resolved["state_store"]["backend"] == "sqlite"
    assert not (tmp_path / ".env").exists()


def test_docker_bootstrap_does_not_reuse_an_external_dsn_password(tmp_path):
    from hermes_cli.postgresql_bootstrap import bootstrap_postgresql_state_store

    passwords = []

    def runner(argv, *, env):
        passwords.append(env["POSTGRES_PASSWORD"])
        return 0

    external_password = "external-test-password"
    bootstrap_postgresql_state_store(
        tmp_path,
        existing_dsn=f"postgresql://external:{external_password}@db.example.test:5432/hermes",
        runner=runner,
        sleep=lambda _: None,
        attempts=1,
    )

    assert passwords
    assert all(password != external_password for password in passwords)


def test_docker_bootstrap_reuses_its_local_password_on_rerun(tmp_path):
    from hermes_cli.postgresql_bootstrap import bootstrap_postgresql_state_store

    passwords = []

    def runner(argv, *, env):
        passwords.append(env["POSTGRES_PASSWORD"])
        return 0

    first_dsn = bootstrap_postgresql_state_store(tmp_path, runner=runner, sleep=lambda _: None, attempts=1)
    second_dsn = bootstrap_postgresql_state_store(
        tmp_path,
        existing_dsn=first_dsn,
        runner=runner,
        sleep=lambda _: None,
        attempts=1,
    )

    assert second_dsn == first_dsn
    assert len(set(passwords)) == 1


def test_bootstrap_writes_loopback_compose_and_waits_for_readiness(tmp_path):
    from hermes_cli.postgresql_bootstrap import bootstrap_postgresql_state_store

    calls = []
    def runner(argv, *, env):
        calls.append((argv, env))
        return 0

    dsn = bootstrap_postgresql_state_store(tmp_path, runner=runner, sleep=lambda _: None, attempts=1)

    compose = yaml.safe_load((tmp_path / "state-store" / "postgresql" / "compose.yaml").read_text())
    service = compose["services"]["postgres"]
    assert service["image"] == "pgvector/pgvector:pg16"
    assert service["environment"]["POSTGRES_PASSWORD"] == "${POSTGRES_PASSWORD}"
    assert service["ports"] == [f"127.0.0.1:{dsn.rsplit(':', 1)[1].split('/', 1)[0]}:5432"]
    assert any("up" in argv for argv, _ in calls)
    assert any("pg_isready" in argv for argv, _ in calls)
    assert all("generated" not in " ".join(argv) for argv, _ in calls)
