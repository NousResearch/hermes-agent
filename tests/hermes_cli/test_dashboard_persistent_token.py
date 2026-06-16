"""End-to-end tests for persistent dashboard session token.

Tests that push_env_into_os_environ() correctly bridges .env file
entries into os.environ so the dashboard's web_server module sees
HERMES_DASHBOARD_SESSION_TOKEN at import time.
"""
import os

import pytest

from hermes_cli.config import push_env_into_os_environ


@pytest.fixture
def temp_env_file(tmp_path):
    return tmp_path / ".env"


def test_push_sets_token_from_env_file(temp_env_file, monkeypatch):
    token = "not-a-real-token-just-a-test-value"
    env_key = "SOME_TEST_ENV_VAR"
    temp_env_file.write_text(env_key + "=" + token + "\n")

    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: temp_env_file)
    monkeypatch.delenv(env_key, raising=False)

    pushed = push_env_into_os_environ()
    assert pushed >= 1
    assert os.environ[env_key] == token


def test_push_respects_existing_os_environ(temp_env_file, monkeypatch):
    env_key = "SOME_TEST_ENV_VAR"
    temp_env_file.write_text(env_key + "=from_dotenv\n")

    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: temp_env_file)
    monkeypatch.setenv(env_key, "from_process")

    pushed = push_env_into_os_environ()
    assert os.environ[env_key] == "from_process"


def test_push_returns_zero_for_missing_env(temp_env_file, monkeypatch):
    missing = temp_env_file.with_suffix(".missing")

    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: missing)

    pushed = push_env_into_os_environ()
    assert pushed == 0


def test_push_returns_zero_for_empty_env(temp_env_file, monkeypatch):
    temp_env_file.write_text("")

    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: temp_env_file)

    pushed = push_env_into_os_environ()
    assert pushed == 0


def test_push_skips_comments_and_blanks(temp_env_file, monkeypatch):
    env_key = "SOME_TEST_ENV_VAR"
    temp_env_file.write_text(
        "# comment line\n"
        "\n"
        + env_key + "=hello_world\n"
        "  # another comment  \n"
    )

    monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: temp_env_file)
    monkeypatch.delenv(env_key, raising=False)

    pushed = push_env_into_os_environ()
    assert pushed == 1
    assert os.environ[env_key] == "hello_world"
