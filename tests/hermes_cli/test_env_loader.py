import importlib
import os
import sys
from pathlib import Path

import pytest

from hermes_cli import env_loader
from hermes_cli.env_loader import load_hermes_dotenv


@pytest.fixture(autouse=True)
def _reset_seeded_keys():
    """Each test starts with a clean Hermes-seeded-keys registry.

    The module-level _HERMES_SEEDED_KEYS set persists across pytest tests
    in the same process, which would let one test's seeded keys affect
    another's pruning behavior.  Snapshot and restore around every test.
    """
    snapshot = set(env_loader._HERMES_SEEDED_KEYS)
    env_loader._HERMES_SEEDED_KEYS.clear()
    try:
        yield
    finally:
        env_loader._HERMES_SEEDED_KEYS.clear()
        env_loader._HERMES_SEEDED_KEYS.update(snapshot)


def test_user_env_overrides_stale_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://new.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"


def test_project_env_overrides_stale_shell_values_when_user_env_missing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://project.example/v1"


def test_project_env_is_sanitized_before_loading(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "TELEGRAM_BOT_TOKEN=0123456789:test"
        "ANTHROPIC_API_KEY=sk-ant-test123\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-test123"


def test_user_env_takes_precedence_over_project_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    project_env = tmp_path / ".env"
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\nOPENAI_API_KEY=project-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [user_env, project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    assert os.getenv("OPENAI_API_KEY") == "project-key"


def test_seeded_key_removed_from_file_is_pruned_on_reload(tmp_path, monkeypatch):
    """A key we placed into os.environ should disappear when removed from .env.

    Regression for the long-standing "ghost env var" bug where a deleted
    key kept its stale value for the lifetime of every long-running
    Hermes process (gateway, dashboard, CLI sessions).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("HERMES_TEST_GHOST_KEY=initial-value\n", encoding="utf-8")

    monkeypatch.delenv("HERMES_TEST_GHOST_KEY", raising=False)

    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("HERMES_TEST_GHOST_KEY") == "initial-value"

    env_file.write_text("OPENAI_BASE_URL=https://example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_TEST_GHOST_KEY") is None
    assert os.getenv("OPENAI_BASE_URL") == "https://example/v1"


def test_shell_exported_key_never_pruned(tmp_path, monkeypatch):
    """Keys we never seeded (genuine shell exports) must survive reloads.

    Even after the user's .env briefly declared then dropped the key,
    we must not pop the shell-exported value out of os.environ.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://example/v1\n", encoding="utf-8")

    # Shell-exported value the user expects us to leave alone.
    monkeypatch.setenv("HERMES_USER_SHELL_VAR", "from-shell")

    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("HERMES_USER_SHELL_VAR") == "from-shell"

    # Reload: still untouched, still present.
    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("HERMES_USER_SHELL_VAR") == "from-shell"


def test_bare_key_without_value_is_pruned_on_reload(tmp_path, monkeypatch):
    """A bare ``KEY`` line (no ``=``) does not seed os.environ — previously
    seeded values for that key should still be pruned.

    ``dotenv_values`` reports ``KEY -> None`` for a bare ``KEY`` line, but
    ``load_dotenv`` does NOT actually update os.environ for it.  Without
    filtering None-valued declarations, the pruner would treat the key as
    "still declared" and skip it, leaving a ghost value behind.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("MY_FLAG=enabled\n", encoding="utf-8")

    monkeypatch.delenv("MY_FLAG", raising=False)

    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("MY_FLAG") == "enabled"

    # Rewrite to a bare key with no value.
    env_file.write_text("MY_FLAG\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("MY_FLAG") is None, (
        "bare KEY (no =) does not seed os.environ; the previously seeded "
        "value must be pruned to avoid a ghost"
    )


def test_empty_value_keeps_key_declared(tmp_path, monkeypatch):
    """A ``KEY=`` line (empty value) IS seeded by load_dotenv as ``""`` and
    must be treated as declared, not pruned.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("MY_FLAG=enabled\n", encoding="utf-8")

    monkeypatch.delenv("MY_FLAG", raising=False)

    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("MY_FLAG") == "enabled"

    env_file.write_text("MY_FLAG=\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home)

    # Empty string, not None — load_dotenv overwrote with "".
    assert os.getenv("MY_FLAG") == ""


def test_seeded_key_pruned_when_only_in_one_of_two_files(tmp_path, monkeypatch):
    """A key declared by user_env then dropped, while a project_env never
    declared it, must be pruned on reload.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    project_env = tmp_path / ".env"
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    user_env.write_text(
        "OPENAI_BASE_URL=https://user.example/v1\nMY_USER_FLAG=on\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MY_USER_FLAG", raising=False)

    load_hermes_dotenv(hermes_home=home, project_env=project_env)
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    assert os.getenv("MY_USER_FLAG") == "on"

    # Drop MY_USER_FLAG from user_env; project_env never had it.
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert os.getenv("MY_USER_FLAG") is None
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"


def test_seeded_key_value_change_propagates_on_reload(tmp_path, monkeypatch):
    """Editing a key's value in .env still updates os.environ on reload."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://v1.example/v1\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("OPENAI_BASE_URL") == "https://v1.example/v1"

    env_file.write_text("OPENAI_BASE_URL=https://v2.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("OPENAI_BASE_URL") == "https://v2.example/v1"


def test_main_import_applies_user_env_over_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://new.example/v1\nHERMES_INFERENCE_PROVIDER=custom\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

    sys.modules.pop("hermes_cli.main", None)
    importlib.import_module("hermes_cli.main")

    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"
    assert os.getenv("HERMES_INFERENCE_PROVIDER") == "custom"
