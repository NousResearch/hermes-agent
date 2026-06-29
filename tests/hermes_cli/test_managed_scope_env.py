"""Env integration tests — managed .env applied last with override."""
import os

import pytest


@pytest.fixture
def env_homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return home, managed


def test_managed_env_beats_user_env(env_homes, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    home, managed = env_homes
    (home / ".env").write_text("OPENAI_API_BASE=https://user.example/v1\n", encoding="utf-8")
    (managed / ".env").write_text("OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["OPENAI_API_BASE"] == "https://org.example/v1"


def test_managed_env_beats_shell(env_homes, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    home, managed = env_homes
    monkeypatch.setenv("OPENAI_API_BASE", "https://shell.example/v1")
    (managed / ".env").write_text("OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["OPENAI_API_BASE"] == "https://org.example/v1"


def test_managed_env_leaves_unmanaged_keys_alone(env_homes, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    home, managed = env_homes
    (home / ".env").write_text("USER_ONLY=keepme\n", encoding="utf-8")
    (managed / ".env").write_text("OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["USER_ONLY"] == "keepme"
    assert os.environ["OPENAI_API_BASE"] == "https://org.example/v1"


def test_no_managed_env_is_noop(env_homes, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    home, managed = env_homes  # managed dir exists but has no .env
    monkeypatch.setenv("SOME_VALUE", "from_shell")
    monkeypatch.setenv("HERMES_DOTENV_OVERRIDE", "1")
    (home / ".env").write_text("SOME_VALUE=from_user\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["SOME_VALUE"] == "from_user"
