"""Env integration tests — managed .env applied last with override."""
import os

import pytest

pytest_plugins = ["tests.agent.test_secret_layer_precedence"]


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


def test_managed_env_beats_user_env(env_homes, profile_store, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    _unused_home, managed = env_homes
    home, seed = profile_store
    seed({"OPENAI_API_BASE": "https://user.example/v1"})
    (managed / ".env").write_text("OPENAI_API_BASE=https://org.example/v1\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["OPENAI_API_BASE"] == "https://org.example/v1"


def test_no_managed_env_is_noop(env_homes, monkeypatch):
    from hermes_cli.env_loader import load_hermes_dotenv

    home, _managed = env_homes  # managed dir exists but has no .env
    monkeypatch.setenv("SOME_VALUE", "from_shell")
    (home / ".env").write_text("SOME_VALUE=from_user\n", encoding="utf-8")
    load_hermes_dotenv(hermes_home=str(home))
    assert os.environ["SOME_VALUE"] == "from_user"
