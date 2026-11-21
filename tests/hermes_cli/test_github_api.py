"""The github_api credential ladder: env → active home's .env → gh CLI → anonymous.

A GUI-launched Desktop spawns the update check via ``--run-module``, which never
runs the CLI's dotenv load — the .env rung is what keeps a token configured via
``hermes config set GITHUB_TOKEN`` visible to that subprocess.
"""
import pytest

from hermes_cli import github_api


@pytest.fixture(autouse=True)
def _clean_ladder(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_api, "_gh_cli_token", lambda: None)
    github_api._dotenv_cache.clear()
    yield


def test_env_token_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("GITHUB_TOKEN=dotenv-token\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "env-token")
    assert github_api.github_token() == "env-token"


def test_dotenv_token_when_env_and_gh_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("GITHUB_TOKEN=dotenv-token\n", encoding="utf-8")
    assert github_api.github_token_from_env() is None
    assert github_api.github_token() == "dotenv-token"


def test_dotenv_gh_token_fallback_and_quoting(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text('GH_TOKEN="quoted-gh-token"\n', encoding="utf-8")
    assert github_api.github_token() == "quoted-gh-token"


def test_missing_or_blank_dotenv_is_anonymous(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert github_api.github_token() is None
    (tmp_path / ".env").write_text("OTHER_KEY=x\nGITHUB_TOKEN=\n", encoding="utf-8")
    github_api._dotenv_cache.clear()
    assert github_api.github_token() is None


def test_unreadable_dotenv_is_anonymous_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_bytes(b"\xff\xfe GITHUB_TOKEN=not-utf8")
    assert github_api.github_token() is None


def test_dotenv_cache_is_keyed_by_home(tmp_path, monkeypatch):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / ".env").write_text("GITHUB_TOKEN=token-a\n", encoding="utf-8")
    (home_b / ".env").write_text("GITHUB_TOKEN=token-b\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert github_api.github_token() == "token-a"
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert github_api.github_token() == "token-b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert github_api.github_token() == "token-a"


def test_dotenv_rung_never_mutates_os_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("GITHUB_TOKEN=dotenv-token\n", encoding="utf-8")
    assert github_api.github_token() == "dotenv-token"
    import os
    assert os.environ.get("GITHUB_TOKEN") is None
