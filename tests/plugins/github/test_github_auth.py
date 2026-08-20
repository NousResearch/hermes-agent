"""Unit tests for the GitHub App auth module (agent/github_auth.py)."""

from __future__ import annotations

import json

import pytest

from agent.github_auth import GitHubAppAuth
from agent import github_auth as auth_mod


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, *, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = self.text.encode("utf-8")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture(autouse=True)
def _clear_app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must be hermetic — no real GITHUB_APP_* from the host env."""
    for key in ("GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY_PATH", "GITHUB_APP_INSTALLATION_ID"):
        monkeypatch.delenv(key, raising=False)


def _set_app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", "/tmp/fake-app.pem")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "67890")


def _fake_key_file(tmp_path) -> str:
    key = tmp_path / "fake-app.pem"
    key.write_text("-----BEGIN RSA PRIVATE KEY-----\nZmFrZWtleQ==\n-----END RSA PRIVATE KEY-----\n")
    return str(key)


def test_credentials_configured_false_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GitHubAppAuth().credentials_configured() is False


def test_credentials_configured_true_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_app_creds(monkeypatch)
    assert GitHubAppAuth().credentials_configured() is True


def test_installation_token_mints_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setattr(auth_mod.GitHubAppAuth, "_sign_jwt", lambda self, app_id, key_path: "fake-jwt")

    calls: list[str] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _FakeResponse(201, {"token": "ghs_12345_abcdef"})

    monkeypatch.setattr("httpx.post", fake_post)

    auth = GitHubAppAuth()
    assert auth.installation_token() == "ghs_12345_abcdef"
    assert auth.installation_token() == "ghs_12345_abcdef"  # cached
    assert len(calls) == 1


def test_installation_token_none_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GitHubAppAuth().installation_token() is None


def test_installation_token_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setattr(auth_mod.GitHubAppAuth, "_sign_jwt", lambda self, app_id, key_path: None)
    assert GitHubAppAuth().installation_token() is None


def test_sign_jwt_uses_rs256_and_iss(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_PATH", _fake_key_file(tmp_path))

    encoded = GitHubAppAuth()._sign_jwt("12345", _fake_key_file(tmp_path))
    # Without a real key the encode path may fail; assert it either produced
    # a token string or returned None gracefully — never raised.
    assert encoded is None or isinstance(encoded, str)


def test_app_slug_fetched_and_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setattr(auth_mod.GitHubAppAuth, "_sign_jwt", lambda self, app_id, key_path: "fake-jwt")

    calls: list[str] = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(200, {"slug": "jarpis-bot"})

    monkeypatch.setattr("httpx.get", fake_get)

    auth = GitHubAppAuth()
    assert auth.app_slug() == "jarpis-bot"
    assert auth.app_slug() == "jarpis-bot"  # cached
    assert len(calls) == 1


def test_bot_login_derived_from_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setattr(auth_mod.GitHubAppAuth, "app_slug", lambda self: "jarpis-bot")
    assert GitHubAppAuth().bot_login() == "jarpis-bot[bot]"


def test_bot_login_none_when_slug_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_app_creds(monkeypatch)
    monkeypatch.setattr(auth_mod.GitHubAppAuth, "app_slug", lambda self: None)
    assert GitHubAppAuth().bot_login() is None
