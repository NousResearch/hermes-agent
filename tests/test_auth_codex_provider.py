import json
import time
import base64
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
    _persist_codex_auth_payload,
    _login_openai_codex,
    login_command,
    get_codex_auth_status,
    get_provider_auth_state,
    read_codex_auth_file,
    resolve_codex_runtime_credentials,
    resolve_provider,
)


def _write_codex_auth(codex_home: Path, *, access_token: str = "access", refresh_token: str = "refresh") -> Path:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_file = codex_home / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "auth_mode": "oauth",
                "last_refresh": "2026-02-26T00:00:00Z",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
            }
        )
    )
    return auth_file


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("utf-8")
    return f"h.{encoded}.s"


def test_read_codex_auth_file_success(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    auth_file = _write_codex_auth(codex_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    payload = read_codex_auth_file()

    assert payload["auth_path"] == auth_file
    assert payload["tokens"]["access_token"] == "access"
    assert payload["tokens"]["refresh_token"] == "refresh"


def test_resolve_codex_runtime_credentials_missing_access_token(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    _write_codex_auth(codex_home, access_token="")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(AuthError) as exc:
        resolve_codex_runtime_credentials()

    assert exc.value.code == "codex_auth_missing_access_token"
    assert exc.value.relogin_required is True


def test_resolve_codex_runtime_credentials_refreshes_expiring_token(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    expiring_token = _jwt_with_exp(int(time.time()) - 10)
    _write_codex_auth(codex_home, access_token=expiring_token, refresh_token="refresh-old")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    called = {"count": 0}

    def _fake_refresh(*, payload, auth_path, timeout_seconds, lock_held=False):
        called["count"] += 1
        assert auth_path == codex_home / "auth.json"
        assert lock_held is True
        return {"access_token": "access-new", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials()

    assert called["count"] == 1
    assert resolved["api_key"] == "access-new"


def test_resolve_codex_runtime_credentials_force_refresh(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    _write_codex_auth(codex_home, access_token="access-current", refresh_token="refresh-old")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    called = {"count": 0}

    def _fake_refresh(*, payload, auth_path, timeout_seconds, lock_held=False):
        called["count"] += 1
        assert lock_held is True
        return {"access_token": "access-forced", "refresh_token": "refresh-new"}

    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert called["count"] == 1
    assert resolved["api_key"] == "access-forced"


def test_resolve_codex_runtime_credentials_uses_file_lock_on_refresh(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    _write_codex_auth(codex_home, access_token="access-current", refresh_token="refresh-old")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    lock_calls = {"enter": 0, "exit": 0}

    @contextmanager
    def _fake_lock(auth_path, timeout_seconds=15.0):
        assert auth_path == codex_home / "auth.json"
        lock_calls["enter"] += 1
        try:
            yield
        finally:
            lock_calls["exit"] += 1

    refresh_calls = {"count": 0}

    def _fake_refresh(*, payload, auth_path, timeout_seconds, lock_held=False):
        refresh_calls["count"] += 1
        assert lock_held is True
        return {"access_token": "access-updated", "refresh_token": "refresh-updated"}

    monkeypatch.setattr("hermes_cli.auth._codex_auth_file_lock", _fake_lock)
    monkeypatch.setattr("hermes_cli.auth._refresh_codex_auth_tokens", _fake_refresh)

    resolved = resolve_codex_runtime_credentials(force_refresh=True, refresh_if_expiring=False)

    assert refresh_calls["count"] == 1
    assert lock_calls["enter"] == 1
    assert lock_calls["exit"] == 1
    assert resolved["api_key"] == "access-updated"


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


def test_persist_codex_auth_payload_writes_atomically(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"stale":true}\n')
    payload = {
        "auth_mode": "oauth",
        "tokens": {
            "access_token": "next-access",
            "refresh_token": "next-refresh",
        },
        "last_refresh": "2026-02-26T00:00:00Z",
    }

    _persist_codex_auth_payload(auth_path, payload)

    stored = json.loads(auth_path.read_text())
    assert stored == payload
    assert list(tmp_path.glob(".auth.json.*.tmp")) == []


def test_get_codex_auth_status_not_logged_in(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex-home"))
    status = get_codex_auth_status()
    assert status["logged_in"] is False
    assert "error" in status


def test_login_openai_codex_persists_provider_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    codex_home = tmp_path / "codex-home"
    _write_codex_auth(codex_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # Mock input() to accept existing credentials
    monkeypatch.setattr("builtins.input", lambda _: "y")

    _login_openai_codex(SimpleNamespace(), PROVIDER_REGISTRY["openai-codex"])

    state = get_provider_auth_state("openai-codex")
    assert state is not None
    assert state["source"] == "codex-auth-json"
    assert state["auth_file"].endswith("auth.json")

    config_path = hermes_home / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["provider"] == "openai-codex"
    assert config["model"]["base_url"] == DEFAULT_CODEX_BASE_URL


def test_login_command_shows_deprecation(monkeypatch, capsys):
    """login_command is deprecated and directs users to hermes model."""
    with pytest.raises(SystemExit) as exc_info:
        login_command(SimpleNamespace())
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "hermes model" in captured.out
