import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.auth import (
    AuthError,
    DEFAULT_CODEX_BASE_URL,
    PROVIDER_REGISTRY,
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


def test_resolve_provider_explicit_codex_does_not_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("openai-codex") == "openai-codex"


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
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr("hermes_cli.auth.subprocess.run", lambda *a, **k: None)

    _login_openai_codex(SimpleNamespace(), PROVIDER_REGISTRY["openai-codex"])

    state = get_provider_auth_state("openai-codex")
    assert state is not None
    assert state["source"] == "codex-auth-json"
    assert state["auth_file"].endswith("auth.json")

    config_path = hermes_home / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["model"]["provider"] == "openai-codex"
    assert config["model"]["base_url"] == DEFAULT_CODEX_BASE_URL


def test_login_command_defaults_to_nous(monkeypatch):
    calls = {"nous": 0, "codex": 0}

    def _fake_nous(args, pconfig):
        calls["nous"] += 1

    def _fake_codex(args, pconfig):
        calls["codex"] += 1

    monkeypatch.setattr("hermes_cli.auth._login_nous", _fake_nous)
    monkeypatch.setattr("hermes_cli.auth._login_openai_codex", _fake_codex)

    login_command(SimpleNamespace())

    assert calls["nous"] == 1
    assert calls["codex"] == 0
