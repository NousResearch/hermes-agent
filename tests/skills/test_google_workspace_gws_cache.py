"""gws token-cache isolation between Google accounts (googleworkspace/cli#572)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


def _fake_gws_run(home: Path):
    """Model gws: one token_cache.json per config dir, not keyed by account."""

    def run(cmd, **kwargs):
        env = kwargs["env"]
        config = Path(env.get("GOOGLE_WORKSPACE_CLI_CONFIG_DIR") or home / ".config" / "gws")
        cache = config / "token_cache.json"
        if cache.exists():
            account = json.loads(cache.read_text(encoding="utf-8"))["account"]
        else:
            creds = Path(env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"])
            account = json.loads(creds.read_text(encoding="utf-8"))["account"]
            config.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"account": account}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"emailAddress": account}), "")

    return run


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_WORKSPACE_CLI_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    spec = importlib.util.spec_from_file_location("gws_cache_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._gws_binary = lambda: "/usr/bin/gws"
    module._ensure_authenticated = lambda: None
    monkeypatch.setattr(module.subprocess, "run", _fake_gws_run(tmp_path))
    return module


def _use_account(module, monkeypatch, home: Path, account: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    token = home / "google_token.json"
    token.write_text(
        json.dumps({
            "type": "authorized_user",
            "client_id": "shared-client",
            "client_secret": "secret",
            "refresh_token": f"1//refresh-{account}",
            "account": account,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "HERMES_HOME", home)
    monkeypatch.setattr(module, "TOKEN_PATH", token)


def _whoami(module) -> str:
    return module._run_gws(["gmail", "users", "getProfile"], params={"userId": "me"})["emailAddress"]


def test_second_account_does_not_reuse_first_accounts_cached_token(api_module, monkeypatch, tmp_path):
    _use_account(api_module, monkeypatch, tmp_path / "profile-a", "a@example.com")
    assert _whoami(api_module) == "a@example.com"

    _use_account(api_module, monkeypatch, tmp_path / "profile-b", "b@example.com")
    assert _whoami(api_module) == "b@example.com"

    _use_account(api_module, monkeypatch, tmp_path / "profile-a", "a@example.com")
    assert _whoami(api_module) == "a@example.com"


def test_explicit_gws_config_dir_is_left_alone(api_module, monkeypatch, tmp_path):
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CONFIG_DIR", str(chosen))
    _use_account(api_module, monkeypatch, tmp_path / "profile-a", "a@example.com")

    assert api_module._gws_env()["GOOGLE_WORKSPACE_CLI_CONFIG_DIR"] == str(chosen)
