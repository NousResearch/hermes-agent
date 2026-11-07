"""Named Google accounts in the google-workspace skill (#15602).

``--account EMAIL`` gives each Google account its own token under
``<HERMES_HOME>/google_tokens/``. Leaving it out keeps the profile's single
``google_token.json``. A named account must never fall back to the default
token, because that would answer one person's request with another's mail.
"""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pm
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/productivity/google-workspace/scripts"
WORK = "work@example.com"


def _load(name: str, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"gw_accounts_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hermes_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_token(path: Path, token: str, *, expired: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    delta = timedelta(hours=-1 if expired else 1)
    path.write_text(
        json.dumps(
            {
                "token": token,
                "refresh_token": f"1//{token}",
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
                "expiry": (datetime.now(UTC) + delta).isoformat(),
                "scopes": ["https://www.googleapis.com/auth/calendar"],
            }
        ),
        encoding="utf-8",
    )


def _run_api(monkeypatch, argv: list[str]) -> dict:
    """Run google_api.py main() with gws stubbed and return the gws call's env."""
    api = _load("google_api", monkeypatch)
    monkeypatch.setattr(api, "_gws_binary", lambda: "/usr/bin/gws")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs["env"])
        return MagicMock(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["google_api.py", *argv])
    api.main()
    assert len(calls) == 1
    return calls[0]


# --- google_api.py -------------------------------------------------------


def test_named_account_runs_gws_with_its_own_token_and_cache(hermes_home, monkeypatch, capsys):
    _write_token(hermes_home / "google_token.json", "default")
    _write_token(hermes_home / "google_tokens" / f"{WORK}.json", "work")

    env = _run_api(monkeypatch, ["--account", "Work@Example.com", "calendar", "list"])

    assert env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"] == str(
        hermes_home / "google_tokens" / f"{WORK}.json"
    )
    config_dir = Path(env["GOOGLE_WORKSPACE_CLI_CONFIG_DIR"])
    assert config_dir.is_dir() and config_dir.is_relative_to(hermes_home / "google_tokens")


def test_omitting_account_keeps_the_profile_token(hermes_home, monkeypatch, capsys):
    _write_token(hermes_home / "google_token.json", "default")
    monkeypatch.delenv("GOOGLE_WORKSPACE_CLI_CONFIG_DIR", raising=False)

    env = _run_api(monkeypatch, ["calendar", "list"])

    assert env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"] == str(hermes_home / "google_token.json")
    assert "GOOGLE_WORKSPACE_CLI_CONFIG_DIR" not in env


def test_unauthorized_account_fails_instead_of_using_the_default(hermes_home, monkeypatch, capsys):
    _write_token(hermes_home / "google_token.json", "default")
    api = _load("google_api", monkeypatch)
    monkeypatch.setattr(api, "_gws_binary", lambda: "/usr/bin/gws")
    monkeypatch.setattr(api.subprocess, "run", Mock(side_effect=AssertionError("ran gws")))
    monkeypatch.setattr(sys, "argv", ["google_api.py", "--account", WORK, "gmail", "search", "x"])

    with pytest.raises(SystemExit) as failure:
        api.main()

    assert failure.value.code == 1
    assert f"--account {WORK}" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["../google_token", "a/b@example.com", "a\\b@example.com", "work"])
def test_account_must_be_a_plain_email(hermes_home, monkeypatch, capsys, value):
    api = _load("google_api", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["google_api.py", "--account", value, "calendar", "list"])

    with pytest.raises(SystemExit) as failure:
        api.main()

    assert failure.value.code == 2
    assert not (hermes_home / "google_tokens").exists()


# --- gws_bridge.py -------------------------------------------------------


def test_bridge_refreshes_the_named_account_token_only(hermes_home, monkeypatch):
    default_path = hermes_home / "google_token.json"
    work_path = hermes_home / "google_tokens" / f"{WORK}.json"
    _write_token(default_path, "default", expired=True)
    _write_token(work_path, "work", expired=True)
    default_before = default_path.read_bytes()
    bridge = _load("gws_bridge", monkeypatch)

    refreshed = io.BytesIO(json.dumps({"access_token": "ya29.work-new", "expires_in": 3600}).encode())
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: refreshed)
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["token"] = cmd, kwargs["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"]
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["gws_bridge.py", "--account", WORK, "gmail", "+triage"])

    with pytest.raises(SystemExit):
        bridge.main()

    assert seen == {"cmd": ["gws", "gmail", "+triage"], "token": "ya29.work-new"}
    assert json.loads(work_path.read_text())["token"] == "ya29.work-new"
    assert default_path.read_bytes() == default_before


# --- setup.py ------------------------------------------------------------


class _FakeFlow:
    """Stands in for google_auth_oauthlib's Flow; records what setup asked of it."""

    last: _FakeFlow | None = None

    def __init__(self, scopes):
        self.code_verifier = "verifier"
        self.auth_kwargs: dict = {}
        self.credentials = MagicMock(granted_scopes=scopes)
        self.credentials.to_json.return_value = json.dumps({"token": "ya29.new", "scopes": scopes})

    @classmethod
    def from_client_secrets_file(cls, path, scopes, **kwargs):
        cls.last = cls(scopes)
        return cls.last

    def authorization_url(self, **kwargs):
        self.auth_kwargs = kwargs
        return "https://accounts.google.com/o/oauth2/auth?x=1", "state-1"

    def fetch_token(self, code):
        pass


@pytest.fixture
def setup_module(hermes_home, monkeypatch):
    (hermes_home / "google_client_secret.json").write_text('{"installed": {}}')
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = _FakeFlow
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", types.ModuleType("google_auth_oauthlib"))
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)
    monkeypatch.setattr(pm, "ensure_import", Mock())
    return _load("setup", monkeypatch)


def _setup(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["setup.py", *argv])
    module.main()


def test_auth_for_a_named_account_hints_it_and_files_the_token_under_it(
    setup_module, hermes_home, monkeypatch, capsys
):
    _write_token(hermes_home / "google_token.json", "default")
    default_before = (hermes_home / "google_token.json").read_bytes()
    monkeypatch.setattr(setup_module, "_authorized_email", lambda creds, scopes: "Work@Example.com")

    _setup(setup_module, monkeypatch, "--account", WORK, "--auth-url")
    assert _FakeFlow.last.auth_kwargs["login_hint"] == WORK
    assert not (hermes_home / "google_oauth_pending.json").exists()

    _setup(setup_module, monkeypatch, "--account", WORK, "--auth-code", "4/code")

    assert json.loads((hermes_home / "google_tokens" / f"{WORK}.json").read_text())["token"] == "ya29.new"
    assert (hermes_home / "google_token.json").read_bytes() == default_before
    capsys.readouterr()
    _setup(setup_module, monkeypatch, "--list-accounts")
    assert capsys.readouterr().out.splitlines() == [
        f"(default) {hermes_home / 'google_token.json'}",
        WORK,
    ]


def test_token_from_a_different_google_account_is_not_saved(setup_module, hermes_home, monkeypatch, capsys):
    monkeypatch.setattr(setup_module, "_authorized_email", lambda creds, scopes: "someone@else.com")

    _setup(setup_module, monkeypatch, "--account", WORK, "--auth-url")
    with pytest.raises(SystemExit) as failure:
        _setup(setup_module, monkeypatch, "--account", WORK, "--auth-code", "4/code")

    assert failure.value.code == 1
    assert "someone@else.com" in capsys.readouterr().out
    assert _saved_tokens(hermes_home) == []


def _saved_tokens(hermes_home: Path) -> list[str]:
    return sorted(p.name for p in hermes_home.rglob("*.json") if p.name != "google_client_secret.json")
