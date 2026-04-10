"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


BRIDGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gws_bridge.py"
)
API_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


@pytest.fixture
def bridge_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_bridge_test", BRIDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def api_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    spec = importlib.util.spec_from_file_location("gws_api_test", API_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_token(path: Path, *, token="ya29.test", expiry=None, **extra):
    data = {
        "token": token,
        "refresh_token": "1//refresh",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        **extra,
    }
    if expiry is not None:
        data["expiry"] = expiry
    path.write_text(json.dumps(data))


def test_bridge_returns_valid_token(bridge_module, tmp_path):
    """Non-expired token is returned without refresh."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"


def test_bridge_refreshes_expired_token(bridge_module, tmp_path):
    """Expired token triggers a refresh via token_uri."""
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.old", expiry=past)

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "access_token": "ya29.refreshed",
        "expires_in": 3600,
    }).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = bridge_module.get_valid_token()

    assert result == "ya29.refreshed"
    # Verify persisted
    saved = json.loads(token_path.read_text())
    assert saved["token"] == "ya29.refreshed"


def test_bridge_exits_on_missing_token(bridge_module):
    """Missing token file causes exit with code 1."""
    with pytest.raises(SystemExit):
        bridge_module.get_valid_token()


def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.injected", expiry=future)

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return MagicMock(returncode=0)

    with patch.object(sys, "argv", ["gws_bridge.py", "gmail", "+triage"]):
        with patch.object(subprocess, "run", side_effect=capture_run):
            with pytest.raises(SystemExit):
                bridge_module.main()

    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"
    assert captured["cmd"] == ["gws", "gmail", "+triage"]


def test_api_calendar_list_uses_agenda_by_default(api_module):
    """calendar list without dates uses +agenda helper."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    with patch.object(subprocess, "run", side_effect=capture_run):
        with pytest.raises(SystemExit):
            api_module.calendar_list(args)

    gws_args = captured["cmd"][2:]  # skip python + bridge path
    assert "calendar" in gws_args
    assert "+agenda" in gws_args
    assert "--days" in gws_args


def test_api_calendar_list_respects_date_range(api_module):
    """calendar list with --start/--end uses raw events list API."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    args = api_module.argparse.Namespace(
        start="2026-04-01T00:00:00Z",
        end="2026-04-07T23:59:59Z",
        max=25,
        calendar="primary",
        func=api_module.calendar_list,
    )

    with patch.object(subprocess, "run", side_effect=capture_run):
        with pytest.raises(SystemExit):
            api_module.calendar_list(args)

    gws_args = captured["cmd"][2:]
    assert "events" in gws_args
    assert "list" in gws_args
    params_idx = gws_args.index("--params")
    params = json.loads(gws_args[params_idx + 1])
    assert params["timeMin"] == "2026-04-01T00:00:00Z"
    assert params["timeMax"] == "2026-04-07T23:59:59Z"
