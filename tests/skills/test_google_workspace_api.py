"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
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
    # Ensure the gws CLI code path is taken even when the binary isn't
    # installed (CI).  Without this, calendar_list() falls through to the
    # Python SDK path which imports ``googleapiclient`` — not in deps.
    module._gws_binary = lambda: "/usr/bin/gws"
    # Bypass authentication check — no real token file in CI.
    module._ensure_authenticated = lambda: None
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


def test_api_calendar_list_uses_events_list(api_module):
    """calendar_list calls _run_gws with events list + params."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    # _gws_binary() returns "/usr/bin/gws", so cmd[0] is that binary
    assert cmd[0] == "/usr/bin/gws"
    assert "calendar" in cmd
    assert "events" in cmd
    assert "list" in cmd
    assert "--params" in cmd
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert "timeMin" in params
    assert "timeMax" in params
    assert params["calendarId"] == "primary"










def test_api_gmail_filter_create_uses_settings_api(api_module, capsys):
    captured = {}

    def fake_run_gws(parts, *, params=None, body=None):
        captured.update(parts=parts, params=params, body=body)
        return {"id": "filter-1", **body}

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(
        from_address="alerts@example.com",
        to="",
        subject="",
        query="has:attachment",
        negated_query="",
        has_attachment=False,
        exclude_chats=True,
        add_labels="Label_1, STARRED",
        remove_labels="INBOX",
        func=api_module.gmail_filter_create,
    )

    api_module.gmail_filter_create(args)

    assert captured == {
        "parts": ["gmail", "users", "settings", "filters", "create"],
        "params": {"userId": "me"},
        "body": {
            "criteria": {
                "from": "alerts@example.com",
                "query": "has:attachment",
                "excludeChats": True,
            },
            "action": {
                "addLabelIds": ["Label_1", "STARRED"],
                "removeLabelIds": ["INBOX"],
            },
        },
    }
    assert json.loads(capsys.readouterr().out)["filter"]["id"] == "filter-1"


def test_api_gmail_filter_list_get_delete_use_settings_api(api_module, capsys):
    calls = []

    def fake_run_gws(parts, *, params=None, body=None):
        calls.append({"parts": parts, "params": params, "body": body})
        if parts[-1] == "list":
            return {"filter": [{"id": "filter-1"}]}
        return {"id": params["id"]}

    api_module._run_gws = fake_run_gws
    api_module.gmail_filter_list(api_module.argparse.Namespace())
    api_module.gmail_filter_get(api_module.argparse.Namespace(filter_id="filter-1"))
    api_module.gmail_filter_delete(api_module.argparse.Namespace(filter_id="filter-1"))

    assert calls == [
        {
            "parts": ["gmail", "users", "settings", "filters", "list"],
            "params": {"userId": "me"},
            "body": None,
        },
        {
            "parts": ["gmail", "users", "settings", "filters", "get"],
            "params": {"userId": "me", "id": "filter-1"},
            "body": None,
        },
        {
            "parts": ["gmail", "users", "settings", "filters", "delete"],
            "params": {"userId": "me", "id": "filter-1"},
            "body": None,
        },
    ]
    output = capsys.readouterr().out
    assert '"id": "filter-1"' in output
    assert '"status": "deleted"' in output


def test_api_gmail_filter_get_uses_python_fallback(api_module, capsys):
    request = MagicMock()
    request.execute.return_value = {"id": "filter-1"}
    service = MagicMock()
    service.users().settings().filters().get.return_value = request
    api_module._gws_binary = lambda: None
    api_module.build_service = MagicMock(return_value=service)

    api_module.gmail_filter_get(api_module.argparse.Namespace(filter_id="filter-1"))

    api_module.build_service.assert_called_once_with("gmail", "v1")
    service.users().settings().filters().get.assert_called_once_with(
        userId="me", id="filter-1"
    )
    assert json.loads(capsys.readouterr().out) == {"id": "filter-1"}


@pytest.mark.parametrize(
    ("criteria", "actions", "message"),
    [
        ({}, {"add_labels": "Label_1"}, "matching criterion"),
        ({"from_address": "alerts@example.com"}, {}, "--add-labels or --remove-labels"),
    ],
)
def test_api_gmail_filter_create_rejects_incomplete_filter(
    api_module, criteria, actions, message
):
    values = {
        "from_address": "",
        "to": "",
        "subject": "",
        "query": "",
        "negated_query": "",
        "has_attachment": False,
        "exclude_chats": False,
        "add_labels": "",
        "remove_labels": "",
        **criteria,
        **actions,
    }

    with pytest.raises(SystemExit, match=message):
        api_module.gmail_filter_create(api_module.argparse.Namespace(**values))


def test_api_get_credentials_refresh_persists_authorized_user_type(api_module, monkeypatch):
    token_path = api_module.TOKEN_PATH
    _write_token(token_path, token="ya29.old")

    class FakeCredentials:
        def __init__(self):
            self.expired = True
            self.refresh_token = "1//refresh"
            self.valid = True

        def refresh(self, request):
            self.expired = False

        def to_json(self):
            return json.dumps({
                "token": "ya29.refreshed",
                "refresh_token": "1//refresh",
                "client_id": "123.apps.googleusercontent.com",
                "client_secret": "secret",
                "token_uri": "https://oauth2.googleapis.com/token",
            })

    class FakeCredentialsModule:
        @staticmethod
        def from_authorized_user_file(filename, scopes):
            assert filename == str(token_path)
            assert scopes == api_module.SCOPES
            return FakeCredentials()

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsModule
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda: object()

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    creds = api_module.get_credentials()

    saved = json.loads(token_path.read_text())
    assert isinstance(creds, FakeCredentials)
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"
