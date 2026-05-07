"""Tests for Google Workspace gws bridge, CLI wrapper, and per-operation scope checks."""

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
def google_api_module(monkeypatch, tmp_path):
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


@pytest.fixture
def api_module(google_api_module):
    return google_api_module


def _write_token(path: Path, *, token="ya29.test", expiry=None, scopes=None, scope=None, **extra):
    data = {
        "token": token,
        "refresh_token": "***",
        "client_id": "123.apps.googleusercontent.com",
        "client_secret": "secret",
        "token_uri": "https://oauth2.googleapis.com/token",
        **extra,
    }
    if expiry is not None:
        data["expiry"] = expiry
    if scopes is not None:
        data["scopes"] = scopes
    if scope is not None:
        data["scope"] = scope
    path.write_text(json.dumps(data))


def test_bridge_returns_valid_token(bridge_module):
    """Non-expired token is returned without refresh."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"


def test_bridge_refreshes_expired_token(bridge_module):
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
    saved = json.loads(token_path.read_text())
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def test_bridge_exits_on_missing_token(bridge_module):
    """Missing token file causes exit with code 1."""
    with pytest.raises(SystemExit):
        bridge_module.get_valid_token()


def test_bridge_main_injects_token_env(bridge_module):
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


def test_api_calendar_list_respects_date_range(api_module):
    """calendar list with --start/--end passes correct time bounds."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout="{}", stderr="")

    args = api_module.argparse.Namespace(
        start="2026-04-01T00:00:00Z",
        end="2026-04-07T23:59:59Z",
        max=25,
        calendar="primary",
        func=api_module.calendar_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.calendar_list(args)

    cmd = captured["cmd"]
    params_idx = cmd.index("--params")
    params = json.loads(cmd[params_idx + 1])
    assert params["timeMin"] == "2026-04-01T00:00:00Z"
    assert params["timeMax"] == "2026-04-07T23:59:59Z"


def test_gmail_send_requires_some_recipient(api_module):
    args = api_module.argparse.Namespace(
        to="",
        cc="",
        bcc="",
        subject="Test",
        body="Hello",
        html=False,
        thread_id="",
        from_header="",
    )

    with pytest.raises(SystemExit, match="at least one recipient"):
        api_module.gmail_send(args)


def test_gmail_send_allows_bcc_only(api_module):
    captured = {}

    def fake_run_gws(path, *, params=None, body=None):
        captured["path"] = path
        captured["params"] = params
        captured["body"] = body
        return {"id": "msg123", "threadId": "thread123"}

    args = api_module.argparse.Namespace(
        to="",
        cc="",
        bcc="hidden@example.com",
        subject="Test",
        body="Hello",
        html=False,
        thread_id="",
        from_header="",
    )

    with patch.object(api_module, "_run_gws", side_effect=fake_run_gws):
        api_module.gmail_send(args)

    raw = captured["body"]["raw"]
    decoded = api_module.base64.urlsafe_b64decode(raw.encode()).decode()
    assert "Bcc: hidden@example.com" in decoded
    assert "Subject: Test" in decoded
    assert captured["path"] == ["gmail", "users", "messages", "send"]


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
class TestCheckOperationScopes:
    def test_gmail_search_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        google_api_module._check_operation_scopes("gmail", "search")

    def test_gmail_search_allowed_with_modify(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.modify"],
        )
        google_api_module._check_operation_scopes("gmail", "search")

    def test_gmail_get_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        google_api_module._check_operation_scopes("gmail", "get")

    def test_gmail_send_blocked_without_send_or_modify(self, google_api_module, capsys):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("gmail", "send")

        err = capsys.readouterr().err
        assert "gmail.send" in err
        assert "gmail.modify" in err

    def test_gmail_send_allowed_with_send_scope(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        )
        google_api_module._check_operation_scopes("gmail", "send")

    def test_gmail_send_allowed_with_modify_scope(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.modify"],
        )
        google_api_module._check_operation_scopes("gmail", "send")

    def test_gmail_reply_blocked_without_send_or_modify(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("gmail", "reply")

    def test_gmail_modify_blocked_without_modify(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("gmail", "modify")

    def test_gmail_labels_allowed_with_labels_scope(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.labels"],
        )
        google_api_module._check_operation_scopes("gmail", "labels")

    def test_calendar_list_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        google_api_module._check_operation_scopes("calendar", "list")

    def test_calendar_create_blocked_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("calendar", "create")

    def test_calendar_create_allowed_with_full_scope(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        google_api_module._check_operation_scopes("calendar", "create")

    def test_drive_search_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        google_api_module._check_operation_scopes("drive", "search")

    def test_sheets_get_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        google_api_module._check_operation_scopes("sheets", "get")

    def test_sheets_update_blocked_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("sheets", "update")

    def test_sheets_update_allowed_with_full_scope(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        google_api_module._check_operation_scopes("sheets", "update")

    def test_docs_get_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/documents.readonly"],
        )
        google_api_module._check_operation_scopes("docs", "get")

    def test_contacts_list_allowed_with_readonly(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/contacts.readonly"],
        )
        google_api_module._check_operation_scopes("contacts", "list")

    def test_unknown_operation_passes(self, google_api_module):
        _write_token(google_api_module.TOKEN_PATH, scopes=[])
        google_api_module._check_operation_scopes("unknown_service", "unknown_action")

    def test_error_message_is_actionable(self, google_api_module, capsys):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        )
        with pytest.raises(SystemExit):
            google_api_module._check_operation_scopes("gmail", "send")

        err = capsys.readouterr().err
        assert "gmail send" in err.lower() or "gmail send" in err
        assert "Re-run setup.py" in err


class TestGrantedScopes:
    def test_returns_scopes_from_list(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar",
            ],
        )
        result = google_api_module._granted_scopes()
        assert result == {
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar",
        }

    def test_returns_empty_set_for_missing_file(self, google_api_module):
        result = google_api_module._granted_scopes()
        assert result == set()

    def test_returns_empty_set_for_no_scopes_key(self, google_api_module):
        google_api_module.TOKEN_PATH.write_text(json.dumps({
            "token": "***",
            "refresh_token": "***",
        }))
        result = google_api_module._granted_scopes()
        assert result == set()

    def test_handles_space_separated_scope_string(self, google_api_module):
        _write_token(
            google_api_module.TOKEN_PATH,
            scope="https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/calendar",
        )
        result = google_api_module._granted_scopes()
        assert "https://www.googleapis.com/auth/gmail.readonly" in result
        assert "https://www.googleapis.com/auth/calendar" in result
