"""Tests for Google Workspace gws bridge and CLI wrapper."""

import base64
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
    assert saved["type"] == "authorized_user"


def test_bridge_refresh_passes_timeout_to_urlopen(bridge_module):
    """Token refresh must pass an explicit timeout so a hung Google endpoint
    cannot block the agent turn indefinitely (no `timeout=` defaults to the
    global socket timeout, which is unset)."""
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

    with patch("urllib.request.urlopen", return_value=mock_resp) as mocked:
        bridge_module.get_valid_token()

    assert mocked.call_count == 1
    _, kwargs = mocked.call_args
    assert kwargs.get("timeout") is not None, (
        "urlopen call must pass timeout= to avoid hanging on unreachable upstream"
    )


def test_bridge_refresh_exits_cleanly_on_network_error(bridge_module):
    """URLError/timeout during refresh exits 1 with a readable message
    instead of crashing with a raw traceback."""
    import urllib.error

    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.old", expiry=past)

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("timed out"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            bridge_module.get_valid_token()

    assert exc_info.value.code == 1


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


def test_walk_attachments_finds_nested_parts(api_module):
    """_walk_attachments surfaces attachments from multi-level multipart payloads."""
    payload = {
        "partId": "0",
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "partId": "0.0",
                "mimeType": "text/plain",
                "body": {"size": 12, "data": "aGVsbG8gd29ybGQ="},
            },
            {
                "partId": "0.1",
                "mimeType": "multipart/related",
                "parts": [
                    {
                        "partId": "0.1.0",
                        "mimeType": "application/pdf",
                        "filename": "report.pdf",
                        "body": {"size": 4096, "attachmentId": "att-1"},
                    },
                ],
            },
            {
                "partId": "0.2",
                "mimeType": "image/png",
                "filename": "logo.png",
                "body": {"size": 2048, "attachmentId": "att-2"},
            },
        ],
    }

    found = api_module._walk_attachments(payload)

    assert len(found) == 2
    assert {a["attachment_id"] for a in found} == {"att-1", "att-2"}
    pdf = next(a for a in found if a["attachment_id"] == "att-1")
    assert pdf["filename"] == "report.pdf"
    assert pdf["mime_type"] == "application/pdf"
    assert pdf["size_bytes"] == 4096
    png = next(a for a in found if a["attachment_id"] == "att-2")
    assert png["filename"] == "logo.png"
    assert png["size_bytes"] == 2048


def test_walk_attachments_empty_when_no_attachments(api_module):
    """A plain text-only payload returns an empty list."""
    payload = {
        "mimeType": "text/plain",
        "body": {"size": 5, "data": "aGVsbG8="},
    }
    assert api_module._walk_attachments(payload) == []


def test_api_gmail_attachment_list_uses_messages_get(api_module, capsys):
    """gmail attachment list reads the full message and surfaces attachments."""
    payload = {
        "parts": [
            {
                "partId": "1",
                "mimeType": "application/pdf",
                "filename": "invoice.pdf",
                "body": {"size": 1234, "attachmentId": "att-xyz"},
            },
        ],
    }
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout=json.dumps({"payload": payload}), stderr="")

    args = api_module.argparse.Namespace(
        message_id="msg-1", func=api_module.gmail_attachment_list,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_attachment_list(args)

    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/gws"
    assert cmd[1:5] == ["gmail", "users", "messages", "get"]
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert params["userId"] == "me"
    assert params["id"] == "msg-1"
    assert params["format"] == "full"

    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1
    assert out[0]["attachment_id"] == "att-xyz"
    assert out[0]["filename"] == "invoice.pdf"
    assert out[0]["size_bytes"] == 1234


def test_api_gmail_attachment_get_decodes_and_writes(api_module, tmp_path, capsys):
    """gmail attachment get base64url-decodes the payload and writes it to disk."""
    raw_bytes = b"%PDF-1.4 fake content\x00\xff"
    encoded = base64.urlsafe_b64encode(raw_bytes).decode()

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout=json.dumps({"size": len(raw_bytes), "data": encoded}), stderr="")

    output = tmp_path / "out" / "doc.pdf"
    args = api_module.argparse.Namespace(
        message_id="msg-1",
        attachment_id="att-xyz",
        output=str(output),
        func=api_module.gmail_attachment_get,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_attachment_get(args)

    cmd = captured["cmd"]
    assert cmd[1:6] == ["gmail", "users", "messages", "attachments", "get"]
    params = json.loads(cmd[cmd.index("--params") + 1])
    assert params["userId"] == "me"
    assert params["messageId"] == "msg-1"
    assert params["id"] == "att-xyz"

    assert output.read_bytes() == raw_bytes
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "saved"
    assert out["path"] == str(output)
    assert out["size_bytes"] == len(raw_bytes)


def test_api_gmail_attachment_get_errors_on_missing_data(api_module, tmp_path, capsys):
    """gmail attachment get surfaces a clear error when the API response has no 'data'."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout=json.dumps({"size": 0}), stderr="")

    output = tmp_path / "doc.pdf"
    args = api_module.argparse.Namespace(
        message_id="msg-1",
        attachment_id="att-xyz",
        output=str(output),
        func=api_module.gmail_attachment_get,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        with pytest.raises(SystemExit) as excinfo:
            api_module.gmail_attachment_get(args)

    assert excinfo.value.code == 1
    assert not output.exists()
