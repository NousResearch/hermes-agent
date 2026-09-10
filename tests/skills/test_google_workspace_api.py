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












def test_api_gmail_search_does_not_filter_by_metadata_headers(api_module):
    """gmail_search must not pass metadataHeaders to the gws get call.

    Regression for #34806: metadataHeaders matches header names
    case-sensitively against the raw MIME source, so a filter of
    ["From", "To", "Subject", "Date"] silently drops headers a sender
    wrote as e.g. "to:" or "SUBJECT:". Omitting the filter (format=metadata
    already excludes body/payload) returns every header and lets the
    existing case-insensitive _headers_dict() do the lookup instead.
    """
    calls = []

    def capture_run(cmd, **kwargs):
        calls.append(cmd)
        if "list" in cmd:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"messages": [{"id": "msg1"}]}),
                stderr="",
            )
        # Simulate a message whose headers came in lower-case, as some
        # senders emit -- this is exactly what a case-sensitive
        # metadataHeaders=["From", "To", "Subject", "Date"] filter drops.
        return MagicMock(
            returncode=0,
            stdout=json.dumps({
                "id": "msg1",
                "threadId": "thread1",
                "snippet": "hi",
                "labelIds": ["INBOX"],
                "payload": {
                    "headers": [
                        {"name": "from", "value": "alice@example.com"},
                        {"name": "to", "value": "bob@example.com"},
                        {"name": "subject", "value": "lower-case headers"},
                        {"name": "date", "value": "Mon, 1 Jun 2026 00:00:00 +0000"},
                    ]
                },
            }),
            stderr="",
        )

    args = api_module.argparse.Namespace(query="is:unread", max=10, func=api_module.gmail_search)

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        with patch("builtins.print") as mock_print:
            api_module.gmail_search(args)

    # The "get" call must not restrict headers by name.
    get_cmd = calls[1]
    if "--params" in get_cmd:
        params = json.loads(get_cmd[get_cmd.index("--params") + 1])
        assert "metadataHeaders" not in params

    # And the parsed output must still surface to/subject despite the
    # lower-case header names in the raw response.
    printed = mock_print.call_args[0][0]
    output = json.loads(printed)
    assert output[0]["to"] == "bob@example.com"
    assert output[0]["subject"] == "lower-case headers"


def test_api_gmail_reply_does_not_filter_by_metadata_headers(api_module):
    """gmail_reply must not pass metadataHeaders to the gws get call either.

    Same root cause as test_api_gmail_search_does_not_filter_by_metadata_headers
    (issue #34806 follow-up): the original message's From/Subject/Message-ID
    headers can come back lower-case, and a metadataHeaders filter drops them
    before _headers_dict() ever sees them -- breaking the reply's To header,
    subject line, and In-Reply-To/References threading.
    """
    calls = []

    def capture_run(cmd, **kwargs):
        calls.append(cmd)
        if "get" in cmd:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "id": "orig1",
                    "threadId": "thread1",
                    "payload": {
                        "headers": [
                            {"name": "from", "value": "alice@example.com"},
                            {"name": "subject", "value": "lower-case headers"},
                            {"name": "message-id", "value": "<abc123@mail.example.com>"},
                        ]
                    },
                }),
                stderr="",
            )
        return MagicMock(
            returncode=0,
            stdout=json.dumps({"id": "sent1", "threadId": "thread1"}),
            stderr="",
        )

    args = api_module.argparse.Namespace(
        message_id="orig1", body="reply body", from_header="", func=api_module.gmail_reply,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        with patch("builtins.print"):
            api_module.gmail_reply(args)

    get_cmd = calls[0]
    if "--params" in get_cmd:
        params = json.loads(get_cmd[get_cmd.index("--params") + 1])
        assert "metadataHeaders" not in params

    # The outgoing raw MIME message must still have picked up the To/Subject
    # despite the lower-case header names in the raw response.
    send_cmd = calls[1]
    body = json.loads(send_cmd[send_cmd.index("--json") + 1])
    raw = base64.urlsafe_b64decode(body["raw"]).decode()
    assert "To: alice@example.com" in raw
    assert "Subject: Re: lower-case headers" in raw
    assert "In-Reply-To: <abc123@mail.example.com>" in raw


def test_api_gmail_search_direct_client_does_not_filter_by_metadata_headers(
    api_module, monkeypatch
):
    """Same regression as the gws-binary test, but for the direct
    googleapiclient fallback path (no gws binary installed).

    Per review on #77067: the shared api_module fixture forces
    _gws_binary() to a real path, so the direct-client branch of
    gmail_search() was never exercised by any existing test.
    """
    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)

    fake_service = MagicMock()
    fake_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "msg1"}]
    }
    fake_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "msg1",
        "threadId": "thread1",
        "snippet": "hi",
        "labelIds": ["INBOX"],
        "payload": {
            "headers": [
                {"name": "from", "value": "alice@example.com"},
                {"name": "to", "value": "bob@example.com"},
                {"name": "subject", "value": "lower-case headers"},
                {"name": "date", "value": "Mon, 1 Jun 2026 00:00:00 +0000"},
            ]
        },
    }
    monkeypatch.setattr(api_module, "build_service", lambda *a, **k: fake_service)

    args = api_module.argparse.Namespace(query="is:unread", max=10, func=api_module.gmail_search)

    with patch("builtins.print") as mock_print:
        api_module.gmail_search(args)

    get_call = fake_service.users.return_value.messages.return_value.get.call_args
    assert "metadataHeaders" not in get_call.kwargs

    printed = mock_print.call_args[0][0]
    output = json.loads(printed)
    assert output[0]["to"] == "bob@example.com"
    assert output[0]["subject"] == "lower-case headers"


def test_api_gmail_reply_direct_client_does_not_filter_by_metadata_headers(
    api_module, monkeypatch
):
    """Direct googleapiclient fallback path for gmail_reply -- see
    test_api_gmail_search_direct_client_does_not_filter_by_metadata_headers.
    """
    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)

    fake_service = MagicMock()
    fake_service.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "orig1",
        "threadId": "thread1",
        "payload": {
            "headers": [
                {"name": "from", "value": "alice@example.com"},
                {"name": "subject", "value": "lower-case headers"},
                {"name": "message-id", "value": "<abc123@mail.example.com>"},
            ]
        },
    }
    fake_service.users.return_value.messages.return_value.send.return_value.execute.return_value = {
        "id": "sent1", "threadId": "thread1",
    }
    monkeypatch.setattr(api_module, "build_service", lambda *a, **k: fake_service)

    args = api_module.argparse.Namespace(
        message_id="orig1", body="reply body", from_header="", func=api_module.gmail_reply,
    )

    with patch("builtins.print"):
        api_module.gmail_reply(args)

    get_call = fake_service.users.return_value.messages.return_value.get.call_args
    assert "metadataHeaders" not in get_call.kwargs

    send_call = fake_service.users.return_value.messages.return_value.send.call_args
    raw = base64.urlsafe_b64decode(send_call.kwargs["body"]["raw"]).decode()
    assert "To: alice@example.com" in raw
    assert "Subject: Re: lower-case headers" in raw
    assert "In-Reply-To: <abc123@mail.example.com>" in raw


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
