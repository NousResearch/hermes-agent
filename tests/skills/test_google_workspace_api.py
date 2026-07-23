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


@pytest.mark.parametrize(
    "header_names",
    [
        ("from", "to", "subject", "date"),
        ("From", "To", "Subject", "Date"),
    ],
)
def test_api_gmail_get_reads_headers_case_insensitively(api_module, capsys, header_names):
    from_name, to_name, subject_name, date_name = header_names

    def fake_run_gws(parts, *, params=None, body=None):
        assert parts == ["gmail", "users", "messages", "get"]
        assert params == {"userId": "me", "id": "msg-1", "format": "full"}
        return {
            "id": "msg-1",
            "threadId": "thread-1",
            "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": from_name, "value": "sender@example.com"},
                    {"name": to_name, "value": "recipient@example.com"},
                    {"name": subject_name, "value": "case bug"},
                    {"name": date_name, "value": "Fri, 29 May 2026 12:00:00 +0000"},
                ],
                "body": {},
            },
        }

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(message_id="msg-1", func=api_module.gmail_get)

    api_module.gmail_get(args)

    result = json.loads(capsys.readouterr().out)
    assert result["from"] == "sender@example.com"
    assert result["to"] == "recipient@example.com"
    assert result["subject"] == "case bug"
    assert result["date"] == "Fri, 29 May 2026 12:00:00 +0000"


@pytest.mark.parametrize(
    "header_names",
    [
        ("from", "to", "subject", "date"),
        ("From", "To", "Subject", "Date"),
    ],
)
def test_api_gmail_search_reads_headers_case_insensitively(
    api_module,
    capsys,
    header_names,
):
    from_name, to_name, subject_name, date_name = header_names
    calls = []

    def fake_run_gws(parts, *, params=None, body=None):
        calls.append({"parts": parts, "params": params, "body": body})
        if parts == ["gmail", "users", "messages", "list"]:
            assert params == {"userId": "me", "q": "from:sender", "maxResults": 5}
            return {"messages": [{"id": "msg-1"}]}

        assert parts == ["gmail", "users", "messages", "get"]
        assert params == {
            "userId": "me",
            "id": "msg-1",
            "format": "metadata",
            "metadataHeaders": ["From", "To", "Subject", "Date"],
        }
        return {
            "id": "msg-1",
            "threadId": "thread-1",
            "labelIds": ["INBOX"],
            "snippet": "preview",
            "payload": {
                "headers": [
                    {"name": from_name, "value": "sender@example.com"},
                    {"name": to_name, "value": "recipient@example.com"},
                    {"name": subject_name, "value": "case bug"},
                    {"name": date_name, "value": "Fri, 29 May 2026 12:00:00 +0000"},
                ],
            },
        }

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(
        query="from:sender",
        max=5,
        func=api_module.gmail_search,
    )

    api_module.gmail_search(args)

    assert len(calls) == 2
    result = json.loads(capsys.readouterr().out)
    assert result == [
        {
            "id": "msg-1",
            "threadId": "thread-1",
            "from": "sender@example.com",
            "to": "recipient@example.com",
            "subject": "case bug",
            "date": "Fri, 29 May 2026 12:00:00 +0000",
            "snippet": "preview",
            "labels": ["INBOX"],
        }
    ]


def test_api_gmail_send_uses_conventional_mime_header_casing(api_module):
    captured = {}

    def fake_run_gws(parts, *, params=None, body=None):
        captured["parts"] = parts
        captured["params"] = params
        captured["body"] = body
        return {"id": "sent-1", "threadId": "thread-1"}

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(
        to="recipient@example.com",
        subject="hello",
        body="body",
        html=False,
        cc="copy@example.com",
        from_header="sender@example.com",
        thread_id="thread-1",
        func=api_module.gmail_send,
    )

    api_module.gmail_send(args)

    raw = api_module.base64.urlsafe_b64decode(captured["body"]["raw"])
    raw_text = raw.decode()
    assert "To: recipient@example.com" in raw_text
    assert "Subject: hello" in raw_text
    assert "Cc: copy@example.com" in raw_text
    assert "From: sender@example.com" in raw_text
    assert "\nto: " not in raw_text
    assert "\nsubject: " not in raw_text


@pytest.mark.parametrize(
    "header_names",
    [
        ("from", "subject", "message-id"),
        ("From", "Subject", "Message-ID"),
    ],
)
def test_api_gmail_reply_reads_headers_case_insensitively_and_uses_conventional_mime_header_casing(
    api_module,
    header_names,
):
    from_name, subject_name, message_id_name = header_names
    calls = []

    def fake_run_gws(parts, *, params=None, body=None):
        calls.append({"parts": parts, "params": params, "body": body})
        if parts == ["gmail", "users", "messages", "get"]:
            assert params == {
                "userId": "me",
                "id": "msg-1",
                "format": "metadata",
                "metadataHeaders": ["From", "Subject", "Message-ID"],
            }
            return {
                "id": "msg-1",
                "threadId": "thread-1",
                "payload": {
                    "headers": [
                        {"name": from_name, "value": "sender@example.com"},
                        {"name": subject_name, "value": "case bug"},
                        {"name": message_id_name, "value": "<msg-1@example.com>"},
                    ],
                },
            }

        assert parts == ["gmail", "users", "messages", "send"]
        assert params == {"userId": "me"}
        return {"id": "sent-1", "threadId": "thread-1"}

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(
        message_id="msg-1",
        body="reply body",
        from_header="recipient@example.com",
        func=api_module.gmail_reply,
    )

    api_module.gmail_reply(args)

    assert len(calls) == 2
    body = calls[1]["body"]
    assert body["threadId"] == "thread-1"
    raw = api_module.base64.urlsafe_b64decode(body["raw"])
    raw_text = raw.decode()
    assert "To: sender@example.com" in raw_text
    assert "Subject: Re: case bug" in raw_text
    assert "From: recipient@example.com" in raw_text
    assert "In-Reply-To: <msg-1@example.com>" in raw_text
    assert "References: <msg-1@example.com>" in raw_text
    assert "\nto: " not in raw_text
    assert "\nsubject: " not in raw_text
    assert "\nin-reply-to: " not in raw_text
    assert "\nreferences: " not in raw_text


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


# ── _extract_message_body / _extract_part_body recursive MIME tests ────


def _b64(text: str) -> str:
    """Base64url-encode a string (no padding) for MIME body.data fields."""
    import base64

    return base64.urlsafe_b64encode(text.encode()).decode()


def test_extract_message_body_simple_text(api_module):
    """Top-level text/plain body is returned directly."""
    msg = {"payload": {"body": {"data": _b64("Hello world")}}}
    assert api_module._extract_message_body(msg) == "Hello world"


def test_extract_message_body_multipart_plain(api_module):
    """text/plain part is preferred inside multipart."""
    msg = {
        "payload": {
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("Plain text")}},
                {"mimeType": "text/html", "body": {"data": _b64("<p>HTML</p>")}},
            ]
        }
    }
    assert api_module._extract_message_body(msg) == "Plain text"


def test_extract_message_body_falls_back_to_html(api_module):
    """When no text/plain exists, text/html is returned."""
    msg = {
        "payload": {
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>HTML</p>")}},
            ]
        }
    }
    assert api_module._extract_message_body(msg) == "<p>HTML</p>"


def test_extract_message_body_forwarded_email_rfc822(api_module):
    """Forwarded email nested inside message/rfc822 is extracted."""
    forwarded_body = "This is the forwarded email content."
    msg = {
        "payload": {
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": _b64("---------- Forwarded message ---------")},
                },
                {
                    "mimeType": "message/rfc822",
                    "payload": {
                        "parts": [
                            {
                                "mimeType": "text/plain",
                                "body": {"data": _b64(forwarded_body)},
                            },
                        ]
                    },
                },
            ]
        }
    }
    # text/plain at top level is found first (wrapper note)
    assert "Forwarded message" in api_module._extract_message_body(msg)


def test_extract_message_body_nested_multipart_mixed(api_module):
    """Deeply nested multipart/mixed → multipart/alternative → text/plain."""
    inner_body = "Inner plain text content"
    msg = {
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": _b64(inner_body)},
                        },
                        {
                            "mimeType": "text/html",
                            "body": {"data": _b64("<p>HTML</p>")},
                        },
                    ],
                }
            ],
        }
    }
    assert api_module._extract_message_body(msg) == inner_body


def test_extract_message_body_empty_on_no_parts(api_module):
    """Empty result when no body data and no parts."""
    msg = {"payload": {}}
    assert api_module._extract_message_body(msg) == ""


def test_extract_part_body_deeply_nested_forward(api_module):
    """Forwarded-as-attachment inside nested multipart:
    multipart/mixed → message/rfc822 → multipart/alternative → text/plain."""
    forward_content = "The actual forwarded email body."
    msg = {
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "message/rfc822",
                    "payload": {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/plain",
                                "body": {"data": _b64(forward_content)},
                            },
                        ],
                    },
                }
            ],
        }
    }
    assert api_module._extract_message_body(msg) == forward_content
