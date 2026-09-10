"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import io
import json
import subprocess
import sys
import types
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
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


def test_maton_gmail_probe_uses_gateway_without_local_oauth(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"messages": [{"id": "message-1"}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    api_module._ensure_authenticated = MagicMock()
    monkeypatch.setattr(api_module, "_maton_urlopen", fake_urlopen)

    api_module.gmail_probe(api_module.argparse.Namespace())

    request = captured["request"]
    assert request.full_url == (
        "https://gateway.maton.ai/google-mail/gmail/v1/users/me/messages?maxResults=1"
    )
    assert request.get_header("Authorization") == "Bearer maton.test"
    assert captured["timeout"] == 30
    api_module._ensure_authenticated.assert_not_called()
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "auth": "maton",
        "messageCount": 1,
    }


def test_gmail_probe_falls_back_to_local_oauth(api_module, monkeypatch, capsys):
    monkeypatch.delenv("MATON_API_KEY", raising=False)
    api_module._gws_binary = lambda: None
    execute = MagicMock(return_value={"messages": []})
    list_request = MagicMock(execute=execute)
    messages = MagicMock()
    messages.list.return_value = list_request
    users = MagicMock()
    users.messages.return_value = messages
    service = MagicMock()
    service.users.return_value = users
    api_module.build_service = MagicMock(return_value=service)

    api_module.gmail_probe(api_module.argparse.Namespace())

    api_module.build_service.assert_called_once_with("gmail", "v1")
    messages.list.assert_called_once_with(userId="me", maxResults=1)
    assert json.loads(capsys.readouterr().out) == {
        "status": "ok",
        "auth": "local_oauth",
        "messageCount": 0,
    }


def test_maton_path_refuses_customer_facing_send(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")

    with pytest.raises(SystemExit) as exc:
        api_module.gmail_send(api_module.argparse.Namespace())

    assert exc.value.code == 1
    assert "approved gated workflow" in capsys.readouterr().err


def _maton_response(payload):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    return FakeResponse()


def test_maton_gmail_search_encodes_query_and_message_path(
    api_module, monkeypatch, capsys,
):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    requests = []
    responses = iter([
        {"messages": [{"id": "message/with space"}]},
        {
            "id": "message/with space",
            "threadId": "thread-1",
            "snippet": "A result",
            "labelIds": ["INBOX"],
            "payload": {"headers": [{"name": "Subject", "value": "Hello"}]},
        },
    ])

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 30
        return _maton_response(next(responses))

    monkeypatch.setattr(api_module, "_maton_urlopen", fake_urlopen)
    api_module.gmail_search(api_module.argparse.Namespace(
        query="is:unread from:a+b@example.com", max=7,
    ))

    assert requests[0].full_url.endswith(
        "users/me/messages?q=is%3Aunread+from%3Aa%2Bb%40example.com&maxResults=7"
    )
    assert requests[1].full_url.endswith(
        "users/me/messages/message%2Fwith%20space?format=metadata"
        "&metadataHeaders=From&metadataHeaders=To&metadataHeaders=Subject"
        "&metadataHeaders=Date"
    )
    assert json.loads(capsys.readouterr().out) == [{
        "id": "message/with space",
        "threadId": "thread-1",
        "from": "",
        "to": "",
        "subject": "Hello",
        "date": "",
        "snippet": "A result",
        "labels": ["INBOX"],
    }]


def test_maton_gmail_get_encodes_path_and_shapes_output(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    captured = {}
    message = {
        "id": "id/with space",
        "threadId": "thread-1",
        "snippet": "Preview",
        "labelIds": ["STARRED"],
        "payload": {
            "headers": [{"name": "From", "value": "sender@example.com"}],
            "body": {"data": "SGVsbG8="},
        },
    }

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return _maton_response(message)

    monkeypatch.setattr(api_module, "_maton_urlopen", fake_urlopen)
    api_module.gmail_get(api_module.argparse.Namespace(message_id="id/with space"))

    assert captured["request"].full_url.endswith(
        "users/me/messages/id%2Fwith%20space?format=full"
    )
    assert json.loads(capsys.readouterr().out) == {
        "id": "id/with space",
        "threadId": "thread-1",
        "from": "sender@example.com",
        "to": "",
        "subject": "",
        "date": "",
        "snippet": "Preview",
        "body": "Hello",
        "labels": ["STARRED"],
    }


def test_maton_gmail_labels_shapes_output(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return _maton_response({"labels": [
            {"id": "INBOX", "name": "Inbox", "type": "system"},
            {"id": "Label_1", "name": "Review"},
        ]})

    monkeypatch.setattr(api_module, "_maton_urlopen", fake_urlopen)
    api_module.gmail_labels(api_module.argparse.Namespace())

    assert captured["request"].full_url.endswith("users/me/labels")
    assert json.loads(capsys.readouterr().out) == [
        {"id": "INBOX", "name": "Inbox", "type": "system"},
        {"id": "Label_1", "name": "Review", "type": ""},
    ]


def test_maton_gmail_modify_posts_body_and_shapes_output(
    api_module, monkeypatch, capsys,
):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return _maton_response({"id": "id/1", "labelIds": ["STARRED"]})

    monkeypatch.setattr(api_module, "_maton_urlopen", fake_urlopen)
    api_module.gmail_modify(api_module.argparse.Namespace(
        message_id="id/1", add_labels="STARRED,IMPORTANT", remove_labels="INBOX",
    ))

    request = captured["request"]
    assert request.full_url.endswith("users/me/messages/id%2F1/modify")
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {
        "addLabelIds": ["STARRED", "IMPORTANT"],
        "removeLabelIds": ["INBOX"],
    }
    assert json.loads(capsys.readouterr().out) == {
        "id": "id/1", "labels": ["STARRED"],
    }


def test_maton_path_refuses_customer_facing_reply(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")

    with pytest.raises(SystemExit) as exc:
        api_module.gmail_reply(api_module.argparse.Namespace())

    assert exc.value.code == 1
    assert "approved gated workflow" in capsys.readouterr().err


def test_maton_gmail_reports_http_error(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    error = urllib.error.HTTPError(
        "https://gateway.maton.ai", 429, "rate limited", {}, io.BytesIO(b"slow down"),
    )
    monkeypatch.setattr(api_module, "_maton_urlopen", MagicMock(side_effect=error))

    with pytest.raises(SystemExit) as exc:
        api_module._run_maton_gmail("users/me/messages")

    assert exc.value.code == 1
    assert "request failed (429): slow down" in capsys.readouterr().err


def test_maton_gmail_reports_url_error(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    error = urllib.error.URLError("gateway unavailable")
    monkeypatch.setattr(api_module, "_maton_urlopen", MagicMock(side_effect=error))

    with pytest.raises(SystemExit) as exc:
        api_module._run_maton_gmail("users/me/messages")

    assert exc.value.code == 1
    assert "request failed: gateway unavailable" in capsys.readouterr().err


def test_maton_gmail_rejects_non_json_response(api_module, monkeypatch, capsys):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    monkeypatch.setattr(
        api_module, "_maton_urlopen",
        MagicMock(return_value=_maton_response(b"not json")),
    )

    with pytest.raises(SystemExit) as exc:
        api_module._run_maton_gmail("users/me/messages")

    assert exc.value.code == 1
    assert "Unexpected non-JSON output" in capsys.readouterr().err


def test_maton_gmail_rejects_cross_origin_redirect_without_forwarding_auth(
    api_module, monkeypatch, capsys,
):
    monkeypatch.setenv("MATON_API_KEY", "maton.test")
    received_headers = {}

    class RedirectTarget(BaseHTTPRequestHandler):
        def do_GET(self):
            received_headers.update(self.headers)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')

        def log_message(self, format, *args):
            pass

    target = HTTPServer(("127.0.0.1", 0), RedirectTarget)

    class RedirectSource(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{target.server_address[1]}/capture",
            )
            self.end_headers()

        def log_message(self, format, *args):
            pass

    source = HTTPServer(("127.0.0.1", 0), RedirectSource)
    threads = [
        Thread(target=server.serve_forever, daemon=True)
        for server in (source, target)
    ]
    for thread in threads:
        thread.start()

    monkeypatch.setattr(
        api_module,
        "MATON_GMAIL_BASE_URL",
        f"http://127.0.0.1:{source.server_address[1]}",
    )
    try:
        with pytest.raises(SystemExit) as exc:
            api_module._run_maton_gmail("users/me/messages")
    finally:
        source.shutdown()
        target.shutdown()
        source.server_close()
        target.server_close()

    assert exc.value.code == 1
    assert received_headers == {}
    assert "request failed (302)" in capsys.readouterr().err












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
