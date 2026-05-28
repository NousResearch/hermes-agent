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


def _decode_raw(raw: str) -> str:
    """Decode a base64url MIME 'raw' back to its text form for assertions."""
    import base64

    return base64.urlsafe_b64decode(raw).decode()


def test_api_gmail_draft_create_uses_drafts_create(api_module):
    """draft create calls _run_gws with drafts/create and a message.raw body."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(
            returncode=0,
            stdout='{"id": "draft1", "message": {"id": "m1", "threadId": "t1"}}',
            stderr="",
        )

    args = api_module.argparse.Namespace(
        to="user@example.com", subject="Hi", body="Hello",
        cc="", from_header="", html=False, func=api_module.gmail_draft_create,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_create(args)

    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/gws"
    assert "drafts" in cmd
    assert "create" in cmd
    body = json.loads(cmd[cmd.index("--json") + 1])
    assert "raw" in body["message"]
    decoded = _decode_raw(body["message"]["raw"])
    assert "user@example.com" in decoded
    assert "Hi" in decoded


def test_api_gmail_draft_create_html_and_cc(api_module):
    """--html marks the MIME as text/html and --cc adds a Cc header."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout='{"id": "draft1", "message": {}}', stderr="")

    args = api_module.argparse.Namespace(
        to="user@example.com", subject="Hi", body="<b>Hello</b>",
        cc="cc@example.com", from_header="", html=True, func=api_module.gmail_draft_create,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_create(args)

    body = json.loads(captured["cmd"][captured["cmd"].index("--json") + 1])
    decoded = _decode_raw(body["message"]["raw"])
    assert "text/html" in decoded
    assert "cc@example.com" in decoded


def test_api_gmail_draft_create_rejects_empty(api_module):
    """draft create with no to/subject/body exits non-zero (empty-draft guard)."""
    args = api_module.argparse.Namespace(
        to="", subject="", body="", cc="", from_header="", html=False,
        func=api_module.gmail_draft_create,
    )

    with pytest.raises(SystemExit) as exc_info:
        api_module.gmail_draft_create(args)

    assert exc_info.value.code == 1


def test_api_gmail_draft_send_uses_drafts_send(api_module):
    """draft send calls _run_gws with drafts/send and an {'id': ...} body."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout='{"id": "m1", "threadId": "t1"}', stderr="")

    args = api_module.argparse.Namespace(draft_id="draft1", func=api_module.gmail_draft_send)

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_send(args)

    cmd = captured["cmd"]
    assert "drafts" in cmd
    assert "send" in cmd
    assert json.loads(cmd[cmd.index("--json") + 1]) == {"id": "draft1"}


def test_api_gmail_draft_list_uses_drafts_list(api_module):
    """draft list calls drafts/list with maxResults, then enriches each draft."""
    cmds = []

    def capture_run(cmd, **kwargs):
        cmds.append(cmd)
        return MagicMock(
            returncode=0,
            stdout='{"drafts": [{"id": "d1", "message": {"id": "m1", "threadId": "t1"}}]}',
            stderr="",
        )

    args = api_module.argparse.Namespace(max=10, func=api_module.gmail_draft_list)

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_list(args)

    list_cmds = [c for c in cmds if "drafts" in c and "list" in c]
    assert list_cmds, "expected a drafts/list call"
    params = json.loads(list_cmds[0][list_cmds[0].index("--params") + 1])
    assert params["maxResults"] == 10


def test_api_gmail_draft_list_skips_enrichment_when_no_message_id(api_module, capsys):
    """A draft with no message stub id yields an id-only row and triggers no
    messages.get fetch."""
    cmds = []

    def capture_run(cmd, **kwargs):
        cmds.append(cmd)
        return MagicMock(returncode=0, stdout='{"drafts": [{"id": "d1"}]}', stderr="")

    args = api_module.argparse.Namespace(max=10, func=api_module.gmail_draft_list)

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_list(args)

    # No per-draft messages.get call should have been made.
    assert not [c for c in cmds if "messages" in c and "get" in c]
    rows = json.loads(capsys.readouterr().out)
    assert rows == [{
        "draftId": "d1", "messageId": "", "to": "", "subject": "", "date": "", "snippet": "",
    }]


def test_api_headers_dict_is_case_insensitive(api_module):
    """Gmail returns headers with whatever case the message used; lookups must
    be case-insensitive (drafts we build carry lowercase to/subject)."""
    msg = {"payload": {"headers": [
        {"name": "to", "value": "a@b.com"},
        {"name": "subject", "value": "Hi"},
        {"name": "Message-Id", "value": "<x@mail>"},
    ]}}
    headers = api_module._headers_dict(msg)
    assert headers.get("To") == "a@b.com"
    assert headers.get("Subject") == "Hi"
    assert headers.get("Message-ID") == "<x@mail>"   # all-caps lookup, mixed-case header
    assert headers["Message-ID"] == "<x@mail>"        # subscript path (reply threading)
    assert "To" in headers


def test_api_gmail_draft_list_populates_to_subject_from_lowercase_headers(api_module, capsys):
    """Regression: a draft whose metadata returns lowercase 'to'/'subject'
    headers (as Gmail does for tool-created drafts) must still populate the
    to/subject fields in the listing."""
    def capture_run(cmd, **kwargs):
        if "drafts" in cmd and "list" in cmd:
            stdout = '{"drafts": [{"id": "d1", "message": {"id": "m1", "threadId": "t1"}}]}'
        else:  # messages.get enrichment — headers come back lowercase
            stdout = ('{"snippet": "hello", "payload": {"headers": ['
                      '{"name": "to", "value": "a@b.com"},'
                      '{"name": "subject", "value": "Hi there"},'
                      '{"name": "Date", "value": "Mon, 1 Jan 2026"}]}}')
        return MagicMock(returncode=0, stdout=stdout, stderr="")

    args = api_module.argparse.Namespace(max=10, func=api_module.gmail_draft_list)
    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_draft_list(args)

    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["to"] == "a@b.com"
    assert rows[0]["subject"] == "Hi there"
    assert rows[0]["date"] == "Mon, 1 Jan 2026"


def test_api_gmail_draft_list_tolerates_enrichment_failure(api_module, monkeypatch, capsys):
    """On the Python-client path, a per-draft metadata fetch that raises falls
    back to an id-only row instead of sinking the whole listing."""
    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)

    fake_service = MagicMock()
    fake_service.users().drafts().list().execute.return_value = {
        "drafts": [{"id": "d1", "message": {"id": "m1", "threadId": "t1"}}]
    }
    fake_service.users().messages().get().execute.side_effect = RuntimeError("boom")
    monkeypatch.setattr(api_module, "build_service", lambda *a, **k: fake_service)

    args = api_module.argparse.Namespace(max=10, func=api_module.gmail_draft_list)
    api_module.gmail_draft_list(args)

    rows = json.loads(capsys.readouterr().out)
    assert rows == [{
        "draftId": "d1", "messageId": "m1", "to": "", "subject": "", "date": "", "snippet": "",
    }]


def test_api_gmail_send_includes_headers_in_raw_mime(api_module):
    """gmail send builds the messages.send body from a base64url raw MIME that
    carries the recipient and subject headers."""
    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stdout='{"id": "m1", "threadId": "t1"}', stderr="")

    args = api_module.argparse.Namespace(
        to="user@example.com", subject="Hi", body="Hello",
        cc="", from_header="", html=False, thread_id="", func=api_module.gmail_send,
    )

    with patch.object(api_module.subprocess, "run", side_effect=capture_run):
        api_module.gmail_send(args)

    cmd = captured["cmd"]
    assert "messages" in cmd
    assert "send" in cmd
    decoded = _decode_raw(json.loads(cmd[cmd.index("--json") + 1])["raw"])
    assert "user@example.com" in decoded
    assert "Hi" in decoded


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
