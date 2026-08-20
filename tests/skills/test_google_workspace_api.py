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


# ---------------------------------------------------------------------------
# gmail reply/send MIME building (_build_reply_mime / _build_send_mime)
# ---------------------------------------------------------------------------

OWN_ADDRESS = "me@example.com"


def _reply_headers(**overrides):
    base = {
        "from": "Support Desk <support@example.com>",
        "to": f"Me <{OWN_ADDRESS}>",
        "subject": "Order 1234",
        "message-id": "<abc@mail.example.com>",
    }
    base.update(overrides)
    return base


def _send_args(api_module, **overrides):
    base = dict(to="user@example.com", subject="Hi", body="Hello", html=False, cc="", from_header="")
    base.update(overrides)
    return api_module.argparse.Namespace(**base)


def _unfolded_to_line(message):
    wire = message.as_bytes().replace(b"\r\n ", b" ").replace(b"\r\n\t", b" ")
    lines = [line for line in wire.split(b"\r\n") if line.lower().startswith(b"to:")]
    assert lines, wire
    return lines[0]


def test_api_reply_targets_from_of_external_sender(api_module):
    msg = api_module._build_reply_mime(_reply_headers(), "body", OWN_ADDRESS)
    assert [a.addr_spec for a in msg["To"].addresses] == ["support@example.com"]


def test_api_reply_to_own_sent_message_targets_original_recipients(api_module):
    """Replying to your own sent message must address the counterparty, not yourself."""
    msg = api_module._build_reply_mime(
        _reply_headers(**{
            "from": f"Me <{OWN_ADDRESS}>",
            "to": "Support Desk <support@example.com>",
        }),
        "body",
        OWN_ADDRESS,
    )
    assert [a.addr_spec for a in msg["To"].addresses] == ["support@example.com"]


def test_api_reply_prefers_reply_to_header(api_module):
    msg = api_module._build_reply_mime(
        _reply_headers(**{"reply-to": "Case Desk <cases@example.com>"}), "body", OWN_ADDRESS
    )
    assert [a.addr_spec for a in msg["To"].addresses] == ["cases@example.com"]


def test_api_reply_non_ascii_display_name_keeps_addr_spec_plain(api_module):
    """Gmail rejects address headers where an RFC 2047 encoded-word spans the
    whole value ("Invalid To header") — only the display name may be encoded."""
    msg = api_module._build_reply_mime(
        _reply_headers(**{"from": '"Jörg Müller" <jorg@example.de>'}), "body", OWN_ADDRESS
    )
    to_line = _unfolded_to_line(msg)
    assert b"<jorg@example.de>" in to_line
    assert msg["To"].addresses[0].display_name == "Jörg Müller"


def test_api_reply_without_recipient_raises(api_module):
    with pytest.raises(RuntimeError):
        api_module._build_reply_mime(
            _reply_headers(**{"from": "", "to": "", "reply-to": ""}), "body", OWN_ADDRESS
        )


def test_api_reply_subject_and_threading_headers(api_module):
    msg = api_module._build_reply_mime(_reply_headers(), "body", OWN_ADDRESS)
    assert str(msg["Subject"]) == "Re: Order 1234"
    assert str(msg["In-Reply-To"]) == "<abc@mail.example.com>"
    assert str(msg["References"]) == "<abc@mail.example.com>"

    already = api_module._build_reply_mime(_reply_headers(subject="Re: x"), "body", OWN_ADDRESS)
    assert str(already["Subject"]) == "Re: x"


def test_api_send_non_ascii_display_name_keeps_addr_spec_plain(api_module):
    msg = api_module._build_send_mime(_send_args(api_module, to='"Jörg Müller" <jorg@example.de>'))
    to_line = _unfolded_to_line(msg)
    assert b"<jorg@example.de>" in to_line
    assert msg["To"].addresses[0].display_name == "Jörg Müller"


def test_api_send_multiple_recipients_and_cc(api_module):
    msg = api_module._build_send_mime(_send_args(
        api_module,
        to='a@example.com, "Bo Ärger" <b@example.de>',
        cc="c@example.com",
    ))
    assert [a.addr_spec for a in msg["To"].addresses] == ["a@example.com", "b@example.de"]
    assert [a.addr_spec for a in msg["Cc"].addresses] == ["c@example.com"]


def test_api_send_html_subtype(api_module):
    msg = api_module._build_send_mime(_send_args(api_module, body="<h1>x</h1>", html=True))
    assert msg.get_content_type() == "text/html"


def test_api_send_invalid_to_raises(api_module):
    with pytest.raises(RuntimeError):
        api_module._build_send_mime(_send_args(api_module, to="not-an-address"))


def test_api_gmail_reply_dry_run_resolves_own_message(api_module, capsys):
    """--dry-run resolves the recipient (via getProfile) and prints headers
    without ever invoking messages.send."""

    def fake_run_gws(parts, params=None, body=None):
        if parts == ["gmail", "users", "messages", "get"]:
            assert "To" in params["metadataHeaders"]
            assert "Reply-To" in params["metadataHeaders"]
            return {
                "threadId": "t1",
                "payload": {"headers": [
                    {"name": "From", "value": f"Me <{OWN_ADDRESS}>"},
                    {"name": "To", "value": "Shop <shop@example.com>"},
                    {"name": "Subject", "value": "Order 1234"},
                    {"name": "Message-ID", "value": "<x@example.com>"},
                ]},
            }
        if parts == ["gmail", "users", "getProfile"]:
            assert params == {"userId": "me"}
            return {"emailAddress": OWN_ADDRESS}
        raise AssertionError(f"unexpected gws call in dry-run: {parts}")

    api_module._run_gws = fake_run_gws
    args = api_module.argparse.Namespace(message_id="m1", body="ping", from_header="", dry_run=True)
    api_module.gmail_reply(args)

    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "dry-run"
    assert out["threadId"] == "t1"
    assert out["headers"]["To"] == "Shop <shop@example.com>"
