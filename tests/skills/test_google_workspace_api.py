"""Tests for Google Workspace gws bridge and CLI wrapper."""

import base64
import importlib.util
import json
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
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
FORMATTING_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/gmail_reply_formatting.py"
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


@pytest.fixture
def formatting_module():
    spec = importlib.util.spec_from_file_location(
        "gmail_reply_formatting_test", FORMATTING_PATH,
    )
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


def _gmail_data(value: str, encoding="utf-8") -> str:
    return base64.urlsafe_b64encode(value.encode(encoding)).decode().rstrip("=")


def _decoded_message(raw: str):
    data = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    return BytesParser(policy=policy.default).parsebytes(data)


def _reply_args(api_module, **overrides):
    values = {
        "message_id": "orig-123",
        "body": "Thanks for the update.",
        "from_header": "",
        "html": False,
        "no_quote_original": False,
        "func": api_module.gmail_reply,
    }
    values.update(overrides)
    return api_module.argparse.Namespace(**values)


def _original_message(*, subject="Status", sender="Sender <sender@example.com>", parts=None):
    return {
        "id": "orig-123",
        "threadId": "thread-456",
        "payload": {
            "mimeType": "multipart/alternative" if parts else "text/plain",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "Date", "value": "Wed, 19 Aug 2026 14:30:00 +0000"},
                {"name": "Subject", "value": subject},
                {"name": "Message-ID", "value": "<current@example.com>"},
                {"name": "In-Reply-To", "value": "<parent@example.com>"},
                {"name": "References", "value": "<root@example.com> <parent@example.com>"},
            ],
            **({"parts": parts} if parts else {"body": {"data": _gmail_data("Original body")}}),
        },
    }


def _run_gws_reply(api_module, original, args, attachments=None):
    captured = {}
    attachments = attachments or {}

    def fake_run(parts, *, params=None, body=None):
        if parts[-2:] == ["messages", "get"]:
            captured["get_params"] = params
            return original
        if parts[-2:] == ["attachments", "get"]:
            return attachments[params["id"]]
        if parts[-2:] == ["messages", "send"]:
            captured["send_body"] = body
            return {"id": "reply-789", "threadId": original["threadId"]}
        raise AssertionError(parts)

    api_module._run_gws = fake_run
    api_module.gmail_reply(args)
    return captured, _decoded_message(captured["send_body"]["raw"])


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


def test_api_gmail_reply_quotes_plain_body_and_preserves_thread_ancestry(api_module):
    captured, message = _run_gws_reply(
        api_module,
        _original_message(sender="Jöhn Döe <sender@example.com>"),
        _reply_args(api_module),
    )

    assert captured["get_params"]["format"] == "full"
    assert captured["send_body"]["threadId"] == "thread-456"
    assert message.get_content_type() == "text/plain"
    assert message.get_content_charset() == "utf-8"
    assert "sender@example.com" in str(message["To"])
    assert message["Subject"] == "Re: Status"
    assert message.get_content() == (
        "Thanks for the update.\n\n"
        "On Wed, 19 Aug 2026 14:30:00 +0000, Jöhn Döe <sender@example.com> wrote:\n"
        "> Original body"
    )
    assert message["In-Reply-To"] == "<current@example.com>"
    assert message["References"] == (
        "<root@example.com> <parent@example.com> <current@example.com>"
    )


def test_api_gmail_reply_trims_existing_plain_quote_chain(api_module):
    original = _original_message()
    original["payload"]["body"]["data"] = _gmail_data(
        "Immediate reply\n\n  -----Original Message-----\nOld chain"
    )
    _, message = _run_gws_reply(api_module, original, _reply_args(api_module))

    assert "> Immediate reply" in message.get_content()
    assert "Old chain" not in message.get_content()


def test_html_sanitizer_closes_safe_ancestors_before_prior_quote(formatting_module):
    value = (
        "<div><table><tbody><tr><td>Current reply"
        '<div id="divreplyfwdmsg">Older quoted chain</div>'
        "</td></tr></tbody></table></div>"
    )

    assert formatting_module.sanitize_quoted_html(value) == (
        "<div><table><tbody><tr><td>Current reply"
        "</td></tr></tbody></table></div>"
    )


@pytest.mark.parametrize(
    "prior_quote",
    [
        '<div id="divreplyfwdmsg">Outlook quoted chain</div>',
        '<div class="protonmail_quote">ProtonMail quoted chain</div>',
    ],
)
def test_html_to_text_uses_all_prior_quote_markers(formatting_module, prior_quote):
    value = f"<p>Current reply</p>{prior_quote}"

    assert formatting_module.html_to_text(value) == "Current reply"


def test_api_gmail_reply_preserves_sanitized_html_by_default(api_module):
    original_html = """
        <div><p style="color: blue; position: fixed">Formatted <strong>history</strong>.</p>
        <table><tr><td>Cell</td></tr></table>
        <a href="javascript:alert(1)" onclick="alert(1)">unsafe link</a>
        <img src="https://tracker.example/pixel.gif">
        <script>alert(1)</script>
        <div class="gmail_quote">Older quoted chain</div></div>
    """
    parts = [{"mimeType": "text/html", "body": {"data": _gmail_data(original_html)}}]
    _, message = _run_gws_reply(
        api_module,
        _original_message(parts=parts),
        _reply_args(api_module, body="Line one\nLine two"),
    )

    body = message.get_content()
    assert message.get_content_type() == "text/html"
    assert body.startswith("Line one<br>\nLine two<br><br>")
    assert '<div class="gmail_quote" data-hermes-quote="original">' in body
    assert '<p style="color: blue">Formatted <strong>history</strong>.</p>' in body
    assert "<table><tr><td>Cell</td></tr></table>" in body
    assert "position" not in body
    assert "javascript:" not in body
    assert "onclick" not in body
    assert "<img" not in body
    assert "<script" not in body
    assert "Older quoted chain" not in body


def test_api_gmail_reply_fetches_attachment_backed_nested_body(api_module):
    parts = [{
        "mimeType": "multipart/mixed",
        "parts": [{
            "mimeType": "text/plain",
            "filename": "private-notes.txt",
            "body": {"data": _gmail_data("Do not quote this attachment")},
        }, {
            "mimeType": "multipart/alternative",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "headers": [{"name": "Content-Type", "value": "text/plain; charset=iso-8859-1"}],
                    "body": {"attachmentId": "plain-part"},
                },
                {
                    "mimeType": "application/pdf",
                    "headers": [{"name": "Content-Disposition", "value": "attachment; filename=report.pdf"}],
                    "body": {"attachmentId": "pdf-part"},
                },
            ],
        }],
    }]
    encoded = base64.urlsafe_b64encode("Résumé original".encode("iso-8859-1")).decode().rstrip("=")
    _, message = _run_gws_reply(
        api_module,
        _original_message(parts=parts),
        _reply_args(api_module),
        attachments={"plain-part": {"data": encoded}},
    )

    assert "> Résumé original" in message.get_content()
    assert "private-notes" not in message.get_content()


def test_api_gmail_reply_no_quote_uses_metadata_and_body_only(api_module):
    captured, message = _run_gws_reply(
        api_module,
        _original_message(),
        _reply_args(api_module, no_quote_original=True),
    )

    assert captured["get_params"]["format"] == "metadata"
    assert captured["get_params"]["metadataHeaders"] == [
        "From", "Subject", "Date", "Message-ID", "In-Reply-To", "References",
    ]
    assert message.get_content() == "Thanks for the update."
    assert message["References"] == (
        "<root@example.com> <parent@example.com> <current@example.com>"
    )


def test_api_gmail_reply_python_backend_supports_authored_html(api_module):
    original = _original_message()
    messages = MagicMock()
    messages.get.return_value.execute.return_value = original
    messages.send.return_value.execute.return_value = {
        "id": "reply-789", "threadId": "thread-456",
    }
    users = MagicMock()
    users.messages.return_value = messages
    service = MagicMock()
    service.users.return_value = users
    api_module._gws_binary = lambda: None
    api_module.build_service = lambda *_args: service

    api_module.gmail_reply(_reply_args(api_module, body="<strong>Thanks</strong>", html=True))

    get_kwargs = messages.get.call_args.kwargs
    send_body = messages.send.call_args.kwargs["body"]
    message = _decoded_message(send_body["raw"])
    assert get_kwargs == {"userId": "me", "id": "orig-123", "format": "full"}
    assert send_body["threadId"] == "thread-456"
    assert message.get_content_type() == "text/html"
    assert message.get_content().startswith("<strong>Thanks</strong><br><br>")
    assert "&gt; Original body" not in message.get_content()
    assert "Original body" in message.get_content()


def test_api_gmail_reply_python_backend_fetches_inline_body(api_module):
    original = _original_message(parts=[{
        "mimeType": "text/plain",
        "body": {"attachmentId": "body-part"},
    }])
    messages = MagicMock()
    messages.get.return_value.execute.return_value = original
    messages.attachments.return_value.get.return_value.execute.return_value = {
        "data": _gmail_data("Python backend original"),
    }
    messages.send.return_value.execute.return_value = {
        "id": "reply-789", "threadId": "thread-456",
    }
    users = MagicMock()
    users.messages.return_value = messages
    service = MagicMock()
    service.users.return_value = users
    api_module._gws_binary = lambda: None
    api_module.build_service = lambda *_args: service

    api_module.gmail_reply(_reply_args(api_module))

    message = _decoded_message(messages.send.call_args.kwargs["body"]["raw"])
    assert "> Python backend original" in message.get_content()
    messages.attachments.return_value.get.assert_called_once_with(
        userId="me", messageId="orig-123", id="body-part",
    )


def test_api_gmail_reply_cli_parses_html_and_quote_opt_out(api_module, monkeypatch):
    captured = {}

    def fake_reply(args):
        captured["args"] = args

    monkeypatch.setattr(api_module, "gmail_reply", fake_reply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "google_api.py", "gmail", "reply", "orig-123",
            "--body", "<p>Thanks</p>", "--html", "--no-quote-original",
        ],
    )

    api_module.main()

    assert captured["args"].message_id == "orig-123"
    assert captured["args"].html is True
    assert captured["args"].no_quote_original is True












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
