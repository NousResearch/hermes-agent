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


# ── gmail send --attach (#72896) ───────────────────────────────────────────

class _SendArgs:
    """Stand-in for argparse's namespace for `gmail send`."""

    def __init__(self, **kw):
        self.to = kw.get("to", "user@example.com")
        self.subject = kw.get("subject", "Subject")
        self.body = kw.get("body", "Body text")
        self.cc = kw.get("cc", "")
        self.from_header = kw.get("from_header", "")
        self.html = kw.get("html", False)
        self.thread_id = kw.get("thread_id", "")
        self.attach = kw.get("attach", [])


def _parts(message):
    return [p for p in message.walk() if p.get_content_maintype() != "multipart"]


class TestGmailAttachments:
    """`gmail send` could not attach a file — the highest-volume outbound
    email use case (cover letter + CV) had no way to include one."""

    def test_no_attachment_still_produces_a_single_part_text_message(self, api_module):
        """Backward compatibility: existing invocations must be unchanged."""
        msg = api_module._build_gmail_message(_SendArgs())

        assert msg.get_content_type() == "text/plain"
        assert not msg.is_multipart()
        assert msg["To"] == "user@example.com"
        assert msg["Subject"] == "Subject"
        assert msg.get_payload() == "Body text"

    def test_html_flag_survives(self, api_module):
        msg = api_module._build_gmail_message(_SendArgs(html=True))
        assert msg.get_content_type() == "text/html"

    def test_one_attachment_round_trips(self, api_module, tmp_path):
        cv = tmp_path / "cv.pdf"
        cv.write_bytes(b"%PDF-1.4 fake cv bytes")

        msg = api_module._build_gmail_message(_SendArgs(attach=[str(cv)]))

        assert msg.get_content_type() == "multipart/mixed"
        parts = _parts(msg)
        assert parts[0].get_content_type() == "text/plain"
        assert parts[0].get_payload() == "Body text"

        att = parts[1]
        assert att.get_filename() == "cv.pdf", "the filename was not preserved"
        assert att.get_content_type() == "application/pdf"
        assert att.get_payload(decode=True) == b"%PDF-1.4 fake cv bytes", (
            "the attachment bytes did not survive the MIME round-trip"
        )

    def test_several_attachments_produce_several_parts(self, api_module, tmp_path):
        a = tmp_path / "cover.txt"; a.write_text("cover letter")
        b = tmp_path / "cv.pdf"; b.write_bytes(b"%PDF cv")

        msg = api_module._build_gmail_message(_SendArgs(attach=[str(a), str(b)]))

        names = [p.get_filename() for p in _parts(msg) if p.get_filename()]
        assert names == ["cover.txt", "cv.pdf"]

    def test_unknown_extension_falls_back_to_octet_stream(self, api_module, tmp_path):
        f = tmp_path / "notes.weirdext"
        f.write_bytes(b"\x00\x01\x02")
        msg = api_module._build_gmail_message(_SendArgs(attach=[str(f)]))
        att = [p for p in _parts(msg) if p.get_filename()][0]
        assert att.get_content_type() == "application/octet-stream"

    def test_missing_file_fails_before_any_api_call(self, api_module, tmp_path):
        with pytest.raises(SystemExit) as exc:
            api_module._build_gmail_message(_SendArgs(attach=[str(tmp_path / "nope.pdf")]))
        assert "not found" in str(exc.value).lower()

    def test_directory_is_rejected(self, api_module, tmp_path):
        with pytest.raises(SystemExit) as exc:
            api_module._build_gmail_message(_SendArgs(attach=[str(tmp_path)]))
        assert "not a regular file" in str(exc.value).lower()

    def _send(self, api_module, args):
        """Drive the real send path with the network stubbed out."""
        api_module._run_gws = lambda path, params=None, body=None, **kw: {
            "id": "m1", "threadId": "",
        }
        api_module.gmail_send(args)

    def test_oversized_attachment_is_refused_with_gmails_limit(self, api_module, tmp_path):
        """Better a clear local error than a rejected API call."""
        big = tmp_path / "huge.bin"
        big.write_bytes(b"\x00" * (26 * 1024 * 1024))
        with pytest.raises(SystemExit) as exc:
            self._send(api_module, _SendArgs(attach=[str(big)]))
        assert "Gmail" in str(exc.value) and "25 MB" in str(exc.value)

    def test_a_large_body_counts_toward_the_limit(self, api_module, tmp_path):
        """The cap is on the whole message, not on the attachment bytes.

        Counting only source files passes this: the attachment is tiny, and it
        is the body that pushes the serialized message over Gmail's limit.
        """
        small = tmp_path / "note.txt"
        small.write_bytes(b"x" * 1024)

        with pytest.raises(SystemExit) as exc:
            self._send(
                api_module,
                _SendArgs(body="A" * (26 * 1024 * 1024), attach=[str(small)]),
            )
        assert "25 MB" in str(exc.value)

    def test_base64_expansion_counts_toward_the_limit(self, api_module, tmp_path):
        """Attachments are base64'd, inflating them ~33% on the wire.

        20 MB of source bytes is under 25 MB, but the encoded message is not.
        """
        f = tmp_path / "big.bin"
        f.write_bytes(b"\x00" * (20 * 1024 * 1024))
        with pytest.raises(SystemExit) as exc:
            self._send(api_module, _SendArgs(attach=[str(f)]))
        assert "25 MB" in str(exc.value)

    def test_a_message_comfortably_under_the_limit_is_sent(self, api_module, tmp_path):
        f = tmp_path / "ok.bin"
        f.write_bytes(b"\x00" * (1024 * 1024))
        self._send(api_module, _SendArgs(body="hi", attach=[str(f)]))   # must not raise

    def test_tilde_paths_are_expanded(self, api_module, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        f = tmp_path / "cv.pdf"; f.write_bytes(b"%PDF")
        msg = api_module._build_gmail_message(_SendArgs(attach=["~/cv.pdf"]))
        assert [p.get_filename() for p in _parts(msg) if p.get_filename()] == ["cv.pdf"]

    def test_send_carries_the_attachment_through_the_gws_path(self, api_module, tmp_path):
        """Both send paths base64 the same message, so both must attach.

        The `gws` binary path is the one taken when the CLI is installed; the
        attachment must not be silently dropped there.
        """
        import base64

        cv = tmp_path / "cv.pdf"
        cv.write_bytes(b"%PDF-1.4 attached")
        captured = {}

        def _fake_run_gws(path, params=None, body=None, **kw):
            captured["raw"] = (body or {}).get("raw", "")
            return {"id": "m1", "threadId": "t1"}

        api_module._run_gws = _fake_run_gws
        api_module.gmail_send(_SendArgs(attach=[str(cv)]))

        decoded = base64.urlsafe_b64decode(captured["raw"].encode())
        assert b"cv.pdf" in decoded, (
            "the gws send path dropped the attachment — it builds its own "
            "message instead of using the shared builder"
        )
        assert b"multipart/mixed" in decoded

    def test_send_without_attachment_is_unchanged_on_the_gws_path(self, api_module):
        import base64

        captured = {}
        api_module._run_gws = lambda path, params=None, body=None, **kw: (
            captured.update(raw=(body or {}).get("raw", "")) or {"id": "m1", "threadId": ""}
        )
        api_module.gmail_send(_SendArgs())

        decoded = base64.urlsafe_b64decode(captured["raw"].encode())
        assert b"multipart" not in decoded
        assert b"Body text" in decoded
