"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import base64
import io
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
    path.write_text(json.dumps(data), encoding="utf-8")


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

    saved = json.loads(token_path.read_text(encoding="utf-8"))
    assert isinstance(creds, FakeCredentials)
    assert saved["token"] == "ya29.refreshed"
    assert saved["type"] == "authorized_user"


def _tabbed_doc():
    """A Doc with two tabs (one nested), as the Docs API returns with includeTabsContent."""
    def body(text):
        return {"content": [
            {"endIndex": len(text) + 2,
             "paragraph": {"elements": [{"textRun": {"content": text + "\n"}}]}},
        ]}
    return {
        "title": "Tabbed",
        "documentId": "doc1",
        "tabs": [
            {
                "tabProperties": {"tabId": "t.0", "title": "First"},
                "documentTab": {"body": body("alpha")},
                "childTabs": [
                    {
                        "tabProperties": {"tabId": "t.0.a", "title": "Nested"},
                        "documentTab": {"body": body("beta")},
                    }
                ],
            },
            {
                "tabProperties": {"tabId": "t.1", "title": "Second"},
                "documentTab": {"body": body("gamma")},
            },
        ],
    }


def test_docs_get_returns_every_tab_of_a_tabbed_doc(api_module, monkeypatch, capsys):
    """A multi-tab Doc must not lose tab content: reads traverse the tabs tree
    (preorder, nested tabs included) instead of only the legacy top-level body."""
    monkeypatch.setattr(
        api_module, "_run_gws",
        lambda parts, params=None, body=None: _tabbed_doc(),
    )
    args = types.SimpleNamespace(doc_id="doc1", tab=None)
    api_module.docs_get(args)
    result = json.loads(capsys.readouterr().out)
    tabs = {t["tabId"]: t for t in result["tabs"]}
    assert set(tabs) == {"t.0", "t.0.a", "t.1"}
    assert tabs["t.0.a"]["body"] == "beta\n"
    assert tabs["t.0.a"]["level"] == 1
    # Multi-tab docs have no single merged "body" — index spaces are independent.
    assert "body" not in result


def test_docs_append_carries_tab_id_and_refuses_ambiguous_writes(api_module, monkeypatch, capsys):
    """Each tab has its own index space, so a write must target exactly one tab:
    the insert location carries the tabId, and an un-targeted write against a
    multi-tab doc errors instead of silently landing in the first tab."""
    monkeypatch.setattr(
        api_module, "_run_gws",
        lambda parts, params=None, body=None: _tabbed_doc(),
    )
    sent = {}
    monkeypatch.setattr(
        api_module, "_docs_insert_text",
        lambda doc_id, text, index, tab_id=None: sent.update(
            {"doc_id": doc_id, "index": index, "tab_id": tab_id}
        ),
    )

    api_module.docs_append(types.SimpleNamespace(doc_id="doc1", text="more", tab="t.1"))
    assert sent["tab_id"] == "t.1"
    assert sent["index"] == len("gamma") + 1  # endIndex - 1 within THAT tab's space
    capsys.readouterr()

    with pytest.raises(SystemExit):
        api_module.docs_append(types.SimpleNamespace(doc_id="doc1", text="more", tab=None))
    err = json.loads(capsys.readouterr().err)
    assert "tabs" in err and len(err["tabs"]) == 3


# --- multiline body handling in gmail send/reply ---
#
# Agents build commands as ``--body "Line one\n\nLine two"``; POSIX shells keep
# the backslash, so Gmail used to send that literal text as the body. The fix is
# a non-lossy multiline interface: ``--body`` stays byte-for-byte, multiline
# bodies come from ``--body-file``/stdin, and escape decoding is opt-in.


def _mime_body(raw):
    """Decode a Gmail ``raw`` payload and return just the message body."""
    decoded = base64.urlsafe_b64decode(raw + "==").decode("utf-8")
    return decoded.split("\n\n", 1)[1]


def _reply_original():
    return {
        "threadId": "t1",
        "payload": {
            "headers": [
                {"name": "From", "value": "them@example.com"},
                {"name": "Subject", "value": "subject"},
                {"name": "Message-ID", "value": "<abc@example.com>"},
            ]
        },
    }


def _capturing_run_gws(captured):
    """Stub _run_gws: serve the reply's metadata read, capture the outgoing raw."""

    def run(parts, params=None, body=None):
        if parts[:4] == ["gmail", "users", "messages", "get"]:
            return _reply_original()
        captured.update(body or {})
        return {"id": "m1", "threadId": "t1"}

    return run


class _FakeExec:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class _FakeService:
    """Minimal googleapiclient stand-in that records the outgoing raw payload."""

    def __init__(self, store):
        self._store = store

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId=None, body=None):
        self._store.update(body or {})
        return _FakeExec({"id": "py1", "threadId": "t1"})

    def get(self, userId=None, id=None, format=None, metadataHeaders=None):
        return _FakeExec(_reply_original())


def _send_args(**overrides):
    args = {
        "to": "me@example.com",
        "subject": "subject",
        "body": None,
        "body_file": None,
        "decode_escapes": False,
        "cc": None,
        "from_header": None,
        "html": False,
        "thread_id": None,
    }
    args.update(overrides)
    return types.SimpleNamespace(**args)


def _reply_args(**overrides):
    args = {
        "message_id": "m0",
        "body": None,
        "body_file": None,
        "decode_escapes": False,
        "from_header": None,
    }
    args.update(overrides)
    return types.SimpleNamespace(**args)


class TestGmailBodyResolution:
    """``--body`` is byte-for-byte; multiline input is explicit and non-lossy."""

    def test_body_source_resolution(self, api_module, tmp_path, monkeypatch):
        resolve = api_module._resolve_body

        # A literal backslash-n survives: --body is never rewritten.
        literal = "code: printf 'a" + "\\" + "n'"
        assert resolve(_send_args(body=literal)) == literal

        # Real newlines from a file reach the payload unchanged.
        body_file = tmp_path / "body.txt"
        body_file.write_text("Line one\n\nLine two\n", encoding="utf-8")
        assert resolve(_send_args(body_file=str(body_file))) == "Line one\n\nLine two\n"

        # ``-`` reads stdin, which is how a shell pipe supplies a multiline body.
        monkeypatch.setattr("sys.stdin", io.StringIO("piped\n\nbody\n"))
        assert resolve(_send_args(body_file="-")) == "piped\n\nbody\n"

        # Escape decoding only happens when explicitly requested.
        assert resolve(_send_args(body="a\\n\\nb", decode_escapes=True)) == "a\n\nb"

    def test_missing_body_is_rejected(self, api_module, capsys):
        with pytest.raises(SystemExit):
            api_module._resolve_body(_send_args())
        assert "error" in json.loads(capsys.readouterr().err)

    @pytest.mark.parametrize("use_gws_binary", [True, False], ids=["gws-binary", "python-client"])
    @pytest.mark.parametrize("command", ["send", "reply"])
    def test_multiline_body_reaches_the_mime_payload(
        self, api_module, monkeypatch, capsys, tmp_path, command, use_gws_binary
    ):
        """Every send/reply client path carries the body-file newlines into MIME."""
        body_file = tmp_path / "body.txt"
        body_file.write_text("Line one\n\nLine two\n", encoding="utf-8")
        captured = {}

        if use_gws_binary:
            monkeypatch.setattr(api_module, "_run_gws", _capturing_run_gws(captured))
        else:
            monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
            monkeypatch.setattr(api_module, "build_service", lambda *a, **k: _FakeService(captured))

        if command == "send":
            api_module.gmail_send(_send_args(body_file=str(body_file)))
        else:
            api_module.gmail_reply(_reply_args(body_file=str(body_file)))
        capsys.readouterr()

        body = _mime_body(captured["raw"])
        assert "Line one\n\nLine two" in body
        assert "\\n" not in body
