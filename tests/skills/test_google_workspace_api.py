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


class _FakeRequest:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class _FakeGmail:
    """Minimal googleapiclient stand-in: users().messages().list/get plus batch requests.
    Ids in ``fail_in_batch`` come back from a batch with an error (as Gmail does for 429s)."""

    def __init__(self, ids, fail_in_batch=()):
        self.msgs = {
            i: {
                "id": i,
                "threadId": f"t-{i}",
                "snippet": f"snippet {i}",
                "labelIds": ["INBOX"],
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [{"name": "Subject", "value": f"subject {i}"}],
                    "body": {"data": "aGk="},
                },
            }
            for i in ids
        }
        self.fail_in_batch = set(fail_in_batch)
        self.batches = 0
        self.single_gets = []

    def users(self):
        return self

    def messages(self):
        return self

    def list(self, userId, q, maxResults):
        return _FakeRequest(lambda: {"messages": [{"id": i} for i in list(self.msgs)[:maxResults]]})

    def get(self, userId, id, format, metadataHeaders=None):
        def run():
            self.single_gets.append(id)
            return self.msgs[id]
        return _FakeRequest(run)

    def new_batch_http_request(self, callback):
        svc = self

        class _Batch:
            def __init__(self):
                self.items = []

            def add(self, request, request_id):
                self.items.append(request_id)

            def execute(self):
                svc.batches += 1
                for rid in self.items:
                    if rid in svc.fail_in_batch:
                        callback(rid, None, Exception("429 rateLimitExceeded"))
                    else:
                        callback(rid, svc.msgs[rid], None)

        return _Batch()


@pytest.fixture
def python_gmail(api_module, monkeypatch):
    """Force the Python SDK path with a fake Gmail service."""
    def install(fake):
        monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
        monkeypatch.setattr(api_module, "build_service", lambda *a, **k: fake)
        return fake
    return install


def test_gmail_search_fetches_metadata_in_one_batch_and_refetches_dropped_items(api_module, python_gmail, capsys):
    """Search must not cost one round trip per message, and items Gmail drops from
    a batch (429) are fetched individually so no result silently disappears."""
    ids = [f"m{i}" for i in range(20)]
    fake = python_gmail(_FakeGmail(ids, fail_in_batch={"m3", "m7"}))
    api_module.gmail_search(types.SimpleNamespace(query="from:x", max=20))
    out = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in out] == ids  # all results, original order
    assert fake.batches == 1
    assert sorted(fake.single_gets) == ["m3", "m7"]  # only the dropped ones
    assert out[0]["subject"] == "subject m0"


def test_gmail_get_accepts_several_ids_in_one_batch(api_module, python_gmail, capsys):
    fake = python_gmail(_FakeGmail(["a", "b", "c"], fail_in_batch={"b"}))
    api_module.gmail_get(types.SimpleNamespace(message_id=["a", "b", "c"]))
    out = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in out] == ["a", "b", "c"]
    assert all("body" in m for m in out)
    assert fake.batches == 1 and fake.single_gets == ["b"]


def test_gmail_get_single_id_keeps_object_output(api_module, python_gmail, capsys):
    fake = python_gmail(_FakeGmail(["a"]))
    api_module.gmail_get(types.SimpleNamespace(message_id=["a"]))
    out = json.loads(capsys.readouterr().out)
    assert isinstance(out, dict) and out["id"] == "a"
    assert fake.batches == 0 and fake.single_gets == ["a"]
