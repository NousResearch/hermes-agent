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
    token_path = bridge_module.get_token_path("jid")
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token("jid")
    assert result == "ya29.valid"


def test_bridge_get_token_path_fails_closed_on_unknown_identity(bridge_module):
    """An unregistered identity must raise, never fall back to jid's path."""
    with pytest.raises(bridge_module.UnknownGoogleIdentityError):
        bridge_module.get_token_path("someone-not-registered")


def test_bridge_get_token_path_fails_closed_on_missing_identity(bridge_module):
    """No identity at all must raise, never default to jid."""
    with pytest.raises(bridge_module.UnknownGoogleIdentityError):
        bridge_module.get_token_path(None)










def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path("jid")
    _write_token(token_path, token="ya29.injected", expiry=future)

    captured = {}

    def capture_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return MagicMock(returncode=0)

    with patch.object(sys, "argv", ["gws_bridge.py", "--identity", "jid", "gmail", "+triage"]):
        with patch.object(subprocess, "run", side_effect=capture_run):
            with pytest.raises(SystemExit):
                bridge_module.main()

    assert captured["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "ya29.injected"
    assert captured["cmd"] == ["gws", "gmail", "+triage"]


def test_bridge_main_requires_identity_flag(bridge_module):
    """Omitting --identity must fail closed, not default to any identity."""
    with patch.object(sys, "argv", ["gws_bridge.py", "gmail", "+triage"]):
        with pytest.raises(SystemExit):
            bridge_module.main()


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












def test_calendar_default_is_all(api_module):
    """--calendar defaults to 'all', not a single calendar — the 2026-08-12
    incident was caused by the old 'primary'-only default silently missing
    every sub-calendar (Family, Birthdays, a person's own named calendar)."""
    captured_args = {}

    def fake_calendar_list(args):
        captured_args["calendar"] = args.calendar

    api_module.calendar_list = fake_calendar_list
    with patch.object(sys, "argv", ["google_api.py", "--identity", "jid", "calendar", "list"]):
        api_module.main()
    assert captured_args["calendar"] == "all"


def test_calendar_list_all_aggregates_across_calendars(api_module):
    """calendar list --calendar all merges events from every calendar, tagged
    with which calendar each event came from, sorted by start time."""
    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="all", func=api_module.calendar_list,
    )

    def fake_all_calendar_ids():
        return [("primary", "primary"), ("family123@group.calendar.google.com", "Family")]

    def fake_events_for_calendar(calendar_id, time_min, time_max, max_results):
        if calendar_id == "primary":
            return [{
                "id": "ev1", "summary": "Primary event",
                "start": {"dateTime": "2026-08-13T10:00:00-04:00"},
                "end": {"dateTime": "2026-08-13T11:00:00-04:00"},
            }]
        return [{
            "id": "ev2", "summary": "Family event",
            "start": {"dateTime": "2026-08-13T08:00:00-04:00"},
            "end": {"dateTime": "2026-08-13T09:00:00-04:00"},
        }]

    api_module._all_calendar_ids = fake_all_calendar_ids
    api_module._events_for_calendar = fake_events_for_calendar

    captured = {}
    def fake_print(s):
        captured["out"] = s
    with patch.object(api_module, "print", fake_print, create=True):
        api_module.calendar_list(args)

    events = json.loads(captured["out"])
    assert len(events) == 2
    # Family event (08:00) sorts before Primary event (10:00).
    assert events[0]["summary"] == "Family event"
    assert events[0]["calendar"] == "Family"
    assert events[1]["summary"] == "Primary event"
    assert events[1]["calendar"] == "primary"


def test_calendar_list_all_survives_one_broken_calendar(api_module):
    """A broken/inaccessible calendar must not blank out every other one."""
    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="all", func=api_module.calendar_list,
    )

    def fake_all_calendar_ids():
        return [("bad-cal", "Broken"), ("primary", "primary")]

    def fake_events_for_calendar(calendar_id, time_min, time_max, max_results):
        if calendar_id == "bad-cal":
            raise RuntimeError("404 not found")
        return [{
            "id": "ev1", "summary": "Still works",
            "start": {"dateTime": "2026-08-13T10:00:00-04:00"},
            "end": {"dateTime": "2026-08-13T11:00:00-04:00"},
        }]

    api_module._all_calendar_ids = fake_all_calendar_ids
    api_module._events_for_calendar = fake_events_for_calendar

    captured = {}
    def fake_print(s, **kwargs):
        if "file" not in kwargs:
            captured["out"] = s
    with patch.object(api_module, "print", fake_print, create=True):
        api_module.calendar_list(args)

    events = json.loads(captured["out"])
    assert len(events) == 1
    assert events[0]["summary"] == "Still works"


def test_calendar_list_specific_calendar_still_works(api_module):
    """Passing an explicit --calendar value (not 'all') still narrows to just
    that one calendar — this is a real narrowing option, not removed."""
    args = api_module.argparse.Namespace(
        start="", end="", max=25, calendar="primary", func=api_module.calendar_list,
    )

    calls = []
    def fake_events_for_calendar(calendar_id, time_min, time_max, max_results):
        calls.append(calendar_id)
        return []

    api_module._events_for_calendar = fake_events_for_calendar
    # _all_calendar_ids must NOT be called when a specific calendar is given.
    def fail_if_called():
        raise AssertionError("_all_calendar_ids should not be called for a specific --calendar value")
    api_module._all_calendar_ids = fail_if_called

    with patch.object(api_module, "print", lambda s: None, create=True):
        api_module.calendar_list(args)

    assert calls == ["primary"]


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
