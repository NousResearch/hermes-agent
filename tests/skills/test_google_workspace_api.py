"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import subprocess
import sys
import types
from datetime import date, datetime, timedelta, timezone
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


def test_contacts_birthdays_pages_sorts_projects_and_filters(api_module, monkeypatch, capsys):
    """Birthday reads include the People fields, page through results, and project dates."""
    pages = [
        {
            "connections": [
                {"names": [{"displayName": "Leap"}], "birthdays": [{"date": {"month": 2, "day": 29, "year": 2000}}]},
                {"names": [{"displayName": "No date"}], "birthdays": [{"date": {"year": 1980}}]},
            ],
            "nextPageToken": "page-2",
        },
        {
            "connections": [
                {"names": [{"displayName": "Soon"}], "birthdays": [{"date": {"month": 3, "day": 2}}]},
                {"names": [{"displayName": "Later"}], "birthdays": [{"date": {"month": 3, "day": 31}}]},
            ],
        },
    ]
    calls = []

    def run_gws(parts, *, params=None, body=None):
        calls.append((parts, params))
        return pages.pop(0)

    monkeypatch.setattr(api_module, "_run_gws", run_gws)
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=10, name=""))
    result = json.loads(capsys.readouterr().out)

    assert [entry["name"] for entry in result] == ["Soon", "Later"]
    assert result[0] == {"name": "Soon", "birthday": "02.03.", "nextDate": "2025-03-02", "daysUntil": 1}
    assert calls[0][0] == ["people", "people", "connections", "list"]
    assert calls[0][1]["personFields"] == "names,birthdays"
    assert calls[0][1]["pageSize"] == 1000
    assert calls[1][1]["pageToken"] == "page-2"


def test_contacts_birthdays_name_filter_and_leap_day(api_module, monkeypatch, capsys):
    """Named queries are case-insensitive and Feb 29 falls on Feb 28 in a common year."""
    monkeypatch.setattr(
        api_module,
        "_people_connections_page",
        lambda page_token=None, **kwargs: {
            "connections": [
                {"names": [{"displayName": "Ada Lovelace"}], "birthdays": [{"date": {"month": 2, "day": 29, "year": 1815}}]},
                {"names": [{"displayName": "Grace Hopper"}], "birthdays": [{"date": {"month": 12, "day": 9}}]},
            ],
        },
    )
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 2, 27))

    api_module.contacts_birthdays(types.SimpleNamespace(days=2, max=10, name="ada"))
    result = json.loads(capsys.readouterr().out)

    assert result == [{
        "name": "Ada Lovelace", "birthday": "29.02.1815", "nextDate": "2025-02-28",
        "daysUntil": 1, "turningAge": 210,
    }]


def test_contacts_birthdays_named_query_defaults_to_an_annual_window(api_module, monkeypatch, capsys):
    """A named birthday question is useful even when the date is outside the 30-day feed."""
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))
    monkeypatch.setattr(api_module, "_gws_binary", lambda: "gws")
    monkeypatch.setattr(api_module, "_people_connections_page", lambda *_args, **_kwargs: {
        "connections": [{"resourceName": "people/ada", "names": [{"displayName": "Ada"}],
                         "birthdays": [{"date": {"month": 9, "day": 10}}]}],
    })

    api_module.contacts_birthdays(types.SimpleNamespace(days=None, max=10, name="Ada"))

    assert json.loads(capsys.readouterr().out) == [{
        "name": "Ada", "birthday": "10.09.", "nextDate": "2025-09-10", "daysUntil": 193,
    }]


def test_contacts_birthdays_emits_one_primary_or_first_valid_birthday_per_person(api_module, monkeypatch, capsys):
    """Multiple People birthday entries must not duplicate one contact in the answer."""
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))
    monkeypatch.setattr(api_module, "_gws_binary", lambda: "gws")
    monkeypatch.setattr(api_module, "_people_connections_page", lambda *_args, **_kwargs: {
        "connections": [{"resourceName": "people/ada", "names": [{"displayName": "Ada"}], "birthdays": [
            {"date": {"month": 3, "day": 2}},
            {"metadata": {"primary": True}, "date": {"month": 3, "day": 3}},
        ]}],
    })

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=10, name=""))

    assert json.loads(capsys.readouterr().out) == [{
        "name": "Ada", "birthday": "03.03.", "nextDate": "2025-03-03", "daysUntil": 2,
    }]


def test_contacts_birthdays_does_not_replace_an_out_of_window_primary_with_a_secondary(api_module, monkeypatch, capsys):
    """Primary-source preference is per contact, rather than a way around --days."""
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))
    monkeypatch.setattr(api_module, "_gws_binary", lambda: "gws")
    monkeypatch.setattr(api_module, "_people_connections_page", lambda *_args, **_kwargs: {
        "connections": [{"names": [{"displayName": "Ada"}], "birthdays": [
            {"date": {"month": 3, "day": 2}},
            {"metadata": {"primary": True}, "date": {"month": 9, "day": 10}},
        ]}],
    })

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=10, name=""))

    assert json.loads(capsys.readouterr().out) == []


def test_contacts_birthdays_uses_python_people_client(api_module, monkeypatch, capsys):
    """The fallback backend requests the same birthday fields."""
    calls = []

    class Connections:
        def list(self, **params):
            calls.append(params)
            return MagicMock(execute=lambda: {
                "connections": [{
                    "names": [{"displayName": "Pat"}],
                    "birthdays": [{"date": {"month": 6, "day": 1}}],
                }],
            })

    service = MagicMock()
    service.people.return_value.connections.return_value = Connections()
    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
    monkeypatch.setattr(api_module, "build_service", lambda service_name, version: service)
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 6, 1))

    api_module.contacts_birthdays(types.SimpleNamespace(days=0, max=1, name=""))

    assert json.loads(capsys.readouterr().out)[0]["name"] == "Pat"
    assert calls == [{
        "resourceName": "people/me", "pageSize": 1000, "personFields": "names,birthdays",
    }]


def test_contacts_birthdays_returns_empty_list_for_no_connections(api_module, monkeypatch, capsys):
    monkeypatch.setattr(api_module, "_people_connections_page", lambda page_token=None, **kwargs: {})

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=100, name=""))

    assert json.loads(capsys.readouterr().out) == []


def test_contacts_birthdays_ignores_invalid_age_year(api_module, monkeypatch, capsys):
    monkeypatch.setattr(
        api_module,
        "_people_connections_page",
        lambda page_token=None, **kwargs: {"connections": [
            {"names": [{"displayName": "Ada"}], "birthdays": [{"date": {"month": 3, "day": 2, "year": "unknown"}}]},
        ]},
    )
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=100, name=""))

    assert json.loads(capsys.readouterr().out) == [{
        "name": "Ada", "birthday": "02.03.unknown", "nextDate": "2025-03-02", "daysUntil": 1
    }]


def test_contacts_birthdays_uses_the_first_nonempty_display_name(api_module, monkeypatch, capsys):
    monkeypatch.setattr(
        api_module,
        "_people_connections_page",
        lambda page_token=None, **kwargs: {"connections": [
            {"names": [{"displayName": "  "}, {"displayName": "Ada"}], "birthdays": [{"date": {"month": 3, "day": 2}}]},
        ]},
    )
    monkeypatch.setattr(api_module, "_today", lambda: date(2025, 3, 1))

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=100, name=""))

    assert json.loads(capsys.readouterr().out)[0]["name"] == "Ada"


def test_contacts_birthdays_detects_the_backend_once_for_all_pages(api_module, monkeypatch, capsys):
    calls = []
    pages = [
        {"connections": [], "nextPageToken": "next"},
        {"connections": []},
    ]
    monkeypatch.setattr(api_module, "_gws_binary", lambda: calls.append("detected") or "/usr/bin/gws")
    monkeypatch.setattr(api_module, "_run_gws", lambda *args, **kwargs: pages.pop(0))

    api_module.contacts_birthdays(types.SimpleNamespace(days=30, max=100, name=""))

    assert calls == ["detected"]
    assert json.loads(capsys.readouterr().out) == []


def test_contacts_birthdays_parser_wires_horizon_limit_and_name(api_module, monkeypatch):
    """The CLI exposes explicit upcoming and named-contact birthday options."""
    captured = {}
    monkeypatch.setattr(api_module, "contacts_birthdays", lambda args: captured.update(vars(args)))
    with patch.object(sys, "argv", ["google_api.py", "contacts", "birthdays", "--days", "7", "--max", "3", "--name", "Ada"]):
        api_module.main()

    assert captured["days"] == 7
    assert captured["max"] == 3
    assert captured["name"] == "Ada"


def test_contacts_birthdays_rejects_non_positive_max(api_module):
    with patch.object(sys, "argv", ["google_api.py", "contacts", "birthdays", "--max", "0"]):
        with pytest.raises(SystemExit):
            api_module.main()
