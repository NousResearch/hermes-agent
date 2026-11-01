"""Tests for Google Workspace gws bridge and CLI wrapper."""

import importlib.util
import json
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone, UTC
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
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    token_path = bridge_module.get_token_path()
    _write_token(token_path, token="ya29.valid", expiry=future)

    result = bridge_module.get_valid_token()
    assert result == "ya29.valid"










def test_bridge_main_injects_token_env(bridge_module, tmp_path):
    """main() sets GOOGLE_WORKSPACE_CLI_TOKEN in subprocess env."""
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
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


def test_gmail_search_empty_result_prints_json_array(
    api_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Python-fallback empty search prints [] like the gws path (#131711)."""
    service = MagicMock()
    service.users().messages().list().execute.return_value = {"resultSizeEstimate": 0}
    monkeypatch.setattr(api_module, "_gws_binary", lambda: None)
    monkeypatch.setattr(api_module, "build_service", lambda *_args: service)

    api_module.gmail_search(api_module.argparse.Namespace(query="is:unread", max=10))

    assert json.loads(capsys.readouterr().out) == []


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


# =========================================================================
# Slides / Docs replace
# =========================================================================


def _deck():
    """A two-slide deck as the Slides API returns it (shape text + speaker notes)."""
    def shape(text):
        return {"shape": {"text": {"textElements": [{"textRun": {"content": text}}]}}}

    return {
        "presentationId": "deck1",
        "title": "QA deck",
        "slides": [
            {
                "objectId": "s1",
                "pageElements": [shape("Alpha\n"), shape("01\n")],
                "notesPage": {"pageElements": [shape("speaker note\n")]},
            },
            {"objectId": "s2", "pageElements": [shape("Beta\n")]},
        ],
    }


def _mock_service(api_module, monkeypatch, **returns):
    """Point build_service() at a mock and seed chained ``.execute()`` results.

    A keyword names the attribute chain with underscores: ``presentations_get``
    seeds ``service.presentations().get().execute()``,
    ``spreadsheets_values_clear`` seeds ``service.spreadsheets().values().clear()``.
    """
    service = MagicMock()
    for path, value in returns.items():
        target = service
        for attr in path.split("_"):
            target = getattr(target, attr)()
        target.execute.return_value = value
    monkeypatch.setattr(api_module, "build_service", lambda *_args: service)
    return service


def test_slides_get_reports_index_object_id_texts_and_notes(api_module, monkeypatch, capsys):
    _mock_service(api_module, monkeypatch, presentations_get=_deck())

    api_module.slides_get(types.SimpleNamespace(presentation_id="deck1", slide=""))

    out = json.loads(capsys.readouterr().out)
    assert out["title"] == "QA deck"
    assert out["slideCount"] == 2
    assert out["slides"][0] == {
        "index": 1,
        "objectId": "s1",
        "texts": ["Alpha", "01"],
        "notes": ["speaker note"],
    }
    # A slide without notes must not grow an empty "notes" key.
    assert out["slides"][1] == {"index": 2, "objectId": "s2", "texts": ["Beta"]}


def test_slides_get_slide_filter_accepts_index_or_object_id(api_module, monkeypatch, capsys):
    """--slide takes a 1-based index or an objectId, so a value read from
    `slides get` can be fed straight back into the write paths."""
    _mock_service(api_module, monkeypatch, presentations_get=_deck())

    for spec, expected in (("2", ["s2"]), ("s2", ["s2"]), ("1,s2", ["s1", "s2"])):
        api_module.slides_get(types.SimpleNamespace(presentation_id="deck1", slide=spec))
        out = json.loads(capsys.readouterr().out)
        assert [s["objectId"] for s in out["slides"]] == expected
        # slideCount stays the deck total, so a filtered read is still anchored.
        assert out["slideCount"] == 2


def test_slides_get_rejects_an_unknown_slide(api_module, monkeypatch):
    _mock_service(api_module, monkeypatch, presentations_get=_deck())

    with pytest.raises(SystemExit):
        api_module.slides_get(types.SimpleNamespace(presentation_id="deck1", slide="9"))


def test_slides_replace_scopes_page_object_ids_and_counts_occurrences(
    api_module, monkeypatch, capsys
):
    service = _mock_service(
        api_module,
        monkeypatch,
        presentations_get=_deck(),
        presentations_batchUpdate={"replies": [{"replaceAllText": {"occurrencesChanged": 2}}]},
    )

    api_module.slides_replace(types.SimpleNamespace(
        presentation_id="deck1", find="old", replace="new", slide="2", ignore_case=False,
    ))

    request = service.presentations().batchUpdate.call_args.kwargs["body"]["requests"][0]
    assert request["replaceAllText"]["containsText"] == {"text": "old", "matchCase": True}
    assert request["replaceAllText"]["replaceText"] == "new"
    assert request["replaceAllText"]["pageObjectIds"] == ["s2"]
    assert json.loads(capsys.readouterr().out)["occurrencesChanged"] == 2


def test_slides_replace_without_slide_filter_covers_the_whole_deck(
    api_module, monkeypatch, capsys
):
    service = _mock_service(
        api_module,
        monkeypatch,
        presentations_batchUpdate={"replies": [{"replaceAllText": {"occurrencesChanged": 1}}]},
    )

    api_module.slides_replace(types.SimpleNamespace(
        presentation_id="deck1", find="old", replace="new", slide="", ignore_case=True,
    ))

    request = service.presentations().batchUpdate.call_args.kwargs["body"]["requests"][0]
    # No pageObjectIds key at all — an empty list would replace nothing.
    assert "pageObjectIds" not in request["replaceAllText"]
    assert request["replaceAllText"]["containsText"]["matchCase"] is False
    assert json.loads(capsys.readouterr().out)["occurrencesChanged"] == 1


def test_slides_thumbnail_uses_flat_thumbnail_size_and_saves_png(
    api_module, monkeypatch, tmp_path, capsys
):
    thumb = {"contentUrl": "https://example.invalid/thumb.png", "width": 1600, "height": 900}
    service = _mock_service(
        api_module,
        monkeypatch,
        presentations_get=_deck(),
        presentations_pages_getThumbnail=thumb,
    )
    downloads = []
    monkeypatch.setattr(
        api_module.urllib.request, "urlretrieve",
        lambda url, path: downloads.append((url, str(path))),
    )

    out_file = tmp_path / "shots" / "slide1.png"
    api_module.slides_thumbnail(types.SimpleNamespace(
        presentation_id="deck1", slide="1", size="LARGE", out=str(out_file),
    ))

    # The python client wants the nested query param in its flat form.
    assert service.presentations().pages().getThumbnail.call_args.kwargs == {
        "presentationId": "deck1",
        "pageObjectId": "s1",
        "thumbnailProperties_thumbnailSize": "LARGE",
    }
    assert downloads == [("https://example.invalid/thumb.png", str(out_file))]
    assert out_file.parent.exists()
    assert json.loads(capsys.readouterr().out)["savedTo"] == str(out_file)


def test_docs_replace_sends_tab_criteria_and_counts_occurrences(api_module, monkeypatch, capsys):
    service = _mock_service(
        api_module, monkeypatch,
        documents_batchUpdate={"replies": [{"replaceAllText": {"occurrencesChanged": 3}}]},
    )

    api_module.docs_replace(types.SimpleNamespace(
        doc_id="doc1", find="old", replace="new", tab="t.1", ignore_case=False,
    ))

    request = service.documents().batchUpdate.call_args.kwargs["body"]["requests"][0]
    assert request["replaceAllText"]["tabsCriteria"] == {"tabIds": ["t.1"]}
    out = json.loads(capsys.readouterr().out)
    assert out["occurrencesChanged"] == 3
    assert out["tab"] == "t.1"


def test_docs_replace_without_tab_omits_tab_criteria(api_module, monkeypatch, capsys):
    service = _mock_service(
        api_module, monkeypatch,
        documents_batchUpdate={"replies": [{"replaceAllText": {"occurrencesChanged": 0}}]},
    )

    api_module.docs_replace(types.SimpleNamespace(
        doc_id="doc1", find="absent", replace="new", tab=None, ignore_case=False,
    ))

    request = service.documents().batchUpdate.call_args.kwargs["body"]["requests"][0]
    assert "tabsCriteria" not in request["replaceAllText"]
    # A find string that matches nothing reports 0 rather than raising.
    assert json.loads(capsys.readouterr().out)["occurrencesChanged"] == 0
