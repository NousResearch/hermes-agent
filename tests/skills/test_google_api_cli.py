"""Tests for the Google Workspace skill CLI wrapper.

These focus on the hybrid backend: prefer the Google Workspace CLI (`gws`) when
available, while preserving the existing Hermes-facing JSON contract.
"""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("google_workspace_api_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


@pytest.fixture
def google_api_module(tmp_path, monkeypatch):
    module = _load_module()
    token_path = tmp_path / "google_token.json"
    token_path.write_text(
        json.dumps(
            {
                "token": "access-token",
                "refresh_token": "refresh-token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
            }
        )
    )
    monkeypatch.setattr(module, "TOKEN_PATH", token_path)
    monkeypatch.setattr(module, "_gws_binary", lambda: "/usr/bin/gws", raising=False)
    monkeypatch.setattr(module, "get_credentials", lambda: SimpleNamespace(token="access-token"))
    monkeypatch.setattr(
        module,
        "build_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy backend should not be used")),
    )
    return module


def test_gmail_search_uses_gws_and_normalizes_results(google_api_module, monkeypatch, capsys):
    calls = []

    def fake_run(cmd, capture_output, text, env):
        calls.append({"cmd": cmd, "env": env})
        if cmd[1:4] == ["gmail", "users", "messages"] and cmd[4] == "list":
            assert json.loads(cmd[6]) == {"userId": "me", "q": "is:unread", "maxResults": 5}
            return _completed(
                json.dumps({"messages": [{"id": "msg-1", "threadId": "thread-1"}]})
            )
        if cmd[1:4] == ["gmail", "users", "messages"] and cmd[4] == "get":
            params = json.loads(cmd[6])
            assert params["id"] == "msg-1"
            return _completed(
                json.dumps(
                    {
                        "id": "msg-1",
                        "threadId": "thread-1",
                        "payload": {
                            "headers": [
                                {"name": "From", "value": "alice@example.com"},
                                {"name": "To", "value": "bob@example.com"},
                                {"name": "Subject", "value": "Hello"},
                                {"name": "Date", "value": "Sat, 15 Mar 2026 10:00:00 +0000"},
                            ]
                        },
                        "snippet": "preview",
                        "labelIds": ["UNREAD"],
                    }
                )
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(google_api_module.subprocess, "run", fake_run)

    google_api_module.gmail_search(Namespace(query="is:unread", max=5))

    out = json.loads(capsys.readouterr().out)
    assert out == [
        {
            "id": "msg-1",
            "threadId": "thread-1",
            "from": "alice@example.com",
            "to": "bob@example.com",
            "subject": "Hello",
            "date": "Sat, 15 Mar 2026 10:00:00 +0000",
            "snippet": "preview",
            "labels": ["UNREAD"],
        }
    ]
    assert calls[0]["env"]["GOOGLE_WORKSPACE_CLI_TOKEN"] == "access-token"


def test_calendar_create_uses_gws_insert_and_normalizes_result(google_api_module, monkeypatch, capsys):
    def fake_run(cmd, capture_output, text, env):
        assert cmd[:5] == ["/usr/bin/gws", "calendar", "events", "insert", "--params"]
        assert json.loads(cmd[5]) == {"calendarId": "primary"}
        body = json.loads(cmd[7])
        assert body == {
            "summary": "Standup",
            "start": {"dateTime": "2026-03-15T09:00:00Z"},
            "end": {"dateTime": "2026-03-15T09:30:00Z"},
            "location": "Room 1",
            "description": "Daily sync",
            "attendees": [{"email": "alice@example.com"}, {"email": "bob@example.com"}],
        }
        return _completed(json.dumps({"id": "evt-1", "summary": "Standup", "htmlLink": "https://calendar/event"}))

    monkeypatch.setattr(google_api_module.subprocess, "run", fake_run)

    google_api_module.calendar_create(
        Namespace(
            summary="Standup",
            start="2026-03-15T09:00:00Z",
            end="2026-03-15T09:30:00Z",
            location="Room 1",
            description="Daily sync",
            attendees="alice@example.com,bob@example.com",
            calendar="primary",
        )
    )

    assert json.loads(capsys.readouterr().out) == {
        "status": "created",
        "id": "evt-1",
        "summary": "Standup",
        "htmlLink": "https://calendar/event",
    }


def test_sheets_append_uses_gws_and_returns_updated_cells(google_api_module, monkeypatch, capsys):
    def fake_run(cmd, capture_output, text, env):
        assert cmd[:6] == ["/usr/bin/gws", "sheets", "spreadsheets", "values", "append", "--params"]
        assert json.loads(cmd[6]) == {
            "spreadsheetId": "sheet-123",
            "range": "Sheet1!A:C",
            "valueInputOption": "USER_ENTERED",
            "insertDataOption": "INSERT_ROWS",
        }
        assert json.loads(cmd[8]) == {"values": [["a", "b", "c"]]}
        return _completed(json.dumps({"updates": {"updatedCells": 3}}))

    monkeypatch.setattr(google_api_module.subprocess, "run", fake_run)

    google_api_module.sheets_append(
        Namespace(sheet_id="sheet-123", range="Sheet1!A:C", values='[["a", "b", "c"]]')
    )

    assert json.loads(capsys.readouterr().out) == {"updatedCells": 3}


def test_docs_get_uses_gws_and_extracts_plain_text(google_api_module, monkeypatch, capsys):
    def fake_run(cmd, capture_output, text, env):
        assert cmd[:6] == ["/usr/bin/gws", "docs", "documents", "get", "--params", '{"documentId": "doc-123"}']
        return _completed(
            json.dumps(
                {
                    "title": "Doc Title",
                    "documentId": "doc-123",
                    "body": {
                        "content": [
                            {
                                "paragraph": {
                                    "elements": [
                                        {"textRun": {"content": "Hello "}},
                                        {"textRun": {"content": "world"}},
                                    ]
                                }
                            }
                        ]
                    },
                }
            )
        )

    monkeypatch.setattr(google_api_module.subprocess, "run", fake_run)

    google_api_module.docs_get(Namespace(doc_id="doc-123"))

    assert json.loads(capsys.readouterr().out) == {
        "title": "Doc Title",
        "documentId": "doc-123",
        "body": "Hello world",
    }
