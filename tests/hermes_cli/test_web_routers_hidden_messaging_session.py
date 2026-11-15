"""A hidden messaging-surface session must not be readable by id.

A Telegram/A2A session can be ``hidden`` (archived) while its transcript stays
in the DB. The sidebar filters such rows, but the detail and transcript routes
served them anyway: a stale tab, a bookmark or a deep link opened the channel
chat, and a turn taken there kept that session's route. Reading was the same
door as writing.

The guard is scoped to messaging sources on purpose: ``hidden`` is also set on
rows the UI legitimately opens (bot chat, subagent chains), and those must keep
answering 200.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

MESSAGING_HIDDEN = "sess-telegram-hidden"
MESSAGING_VISIBLE = "sess-telegram-visible"
LOCAL_HIDDEN = "sess-local-hidden"
LOCAL_VISIBLE = "sess-local-visible"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB writes neither the real store nor a shared one."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture()
def db_path(home):
    from hermes_state import SessionDB

    path = home / "state.db"
    SessionDB(path).close()  # create the schema
    con = sqlite3.connect(path)
    rows = [
        (MESSAGING_HIDDEN, "telegram", 1),
        (MESSAGING_VISIBLE, "telegram", 0),
        (LOCAL_HIDDEN, "local", 1),
        (LOCAL_VISIBLE, "local", 0),
    ]
    con.executemany(
        "INSERT INTO sessions (id, source, started_at, hidden, archived, message_count) "
        "VALUES (?, ?, 1, ?, ?, 0)",
        [(sid, source, hidden, hidden) for sid, source, hidden in rows],
    )
    con.commit()
    con.close()
    return path


@pytest.fixture()
def client(db_path, monkeypatch):
    from hermes_cli.web_routers import sessions

    monkeypatch.setattr(
        sessions, "_open_session_db_for_profile",
        lambda profile, read_only: __import__("hermes_state").SessionDB(db_path),
    )
    monkeypatch.setattr(sessions, "_serving_profile", lambda profile: "default")
    app = FastAPI()
    app.include_router(sessions.manage_router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("session_id", [MESSAGING_HIDDEN])
def test_hidden_messaging_session_detail_is_404(client, session_id) -> None:
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


@pytest.mark.parametrize("session_id", [MESSAGING_HIDDEN])
def test_hidden_messaging_session_transcript_is_404(client, session_id) -> None:
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


@pytest.mark.parametrize("session_id", [LOCAL_VISIBLE, MESSAGING_VISIBLE])
def test_visible_sessions_still_answer(client, session_id) -> None:
    assert client.get(f"/api/sessions/{session_id}").status_code == 200
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 200


@pytest.mark.parametrize("session_id", [LOCAL_HIDDEN])
def test_hidden_non_messaging_session_keeps_working(client, session_id) -> None:
    """``hidden`` alone is not the trigger — bot chat and subagent rows carry it too."""
    assert client.get(f"/api/sessions/{session_id}").status_code == 200


def test_unknown_session_is_still_404(client) -> None:
    assert client.get("/api/sessions/does-not-exist").status_code == 404
