"""Browser turn history uses a display-only projection without changing model input."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def browser_rows():
    envelope = json.dumps({
        "protocol": "hermes.browser.turn.v2",
        "human_input": {"source": "composer", "text": "Hello from the browser"},
        "browser_context": {"delivery": "full", "payload": {"text": "page context"}},
    })
    lookalike = json.dumps({"protocol": "hermes.browser.turn.v2", "human_input": {"text": 42}})
    return [
        {"role": "user", "content": envelope},
        {"role": "user", "content": "ordinary text"},
        {"role": "user", "content": lookalike},
        {"role": "assistant", "content": envelope},
    ]


@pytest.fixture
def client(tmp_path, monkeypatch, browser_rows):
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id="browser-chat", source="browser")
        db.append_messages_batch("browser-chat", browser_rows)
    finally:
        db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as test_client:
        yield test_client, home / "state.db"


def test_browser_turn_is_projected_without_changing_stored_or_response_content(client, browser_rows):
    from hermes_state import SessionDB

    test_client, db_path = client
    response = test_client.get("/api/sessions/browser-chat/messages?limit=10&order=oldest")
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert messages[0]["display_content"] == "Hello from the browser"
    assert [row["content"] for row in messages] == [row["content"] for row in browser_rows]
    assert all("display_content" not in row for row in messages[1:])
    db = SessionDB(db_path)
    try:
        assert [row["content"] for row in db.get_messages("browser-chat")] == [
            row["content"] for row in browser_rows
        ]
    finally:
        db.close()
