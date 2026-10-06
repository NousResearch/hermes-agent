"""Persisted steer text has the same display contract on REST and resume."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.prompt_builder import steer_user_row
from hermes_state import SessionDB


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=home / "state.db") as db:
        db.create_session(session_id="steer-display", source="desktop")
        with TestClient(app) as client:
            yield db, client


@pytest.mark.parametrize("inline_images", [True, False])
@pytest.mark.parametrize("around", [False, True])
def test_persisted_steer_projects_inner_text_without_rewriting_storage(store, inline_images, around):
    db, client = store
    text = "Check the tests too — 中文"
    steer = steer_user_row(text)
    db.append_messages_batch("steer-display", [
        {"role": "user", "content": "Investigate"},
        {"role": "assistant", "content": "Before correction"},
        steer,
        {"role": "assistant", "content": "After correction"},
    ])
    before = db.get_messages("steer-display")
    row_id = before[2]["id"]
    route = "/api/sessions/steer-display/messages"
    params = {"inline_images": str(inline_images).lower()}
    if around:
        route += "/around"
        params["row_id"] = row_id
    response = client.get(route, params=params)
    assert response.status_code == 200
    rows = response.json()["messages"]
    shown = next(row for row in rows if row["id"] == row_id)
    assert shown.get("display_content", shown["content"]) == text
    assert shown["content"] == steer["content"]
    assert shown["display_kind"] == "steer"
    expected_page = before[2:] if around else before
    assert [row["id"] for row in rows] == [row["id"] for row in expected_page]
    assert db.get_messages("steer-display") == before


@pytest.mark.parametrize("role,kind,content", [
    ("user", None, steer_user_row("literal example")["content"]),
    ("assistant", "steer", steer_user_row("quoted example")["content"]),
    ("user", "steer", "unwrapped legacy correction"),
    ("user", "steer", ""),
    ("user", "steer", "[OUT-OF-BAND USER MESSAGE — incomplete"),
])
def test_non_steer_and_unextractable_rows_keep_original_content(store, role, kind, content):
    db, client = store
    message = {"role": role, "content": content}
    if kind:
        message["display_kind"] = kind
    db.append_messages_batch("steer-display", [message])
    before = db.get_messages("steer-display")
    response = client.get("/api/sessions/steer-display/messages")
    assert response.status_code == 200
    row = response.json()["messages"][0]
    assert row.get("display_content", row["content"]) == content
    assert db.get_messages("steer-display") == before
