"""``GET /api/sessions/{id}/messages`` honors ``inline_images=false`` (#116511).

The REST pages carry ``content`` verbatim (a parts list with inline ``data:image`` URIs) so
Desktop's ``extractEmbeddedImages`` can pull them out of the text. A client that reads over
a network had no way to ask for less: the measured conversation was 26.33 MiB per page read.
``inline_images=false`` (default ``true``) routes the content through the same
``render_message_content(image_urls=False)`` projection ``session.resume`` uses, rendering
``[image]`` in place of the data URI.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

DATA_URI = "data:image/png;base64," + "a" * 128
IMAGE_CONTENT = [
    {"type": "text", "text": "what is this?"},
    {"type": "image_url", "image_url": {"url": DATA_URI}},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")

    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id="img-chat", source="desktop")
        db.append_messages_batch("img-chat", [
            {"role": "user", "content": IMAGE_CONTENT},
            {"role": "assistant", "content": "a chart"},
        ])
    finally:
        db.close()

    from hermes_cli.web_routers.sessions import manage_router

    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as test_client:
        yield test_client


def test_messages_default_inlines_images(client):
    page = client.get("/api/sessions/img-chat/messages?limit=10&order=oldest").json()
    user_row = next(m for m in page["messages"] if m["role"] == "user")
    assert DATA_URI in str(user_row["content"])


def test_messages_inline_images_false_renders_placeholder(client):
    page = client.get(
        "/api/sessions/img-chat/messages?limit=10&order=oldest&inline_images=false").json()
    user_row = next(m for m in page["messages"] if m["role"] == "user")
    assert "[image]" in user_row["content"]
    assert DATA_URI not in user_row["content"]
    # Non-image rows are untouched.
    assistant_row = next(m for m in page["messages"] if m["role"] == "assistant")
    assert assistant_row["content"] == "a chart"
