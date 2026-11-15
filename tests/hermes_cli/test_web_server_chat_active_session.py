"""``GET /api/chat/active-session``: which session the dashboard's ``/chat`` PTY is on.

A fresh dashboard chat mints its session inside the PTY, so the SPA never sees the id; the TUI
writes it to the per-channel marker file and this read-only route exposes it. Actions that need the
transcript without typing into the composer ("copy last response", #149751) ride on it.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous


@pytest.fixture
def bind_marker(tmp_path):
    """Write the marker a dashboard TUI would, and drop the binding afterwards."""
    files = web_server._get_pty_active_session_files(web_server.app)
    bound: list[str] = []

    def _bind(channel: str, payload: dict) -> None:
        path = tmp_path / f"{channel}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        files[channel] = path
        bound.append(channel)

    try:
        yield _bind, files
    finally:
        for channel in bound:
            files.pop(channel, None)


def test_reports_the_session_the_chat_is_on(client, bind_marker):
    bind, _files = bind_marker
    bind("chan-a", {"session_id": "sess-1"})

    assert client.get("/api/chat/active-session", params={"channel": "chan-a"}).json() == {
        "session_id": "sess-1"}


def test_a_chat_without_a_turn_yet_answers_null(client, bind_marker):
    """The marker exists from PTY start; the TUI writes the id on the first session."""
    bind, _files = bind_marker
    bind("chan-b", {})

    assert client.get("/api/chat/active-session", params={"channel": "chan-b"}).json() == {
        "session_id": None}


def test_an_unknown_channel_answers_null_and_creates_nothing(client, bind_marker):
    """Read-only: the PTY websocket owns the marker's lifetime, a GET must not mint one."""
    _bind, files = bind_marker

    response = client.get("/api/chat/active-session", params={"channel": "chan-unknown"})

    assert response.status_code == 200
    assert response.json() == {"session_id": None}
    assert "chan-unknown" not in files


def test_rejects_an_invalid_channel(client):
    response = client.get("/api/chat/active-session", params={"channel": "../etc/passwd"})

    assert response.status_code == 400


def test_a_vanished_marker_answers_null(client, bind_marker):
    """The dashboard reaps chat markers on shutdown; a stale binding must not 500."""
    bind, _files = bind_marker
    bind("chan-c", {"session_id": "sess-gone"})
    _files["chan-c"].unlink()

    assert client.get("/api/chat/active-session", params={"channel": "chan-c"}).json() == {
        "session_id": None}
