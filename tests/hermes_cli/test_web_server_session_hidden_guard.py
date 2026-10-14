"""Tests: the Desktop REST PATCH ``hidden`` guard (#125534).

``sessions.hidden`` is reserved for internal plumbing identities, so the dashboard's
PATCH /api/sessions/{id} must refuse to hide an ordinary conversation (and leave its
row visible), while a plumbing-titled row — the only thing Bot Mode's hide sweep, the
flag's one desktop caller, ever targets — keeps hiding through the same seam.
"""

import pytest

from hermes_state_sessions import HIDDEN_RESERVED_FOR_PLUMBING


@pytest.fixture
def auth_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
    )

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _seed(session_id: str, title: str = "") -> None:
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session(session_id=session_id, source="cli")
        db.append_message(session_id=session_id, role="user", content="hi")
        if title:
            assert db.set_session_title(session_id, title)
    finally:
        db.close()


def test_patch_hidden_refuses_ordinary_chat(auth_client):
    _seed("ordinary-chat")

    resp = auth_client.patch("/api/sessions/ordinary-chat", json={"hidden": True})
    assert resp.status_code == 400
    assert resp.json()["detail"] == HIDDEN_RESERVED_FOR_PLUMBING

    # The refused hide never wrote the flag: the row stays in the default list.
    rows = auth_client.get("/api/sessions?limit=100").json()["sessions"]
    assert bool(next(s for s in rows if s["id"] == "ordinary-chat")["hidden"]) is False

    # Unhiding is never restricted.
    restored = auth_client.patch("/api/sessions/ordinary-chat", json={"hidden": False})
    assert restored.status_code == 200


def test_patch_hidden_allows_plumbing_row(auth_client):
    _seed("bot-chat", title="Bot Chat")

    resp = auth_client.patch("/api/sessions/bot-chat", json={"hidden": True})
    assert resp.status_code == 200
    assert resp.json()["hidden"] is True

    rows = auth_client.get("/api/sessions?limit=100").json()["sessions"]
    assert all(s["id"] != "bot-chat" for s in rows)
