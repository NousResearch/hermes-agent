"""Archived profile pages count archived rows even when live pins back-fill."""

from starlette.testclient import TestClient


def test_archived_profiles_pages_do_not_lose_rows_to_non_archived_pins(_isolate_hermes_home, monkeypatch):
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    db = SessionDB()
    try:
        for sid in ("archive-a", "archive-b"):
            db.create_session(session_id=sid, source="cli")
            db.append_message(session_id=sid, role="user", content="hi")
            db.set_session_archived(sid, True)
        db.create_session(session_id="live-pin", source="cli")
        db.append_message(session_id="live-pin", role="user", content="hi")
        db.set_session_pinned("live-pin", True)
    finally:
        db.close()

    for route in ("/api/profiles/sessions?profile=default", "/api/sessions?profile=default"):
        pages = [client.get(f"{route}&limit=1&offset={offset}&archived=only") for offset in (0, 1)]
        assert all(response.status_code == 200 for response in pages)
        assert [response.json()["total"] for response in pages] == [2, 2]
        assert {row["id"] for response in pages for row in response.json()["sessions"] if row["archived"]} == {"archive-a", "archive-b"}
        # Desktop keeps the first limit rows plus pins. Backfill must follow
        # the archived window or that trim discards an archived row forever.
        assert all(response.json()["sessions"][0]["archived"] for response in pages)
        assert all(any(row["id"] == "live-pin" for row in response.json()["sessions"]) for response in pages)
