"""Dashboard archive and restore use the existing lineage-aware session API."""

import time

from fastapi.testclient import TestClient


def test_dashboard_archive_roundtrip_keeps_compressed_history(
    monkeypatch, _isolate_hermes_home
):
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_state import SessionDB

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
    )
    db = SessionDB()
    try:
        db.create_session("root", source="cli")
        db.append_message("root", role="user", content="archive needle")
        db.create_session("tip", source="cli", parent_session_id="root")
        db.append_message("tip", role="assistant", content="history retained")
        base = time.time() - 100
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' WHERE id = 'root'",
            (base, base + 10),
        )
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = 'tip'", (base + 20,)
        )
        db._conn.commit()
    finally:
        db.close()

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def ids(archived):
        response = client.get("/api/sessions", params={"archived": archived})
        assert response.status_code == 200
        return {row["id"] for row in response.json()["sessions"]}

    assert "tip" in ids("exclude")
    response = client.patch("/api/sessions/tip", json={"archived": True})
    assert response.status_code == 200
    assert response.json()["archived"] is True
    assert "tip" not in ids("exclude")
    assert "tip" in ids("only")
    assert client.get(
        "/api/sessions/search", params={"q": "needle", "archived": "exclude"}
    ).json()["results"] == []
    archived_hits = client.get(
        "/api/sessions/search", params={"q": "needle", "archived": "only"}
    )
    assert archived_hits.status_code == 200
    assert [row["id"] for row in archived_hits.json()["results"]] == ["tip"]

    tip_messages = client.get("/api/sessions/tip/messages")
    assert tip_messages.status_code == 200
    assert [row["content"] for row in tip_messages.json()["messages"]] == ["history retained"]
    db = SessionDB()
    try:
        assert [row["content"] for row in db.get_messages("root")] == ["archive needle"]
    finally:
        db.close()

    response = client.patch("/api/sessions/tip", json={"archived": False})
    assert response.status_code == 200
    assert "tip" in ids("exclude")
    assert "tip" not in ids("only")
    db = SessionDB()
    try:
        assert db.get_session("root")["archived"] == 0
        assert db.get_session("tip")["archived"] == 0
        assert db.message_count() == 2
    finally:
        db.close()
