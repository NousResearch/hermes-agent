"""Desktop HTTP tags use persistent profile stores and exact conversation ids."""
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    other = home / "profiles" / "work"
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    for path in (home, other):
        with SessionDB(path / "state.db") as db:
            db.create_session("root", source="desktop")
            db.append_message("root", "user", "hello")
            db.end_session("root", "compression")
            db.create_session("tip", source="desktop", parent_session_id="root")
            db.append_message("tip", "user", "continued")
    from hermes_cli.web_routers.sessions import list_router, manage_router, search_router
    from hermes_cli.web_routers.profiles import sessions_router
    app = FastAPI()
    app.include_router(list_router)
    app.include_router(sessions_router)
    app.include_router(search_router)
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield client


def test_http_catalogue_lineage_and_profile_rows(client):
    for profile, tag in [("default", "A"), ("work", "B"), ("default", "A2")]:
        response = client.put(f"/api/sessions/tip/tags?profile={profile}", json={"tag": tag, "assigned": True})
        assert response.status_code == 200, response.text
    for profile, tags in [("default", ["A", "A2"]), ("work", ["B"]), ("default", ["A", "A2"])]:
        assert client.get(f"/api/sessions/tags?profile={profile}").json() == {"tags": tags}
        rows = client.get(f"/api/sessions?profile={profile}").json()["sessions"]
        assert rows[0]["tags"] == tags
        rows = client.get(f"/api/profiles/sessions?profile={profile}").json()["sessions"]
        assert rows[0]["tags"] == tags
        assert client.get(f"/api/sessions/tip?profile={profile}").json()["tags"] == tags
    response = client.put("/api/sessions/root/tags", json={"tag": "A", "assigned": False})
    assert response.json() == {"tags": ["A2"]}
    assert client.get("/api/sessions/tags").json() == {"tags": ["A", "A2"]}


def test_http_search_preserves_tags_in_id_and_content_results(client, monkeypatch):
    client.put("/api/sessions/tip/tags", json={"tag": "Searchable", "assigned": True})
    batches = []
    original = SessionDB.get_session_tags_batch
    def batch(self, ids):
        batches.append(ids)
        return original(self, ids)
    monkeypatch.setattr(SessionDB, "get_session_tags_batch", batch)
    for query in ("tip", "hello"):
        batches.clear()
        results = client.get("/api/sessions/search", params={"q": query}).json()["results"]
        assert results[0]["session_id"] == "tip"
        assert results[0]["tags"] == ["Searchable"]
        # ID search hydrates its candidate list once; final projection batches
        # all deduplicated hits once, never one extra lookup per result.
        assert batches[-1] == ["tip"]
        assert len(batches) <= 2


def test_http_search_tag_hydration_is_batched_for_many_results(client, monkeypatch):
    with SessionDB(Path.home() / ".hermes" / "state.db") as db:
        for i in range(30):
            sid = f"batch-{i}"
            db.create_session(sid, source="desktop")
            db.append_message(sid, "user", "batchneedle")
            db.set_session_tag(sid, "Batch", True)
    batches = []
    original = SessionDB.get_session_tags_batch
    def batch(self, ids):
        batches.append(list(ids))
        return original(self, ids)
    monkeypatch.setattr(SessionDB, "get_session_tags_batch", batch)
    rows = client.get("/api/sessions/search", params={"q": "batchneedle", "limit": 100}).json()["results"]
    assert len(rows) == 30
    assert all(row["tags"] == ["Batch"] for row in rows)
    assert len(batches) <= 2
    assert set(batches[-1]) == {row["session_id"] for row in rows}


def test_http_rejects_invalid_input_without_catalogue_mutation(client):
    for body in [{"tag": tag, "assigned": True} for tag in [None, 1, "", "\ttag", "x" * 65]] + [
        {"tag": "Valid", "assigned": value} for value in [1, "false", None]
    ] + [{"tag": "Valid", "assigned": True, "unknown": 1}]:
        assert client.put("/api/sessions/root/tags", json=body).status_code == 422
    assert client.put("/api/sessions/roo/tags", json={"tag": "Ghost", "assigned": True}).status_code == 404
    assert client.get("/api/sessions/tags?profile=absent").status_code == 404
    assert client.get("/api/sessions/tags").json() == {"tags": []}
