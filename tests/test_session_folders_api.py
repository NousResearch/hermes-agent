"""Integration tests for session folder REST API endpoints.

Uses FastAPI TestClient against the real web_server app with a temp SQLite DB.
The `_open_session_db_for_profile` helper is monkeypatched to return a
test SessionDB pointing at a temp file.
"""

from __future__ import annotations

import json
import os
import sys
import threading

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def tmp_db_path(tmp_path):
    """Path for a temporary state.db."""
    return str(tmp_path / "test_state.db")


@pytest.fixture
def test_app(tmp_db_path):
    """Return a TestClient bound to the web_server app with a temp DB."""
    import os
    os.environ["HERMES_DASHBOARD_SESSION_TOKEN"] = "test-session-token-for-testing"
    from fastapi.testclient import TestClient
    from hermes_state import SessionDB
    from pathlib import Path
    from hermes_cli import web_server as ws

    import importlib
    importlib.reload(ws)

    db_path = Path(tmp_db_path)
    db = SessionDB(db_path=db_path)
    original_opener = ws._open_session_db_for_profile

    def test_opener(profile=None):
        return SessionDB(db_path=db_path)

    ws._open_session_db_for_profile = test_opener
    client = TestClient(ws.app)
    client.headers["X-Hermes-Session-Token"] = "test-session-token-for-testing"

    yield client, db

    ws._open_session_db_for_profile = original_opener
    db.close()


@pytest.fixture
def db_with_sessions(test_app):
    """Return a test app + client + a SessionDB with 3 sessions and a folder."""
    client, db = test_app
    for i in range(3):
        sid = f"s{i}"
        db.create_session(session_id=sid, source="cli")
        db.append_message(sid, role="user", content=f"Hello {i}")
    folder = db.create_folder(name="Test Folder")
    return client, db, folder


# =====================================================================
# Folder CRUD
# =====================================================================


class TestCreateFolder:
    def test_create(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={"name": "Bug Reports"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Bug Reports"
        assert data["session_count"] == 0
        assert data["id"].startswith("sf_")

    def test_empty_name_returns_400(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={"name": ""})
        assert resp.status_code == 400

    def test_missing_name_returns_422(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={})
        assert resp.status_code == 422


class TestListFolders:
    def test_list_empty(self, test_app):
        client, db = test_app
        resp = client.get("/api/session-folders")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_folders(self, test_app):
        client, db = test_app
        client.post("/api/session-folders", json={"name": "A"})
        client.post("/api/session-folders", json={"name": "B"})
        resp = client.get("/api/session-folders")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = [f["name"] for f in data]
        assert "A" in names
        assert "B" in names

    def test_session_count_reflects_members(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0", "s1"])
        resp = client.get("/api/session-folders")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        target = [f for f in data if f["id"] == folder["id"]][0]
        assert target["session_count"] == 2


class TestRenameFolder:
    def test_rename(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={"name": "Old"})
        assert resp.status_code == 200
        create = resp.json()
        resp = client.patch(f"/api/session-folders/{create['id']}", json={"name": "New"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        folders = client.get("/api/session-folders").json()
        renamed = [f for f in folders if f["id"] == create["id"]][0]
        assert renamed["name"] == "New"

    def test_rename_nonexistent(self, test_app):
        client, db = test_app
        resp = client.patch("/api/session-folders/nonexistent", json={"name": "X"})
        assert resp.status_code == 404

    def test_rename_empty_name(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={"name": "X"})
        assert resp.status_code == 200
        create = resp.json()
        resp = client.patch(f"/api/session-folders/{create['id']}", json={"name": ""})
        assert resp.status_code == 400


class TestDeleteFolder:
    def test_delete(self, test_app):
        client, db = test_app
        resp = client.post("/api/session-folders", json={"name": "To Delete"})
        assert resp.status_code == 200
        create = resp.json()
        resp = client.delete(f"/api/session-folders/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        folders = client.get("/api/session-folders").json()
        assert len(folders) == 0

    def test_delete_nonexistent(self, test_app):
        client, db = test_app
        resp = client.delete("/api/session-folders/nonexistent")
        assert resp.status_code == 404

    def test_delete_does_not_delete_sessions(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0"])
        resp = client.delete(f"/api/session-folders/{folder['id']}")
        assert resp.status_code == 200
        assert db.get_session("s0") is not None


# =====================================================================
# Session membership
# =====================================================================


class TestAddSessions:
    def test_add(self, db_with_sessions):
        client, db, folder = db_with_sessions
        resp = client.post(
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["s0", "s1"]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_add_idempotent(self, db_with_sessions):
        client, db, folder = db_with_sessions
        client.post(
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["s0"]},
        )
        resp = client.post(
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["s0"]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_add_nonexistent_folder(self, test_app):
        client, db = test_app
        resp = client.post(
            "/api/session-folders/bad/sessions",
            json={"session_ids": ["s1"]},
        )
        assert resp.status_code == 500

    def test_add_with_profile(self, db_with_sessions):
        client, db, folder = db_with_sessions
        resp = client.post(
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["s0"], "profile": "default"},
        )
        assert resp.status_code == 200


class TestRemoveSessions:
    def test_remove(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0", "s1"])
        resp = client.request("DELETE",
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["s0"]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_remove_nonexistent(self, db_with_sessions):
        client, db, folder = db_with_sessions
        resp = client.request("DELETE",
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": ["ghost"]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_remove_empty_list(self, db_with_sessions):
        client, db, folder = db_with_sessions
        resp = client.request("DELETE",
            f"/api/session-folders/{folder['id']}/sessions",
            json={"session_ids": []},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# =====================================================================
# Folder map
# =====================================================================


class TestFolderMap:
    def test_get_map(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0", "s1"])
        resp = client.get("/api/session-folders/map", params={"session_ids": "s0,s1,s2"})
        assert resp.status_code == 200
        data = resp.json()
        assert folder["id"] in data.get("s0", [])
        assert folder["id"] in data.get("s1", [])
        assert "s2" not in data or data["s2"] == []

    def test_get_map_empty(self, test_app):
        client, db = test_app
        resp = client.get("/api/session-folders/map")
        assert resp.status_code == 200
        assert resp.json() == {}


# =====================================================================
# Folder rename endpoint tests (via API, not just tool handlers)
# =====================================================================


class TestFolderRenameEndpoint:
    def test_rename_via_api(self, test_app):
        client, db = test_app
        create = client.post("/api/session-folders", json={"name": "Original"}).json()
        resp = client.patch(f"/api/session-folders/{create['id']}", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        folders = client.get("/api/session-folders").json()
        assert any(f["name"] == "Renamed" for f in folders)

    def test_rename_nonexistent_via_api(self, test_app):
        client, db = test_app
        resp = client.patch("/api/session-folders/bad", json={"name": "X"})
        assert resp.status_code == 404

    def test_rename_empty_name_via_api(self, test_app):
        client, db = test_app
        create = client.post("/api/session-folders", json={"name": "X"}).json()
        resp = client.patch(f"/api/session-folders/{create['id']}", json={"name": ""})
        assert resp.status_code == 400


# =====================================================================
# Folder delete endpoint tests
# =====================================================================


class TestFolderDeleteEndpoint:
    def test_delete_via_api(self, test_app):
        client, db = test_app
        create = client.post("/api/session-folders", json={"name": "Temp"}).json()
        resp = client.delete(f"/api/session-folders/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        folders = client.get("/api/session-folders").json()
        assert len(folders) == 0

    def test_delete_nonexistent_via_api(self, test_app):
        client, db = test_app
        resp = client.delete("/api/session-folders/nonexistent")
        assert resp.status_code == 404

    def test_delete_preserves_sessions_via_api(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0"])
        resp = client.delete(f"/api/session-folders/{folder['id']}")
        assert resp.status_code == 200
        assert db.get_session("s0") is not None  # session survived


# =====================================================================
# Folder listing field tests
# =====================================================================


class TestFolderListFields:
    def test_list_includes_session_ids(self, db_with_sessions):
        client, db, folder = db_with_sessions
        db.add_sessions_to_folder(folder["id"], ["s0", "s1"])
        resp = client.get("/api/session-folders")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        target = [f for f in data if f["id"] == folder["id"]][0]
        assert "session_ids" in target
        assert isinstance(target["session_ids"], list)
        assert "s0" in target["session_ids"]
        assert "s1" in target["session_ids"]

    def test_list_session_ids_empty_for_empty_folder(self, test_app):
        client, db = test_app
        create = client.post("/api/session-folders", json={"name": "Empty"}).json()
        resp = client.get("/api/session-folders")
        data = resp.json()
        target = [f for f in data if f["id"] == create["id"]][0]
        assert target["session_ids"] == []
