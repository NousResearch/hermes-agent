"""Tests for Phase 3 Hermes Code Mode: CodeSessionService."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# =============================================================================
# Helpers
# =============================================================================


def _make_workspace(db, tmp_path, name="myproject"):
    project = tmp_path / name
    project.mkdir(exist_ok=True)
    return db.upsert_workspace(
        path=str(project),
        name=name,
        is_git_repo=True,
        branch="main",
        detected_stack=["python"],
    )


# =============================================================================
# CodeSessionDB — persistence layer
# =============================================================================


class TestCodeSessionDB:
    @pytest.fixture()
    def db(self, tmp_path):
        from hermes_state import CodeSessionDB

        d = CodeSessionDB(db_path=tmp_path / "state.db")
        yield d
        d.close()

    @pytest.fixture()
    def wdb(self, tmp_path):
        from hermes_state import WorkspaceDB

        d = WorkspaceDB(db_path=tmp_path / "state.db")
        yield d
        d.close()

    # Re-use same db file
    @pytest.fixture()
    def db_and_wdb(self, tmp_path):
        from hermes_state import CodeSessionDB, WorkspaceDB

        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        db = CodeSessionDB(db_path=db_path)
        yield db, wdb
        db.close()
        wdb.close()

    def test_create_session_planning_status(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        session = db.create_session(workspace_id=ws["id"])
        assert session["status"] == "planning"
        assert session["workspace_id"] == ws["id"]
        assert session["id"]

    def test_create_session_copies_branch(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        session = db.create_session(workspace_id=ws["id"], branch="feature/x")
        assert session["branch"] == "feature/x"

    def test_create_session_stores_provider_model(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        session = db.create_session(
            workspace_id=ws["id"],
            provider="openai",
            model="gpt-4o",
        )
        assert session["provider"] == "openai"
        assert session["model"] == "gpt-4o"

    def test_create_session_stores_metadata(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        session = db.create_session(
            workspace_id=ws["id"],
            metadata={"source": "code_cockpit_chat", "execution_mode": "approval"},
        )
        assert session["metadata"] == {
            "source": "code_cockpit_chat",
            "execution_mode": "approval",
        }

    def test_list_sessions_returns_created(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        db.create_session(workspace_id=ws["id"], title="s1")
        db.create_session(workspace_id=ws["id"], title="s2")
        sessions = db.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_by_workspace(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws1 = _make_workspace(wdb, tmp_path, name="proj1")
        ws2 = _make_workspace(wdb, tmp_path, name="proj2")
        db.create_session(workspace_id=ws1["id"])
        db.create_session(workspace_id=ws2["id"])
        assert len(db.list_sessions(workspace_id=ws1["id"])) == 1
        assert len(db.list_sessions(workspace_id=ws2["id"])) == 1

    def test_list_sessions_filter_by_status(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        db.update_session(s["id"], {"status": "coding"})
        assert len(db.list_sessions(status="planning")) == 0
        assert len(db.list_sessions(status="coding")) == 1

    def test_get_session_returns_none_for_unknown(self, db):
        assert db.get_session("nonexistent") is None

    def test_update_session_changes_fields(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        updated = db.update_session(s["id"], {"status": "coding", "summary": "working"})
        assert updated["status"] == "coding"
        assert updated["summary"] == "working"
        assert updated["updated_at"] >= s["updated_at"]

    def test_update_session_accepts_web_cockpit_statuses(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])

        running = db.update_session(s["id"], {"status": "running"})
        completed = db.update_session(s["id"], {"status": "completed"})

        assert running["status"] == "running"
        assert completed["status"] == "completed"

    def test_update_session_invalid_status_raises(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        with pytest.raises(ValueError, match="Invalid status"):
            db.update_session(s["id"], {"status": "flying"})

    def test_update_session_metadata_serialized(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        updated = db.update_session(s["id"], {"metadata": {"key": "val"}})
        assert updated["metadata"]["key"] == "val"

    def test_cancel_sets_cancelled_status(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        db.update_session(
            s["id"],
            {"status": "cancelled", "completed_at": "2026-04-24T00:00:00+00:00"},
        )
        updated = db.get_session(s["id"])
        assert updated["status"] == "cancelled"
        assert updated["completed_at"] is not None

    def test_complete_sets_done_status(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        db.update_session(
            s["id"],
            {
                "status": "done",
                "completed_at": "2026-04-24T00:00:00+00:00",
                "summary": "all done",
            },
        )
        updated = db.get_session(s["id"])
        assert updated["status"] == "done"
        assert updated["summary"] == "all done"
        assert updated["completed_at"] is not None


# =============================================================================
# Events
# =============================================================================


class TestCodeSessionEvents:
    @pytest.fixture()
    def db_and_wdb(self, tmp_path):
        from hermes_state import CodeSessionDB, WorkspaceDB

        db_path = tmp_path / "state.db"
        wdb = WorkspaceDB(db_path=db_path)
        db = CodeSessionDB(db_path=db_path)
        yield db, wdb
        db.close()
        wdb.close()

    def test_add_event_persisted(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        event = db.add_event(s["id"], "note.added", message="hello", payload={"x": 1})
        assert event["id"]
        assert event["type"] == "note.added"
        assert event["message"] == "hello"
        assert event["payload"]["x"] == 1

    def test_list_events_empty_initially(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        assert db.list_events(s["id"]) == []

    def test_list_events_ordered_by_created_at(self, db_and_wdb, tmp_path):
        db, wdb = db_and_wdb
        ws = _make_workspace(wdb, tmp_path)
        s = db.create_session(workspace_id=ws["id"])
        db.add_event(s["id"], "first")
        db.add_event(s["id"], "second")
        events = db.list_events(s["id"])
        assert events[0]["type"] == "first"
        assert events[1]["type"] == "second"


# =============================================================================
# Artifacts association
# =============================================================================


class TestArtifactAssociation:
    @pytest.fixture()
    def dbs(self, tmp_path):
        from hermes_state import CodeSessionDB, WorkspaceDB, SessionDB

        db_path = tmp_path / "state.db"
        sdb = SessionDB(db_path=db_path)
        wdb = WorkspaceDB(db_path=db_path)
        cdb = CodeSessionDB(db_path=db_path)
        yield sdb, wdb, cdb
        sdb.close()
        wdb.close()
        cdb.close()

    def test_list_artifacts_empty_when_none(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        cs = cdb.create_session(workspace_id=ws["id"])
        assert cdb.list_artifacts_for_code_session(cs["id"]) == []

    def test_list_artifacts_fallback_by_hermes_session_id(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)

        # Create Hermes session + artifact
        sdb.create_session("hsess1", "cli")
        sdb.create_artifact(
            session_id="hsess1",
            tool_name="patch",
            path="foo.py",
            status="modified",
        )

        # Create CodeSession linked to Hermes session
        cs = cdb.create_session(workspace_id=ws["id"], hermes_session_id="hsess1")
        artifacts = cdb.list_artifacts_for_code_session(
            cs["id"], hermes_session_id="hsess1"
        )
        assert len(artifacts) == 1
        assert artifacts[0]["path"] == "foo.py"

    def test_link_artifact_sets_code_session_id(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        sdb.create_session("hsess2", "cli")
        art = sdb.create_artifact(
            session_id="hsess2",
            tool_name="write_file",
            path="bar.py",
            status="added",
        )
        cs = cdb.create_session(workspace_id=ws["id"])
        updated_art = cdb.link_artifact_to_session(art["id"], cs["id"])
        assert updated_art["code_session_id"] == cs["id"]

        # Now direct query finds it
        linked = cdb.list_artifacts_for_code_session(cs["id"])
        assert len(linked) == 1
        assert linked[0]["path"] == "bar.py"

    def test_link_artifact_unknown_artifact_returns_none(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        cs = cdb.create_session(workspace_id=ws["id"])
        assert cdb.link_artifact_to_session("nonexistent", cs["id"]) is None


# =============================================================================
# CodeSessionService
# =============================================================================


class TestCodeSessionService:
    @pytest.fixture()
    def dbs(self, tmp_path):
        from hermes_state import CodeSessionDB, WorkspaceDB, SessionDB

        db_path = tmp_path / "state.db"
        sdb = SessionDB(db_path=db_path)
        wdb = WorkspaceDB(db_path=db_path)
        cdb = CodeSessionDB(db_path=db_path)
        yield sdb, wdb, cdb
        sdb.close()
        wdb.close()
        cdb.close()

    def test_create_session_unknown_workspace_raises(self, tmp_path):
        from hermes_cli.code.session_service import CodeSessionService

        svc = CodeSessionService(db_path=tmp_path / "state.db")
        with pytest.raises(ValueError, match="Workspace not found"):
            svc.create_session(workspace_id="bad_id")

    def test_cancel_session_sets_status(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        cs = cdb.create_session(workspace_id=ws["id"])

        from hermes_cli.code.session_service import CodeSessionService

        svc = CodeSessionService(db_path=tmp_path / "state.db")
        cancelled = svc.cancel_session(cs["id"], reason="user request")
        assert cancelled["status"] == "cancelled"
        assert cancelled["completed_at"] is not None

        events = svc.list_events(cs["id"])
        event_types = [e["type"] for e in events]
        assert "code_session.cancelled" in event_types

    def test_complete_session_sets_done(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        cs = cdb.create_session(workspace_id=ws["id"])

        from hermes_cli.code.session_service import CodeSessionService

        svc = CodeSessionService(db_path=tmp_path / "state.db")
        done = svc.complete_session(cs["id"], summary="all tests pass")
        assert done["status"] == "done"
        assert done["summary"] == "all tests pass"
        assert done["completed_at"] is not None

    def test_update_status_emits_event(self, dbs, tmp_path):
        sdb, wdb, cdb = dbs
        ws = _make_workspace(wdb, tmp_path)
        cs = cdb.create_session(workspace_id=ws["id"])

        from hermes_cli.code.session_service import CodeSessionService

        svc = CodeSessionService(db_path=tmp_path / "state.db")
        svc.update_session(cs["id"], status="coding")

        events = svc.list_events(cs["id"])
        event_types = [e["type"] for e in events]
        assert "code_session.status_changed" in event_types


# =============================================================================
# Schema
# =============================================================================


class TestCodeSessionSchema:
    def test_code_sessions_table_exists(self, tmp_path):
        from hermes_state import CodeSessionDB

        db = CodeSessionDB(db_path=tmp_path / "state.db")
        try:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='code_sessions'"
            )
            assert cursor.fetchone() is not None
        finally:
            db.close()

    def test_code_session_events_table_exists(self, tmp_path):
        from hermes_state import CodeSessionDB

        db = CodeSessionDB(db_path=tmp_path / "state.db")
        try:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='code_session_events'"
            )
            assert cursor.fetchone() is not None
        finally:
            db.close()

    def test_artifacts_code_session_id_column_exists(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            cursor = db._conn.execute("PRAGMA table_info(artifacts)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "code_session_id" in cols
        finally:
            db.close()

    def test_schema_version_is_18(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            cursor = db._conn.execute("SELECT version FROM schema_version")
            assert cursor.fetchone()[0] == 18
        finally:
            db.close()


# =============================================================================
# REST endpoints
# =============================================================================


class TestCodeSessionEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("starlette not installed")

        from hermes_cli.web_server import app, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers["Authorization"] = f"Bearer {_SESSION_TOKEN}"

    def test_list_sessions_returns_empty(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        monkeypatch.setattr(
            svc_mod.CodeSessionService, "list_sessions", lambda self, **kw: []
        )
        resp = self.client.get("/api/code/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": [], "total": 0}

    def test_list_sessions_requires_auth(self):
        import hermes_cli.web_server as web_server
        from starlette.testclient import TestClient

        client = TestClient(web_server.app)
        resp = client.get("/api/code/sessions")
        assert resp.status_code == 401

    def test_create_session_unknown_workspace_returns_400(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        def _raise(self, workspace_id, **kw):
            raise ValueError(f"Workspace not found: {workspace_id}")

        monkeypatch.setattr(svc_mod.CodeSessionService, "create_session", _raise)
        resp = self.client.post(
            "/api/code/sessions",
            json={"workspace_id": "bad_id"},
        )
        assert resp.status_code == 400

    def test_create_session_returns_session(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake_session = {
            "id": "cs1",
            "workspace_id": "ws1",
            "status": "planning",
            "title": "Fix chat bug",
            "branch": "main",
            "provider": None,
            "model": None,
            "hermes_session_id": None,
            "task_id": None,
            "summary": None,
            "metadata": {},
            "completed_at": None,
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T00:00:00+00:00",
            "started_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "create_session",
            lambda self, **kw: fake_session,
        )
        resp = self.client.post(
            "/api/code/sessions",
            json={"workspace_id": "ws1", "title": "Fix chat bug"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code_session"]["id"] == "cs1"
        assert data["code_session"]["status"] == "planning"

    def test_get_session_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        monkeypatch.setattr(
            svc_mod.CodeSessionService, "get_session", lambda self, sid: None
        )
        resp = self.client.get("/api/code/sessions/bad_id")
        assert resp.status_code == 404

    def test_get_session_returns_session(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake_session = {
            "id": "cs1",
            "workspace_id": "ws1",
            "status": "coding",
            "branch": "feature/x",
            "title": None,
            "provider": "openai",
            "model": "gpt-4o",
            "hermes_session_id": None,
            "task_id": None,
            "summary": None,
            "metadata": {},
            "completed_at": None,
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T00:00:00+00:00",
            "started_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "get_session",
            lambda self, sid: fake_session,
        )
        resp = self.client.get("/api/code/sessions/cs1")
        assert resp.status_code == 200
        assert resp.json()["code_session"]["model"] == "gpt-4o"

    def test_patch_session_invalid_status_returns_400(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        def _raise(self, sid, **updates):
            raise ValueError("Invalid status 'flying'. Allowed: ...")

        monkeypatch.setattr(svc_mod.CodeSessionService, "update_session", _raise)
        resp = self.client.patch(
            "/api/code/sessions/cs1",
            json={"status": "flying"},
        )
        assert resp.status_code == 400

    def test_patch_session_returns_updated(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake = {
            "id": "cs1",
            "workspace_id": "ws1",
            "status": "coding",
            "title": None,
            "branch": None,
            "provider": None,
            "model": None,
            "hermes_session_id": None,
            "task_id": None,
            "summary": None,
            "metadata": {},
            "completed_at": None,
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T01:00:00+00:00",
            "started_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "update_session",
            lambda self, sid, **upd: fake,
        )
        resp = self.client.patch("/api/code/sessions/cs1", json={"status": "coding"})
        assert resp.status_code == 200
        assert resp.json()["code_session"]["status"] == "coding"

    def test_cancel_session_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        def _raise(self, sid, **kw):
            raise ValueError(f"CodeSession not found: {sid}")

        monkeypatch.setattr(svc_mod.CodeSessionService, "cancel_session", _raise)
        resp = self.client.post("/api/code/sessions/bad_id/cancel")
        assert resp.status_code == 404

    def test_cancel_session_returns_cancelled(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake = {
            "id": "cs1",
            "workspace_id": "ws1",
            "status": "cancelled",
            "completed_at": "2026-04-24T12:00:00+00:00",
            "title": None,
            "branch": None,
            "provider": None,
            "model": None,
            "hermes_session_id": None,
            "task_id": None,
            "summary": None,
            "metadata": {},
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T12:00:00+00:00",
            "started_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "cancel_session",
            lambda self, sid, **kw: fake,
        )
        resp = self.client.post(
            "/api/code/sessions/cs1/cancel", json={"reason": "user cancelled"}
        )
        assert resp.status_code == 200
        assert resp.json()["code_session"]["status"] == "cancelled"
        assert resp.json()["code_session"]["completed_at"] is not None

    def test_complete_session_returns_done(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake = {
            "id": "cs1",
            "workspace_id": "ws1",
            "status": "done",
            "completed_at": "2026-04-24T12:00:00+00:00",
            "summary": "all good",
            "title": None,
            "branch": None,
            "provider": None,
            "model": None,
            "hermes_session_id": None,
            "task_id": None,
            "metadata": {},
            "created_at": "2026-04-24T00:00:00+00:00",
            "updated_at": "2026-04-24T12:00:00+00:00",
            "started_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "complete_session",
            lambda self, sid, **kw: fake,
        )
        resp = self.client.post(
            "/api/code/sessions/cs1/complete",
            json={"summary": "all good"},
        )
        assert resp.status_code == 200
        assert resp.json()["code_session"]["status"] == "done"
        assert resp.json()["code_session"]["summary"] == "all good"

    def test_list_events_returns_list(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        events = [
            {
                "id": "e1",
                "code_session_id": "cs1",
                "type": "code_session.created",
                "message": None,
                "payload": {},
                "created_at": "2026-04-24T00:00:00+00:00",
            }
        ]
        monkeypatch.setattr(
            svc_mod.CodeSessionService, "get_session", lambda self, sid: {"id": sid}
        )
        monkeypatch.setattr(
            svc_mod.CodeSessionService, "list_events", lambda self, sid: events
        )
        resp = self.client.get("/api/code/sessions/cs1/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["events"][0]["type"] == "code_session.created"

    def test_list_events_session_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        monkeypatch.setattr(
            svc_mod.CodeSessionService, "get_session", lambda self, sid: None
        )
        resp = self.client.get("/api/code/sessions/bad/events")
        assert resp.status_code == 404

    def test_list_artifacts_returns_list(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        artifacts = [
            {
                "id": "a1",
                "path": "src/main.py",
                "status": "modified",
                "tool_name": "patch",
                "additions": 5,
                "deletions": 1,
                "diff": "",
                "timestamp": 1234567890.0,
                "tool_call_id": "",
                "code_session_id": "cs1",
            }
        ]
        monkeypatch.setattr(
            svc_mod.CodeSessionService, "list_artifacts", lambda self, sid: artifacts
        )
        resp = self.client.get("/api/code/sessions/cs1/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["artifacts"][0]["path"] == "src/main.py"
        assert data["code_session_id"] == "cs1"

    def test_list_artifacts_session_not_found_returns_404(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        def _raise(self, sid):
            raise ValueError(f"CodeSession not found: {sid}")

        monkeypatch.setattr(svc_mod.CodeSessionService, "list_artifacts", _raise)
        resp = self.client.get("/api/code/sessions/bad/artifacts")
        assert resp.status_code == 404

    def test_add_event_endpoint(self, monkeypatch):
        import hermes_cli.code.session_service as svc_mod

        fake_event = {
            "id": "e1",
            "code_session_id": "cs1",
            "type": "note.added",
            "message": "user note",
            "payload": {"x": 1},
            "created_at": "2026-04-24T00:00:00+00:00",
        }
        monkeypatch.setattr(
            svc_mod.CodeSessionService,
            "add_event",
            lambda self, sid, event_type, message=None, payload=None: fake_event,
        )
        resp = self.client.post(
            "/api/code/sessions/cs1/events",
            json={"type": "note.added", "message": "user note", "payload": {"x": 1}},
        )
        assert resp.status_code == 200
        assert resp.json()["event"]["type"] == "note.added"
