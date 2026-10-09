"""The dashboard's archive PATCH demands explicit confirmation for a cascade (#70185).

Archiving one session in a compression lineage flips the whole chain, so
``PATCH /api/sessions/{id}`` with ``{"archived": true}`` must refuse — 409 with the
blast-radius payload — unless the caller opts in with ``confirm_cascade: true``.
Single-session archives and unarchives pass through unchanged.
"""

import time

import pytest


class TestArchiveCascadeGate:
    """PATCH /api/sessions/{id} gates cascade archives behind confirm_cascade."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_state import SessionDB

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        db = SessionDB()
        try:
            base = time.time() - 1000
            db.create_session("root", source="desktop")
            db.create_session("tip", source="desktop", parent_session_id="root")
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' "
                "WHERE id = 'root'",
                (base, base + 10),
            )
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'tip'", (base + 20,))
            db._conn.commit()
        finally:
            db.close()

    def _archive_state(self, session_id):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            return db.get_session(session_id)["archived"]
        finally:
            db.close()

    def test_cascade_archive_without_confirm_returns_409(self):
        resp = self.client.patch("/api/sessions/tip", json={"archived": True})
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["cascade_count"] == 2
        assert set(detail["affected_ids"]) == {"root", "tip"}
        # Refused: nothing flipped.
        assert self._archive_state("root") == 0

    def test_cascade_archive_with_confirm_archives_lineage(self):
        resp = self.client.patch(
            "/api/sessions/tip", json={"archived": True, "confirm_cascade": True})
        assert resp.status_code == 200
        assert resp.json()["archived"] is True
        assert self._archive_state("root") == 1

    def test_unarchive_bypasses_the_gate(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            assert db.set_session_archived("root", True)
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/root", json={"archived": False})
        assert resp.status_code == 200

    def test_single_session_archive_passes_through(self):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.create_session("solo", source="desktop")
        finally:
            db.close()

        resp = self.client.patch("/api/sessions/solo", json={"archived": True})
        assert resp.status_code == 200
        assert self._archive_state("solo") == 1
