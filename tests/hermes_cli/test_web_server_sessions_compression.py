"""Dashboard sessions endpoints vs compression-continuation children (#54298, #57543).

The sessions list collapses each compression chain to one row carrying the *tip's*
id; these tests pin the paired stats count and both delete endpoints to that
projection. Kept in a topical sibling (not ``test_web_server.py``, which is over
its file-lines ratchet cap and may only go down).
"""

import pytest


class _SessionsCompressionEndpoints:
    """Shared harness: TestClient pair against the isolated HERMES_HOME store."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(
            hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db"
        )
        self.client = TestClient(app)
        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN


class TestSessionStatsCompressionChildren(_SessionsCompressionEndpoints):
    """``GET /api/sessions/stats`` must pair with the list (#54298)."""

    def test_session_stats_excludes_compression_children(self):
        """Stats counts every physical row while the paired list collapses each
        chain to one row — total/active_store/archived were inflated by the
        hidden chain links and never matched the listed rows."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            # compression chain: root -> child1 -> child2 -> live tip
            db.create_session(session_id="comp-root", source="cli")
            db.append_message("comp-root", role="user", content="root")
            db.end_session("comp-root", end_reason="compression")
            db.create_session(session_id="comp-child1", source="cli", parent_session_id="comp-root")
            db.append_message("comp-child1", role="user", content="child1")
            db.end_session("comp-child1", end_reason="compression")
            db.create_session(session_id="comp-child2", source="cli", parent_session_id="comp-child1")
            db.append_message("comp-child2", role="user", content="child2")
            db.end_session("comp-child2", end_reason="compression")
            db.create_session(session_id="comp-tip", source="cli", parent_session_id="comp-child2")
            db.append_message("comp-tip", role="user", content="tip")
            # an archived standalone session
            db.create_session(session_id="archived", source="cli")
            db.append_message("archived", role="user", content="old")
            db.set_session_archived("archived", True)
            # an active standalone session
            db.create_session(session_id="standalone", source="cli")
            db.append_message("standalone", role="user", content="solo")
        finally:
            db.close()

        stats = self.auth_client.get("/api/sessions/stats").json()
        # 6 rows in the DB; the list surfaces 3 conversations: the compression
        # chain (collapsed to its tip, counted once at the root), the archived
        # standalone, and the active standalone.
        assert stats["total"] == 3, stats
        assert stats["active_store"] == 2, stats
        assert stats["archived"] == 1, stats


class TestDeleteSessionCompressionChain(_SessionsCompressionEndpoints):
    """``DELETE /api/sessions/{session_id}`` — single-row flavour of #57543."""

    def test_delete_removes_whole_compression_chain(self, monkeypatch):
        """The list row the user clicked carries the chain tip's id; deleting it
        must take the root with it, or the conversation resurfaces as the
        previous chain link on the next reload."""
        from types import SimpleNamespace

        from gateway.session_authority import LiveSession, SessionAuthority
        from hermes_cli.web_server import app
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
        from hermes_state_runtime import begin_runtime_epoch

        db = SessionDB()
        try:
            db.create_session("conv_root", source="tui")
            db.end_session("conv_root", end_reason="compression")
            db.create_session(
                session_id="conv_tip", source="tui", parent_session_id="conv_root"
            )
            # The dashboard never opens a second writer: DELETE admits through the owner.
            authority = SessionAuthority(
                SimpleNamespace(_draining=False), profile_id=str(get_hermes_home()),
                instance_id="owner", db=db, epoch=begin_runtime_epoch(db, instance_id="owner"))
            authority.sessions["conv_tip"] = LiveSession(None, "route")
            monkeypatch.setattr(app.state, "session_authority", authority, raising=False)

            snap = self.auth_client.get("/api/sessions/conv_tip/mutation-snapshot").json()
            resp = self.auth_client.delete("/api/sessions/conv_tip", params={
                "request_id": "delete-chain", "expected_revision": snap["runtime_revision"],
                "expected_generation": snap["runtime_generation"]})
            assert resp.status_code == 200, resp.text
            assert resp.json().get("ok") is True

            assert db.get_session("conv_tip") is None
            assert db.get_session("conv_root") is None
        finally:
            db.close()


class TestBulkDeleteCompressionChain(_SessionsCompressionEndpoints):
    """``POST /api/sessions/bulk-delete`` — the #57543 multi-select repro."""

    def test_deletes_whole_compression_chain_of_each_selected_root(self):
        """The sessions list surfaces a compressed conversation as ONE row
        carrying the chain tip's id. Bulk-deleting that id must remove the whole
        chain — otherwise the root resurfaces as the previous link on the next
        reload and the user sees the delete "not work". ``deleted`` counts
        selected rows, and other lineages / standalone rows survive untouched."""
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            # chain A (selected): a_root -> a_tip
            db.create_session("a_root", source="tui")
            db.end_session("a_root", end_reason="compression")
            db.create_session("a_tip", source="tui", parent_session_id="a_root")
            # chain B (selected): b_root -> b_mid -> b_tip
            db.create_session("b_root", source="tui")
            db.end_session("b_root", end_reason="compression")
            db.create_session("b_mid", source="tui", parent_session_id="b_root")
            db.end_session("b_mid", end_reason="compression")
            db.create_session("b_tip", source="tui", parent_session_id="b_mid")
            # chain C — NOT selected
            db.create_session("c_root", source="tui")
            db.end_session("c_root", end_reason="compression")
            db.create_session("c_tip", source="tui", parent_session_id="c_root")
            # standalones — NOT selected
            db.create_session("solo", source="tui")
        finally:
            db.close()

        resp = self.auth_client.post(
            "/api/sessions/bulk-delete", json={"ids": ["a_tip", "b_tip"]}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 2, "skipped_active": []}

        db = SessionDB()
        try:
            for gone in ("a_root", "a_tip", "b_root", "b_mid", "b_tip"):
                assert db.get_session(gone) is None, gone
            for kept in ("c_root", "c_tip", "solo"):
                assert db.get_session(kept) is not None, kept
        finally:
            db.close()
