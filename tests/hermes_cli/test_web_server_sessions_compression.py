"""Dashboard sessions endpoints vs compression-continuation children (#54298, #57543).

The sessions list collapses each compression chain to one row carrying the *tip's*
id; these tests pin the paired stats count and both delete endpoints to that
projection. Kept in a topical sibling (not ``test_web_server.py``, which is over
its file-lines ratchet cap and may only go down).
"""

import pytest
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


class _SessionsCompressionEndpoints:
    """Shared harness: TestClient pair against the isolated HERMES_HOME store."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.auth_client = TestClient(app)
        self.auth_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN


class TestSessionStatsCompressionChildren(_SessionsCompressionEndpoints):
    """``GET /api/sessions/stats`` must pair with the list (#54298)."""

    def test_session_stats_excludes_compression_children(self):
        """Stats counts every physical row while the paired list collapses each
        chain to one row — total/active_store/archived were inflated by the
        hidden chain links and never matched the listed rows."""
        from persistence_assembly import open_persistence, profile_scope

        db = open_persistence(profile_scope())
        try:
            # compression chain: root -> child1 -> child2 -> live tip
            seed_session(db, "comp-root", "cli")
            seed_messages(db, "comp-root", {"role": "user", "content": "root"})
            db.sessions.end("comp-root", "compression")
            seed_session(db, "comp-child1", "cli", parent_session_id="comp-root")
            seed_messages(db, "comp-child1", {"role": "user", "content": "child1"})
            db.sessions.end("comp-child1", "compression")
            seed_session(db, "comp-child2", "cli", parent_session_id="comp-child1")
            seed_messages(db, "comp-child2", {"role": "user", "content": "child2"})
            db.sessions.end("comp-child2", "compression")
            seed_session(db, "comp-tip", "cli", parent_session_id="comp-child2")
            seed_messages(db, "comp-tip", {"role": "user", "content": "tip"})
            # an archived standalone session
            seed_session(db, "archived", "cli")
            seed_messages(db, "archived", {"role": "user", "content": "old"})
            db.sessions.set_archived("archived", True)
            # an active standalone session
            seed_session(db, "standalone", "cli")
            seed_messages(db, "standalone", {"role": "user", "content": "solo"})
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
        from persistence_assembly import open_persistence, profile_scope

        db = open_persistence(profile_scope())
        try:
            seed_session(db, "conv_root", "tui")
            db.sessions.end("conv_root", "compression")
            seed_session(
                db, "conv_tip", "tui", parent_session_id="conv_root"
            )
            # The dashboard never opens a second writer: DELETE admits through the owner.
            authority = SessionAuthority(
                SimpleNamespace(_draining=False), profile_id=str(get_hermes_home()),
                instance_id="owner", epoch=db.runtime.begin_epoch(instance_id="owner"),
                persistence=storage_handle(get_hermes_home()))
            authority.sessions["conv_tip"] = LiveSession(None, "route")
            monkeypatch.setattr(app.state, "session_authority", authority, raising=False)

            snap = self.auth_client.get("/api/sessions/conv_tip/mutation-snapshot").json()
            resp = self.auth_client.delete("/api/sessions/conv_tip", params={
                "request_id": "delete-chain", "expected_revision": snap["runtime_revision"],
                "expected_generation": snap["runtime_generation"]})
            assert resp.status_code == 200, resp.text
            assert resp.json().get("ok") is True

            assert session_row(db, "conv_tip") is None
            assert session_row(db, "conv_root") is None
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
        from persistence_assembly import open_persistence, profile_scope

        db = open_persistence(profile_scope())
        try:
            # chain A (selected): a_root -> a_tip
            seed_session(db, "a_root", "tui")
            db.sessions.end("a_root", "compression")
            seed_session(db, "a_tip", "tui", parent_session_id="a_root")
            # chain B (selected): b_root -> b_mid -> b_tip
            seed_session(db, "b_root", "tui")
            db.sessions.end("b_root", "compression")
            seed_session(db, "b_mid", "tui", parent_session_id="b_root")
            db.sessions.end("b_mid", "compression")
            seed_session(db, "b_tip", "tui", parent_session_id="b_mid")
            # chain C — NOT selected
            seed_session(db, "c_root", "tui")
            db.sessions.end("c_root", "compression")
            seed_session(db, "c_tip", "tui", parent_session_id="c_root")
            # standalones — NOT selected
            seed_session(db, "solo", "tui")
        finally:
            db.close()

        resp = self.auth_client.post(
            "/api/sessions/bulk-delete", json={"ids": ["a_tip", "b_tip"]}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "deleted": 2, "skipped_active": []}

        db = open_persistence(profile_scope())
        try:
            for gone in ("a_root", "a_tip", "b_root", "b_mid", "b_tip"):
                assert session_row(db, gone) is None, gone
            for kept in ("c_root", "c_tip", "solo"):
                assert session_row(db, kept) is not None, kept
        finally:
            db.close()
