"""PATCH/GET /api/sessions stamp round-trip.

The stamp lives in state.db, not in the client: the PATCH endpoint sets and
clears it, GET /api/sessions must project the column into every list row (the
Desktop renders from that projection and never sees raw SQL).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hermes_state import SessionDB  # noqa: E402

_SID = "20260915_120000_stamp1"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "stamp-test-token")
    # Seed the store the endpoints resolve (same HERMES_HOME → same state.db).
    seed = SessionDB()
    seed.create_session(_SID, source="cli")
    seed.close()

    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = "Bearer stamp-test-token"
        yield c


def _stamp_in_db() -> object:
    db = SessionDB()
    try:
        return db.get_session_stamp(_SID)
    finally:
        db.close()


class TestSessionStampEndpoint:
    def test_patch_sets_then_empty_string_clears(self, client):
        r = client.patch(f"/api/sessions/{_SID}", json={"stamp": "  Review  "})
        assert r.status_code == 200, r.text
        assert r.json()["stamp"] == "Review"
        assert _stamp_in_db() == "Review"

        r = client.patch(f"/api/sessions/{_SID}", json={"stamp": ""})
        assert r.status_code == 200, r.text
        assert r.json()["stamp"] == ""
        assert _stamp_in_db() is None

    def test_patch_leaves_the_stamp_alone_when_the_field_is_absent(self, client):
        client.patch(f"/api/sessions/{_SID}", json={"stamp": "Hold"})

        r = client.patch(f"/api/sessions/{_SID}", json={"pinned": True})

        assert r.status_code == 200, r.text
        assert r.json()["stamp"] == "Hold"
        assert _stamp_in_db() == "Hold"

    @pytest.mark.parametrize("bad", ["x" * 25, "WIP\nHold", "WIP\tHold"])
    def test_patch_rejects_invalid_stamps_with_400(self, client, bad):
        r = client.patch(f"/api/sessions/{_SID}", json={"stamp": bad})

        assert r.status_code == 400, r.text
        assert _stamp_in_db() is None

    def test_patch_with_nothing_to_update_mentions_stamp(self, client):
        r = client.patch(f"/api/sessions/{_SID}", json={})

        assert r.status_code == 400
        assert "stamp" in r.json()["detail"]

    def test_get_sessions_row_carries_the_stamp(self, client):
        client.patch(f"/api/sessions/{_SID}", json={"stamp": "Handoff"})

        r = client.get("/api/sessions")

        assert r.status_code == 200, r.text
        rows = [s for s in r.json()["sessions"] if s["id"] == _SID]
        assert len(rows) == 1
        assert rows[0]["stamp"] == "Handoff"
