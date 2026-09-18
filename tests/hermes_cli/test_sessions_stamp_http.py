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


def _stamps_in_db() -> list:
    db = SessionDB()
    try:
        return db.get_session_stamps(_SID)
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


class TestSessionStampsListEndpoint:
    """The list door (`stamps`): ordered, capped at three, and projected to every list row."""

    def test_patch_sets_an_ordered_deduped_list_and_the_row_projects_it(self, client):
        r = client.patch(f"/api/sessions/{_SID}", json={"stamps": ["WIP", " Review ", "wip"]})

        assert r.status_code == 200, r.text
        # Case-insensitive dedupe onto the FIRST spelling; order is the order sent.
        assert r.json()["stamps"] == ["WIP", "Review"]
        assert _stamps_in_db() == ["WIP", "Review"]

        rows = [s for s in client.get("/api/sessions").json()["sessions"] if s["id"] == _SID]
        assert len(rows) == 1
        # A real list, not the stored JSON string: the Desktop renders this directly.
        assert rows[0]["stamps"] == ["WIP", "Review"]

    def test_an_empty_list_clears_and_an_absent_one_leaves_the_list_alone(self, client):
        client.patch(f"/api/sessions/{_SID}", json={"stamps": ["WIP", "Hold"]})

        r = client.patch(f"/api/sessions/{_SID}", json={"pinned": True})

        assert r.status_code == 200, r.text
        assert r.json()["stamps"] == ["WIP", "Hold"]

        r = client.patch(f"/api/sessions/{_SID}", json={"stamps": []})

        assert r.status_code == 200, r.text
        assert r.json()["stamps"] == []
        assert _stamps_in_db() == []

    def test_a_fourth_stamp_is_refused_with_400_and_changes_nothing(self, client):
        client.patch(f"/api/sessions/{_SID}", json={"stamps": ["WIP", "Hold"]})

        r = client.patch(f"/api/sessions/{_SID}", json={"stamps": ["a", "b", "c", "d"]})

        assert r.status_code == 400, r.text
        # The refusal names the cap instead of silently dropping a label.
        assert "max 3" in r.json()["detail"]
        assert _stamps_in_db() == ["WIP", "Hold"]

    def test_the_list_wins_when_a_request_carries_both_doors(self, client):
        """An older client sends `stamp`, one that knows the list sends `stamps`; a request
        carrying both means the richer field, and the singular column stays its first label."""
        r = client.patch(
            f"/api/sessions/{_SID}", json={"stamp": "Review", "stamps": ["WIP", "Hold"]})

        assert r.status_code == 200, r.text
        assert r.json()["stamps"] == ["WIP", "Hold"]
        assert r.json()["stamp"] == "WIP"
        assert _stamps_in_db() == ["WIP", "Hold"]

    def test_the_single_label_door_still_replaces_the_whole_list(self, client):
        client.patch(f"/api/sessions/{_SID}", json={"stamps": ["WIP", "Hold"]})

        r = client.patch(f"/api/sessions/{_SID}", json={"stamp": "Merged"})

        assert r.status_code == 200, r.text
        assert r.json()["stamps"] == ["Merged"]
