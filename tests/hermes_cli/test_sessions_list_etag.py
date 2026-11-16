"""ETag / If-None-Match on GET /api/sessions (#C3, t_406dbcb3).

The reconnect critical path re-fetches the session list even when nothing
moved. The route now stamps a content ETag over the exact response bytes and
answers a matching conditional GET with 304 and no body. These are the
contract pins: the stamp, the hit, the miss after a change, and the fact that
EVERY field the response carries lives inside the hashed body (a change in
``total`` or in the volatile ``is_active`` flag must rotate the ETag — a 304
that hides a change would be a correctness bug, not a performance win).
"""

import time

import pytest


class TestSessionsListEtag:
    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.home = get_hermes_home()
        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", self.home / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def _seed(self, *ids):
        from hermes_state import SessionDB

        db = SessionDB(db_path=self.home / "state.db")
        try:
            for sid in ids:
                db.create_session(sid, source="cli")
        finally:
            db.close()

    def test_list_carries_a_content_etag_and_no_cache(self):
        self._seed("s-etag-1")

        response = self.client.get("/api/sessions?limit=10&offset=0")

        assert response.status_code == 200
        etag = response.headers.get("etag")
        assert etag and etag.startswith('"') and etag.endswith('"')
        assert response.headers.get("cache-control") == "no-cache"
        # Stable across identical reads (the hash covers the body, not time):
        assert self.client.get("/api/sessions?limit=10&offset=0").headers["etag"] == etag

    def test_matching_if_none_match_answers_304_with_no_body(self):
        self._seed("s-etag-2")

        first = self.client.get("/api/sessions?limit=10&offset=0")
        etag = first.headers["etag"]

        again = self.client.get(
            "/api/sessions?limit=10&offset=0", headers={"If-None-Match": etag})

        assert again.status_code == 304
        assert again.content == b""
        # RFC 9110: a 304 must carry the current validators of the representation.
        assert again.headers["etag"] == etag
        assert again.headers["cache-control"] == "no-cache"

    def test_stale_etag_after_a_new_session_answers_200_with_the_new_body(self):
        self._seed("s-etag-3a")
        old = self.client.get("/api/sessions?limit=10&offset=0").headers["etag"]

        self._seed("s-etag-3b")
        changed = self.client.get("/api/sessions?limit=10&offset=0")

        assert changed.status_code == 200
        assert changed.headers["etag"] != old
        assert {s["id"] for s in changed.json()["sessions"]} == {"s-etag-3a", "s-etag-3b"}

        # A conditional GET with the OLD validator never gets the old body back.
        fresh = self.client.get(
            "/api/sessions?limit=10&offset=0", headers={"If-None-Match": old})
        assert fresh.status_code == 200

    def test_etag_covers_total_not_just_the_page_window(self):
        # Same 1-row page under both store sizes; only `total` differs. If the
        # ETag hashed the rows alone, the second read would answer 304 and the
        # client would keep a stale total forever.
        self._seed("s-window-a")
        first = self.client.get("/api/sessions?limit=1&offset=0&order=created")
        etag = first.headers["etag"]
        assert first.json()["total"] == 1

        self._seed("s-window-b")
        changed = self.client.get(
            "/api/sessions?limit=1&offset=0&order=created",
            headers={"If-None-Match": etag})

        assert changed.status_code == 200
        assert changed.json()["total"] == 2

    def test_wildcard_and_list_headers_hit_and_miss_correctly(self):
        self._seed("s-etag-5")
        etag = self.client.get("/api/sessions?limit=10&offset=0").headers["etag"]

        star = self.client.get("/api/sessions?limit=10&offset=0",
                               headers={"If-None-Match": "*"})
        assert star.status_code == 304

        listed = self.client.get(
            "/api/sessions?limit=10&offset=0",
            headers={"If-None-Match": f'"other", {etag}'})
        assert listed.status_code == 304

        miss = self.client.get(
            "/api/sessions?limit=10&offset=0", headers={"If-None-Match": '"other"'})
        assert miss.status_code == 200

    def test_volatile_is_active_flip_rotates_the_etag(self):
        self._seed("s-etag-active")
        etag = self.client.get("/api/sessions?limit=10&offset=0").headers["etag"]

        # Age the session past the active window without touching anything the
        # ETag could miss: with no messages its effective last activity is
        # started_at, so this flips is_active true→false — and the body a 304
        # would protect must therefore differ.
        from hermes_state import SessionDB

        db = SessionDB(db_path=self.home / "state.db")
        try:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 3600, "s-etag-active"))
        finally:
            db.close()

        aged = self.client.get(
            "/api/sessions?limit=10&offset=0", headers={"If-None-Match": etag})

        assert aged.status_code == 200
        row = next(s for s in aged.json()["sessions"] if s["id"] == "s-etag-active")
        assert row["is_active"] is False

    def test_error_paths_stay_error_paths_under_a_conditional_get(self):
        # A 400 (bad param) must not be turned into a 304 by a validator, and
        # must not carry cacheable validators of its own.
        response = self.client.get(
            "/api/sessions?order=nonsense", headers={"If-None-Match": "*"})

        assert response.status_code == 400
        assert "etag" not in response.headers
