"""ETag / If-None-Match on the profiles session-aggregate routes (#C3, t_2a76b2d8).

The desktop SIDEBAR's hot reads are the batched profiles routes —
``GET /api/profiles/sessions`` (the aggregate the registry splice fans into) and
``GET /api/profiles/sessions/sidebar``. They carried no ETag, so those reads
could never 304 and the desktop re-transferred the whole aggregate on every
refresh. This stamps the EXACT same treatment ``GET /api/sessions`` got in
t_406dbcb3 (PR #9): a strong content ETag over the exact response bytes,
``Cache-Control: no-cache``, and a matching ``If-None-Match`` (single, list or
``*``, weakness-insensitive) answered with 304 and a zero-byte body.

The sidebar's singleflight cache is the interesting half: its key is the query
parameters only, so a cached read must still honor the validator — the
conditional response is stamped on the (possibly cached) payload, per request.
"""

import pytest


class _ProfilesEtagBase:
    @property
    def _route(self):
        raise NotImplementedError

    @property
    def _query(self):
        # The desktop's own read shape: the all-profiles aggregate / default sidebar.
        return "" if self._route.endswith("/sidebar") else "?profile=all"

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

    def _url(self, extra: str = ""):
        base = f"{self._route}{self._query}"
        join = "&" if "?" in base else "?"
        return f"{base}{join}{extra}" if extra else base

    def _seed(self, *ids):
        from hermes_state import SessionDB

        db = SessionDB(db_path=self.home / "state.db")
        try:
            for sid in ids:
                db.create_session(sid, source="cli")
                # min_messages=1 is the sidebar's slice floor: a bare session is invisible to it.
                db.append_message(session_id=sid, role="user", content="hi")
        finally:
            db.close()

    def _get(self, headers=None, extra: str = ""):
        return self.client.get(self._url(extra), headers=headers or {})

    def test_list_carries_a_content_etag_and_no_cache(self):
        self._seed("p-etag-1")

        response = self._get()

        assert response.status_code == 200
        etag = response.headers.get("etag")
        assert etag and etag.startswith('"') and etag.endswith('"')
        assert response.headers.get("cache-control") == "no-cache"
        # Stable across identical reads (the hash covers the body, not time):
        assert self._get().headers["etag"] == etag

    def test_matching_if_none_match_answers_304_with_no_body(self):
        self._seed("p-etag-2")

        first = self._get()
        etag = first.headers["etag"]

        again = self._get(headers={"If-None-Match": etag})

        assert again.status_code == 304
        assert again.content == b""
        # RFC 9110: a 304 must carry the current validators of the representation.
        assert again.headers["etag"] == etag
        assert again.headers["cache-control"] == "no-cache"

    def test_stale_etag_after_a_new_session_answers_200_with_the_new_body(self):
        self._seed("p-etag-3a")
        old = self._get().headers["etag"]

        self._seed("p-etag-3b")
        changed = self._get()

        assert changed.status_code == 200
        assert changed.headers["etag"] != old
        assert {s["id"] for s in changed.json()["sessions"]} >= {"p-etag-3a", "p-etag-3b"}

        # A conditional GET with the OLD validator never gets the old body back.
        fresh = self._get(headers={"If-None-Match": old})
        assert fresh.status_code == 200

    def test_wildcard_and_list_headers_hit_and_miss_correctly(self):
        self._seed("p-etag-5")
        etag = self._get().headers["etag"]

        star = self._get(headers={"If-None-Match": "*"})
        assert star.status_code == 304

        listed = self._get(headers={"If-None-Match": f'"other", {etag}'})
        assert listed.status_code == 304

        miss = self._get(headers={"If-None-Match": '"other"'})
        assert miss.status_code == 200


class TestProfilesSessionsEtag(_ProfilesEtagBase):
    @property
    def _route(self):
        return "/api/profiles/sessions"

    def test_etag_covers_totals_not_just_the_page_window(self):
        # Same 1-row page under both store sizes; only `total` differs (a SQL
        # count, independent of the window). If the ETag hashed the rows alone,
        # the second read would answer 304 and the client would keep a stale
        # total forever.
        self._seed("p-window-a")
        first = self._get(extra="limit=1&offset=0&order=created")
        etag = first.headers["etag"]
        assert first.json()["total"] == 1

        self._seed("p-window-b")
        changed = self._get(headers={"If-None-Match": etag}, extra="limit=1&offset=0&order=created")

        assert changed.status_code == 200
        assert changed.json()["total"] == 2

    def test_error_paths_stay_error_paths_under_a_conditional_get(self):
        # A 400 (bad param) must not be turned into a 304 by a validator, and
        # must not carry cacheable validators of its own.
        response = self.client.get(
            self._route + "?profile=all&order=nonsense", headers={"If-None-Match": "*"})

        assert response.status_code == 400
        assert "etag" not in response.headers


class TestProfilesSessionsSidebarEtag(_ProfilesEtagBase):
    @property
    def _route(self):
        return "/api/profiles/sessions/sidebar"

    def _no_cache(self, monkeypatch):
        # The 5s singleflight TTL would serve the pre-change payload; zero it so
        # the payload-change contract is measured against fresh scans.
        from hermes_cli.web_routers import profiles

        monkeypatch.setattr(profiles, "_SIDEBAR_CACHE_TTL_SECONDS", 0)

    def test_stale_etag_after_a_new_session_answers_200_with_the_new_body(self, monkeypatch):
        self._no_cache(monkeypatch)
        self._seed("p-etag-3a")
        old = self._get().headers["etag"]

        self._seed("p-etag-3b")
        changed = self._get()

        assert changed.status_code == 200
        assert changed.headers["etag"] != old
        assert {s["id"] for s in changed.json()["recents"]["sessions"]} >= {"p-etag-3a", "p-etag-3b"}

        # A conditional GET with the OLD validator never gets the old body back.
        fresh = self._get(headers={"If-None-Match": old})
        assert fresh.status_code == 200

    def test_etag_covers_usage_not_just_the_page_window(self, monkeypatch):
        # Same 1-row recents page under both stores; only ``profiles_usage``
        # differs — it is aggregated in SQL across the WHOLE store, so a window
        # alone never shows it. If the ETag hashed the rows alone, the second
        # read would answer 304 and the client would keep a stale usage total.
        self._no_cache(monkeypatch)
        self._seed("p-uwin-a")
        first = self._get(extra="recents_limit=1")
        etag = first.headers["etag"]
        assert first.json()["recents"]["profiles_usage"]["default"]["tokens"] == 0

        from hermes_state import SessionDB

        db = SessionDB(db_path=self.home / "state.db")
        try:
            db._conn.execute(
                "UPDATE sessions SET input_tokens = 42 WHERE id = ?", ("p-uwin-a",))
        finally:
            db.close()

        changed = self._get(headers={"If-None-Match": etag}, extra="recents_limit=1")

        assert changed.status_code == 200
        assert changed.json()["recents"]["profiles_usage"]["default"]["tokens"] == 42

    def test_changed_payload_rotates_the_etag(self, monkeypatch):
        # TTL to zero: the singleflight wrapper goes pass-through, so this is the
        # plain payload-change contract (same as the aggregate route).
        self._no_cache(monkeypatch)

        self._seed("p-side-1")
        old = self._get().headers["etag"]

        self._seed("p-side-2")
        changed = self._get()

        assert changed.status_code == 200
        assert changed.headers["etag"] != old
        assert {s["id"] for s in changed.json()["recents"]["sessions"]} >= {"p-side-1", "p-side-2"}

    def test_conditional_hit_is_served_from_the_singleflight_cache(self):
        # Within the 5s TTL the sidebar body is served from the cache: the scan
        # must run ONCE, yet every read — validator or not — must still get the
        # conditional treatment. A 304 off the cached payload is the whole point
        # (no re-scan, zero bytes); a cache that ignores the validator serves a
        # full 200 body where the contract says 304.
        from hermes_cli.web_routers import profiles

        self._seed("p-cache-1")
        calls = []
        original = profiles._read_profile_db

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(profiles, "_read_profile_db", counting)
        try:
            first = self._get()
            assert first.status_code == 200
            etag = first.headers["etag"]

            cached = self._get()
            assert cached.status_code == 200
            assert cached.headers["etag"] == etag

            again = self._get(headers={"If-None-Match": etag})
            assert again.status_code == 304
            assert again.content == b""
        finally:
            monkey.undo()

        assert len(calls) == 1, "the TTL window must be served from the singleflight cache"
