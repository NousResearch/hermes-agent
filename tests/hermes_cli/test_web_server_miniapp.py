"""Telegram Mini App dashboard: token routes, admin-tier gates and per-row session ownership.

These drive the real dashboard app (``hermes_cli.web_server``) through its middleware stack with Mini
App bearer tokens, alongside the desktop dashboard's cookie-session caller.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.web_server_gateway as _web_server_gateway


class TestTokenAuthOptionalFallthroughIntegration:
    """Integration coverage for register_token_route(required=False)'s
    fallthrough contract, against the REAL Starlette middleware stack.

    tests/hermes_cli/test_dashboard_token_auth.py covers the same contract
    at the unit level (a _FakeRequest and a fake call_next that always
    "succeeds"). That's enough to pin is_token_route_required()'s merge
    semantics, but it CANNOT prove the actual claim this fix depends on:
    that a required=False route with no Authorization header really does
    fall through _token_auth_seam (registered last -> runs first, per
    Starlette's outermost-last ordering) to auth_middleware /
    _dashboard_auth_gate below it, and that THAT gate is what decides the
    request — a fake call_next always returns 200 regardless of whether
    fallthrough actually happened. This class hits the real mounted app
    through Starlette's TestClient so that claim is verified, not asserted.

    /api/skills is used as the route under test: a real, always-mounted,
    non-public route (unlike /api/status, which is in PUBLIC_API_PATHS and
    would return 200 regardless of any auth gate, proving nothing) — a
    realistic stand-in for the Mini App plugin's own required=False
    registrations, without needing that plugin enabled for this test.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        class _StubTokenProvider(DashboardAuthProvider):
            """Minimal token-only provider: accepts exactly "good-token"."""

            name = "test-optional-token-provider"
            display_name = "Test Optional Token Provider"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "good-token":
                    return TokenPrincipal(
                        principal="test:1",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubTokenProvider())
        # Route under test: required=False, the fix being verified.
        token_auth.register_token_route("/api/skills", required=False)
        # Contrast route: required=True (default / drain's original
        # contract) — pins that the fix is genuinely opt-in and doesn't
        # loosen the exclusive-token contract for routes that don't ask
        # for the fallthrough behaviour.
        token_auth.register_token_route("/api/cron/jobs", required=True)

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield

        clear_providers()
        token_auth.clear_token_routes()

    def test_required_false_no_token_no_session_falls_through_to_cookie_gate_401(self):
        """No Authorization header at all -> the seam must be a no-op and
        fall through; auth_middleware (the cookie/session gate, engaged
        here via the legacy loopback session-token path) is what actually
        decides, with ITS OWN 401 body shape ({"detail": ...}, no "error"
        key) — distinct from the seam's own 401
        ({"error": "unauthenticated", "detail": ...}). If the seam had
        wrongly kept owning the decision instead of falling through, this
        response would carry the "error" key.
        """
        resp = self.client.get("/api/skills")
        assert resp.status_code == 401
        body = resp.json()
        assert "error" not in body
        assert body.get("detail") == "Unauthorized"

    def test_required_false_no_token_with_session_falls_through_and_gate_grants_access(self):
        """The load-bearing positive case. No bearer token is presented at
        all, but a valid session credential (the desktop dashboard
        operator's auth mechanism) IS presented. If the seam still owned
        the decision (the bug this fix exists to close), it would see "no
        token" and reject with 401 regardless of the session credential.
        A 200 here is only possible if the seam genuinely fell through and
        let the downstream gate decide on its own terms.
        """
        resp = self.session_client.get("/api/skills")
        assert resp.status_code == 200

    def test_required_false_invalid_presented_token_still_401s_from_seam(self):
        """A PRESENTED (even if wrong) bearer token is still fully owned by
        the seam — required=False only changes behaviour for a fully
        ABSENT token. Even with a valid session credential also on the
        request, an invalid bearer token must still 401 with the seam's
        own error shape, proving required=False isn't a blanket bypass for
        every request to the route.
        """
        self.session_client.headers["authorization"] = "Bearer wrong-token"
        resp = self.session_client.get("/api/skills")
        assert resp.status_code == 401
        assert resp.json().get("error") == "unauthenticated"

    def test_required_false_valid_presented_token_grants_access_without_session(self):
        """A genuine token-bearing caller (e.g. the Mini App) with no
        session credential at all still gets in — required=False doesn't
        change the contract for the case it's actually meant to serve.
        """
        resp = self.client.get(
            "/api/skills", headers={"authorization": "Bearer good-token"}
        )
        assert resp.status_code == 200

    def test_required_true_route_does_not_fall_through_even_with_valid_session(self):
        """Contrast case: a required=True (default, drain-style)
        registration must NOT fall through, even when a valid session
        credential is present on the same request — confirms
        required=False is opt-in per registration, not a global loosening
        of the seam's exclusivity contract.
        """
        resp = self.session_client.get("/api/cron/jobs")
        assert resp.status_code == 401
        assert resp.json().get("error") == "unauthenticated"

    def test_required_true_route_unchanged_for_valid_token_no_session(self):
        """Sanity check: required=True's happy path (valid token, no
        session) is unaffected by this change.
        """
        resp = self.client.get(
            "/api/cron/jobs", headers={"authorization": "Bearer good-token"}
        )
        assert resp.status_code == 200


class TestTokenRouteMethodRestrictionIntegration:
    """Security regression: a read-only (methods={"GET","HEAD"}) token
    registration must NOT authenticate a mutating verb that shares the same
    path as the registered read route, against the REAL middleware stack.

    Several dashboard paths mount both a reader and a mutator on the same
    path — GET vs POST /api/skills (create-skill, agent-executed → code
    execution), GET vs PUT /api/skills/content, GET vs POST /api/cron/jobs,
    GET vs DELETE/PATCH /api/sessions/{id}. Route matching is path-based and
    method-blind, so without the method restriction the Telegram Mini App's
    GET registrations would also make those write verbs token-authable for a
    merely-paired principal (the seam would set token_authenticated and the
    cookie gate would then skip), silently turning a "read-only" tier into
    full write + RCE. These tests present a VALID token on each write verb
    and assert the seam does not own the decision — the request falls through
    to the cookie gate (its own 401 body shape, no "error" key), and the
    desktop cookie operator can still perform the write.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        class _StubTokenProvider(DashboardAuthProvider):
            name = "test-readonly-token-provider"
            display_name = "Test Read-Only Token Provider"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "good-token":
                    return TokenPrincipal(
                        principal="test:1",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubTokenProvider())
        # Mirror the plugin's real read-only registrations exactly.
        read = ("GET", "HEAD")
        token_auth.register_token_route("/api/skills", required=False, methods=read)
        token_auth.register_token_route("/api/skills/content", required=False, methods=read)
        token_auth.register_token_route("/api/cron/jobs", required=False, methods=read)
        token_auth.register_token_route(
            r"/api/sessions/\d{8}_\d{6}_[0-9a-f]{6,8}",
            match="regex",
            required=False,
            methods=read,
        )

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self._sid = "20260702_100000_aaaaaaaa"

        yield

        clear_providers()
        token_auth.clear_token_routes()

    def _assert_fell_through_to_cookie_gate(self, resp):
        # The seam did NOT own this decision: cookie-gate 401 body shape
        # ({"detail": ...}, no "error"), distinct from the seam's own 401.
        assert resp.status_code == 401
        assert "error" not in resp.json()

    # ---- valid token must NOT authenticate a write verb ----

    def test_valid_token_post_skills_is_not_token_authed(self):
        resp = self.client.post(
            "/api/skills",
            headers={"authorization": "Bearer good-token"},
            json={"name": "x", "content": "y"},
        )
        self._assert_fell_through_to_cookie_gate(resp)

    def test_valid_token_put_skills_content_is_not_token_authed(self):
        resp = self.client.put(
            "/api/skills/content",
            headers={"authorization": "Bearer good-token"},
            json={"name": "x", "content": "y"},
        )
        self._assert_fell_through_to_cookie_gate(resp)

    def test_valid_token_post_cron_jobs_is_not_token_authed(self):
        resp = self.client.post(
            "/api/cron/jobs",
            headers={"authorization": "Bearer good-token"},
            json={},
        )
        self._assert_fell_through_to_cookie_gate(resp)

    def test_valid_token_delete_session_is_not_token_authed(self):
        resp = self.client.delete(
            f"/api/sessions/{self._sid}",
            headers={"authorization": "Bearer good-token"},
        )
        self._assert_fell_through_to_cookie_gate(resp)

    def test_valid_token_patch_session_is_not_token_authed(self):
        resp = self.client.patch(
            f"/api/sessions/{self._sid}",
            headers={"authorization": "Bearer good-token"},
            json={"title": "pwned"},
        )
        self._assert_fell_through_to_cookie_gate(resp)

    # ---- read verb on the same paths is still token-authed (functionality) ----

    def test_valid_token_get_skills_is_token_authed(self):
        resp = self.client.get(
            "/api/skills", headers={"authorization": "Bearer good-token"}
        )
        assert resp.status_code == 200

    # ---- the desktop cookie operator is NOT locked out of the write verbs ----

    def test_cookie_operator_can_still_post_skills(self):
        # Reaches the handler (any non-auth status: 200/400/422/…), proving
        # the method restriction only gates the TOKEN path, not the cookie
        # path. The one thing it must never be is an auth rejection.
        resp = self.session_client.post(
            "/api/skills", json={"name": "x", "content": "y"}
        )
        assert resp.status_code not in (401, 403)

    def test_cookie_operator_can_still_delete_session(self):
        resp = self.session_client.delete(f"/api/sessions/{self._sid}")
        assert resp.status_code not in (401, 403)


class TestDashboardRequesterScopeAuthPathAwareness:
    """Confirms _dashboard_requester_scope() (the DM-scope filter) never
    restricts a cookie/session-authenticated caller — only a non-admin
    telegram-miniapp *token* principal is scoped.

    Alongside the required=False fallthrough test: a new ownership/scoping gate
    applied to a handler that previously served two different trust levels
    identically (the single-owner desktop dashboard operator, and now a
    DM-scoped Mini App token principal) must be proven to distinguish
    which auth path produced the current principal — not assumed to, just
    because it happens to key off request.state.token_principal. Both the
    pure-function unit tests and the end-to-end GET /api/sessions tests
    below hit this directly.
    """

    # ---- unit-level: _dashboard_requester_scope() in isolation -----------

    def _fake_request(self, *, token_principal=None):
        from types import SimpleNamespace
        return SimpleNamespace(state=SimpleNamespace(token_principal=token_principal))

    def test_no_token_principal_is_unscoped(self):
        """The defining guarantee: a cookie/session-authenticated request
        never sets request.state.token_principal (only the token-auth seam
        does, and only for a route it decided). Absent that attribute
        entirely, scope must be (None, None) — unrestricted, matching the
        desktop dashboard's existing single-owner-operator contract.
        """
        from hermes_cli.web_server import _dashboard_requester_scope

        req = self._fake_request(token_principal=None)
        assert _dashboard_requester_scope(req) == (None, None)

    def test_missing_token_principal_attribute_is_unscoped(self):
        """Same guarantee, but for a request.state that never had the
        attribute set at all (getattr default path) rather than an
        explicit None — this is the actual shape of a real cookie request,
        since nothing ever assigns token_principal on that path.
        """
        from types import SimpleNamespace
        from hermes_cli.web_server import _dashboard_requester_scope

        req = SimpleNamespace(state=SimpleNamespace())
        assert _dashboard_requester_scope(req) == (None, None)

    def test_admin_scoped_token_principal_is_unrestricted(self):
        from hermes_cli.dashboard_auth import TokenPrincipal
        from hermes_cli.web_server import _dashboard_requester_scope

        principal = TokenPrincipal(
            principal="telegram:999",
            provider="telegram-miniapp",
            scopes=("dashboard:read", "dashboard:admin"),
        )
        req = self._fake_request(token_principal=principal)
        assert _dashboard_requester_scope(req) == ("admin", None)

    def test_non_admin_telegram_miniapp_principal_is_own_scoped(self):
        from hermes_cli.dashboard_auth import TokenPrincipal
        from hermes_cli.web_server import _dashboard_requester_scope

        principal = TokenPrincipal(
            principal="telegram:12345",
            provider="telegram-miniapp",
            scopes=("dashboard:read",),
        )
        req = self._fake_request(token_principal=principal)
        assert _dashboard_requester_scope(req) == ("own", "12345")

    def test_non_telegram_miniapp_token_principal_is_own_scoped_with_no_id(self):
        """A token principal from some other (non Mini App) provider —
        drain's bearer secret, or a future service credential — has no
        Telegram user id to scope by. Must resolve to a deny-by-default
        ("own", None), never fall back to unrestricted access just because
        it isn't the Mini App provider.
        """
        from hermes_cli.dashboard_auth import TokenPrincipal
        from hermes_cli.web_server import _dashboard_requester_scope

        principal = TokenPrincipal(
            principal="drain-secret", provider="drain-bearer-secret", scopes=("drain",)
        )
        req = self._fake_request(token_principal=principal)
        assert _dashboard_requester_scope(req) == ("own", None)

    # ---- integration-level: GET /api/sessions end to end ------------------

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        class _StubMiniAppProvider(DashboardAuthProvider):
            """Stands in for TelegramMiniAppProvider: "own-token" resolves
            to a non-admin DM-scoped principal for user 12345, "admin-token"
            to an admin-scoped principal.
            """

            name = "telegram-miniapp"
            display_name = "Test Telegram Mini App"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "own-token":
                    return TokenPrincipal(
                        principal="telegram:12345",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                if token == "admin-token":
                    return TokenPrincipal(
                        principal="telegram:999",
                        provider=self.name,
                        scopes=("dashboard:read", "dashboard:admin"),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubMiniAppProvider())
        token_auth.register_token_route("/api/sessions", required=False)

        db = hermes_state.SessionDB()
        try:
            db.create_session(
                session_id="dm-owned-by-12345",
                source="telegram",
                user_id="12345",
                chat_id="12345",
                chat_type="dm",
            )
            db.create_session(
                session_id="dm-owned-by-other-user",
                source="telegram",
                user_id="67890",
                chat_id="67890",
                chat_type="dm",
            )
            db.create_session(
                session_id="cli-owned-by-nobody",
                source="cli",
            )
        finally:
            db.close()

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield

        clear_providers()
        token_auth.clear_token_routes()

    def _ids(self, resp):
        return {s["id"] for s in resp.json()["sessions"]}

    def test_cookie_session_caller_sees_every_session_unscoped(self):
        """The actual guarantee under audit: the existing desktop dashboard
        operator (session-token/cookie auth, no bearer token at all) must
        see every session regardless of source/chat_id/owner — exactly the
        unrestricted access it always had, unaffected by the new DM-scope
        filter that exists only for Mini App token principals.
        """
        resp = self.session_client.get("/api/sessions?limit=50")
        assert resp.status_code == 200
        ids = self._ids(resp)
        assert {"dm-owned-by-12345", "dm-owned-by-other-user", "cli-owned-by-nobody"} <= ids

    def test_non_admin_miniapp_token_caller_sees_only_own_dm_session(self):
        resp = self.client.get(
            "/api/sessions?limit=50", headers={"authorization": "Bearer own-token"}
        )
        assert resp.status_code == 200
        ids = self._ids(resp)
        assert ids == {"dm-owned-by-12345"}

    def test_admin_miniapp_token_caller_sees_every_session_unscoped(self):
        resp = self.client.get(
            "/api/sessions?limit=50", headers={"authorization": "Bearer admin-token"}
        )
        assert resp.status_code == 200
        ids = self._ids(resp)
        assert {"dm-owned-by-12345", "dm-owned-by-other-user", "cli-owned-by-nobody"} <= ids


class TestStatusGatewayStartTimeAdminOnly:
    """GET /api/status's gateway_start_time AND telegram_allowlist_updated_at
    fields -- same admin-tier gate, same reasoning, added together so the
    Mini App admin view can compute
    needs_restart = telegram_allowlist_updated_at > gateway_start_time.

    PRODUCTION RUNS WITH app.state.auth_required = True, ALWAYS, for this
    feature: the Mini App requires exposing the dashboard on a non-loopback
    bind (Telegram's clients must be able to reach it), and
    should_require_auth(host) (hermes_cli/web_server.py:384-403) returns True
    for any non-loopback host. auth_required=False is the LOOPBACK/dev case,
    which the Mini App cannot reach at all. Every test below pins
    app.state.auth_required = True explicitly -- do not remove this or let a
    future edit revert to the ambient test-module default (which is False,
    matching test_get_status's own baseline). Left unpinned, these tests
    would pass only because auth_required
    happened to be False in that shared ambient state -- a code path the Mini
    App can never reach. Pin it every time, in every case, so a green run
    here actually means something.

    The real security boundary is _dashboard_requester_scope, not
    auth_required (see hermes_cli/web_server.py's comment above the field for
    the full reasoning) -- these tests exercise that boundary specifically
    under the auth_required value that makes it the only thing standing
    between the field and an anonymous caller.

    One more thing pinning auth_required=True surfaces: /api/status is in
    PUBLIC_API_PATHS, and gated_auth_middleware skips ALL session-cookie
    verification for public paths (hermes_cli/dashboard_auth/middleware.py:
    270-271) -- request.state.session is never set on this endpoint, for any
    caller, cookie-holder or not. So a "cookie session" caller and a fully
    anonymous caller are indistinguishable here and must both be denied the
    field; only a verified admin-tier Mini App TokenPrincipal (which the
    token-auth seam verifies on token presence, unaffected by the public-path
    cookie-gate skip) can prove admin status on this specific endpoint.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        # Deterministic, non-None gateway_start_time to assert on -- mirrors
        # the real runtime-status file's "start_epoch" field (a wall-clock
        # Unix epoch; NOT "start_time", which is gateway/status.py's
        # boot-relative PID-reuse-guard fingerprint and must never be
        # surfaced as a timestamp -- see gateway/status.py's
        # _get_process_start_epoch docstring for why the two are distinct).
        # get_status's runtime read is a late-binding proxy to gateway.status
        # (hermes_cli/web_routers/status.py), not a direct hermes_cli.web_server
        # attribute -- patching the web_server compat-shim name is a no-op for
        # the router's own lookup. See hermes_cli/web_deps.py's late()/_server().
        monkeypatch.setattr(
            "gateway.status.read_runtime_status",
            lambda *a, **kw: {"gateway_state": "running", "start_epoch": 1234567890},
        )
        monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **kw: 4242)
        monkeypatch.setattr("gateway.status.get_running_pid_cached", lambda *a, **kw: 4242)

        # A real .env file so telegram_allowlist_updated_at has a real,
        # non-None mtime to assert on -- a fresh test home has no .env at
        # all until something writes one.
        (get_hermes_home() / ".env").write_text("TELEGRAM_ALLOWED_USERS=\n")

        # Pin to the value production actually runs under for this feature —
        # see the class docstring. Every test in this class relies on this.
        # raising=False: app.state.auth_required is only ever set by
        # start_server() (never called in this test process), so the
        # attribute may not exist yet on the shared State object.
        monkeypatch.setattr(app.state, "auth_required", True, raising=False)

        class _StubMiniAppProvider(DashboardAuthProvider):
            name = "telegram-miniapp"
            display_name = "Test Telegram Mini App"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "own-token":
                    return TokenPrincipal(
                        principal="telegram:12345",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                if token == "admin-token":
                    return TokenPrincipal(
                        principal="telegram:999",
                        provider=self.name,
                        scopes=("dashboard:read", "dashboard:admin"),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubMiniAppProvider())
        # Call the REAL production registration function, not a hand-copied
        # approximation of its parameters — a test that duplicates
        # required=False/methods=("GET","HEAD") by hand would keep passing
        # even if the real plugin's registration later drifted (e.g. to
        # required=True, or methods including POST), silently stopping this
        # test from proving anything about the actual token-route seam.
        from plugins.dashboard_auth.telegram_miniapp import _register_miniapp_token_routes

        _register_miniapp_token_routes()

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield

        clear_providers()
        token_auth.clear_token_routes()

    def test_non_admin_miniapp_token_caller_does_not_see_gateway_start_time(self):
        resp = self.client.get("/api/status", headers={"authorization": "Bearer own-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert "gateway_start_time" not in body
        assert "telegram_allowlist_updated_at" not in body
        assert "gateway_pid" not in body  # gated bind: withheld from everyone, own gate

    def test_admin_miniapp_token_caller_sees_gateway_start_time(self):
        """The one case where the field appears under auth_required=True --
        the production condition this feature actually needs to satisfy.
        """
        resp = self.client.get("/api/status", headers={"authorization": "Bearer admin-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["gateway_start_time"] == 1234567890
        assert isinstance(body["telegram_allowlist_updated_at"], int)
        # gateway_pid stays withheld regardless of admin status -- its own
        # gate (`if not auth_required:`) is unconditional on a gated bind,
        # by design, for a different reason (external liveness-probe safety,
        # not admin-tier). Confirms the two fields' independent gates didn't
        # bleed into each other while fixing one of them.
        assert "gateway_pid" not in body

    def test_cookie_session_caller_does_not_see_gateway_start_time(self):
        """A legacy session-token header does not unlock the gated field.

        A caller presenting the legacy loopback session-token header still
        gets NOTHING here on a gated bind: /api/status's public-path skip in
        gated_auth_middleware runs before any credential is even inspected,
        so this header is not the mechanism that would matter even if it
        were the real gated-deployment session mechanism (it isn't -- that's
        cookies verified by gated_auth_middleware, which never runs for this
        path either). There is no way to prove "cookie-authenticated" on
        this endpoint under a gated bind; only a verified admin Mini App
        token can.
        """
        resp = self.session_client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "gateway_start_time" not in body
        assert "telegram_allowlist_updated_at" not in body
        assert "gateway_pid" not in body

    def test_fully_anonymous_caller_does_not_see_gateway_start_time(self):
        """The property that actually matters: on a gated bind, a caller
        presenting zero credentials at all (the exact shape of an external
        uptime-probe or a random internet visitor -- /api/status is in
        PUBLIC_API_PATHS and reachable by anyone regardless of auth_required)
        must not learn the gateway's process start time. Indistinguishable
        from the cookie-session case above at this specific endpoint (see
        class docstring) -- both must be denied, and both are, by the same
        `requester_scope == "admin" or not auth_required` condition.
        """
        resp = self.client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "gateway_start_time" not in body
        assert "telegram_allowlist_updated_at" not in body
        assert "gateway_pid" not in body

    def test_telegram_allowlist_updated_at_advances_after_put_api_env_no_restart(self, monkeypatch):
        """The load-bearing case for this field's whole purpose: editing
        TELEGRAM_ALLOWED_USERS via the REAL PUT /api/env endpoint (not a
        direct file write -- exercising the actual write path) must advance telegram_allowlist_updated_at past the
        gateway's (unchanged, still-mocked) start_time -- with zero gateway
        restart anywhere in this test. gateway_start_time stays pinned at
        1234567890 throughout (mocked read_runtime_status never changes);
        only the .env write moves telegram_allowlist_updated_at, and it
        moves to "now" (2026, a much larger epoch value), proving the
        comparison the frontend's needs_restart banner depends on
        (telegram_allowlist_updated_at > gateway_start_time) actually holds
        the instant an operator edits the allowlist, no restart required to
        observe it.

        PUT /api/env is NOT a Mini App token route (it's absent
        from _register_miniapp_token_routes()'s allowlist) -- it's the
        desktop dashboard operator's own cookie-gated endpoint, editing env
        vars is not something the Mini App can do at all. So this test
        cannot use the admin bearer token for the PUT call the way every
        other test in this class does for GET /api/status -- it needs a
        genuine cookie-session caller instead. /api/env is NOT in
        PUBLIC_API_PATHS (unlike /api/status), so gated_auth_middleware
        actually attempts real cookie verification for it under
        auth_required=True; faking that verification (rather than the
        legacy _SESSION_TOKEN header, which only ever mattered for the
        loopback gate that auth_required=True bypasses) is what makes this
        PUT reach the handler at all.
        """
        from types import SimpleNamespace

        async def _fake_gated_auth_middleware(request, call_next):
            request.state.session = SimpleNamespace(user_id="test-operator")
            return await call_next(request)

        monkeypatch.setattr(
            "hermes_cli.dashboard_auth.middleware.gated_auth_middleware",
            _fake_gated_auth_middleware,
        )

        admin_headers = {"authorization": "Bearer admin-token"}

        before = self.client.get("/api/status", headers=admin_headers).json()
        assert before["gateway_start_time"] == 1234567890

        put_resp = self.client.put(
            "/api/env",
            json={"key": "TELEGRAM_ALLOWED_USERS", "value": "42,99"},
        )
        assert put_resp.status_code == 200

        after = self.client.get("/api/status", headers=admin_headers).json()
        # No restart happened anywhere in this test -- same TestClient/app,
        # same mocked runtime-status lambda -- so gateway_start_time is
        # bit-for-bit unchanged.
        assert after["gateway_start_time"] == 1234567890
        assert after["telegram_allowlist_updated_at"] > after["gateway_start_time"]
        # is_paired_or_allowlisted picks up this exact write with zero dashboard restart either.
        from plugins.dashboard_auth.telegram_miniapp.tiers import is_paired_or_allowlisted

        pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
        assert is_paired_or_allowlisted("42", pairing_store=pairing_store) is True
        assert is_paired_or_allowlisted("7", pairing_store=pairing_store) is False

    def test_paired_non_admin_token_gets_neither_field_on_loopback_no_gate_bind(
        self, monkeypatch
    ):
        """The gap this whole class exists to prevent, closed: an earlier
        version of get_status()'s gate was
        `requester_scope == "admin" or not auth_required` -- on a loopback/
        no-gate bind (auth_required=False), that `or` unconditionally leaked
        both admin-tier fields to a PAIRED, NON-ADMIN Mini App token too,
        since `not auth_required` doesn't care what requester_scope actually
        is. This is the test that would have caught that originally and
        didn't exist: override this class's auth_required=True pin
        specifically to False (the exact condition the leak needed), use the
        non-admin "own-token", and confirm both fields are absent. The gate
        is now requester_scope == "admin" alone, no auth_required branch, so
        this must pass regardless of bind type.
        """
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        resp = self.client.get("/api/status", headers={"authorization": "Bearer own-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert "gateway_start_time" not in body
        assert "telegram_allowlist_updated_at" not in body

    def test_admin_token_still_sees_both_fields_on_loopback_no_gate_bind(self, monkeypatch):
        """The other half of the same fix: dropping the `or not
        auth_required` branch must not ALSO accidentally deny admin -- the
        sole remaining gate (requester_scope == "admin") is bind-type
        independent by construction, but confirm it directly rather than
        just inferring it from the code.
        """
        from hermes_cli.web_server import app

        monkeypatch.setattr(app.state, "auth_required", False, raising=False)
        resp = self.client.get("/api/status", headers={"authorization": "Bearer admin-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["gateway_start_time"] == 1234567890
        assert body["telegram_allowlist_updated_at"] is not None


class TestSessionDetailAndMessagesOwnership:
    """GET /api/sessions/{id} and .../messages: the per-row ownership
    check (_enforce_session_ownership), reusing _dashboard_requester_scope
    for trust classification exactly as GET /api/sessions does.

    This registers the *actual production* _SESSION_ID_RE/_SESSION_MESSAGES_RE
    patterns (imported straight from plugins.dashboard_auth.telegram_miniapp)
    with required=False, so it exercises the same seam configuration that
    ships in production rather than a test-only shortcut.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        class _StubMiniAppProvider(DashboardAuthProvider):
            name = "telegram-miniapp"
            display_name = "Test Telegram Mini App"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "own-token":
                    return TokenPrincipal(
                        principal="telegram:12345",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                if token == "admin-token":
                    return TokenPrincipal(
                        principal="telegram:999",
                        provider=self.name,
                        scopes=("dashboard:read", "dashboard:admin"),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubMiniAppProvider())
        from plugins.dashboard_auth.telegram_miniapp import _SESSION_ID_RE, _SESSION_MESSAGES_RE

        token_auth.register_token_route(_SESSION_ID_RE, match="regex", required=False)
        token_auth.register_token_route(_SESSION_MESSAGES_RE, match="regex", required=False)

        db = hermes_state.SessionDB()
        try:
            db.create_session(
                session_id="20260702_100000_aaaaaaaa",
                source="telegram",
                user_id="12345",
                chat_id="12345",
                chat_type="dm",
            )
            db.append_message(session_id="20260702_100000_aaaaaaaa", role="user", content="hi own")
            db.create_session(
                session_id="20260702_110000_bbbbbbbb",
                source="telegram",
                user_id="67890",
                chat_id="67890",
                chat_type="dm",
            )
            db.append_message(session_id="20260702_110000_bbbbbbbb", role="user", content="hi other")
            # Cron-run shape (cron/scheduler.py's run_job:
            # "cron_{job_id}_{timestamp}", job_id = uuid4().hex[:12]) --
            # distinct from the ordinary interactive-session id shape above.
            # source="cron" so session_row_is_own_dm never matches it for a
            # non-admin token regardless of chat_id (a scheduled job isn't
            # "owned" by any one Telegram DM) -- only the admin token and the
            # cookie-authenticated desktop caller should reach it.
            db.create_session(
                session_id="cron_abcdef012345_20260707_093920",
                source="cron",
                chat_id="12345",
                chat_type="dm",
            )
            db.append_message(
                session_id="cron_abcdef012345_20260707_093920", role="user", content="cron run"
            )
        finally:
            db.close()

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield

        clear_providers()
        token_auth.clear_token_routes()

    # ---- detail ----

    def test_cookie_caller_can_see_any_session_detail(self):
        resp = self.session_client.get("/api/sessions/20260702_110000_bbbbbbbb")
        assert resp.status_code == 200
        assert resp.json()["id"] == "20260702_110000_bbbbbbbb"

    def test_admin_token_caller_can_see_any_session_detail(self):
        resp = self.client.get(
            "/api/sessions/20260702_110000_bbbbbbbb", headers={"authorization": "Bearer admin-token"}
        )
        assert resp.status_code == 200

    def test_own_token_caller_can_see_own_session_detail(self):
        resp = self.client.get(
            "/api/sessions/20260702_100000_aaaaaaaa", headers={"authorization": "Bearer own-token"}
        )
        assert resp.status_code == 200

    def test_own_token_caller_cannot_see_other_users_session_detail(self):
        resp = self.client.get(
            "/api/sessions/20260702_110000_bbbbbbbb", headers={"authorization": "Bearer own-token"}
        )
        assert resp.status_code == 404

    def test_own_token_caller_gets_same_404_for_nonexistent_session(self):
        # The IDOR-hardening property: an "exists but not yours" response
        # must be indistinguishable from "doesn't exist" at all.
        owned_absent = self.client.get(
            "/api/sessions/20260702_999999_ffffffff", headers={"authorization": "Bearer own-token"}
        )
        owned_mismatch = self.client.get(
            "/api/sessions/20260702_110000_bbbbbbbb", headers={"authorization": "Bearer own-token"}
        )
        assert owned_absent.status_code == owned_mismatch.status_code == 404
        assert owned_absent.json() == owned_mismatch.json()

    # ---- messages ----

    def test_cookie_caller_can_see_any_session_messages(self):
        resp = self.session_client.get("/api/sessions/20260702_110000_bbbbbbbb/messages")
        assert resp.status_code == 200

    def test_admin_token_caller_can_see_any_session_messages(self):
        resp = self.client.get(
            "/api/sessions/20260702_110000_bbbbbbbb/messages",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 200

    def test_own_token_caller_can_see_own_session_messages(self):
        resp = self.client.get(
            "/api/sessions/20260702_100000_aaaaaaaa/messages",
            headers={"authorization": "Bearer own-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["messages"]

    def test_own_token_caller_cannot_see_other_users_session_messages(self):
        resp = self.client.get(
            "/api/sessions/20260702_110000_bbbbbbbb/messages",
            headers={"authorization": "Bearer own-token"},
        )
        assert resp.status_code == 404

    # ---- cron-shaped session id (regression: this shape used to fail
    # _SESSION_ID_RE's fullmatch entirely, falling through to the downstream
    # cookie gate, which 401'd a Mini App token caller -- surfaced as an
    # instant, incorrect "session expired" screen for any cron/scheduled
    # session opened from the Mini App) ----

    def test_admin_token_caller_can_see_cron_session_detail(self):
        resp = self.client.get(
            "/api/sessions/cron_abcdef012345_20260707_093920",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "cron_abcdef012345_20260707_093920"

    def test_admin_token_caller_can_see_cron_session_messages(self):
        resp = self.client.get(
            "/api/sessions/cron_abcdef012345_20260707_093920/messages",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["messages"]

    def test_cookie_caller_can_see_cron_session_detail(self):
        resp = self.session_client.get("/api/sessions/cron_abcdef012345_20260707_093920")
        assert resp.status_code == 200

    def test_own_token_caller_cannot_see_cron_session_detail(self):
        # Not a regression from broadening the id shape: source="cron" never
        # satisfies session_row_is_own_dm, so this stays 404 for a non-admin
        # token exactly as it would for any other not-mine session.
        resp = self.client.get(
            "/api/sessions/cron_abcdef012345_20260707_093920",
            headers={"authorization": "Bearer own-token"},
        )
        assert resp.status_code == 404


class TestSessionResumeEndpoint:
    """POST /api/sessions/{id}/resume: writes a gateway/resume_control.py
    marker rather than mutating any live gateway state directly (see that
    module's docstring for why) -- admin-tier only, same shape as the other
    instance-wide mutating actions (_require_dashboard_admin), not the
    per-row ownership check GET/DELETE use.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        self.home = get_hermes_home()

        class _StubMiniAppProvider(DashboardAuthProvider):
            name = "telegram-miniapp"
            display_name = "Test Telegram Mini App"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "own-token":
                    return TokenPrincipal(
                        principal="telegram:12345", provider=self.name, scopes=("dashboard:read",)
                    )
                if token == "admin-token":
                    return TokenPrincipal(
                        principal="telegram:999",
                        provider=self.name,
                        scopes=("dashboard:read", "dashboard:admin"),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubMiniAppProvider())
        from plugins.dashboard_auth.telegram_miniapp import _SESSION_RESUME_RE

        token_auth.register_token_route(_SESSION_RESUME_RE, match="regex", required=False, methods=("POST",))

        db = hermes_state.SessionDB()
        try:
            db.create_session(
                session_id="20260702_100000_aaaaaaaa",
                source="telegram",
                user_id="12345",
                chat_id="12345",
                chat_type="dm",
            )
            db.append_message(session_id="20260702_100000_aaaaaaaa", role="user", content="hi")
            db.create_session(
                session_id="cron_abcdef012345_20260707_093920",
                source="cron",
                chat_id="12345",
                chat_type="dm",
            )
        finally:
            db.close()

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

        yield

        clear_providers()
        token_auth.clear_token_routes()

    def test_non_admin_token_cannot_resume_session(self):
        resp = self.client.post(
            "/api/sessions/20260702_100000_aaaaaaaa/resume",
            headers={"authorization": "Bearer own-token"},
        )
        assert resp.status_code == 403

    def test_admin_token_can_resume_own_dm_session(self):
        resp = self.client.post(
            "/api/sessions/20260702_100000_aaaaaaaa/resume",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["target_session_id"] == "20260702_100000_aaaaaaaa"
        assert body["session_key"] == "agent:main:telegram:dm:12345"

        import gateway.resume_control as rc
        pending = rc.pending_resume_requests(home=self.home)
        assert pending == {"agent:main:telegram:dm:12345": "20260702_100000_aaaaaaaa"}

    def test_cookie_caller_can_resume_session(self):
        resp = self.session_client.post("/api/sessions/20260702_100000_aaaaaaaa/resume")
        assert resp.status_code == 200

    def test_admin_token_cannot_resume_non_telegram_session(self):
        resp = self.client.post(
            "/api/sessions/cron_abcdef012345_20260707_093920/resume",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 400

    def test_admin_token_resume_nonexistent_session_is_404(self):
        resp = self.client.post(
            "/api/sessions/20260702_999999_ffffffff/resume",
            headers={"authorization": "Bearer admin-token"},
        )
        assert resp.status_code == 404


class TestMiniAppAdminOnlyMutatingEndpoints:
    """Every new mutating endpoint the Mini App needs
    (_require_dashboard_admin, hermes_cli/web_server.py): cron
    pause/resume/trigger, skills toggle, session archive/delete, the
    Telegram allowlist, gateway restart, hermes update, and the
    /api/plugins/telegram-miniapp/me tier-detection endpoint.

    Calls the REAL _register_miniapp_token_routes() (not a hand-copied
    approximation of its registrations) so this exercises the actual
    production route set, same discipline as
    TestSessionDetailAndMessagesOwnership above.

    The property under test throughout: a non-admin paired Mini App token
    must get 403 from the HANDLER (not just "dispatch not registered" --
    these routes ARE registered, since the admin tier needs to reach them;
    the gate is _require_dashboard_admin running inside the handler body).
    A cookie/session caller and an admin-tier token must both pass.
    """

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_cli.dashboard_auth import (
            DashboardAuthProvider,
            LoginStart,
            TokenPrincipal,
            clear_providers,
            register_provider,
        )
        from hermes_cli.dashboard_auth import token_auth

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        class _StubMiniAppProvider(DashboardAuthProvider):
            name = "telegram-miniapp"
            display_name = "Test Telegram Mini App"
            supports_token = True

            def start_login(self, *, redirect_uri):
                return LoginStart(redirect_url="x", cookie_payload={})

            def complete_login(self, *, code, state, code_verifier, redirect_uri):
                raise NotImplementedError

            def verify_session(self, *, access_token):
                return None

            def refresh_session(self, *, refresh_token):
                raise NotImplementedError

            def revoke_session(self, *, refresh_token):
                return None

            def verify_token(self, *, token):
                if token == "own-token":
                    return TokenPrincipal(
                        principal="telegram:12345",
                        provider=self.name,
                        scopes=("dashboard:read",),
                    )
                if token == "admin-token":
                    return TokenPrincipal(
                        principal="telegram:999",
                        provider=self.name,
                        scopes=("dashboard:read", "dashboard:admin"),
                    )
                return None

        clear_providers()
        token_auth.clear_token_routes()
        register_provider(_StubMiniAppProvider())
        from plugins.dashboard_auth.telegram_miniapp import _register_miniapp_token_routes

        _register_miniapp_token_routes()

        self.client = TestClient(app)
        self.session_client = TestClient(app)
        self.session_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self.own = {"authorization": "Bearer own-token"}
        self.admin = {"authorization": "Bearer admin-token"}

        yield

        clear_providers()
        token_auth.clear_token_routes()

    # ---- cron pause/resume/trigger ----

    def test_non_admin_token_cannot_pause_cron_job(self):
        resp = self.client.post("/api/cron/jobs/abcdef123456/pause", headers=self.own)
        assert resp.status_code == 403

    def test_non_admin_token_cannot_resume_cron_job(self):
        resp = self.client.post("/api/cron/jobs/abcdef123456/resume", headers=self.own)
        assert resp.status_code == 403

    def test_non_admin_token_cannot_trigger_cron_job(self):
        resp = self.client.post("/api/cron/jobs/abcdef123456/trigger", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_reaches_cron_pause_handler(self):
        # The job doesn't exist, so this 404s past the gate -- proving the
        # 403 above came from the admin gate specifically, not from the job
        # lookup also legitimately 404ing for a bogus id either way.
        resp = self.client.post("/api/cron/jobs/abcdef123456/pause", headers=self.admin)
        assert resp.status_code == 404

    def test_cookie_caller_reaches_cron_pause_handler(self):
        resp = self.session_client.post("/api/cron/jobs/abcdef123456/pause")
        assert resp.status_code == 404

    # ---- cron delete ----

    def test_non_admin_token_cannot_delete_cron_job(self):
        resp = self.client.delete("/api/cron/jobs/abcdef123456", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_reaches_cron_delete_handler(self):
        # Same proof shape as the pause case above: a bogus job id 404s past
        # the admin gate rather than the request never reaching the handler.
        resp = self.client.delete("/api/cron/jobs/abcdef123456", headers=self.admin)
        assert resp.status_code == 404

    def test_cookie_caller_reaches_cron_delete_handler(self):
        resp = self.session_client.delete("/api/cron/jobs/abcdef123456")
        assert resp.status_code == 404

    # ---- skills toggle ----

    def test_non_admin_token_cannot_toggle_skill(self):
        resp = self.client.put(
            "/api/skills/toggle",
            headers=self.own,
            json={"name": "deep-research", "enabled": False},
        )
        assert resp.status_code == 403

    def test_admin_token_can_toggle_skill(self, monkeypatch):
        import hermes_cli.web_server as web_server

        monkeypatch.setattr(web_server, "load_config", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.skills_config.get_disabled_skills", lambda cfg: set()
        )
        monkeypatch.setattr(
            "hermes_cli.skills_config.save_disabled_skills", lambda cfg, disabled: None
        )
        resp = self.client.put(
            "/api/skills/toggle",
            headers=self.admin,
            json={"name": "deep-research", "enabled": False},
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "name": "deep-research", "enabled": False}

    def test_non_admin_token_cannot_read_skill_content(self):
        """Direct-route proof, not a "button hidden" assertion: a paired
        (non-admin) Mini App token must get 403 from the HANDLER when it
        hits GET /api/skills/content itself, regardless of what the
        frontend renders. The skill name doesn't need to exist for this --
        _require_dashboard_admin must reject before _find_skill ever runs.
        """
        resp = self.client.get(
            "/api/skills/content", params={"name": "deep-research"}, headers=self.own
        )
        assert resp.status_code == 403

    def test_admin_token_can_read_skill_content(self, monkeypatch):
        from tools import skill_manager_tool

        monkeypatch.setattr(
            skill_manager_tool,
            "_find_skill",
            lambda name: {"path": Path("/nonexistent/skill/dir")},
        )
        resp = self.client.get(
            "/api/skills/content", params={"name": "deep-research"}, headers=self.admin
        )
        # Reaches the real handler past the admin gate -- 404 here because
        # the patched path has no SKILL.md, proving the gate (not the skill
        # lookup) is what rejected the non-admin case above.
        assert resp.status_code == 404

    def test_cookie_caller_can_read_skill_content(self, monkeypatch):
        from tools import skill_manager_tool

        monkeypatch.setattr(
            skill_manager_tool,
            "_find_skill",
            lambda name: {"path": Path("/nonexistent/skill/dir")},
        )
        resp = self.session_client.get("/api/skills/content", params={"name": "deep-research"})
        assert resp.status_code == 404

    # ---- session archive / delete ----

    def test_non_admin_token_cannot_archive_session(self):
        resp = self.client.patch(
            "/api/sessions/20260702_100000_aaaaaaaa",
            headers=self.own,
            json={"archived": True},
        )
        assert resp.status_code == 403

    def test_non_admin_token_cannot_delete_session(self):
        resp = self.client.delete("/api/sessions/20260702_100000_aaaaaaaa", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_reaches_session_archive_handler(self):
        # Session doesn't exist in this test's DB either -- 404 past the
        # gate, same proof shape as the cron case above.
        resp = self.client.patch(
            "/api/sessions/20260702_100000_aaaaaaaa",
            headers=self.admin,
            json={"archived": True},
        )
        assert resp.status_code == 404

    def test_admin_token_can_delete_nonexistent_session_idempotently(self):
        # delete_session_endpoint's own contract: absent id -> ok, already_absent.
        resp = self.client.delete("/api/sessions/20260702_100000_aaaaaaaa", headers=self.admin)
        assert resp.status_code == 200
        assert resp.json()["already_absent"] is True

    # ---- telegram allowlist ----

    def test_non_admin_token_cannot_read_allowlist(self):
        resp = self.client.get("/api/plugins/telegram-miniapp/allowlist", headers=self.own)
        assert resp.status_code == 403

    def test_non_admin_token_cannot_add_to_allowlist(self):
        resp = self.client.post(
            "/api/plugins/telegram-miniapp/allowlist", headers=self.own, json={"user_id": "555001122"}
        )
        assert resp.status_code == 403

    def test_non_admin_token_cannot_remove_from_allowlist(self):
        resp = self.client.delete("/api/plugins/telegram-miniapp/allowlist/555001122", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_can_add_and_read_allowlist(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        add = self.client.post(
            "/api/plugins/telegram-miniapp/allowlist", headers=self.admin, json={"user_id": "555001122"}
        )
        assert add.status_code == 200
        assert add.json() == {"ok": True, "user_id": "555001122"}

        read = self.client.get("/api/plugins/telegram-miniapp/allowlist", headers=self.admin)
        assert read.status_code == 200
        ids = {e["user_id"] for e in read.json()["allowlist"]}
        assert "555001122" in ids

    def test_admin_add_rejects_non_numeric_id(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        resp = self.client.post(
            "/api/plugins/telegram-miniapp/allowlist", headers=self.admin, json={"user_id": "not-a-number"}
        )
        assert resp.status_code == 400

    def test_admin_can_add_when_allowlist_currently_unset(self, monkeypatch):
        """The deliberate divergence from gateway/pairing.py's
        _sync_allowlist_add: that helper no-ops when the var is empty (to
        avoid a passive pairing approval silently locking an open gateway).
        This is an explicit admin action from the Users tab -- it must
        always take effect, empty-var case included.
        """
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        resp = self.client.post(
            "/api/plugins/telegram-miniapp/allowlist", headers=self.admin, json={"user_id": "555001122"}
        )
        assert resp.status_code == 200
        assert os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip() != ""

    def test_admin_can_remove_from_allowlist(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "555001122,42")
        resp = self.client.delete("/api/plugins/telegram-miniapp/allowlist/555001122", headers=self.admin)
        assert resp.status_code == 200
        remaining = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        assert "555001122" not in remaining.split(",")
        assert "42" in remaining.split(",")

    def test_allowlist_merges_pairing_store_and_env_only_entries(self, monkeypatch, tmp_path):
        import gateway.pairing as pairing_mod
        from gateway.pairing import PairingStore

        # gateway.pairing.PAIRING_DIR is a module-level constant captured at
        # import time from whichever HERMES_HOME was active then -- the
        # _isolate_hermes_home fixture's per-test HERMES_HOME redirection
        # doesn't retroactively move it (see
        # tests/gateway/test_internal_event_bypass_pairing.py for the same
        # note). Without this override, PairingStore() below writes into
        # the real, live ~/.hermes/platforms/pairing/telegram-approved.json
        # instead of an isolated tmp dir.
        monkeypatch.setattr(pairing_mod, "PAIRING_DIR", tmp_path)

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "88121904")
        store = PairingStore()
        with store._lock:
            store._approve_user("telegram", "224918330", user_name="Dana Kim")

        resp = self.client.get("/api/plugins/telegram-miniapp/allowlist", headers=self.admin)
        assert resp.status_code == 200
        by_id = {e["user_id"]: e for e in resp.json()["allowlist"]}
        assert by_id["224918330"]["source"] == "pairing"
        assert by_id["224918330"]["name"] == "Dana Kim"
        assert by_id["88121904"]["source"] == "env"
        assert by_id["88121904"]["name"] is None

    # ---- gateway restart / hermes update ----

    def test_non_admin_token_cannot_restart_gateway(self):
        resp = self.client.post("/api/gateway/restart", headers=self.own)
        assert resp.status_code == 403

    def test_non_admin_token_cannot_update_hermes(self):
        resp = self.client.post("/api/hermes/update", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_can_restart_gateway(self, monkeypatch):
        import hermes_cli.web_server as web_server

        class _Proc:
            pid = 4242

            def poll(self):
                return None

        _web_server_gateway._ACTION_PROCS.pop("gateway-restart", None)
        _web_server_gateway._ACTION_COMMANDS.pop("gateway-restart", None)
        # restart_gateway (hermes_cli/web_routers/*) resolves this via a late-binding proxy to its
        # OWN module, not hermes_cli.web_server -- patching the web_server compat-shim name is a
        # no-op for the router's actual lookup. See hermes_cli/web_deps.py's late()/_server(), and
        # test_admin_token_can_update_hermes below for the same fix on a sibling action endpoint.
        monkeypatch.setattr(
            "hermes_cli.web_server_gateway._spawn_hermes_action", lambda *_a, **_kw: _Proc())

        resp = self.client.post("/api/gateway/restart", headers=self.admin)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_admin_token_can_update_hermes(self, monkeypatch):
        import hermes_cli.web_server as web_server

        class _Proc:
            pid = 4343

            def poll(self):
                return None

        _web_server_gateway._ACTION_PROCS.pop("hermes-update", None)
        _web_server_gateway._ACTION_RESULTS.pop("hermes-update", None)
        # update_hermes (hermes_cli/web_routers/actions.py) resolves both of
        # these via late-binding proxies to their OWN modules, not
        # hermes_cli.web_server -- patching the web_server compat-shim name
        # is a no-op for the router's actual lookup. See
        # hermes_cli/web_deps.py's late()/_server().
        monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda _root: "git")
        monkeypatch.setattr(web_server.secrets, "token_hex", lambda _size: "c" * 32)
        monkeypatch.setattr(
            "hermes_cli.web_server_gateway._spawn_hermes_action", lambda *_a, **_kw: _Proc())

        resp = self.client.post("/api/hermes/update", headers=self.admin)
        assert resp.status_code == 200
        assert resp.json() == {
            "ok": True,
            "pid": 4343,
            "name": "hermes-update",
            "action_id": "c" * 32,
        }

    # ---- /api/plugins/telegram-miniapp/me ----

    def test_miniapp_me_reports_admin_for_admin_token(self):
        resp = self.client.get("/api/plugins/telegram-miniapp/me", headers=self.admin)
        assert resp.status_code == 200
        assert resp.json() == {"tier": "admin", "user_id": None}

    def test_miniapp_me_reports_paired_for_own_token(self):
        resp = self.client.get("/api/plugins/telegram-miniapp/me", headers=self.own)
        assert resp.status_code == 200
        assert resp.json() == {"tier": "paired", "user_id": "12345"}

    def test_miniapp_me_reports_admin_for_cookie_caller(self):
        resp = self.session_client.get("/api/plugins/telegram-miniapp/me")
        assert resp.status_code == 200
        assert resp.json() == {"tier": "admin", "user_id": None}

    def test_miniapp_me_401s_for_no_credential_at_all(self):
        """Unlike /api/status, /api/plugins/telegram-miniapp/me is NOT in PUBLIC_API_PATHS --
        a request with no bearer token AND no cookie never reaches this
        handler at all; the legacy session-token gate (auth_middleware,
        active regardless of auth_required) rejects it first with its own
        401. A 200 + {"tier": None} is only reachable
        when a token IS presented but resolves to a non-Telegram-miniapp
        principal with no derivable id (see
        test_non_telegram_miniapp_token_principal_is_own_scoped_with_no_id
        in TestDashboardRequesterScopeAuthPathAwareness for that branch) --
        not the "no token presented" case this test actually exercises.
        """
        resp = self.client.get("/api/plugins/telegram-miniapp/me")
        assert resp.status_code == 401

    # ---- sessions search ----

    def test_non_admin_token_cannot_search_sessions(self):
        resp = self.client.get("/api/sessions/search", params={"q": "x"}, headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_can_search_sessions(self):
        resp = self.client.get("/api/sessions/search", params={"q": "x"}, headers=self.admin)
        assert resp.status_code == 200

    def test_cookie_caller_can_search_sessions(self):
        resp = self.session_client.get("/api/sessions/search", params={"q": "x"})
        assert resp.status_code == 200

    # ---- logs ----
    # Reuses the pre-existing GET /api/logs (hermes_cli/logs.py's LOG_FILES +
    # _read_tail) the desktop dashboard already has -- the Mini App just
    # gains admin-tier dispatch access to the SAME endpoint, not a
    # parallel one. get_hermes_home() is already overridden by
    # _isolate_hermes_home, and get_logs() re-resolves it per call (no
    # module-level path caching to fight), so no extra monkeypatching here.

    def test_non_admin_token_cannot_list_logs(self):
        resp = self.client.get("/api/logs", headers=self.own)
        assert resp.status_code == 403

    def test_admin_token_can_read_agent_log(self):
        from hermes_constants import get_hermes_home
        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "agent.log").write_text("2026-07-07 12:00:00 INFO hello\n")

        resp = self.client.get("/api/logs", params={"file": "agent"}, headers=self.admin)
        assert resp.status_code == 200
        assert resp.json()["lines"]

    def test_admin_token_can_read_new_update_log_key(self):
        # The Mini App's admin log viewer surfaces update.log next to
        # agent/errors/gateway.
        from hermes_constants import get_hermes_home
        log_dir = get_hermes_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "update.log").write_text("=== hermes update started ===\n")

        resp = self.client.get("/api/logs", params={"file": "update"}, headers=self.admin)
        assert resp.status_code == 200
        assert resp.json()["lines"]

    def test_cookie_caller_can_list_logs(self):
        resp = self.session_client.get("/api/logs")
        assert resp.status_code == 200

    def test_admin_token_unknown_file_is_400(self):
        resp = self.client.get("/api/logs", params={"file": "does-not-exist"}, headers=self.admin)
        assert resp.status_code == 400
