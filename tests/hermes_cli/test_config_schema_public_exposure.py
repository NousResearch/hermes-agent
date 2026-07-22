"""``/api/config/schema`` is public — it must not leak config.yaml contents.

The route sits on ``PUBLIC_API_PATHS`` so the SPA can render the Config page
shell before login. That allowlist's contract is explicit: every entry has to
be safe to hand to "anyone who happens to curl the hostname", because a
wildcard-subdomain deployment is publicly reachable (the same reachability
that makes ``/api/status`` the portal's liveness probe).

The static ``CONFIG_SCHEMA`` satisfies that. The per-request voice-provider
merge does not: it reads ``tts.providers.*`` / ``stt.providers.*`` out of the
operator's config.yaml, and those keys are user data — internal vendor and
host identifiers — not schema defaults. Once the options became config-derived
an anonymous caller could enumerate them.

Pinned here: anonymous callers get the plain schema, authenticated callers
still get the enriched options, and the anonymous response stays usable so the
pre-login SPA is unaffected.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(monkeypatch, tmp_path):
    starlette = pytest.importorskip("starlette.testclient")
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "tts:\n"
        "  providers:\n"
        "    mycompany-voice:\n"
        "      type: command\n"
        "      command: '/opt/acme/bin/tts'\n"
        "stt:\n"
        "  providers:\n"
        "    internal-stt:\n"
        "      type: command\n"
        "      command: '/opt/acme/bin/stt'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()
    from hermes_cli.web_server import app

    return starlette.TestClient(app)


def _token():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    return {_SESSION_HEADER_NAME: _SESSION_TOKEN}


CUSTOM_NAMES = ("mycompany-voice", "internal-stt")


def test_anonymous_caller_gets_no_config_derived_options(client):
    resp = client.get("/api/config/schema")
    assert resp.status_code == 200
    leaked = [n for n in CUSTOM_NAMES if n in resp.text]
    assert not leaked, f"config.yaml provider names exposed anonymously: {leaked}"


def test_authenticated_caller_still_sees_custom_providers(client):
    resp = client.get("/api/config/schema", headers=_token())
    assert resp.status_code == 200
    options = resp.json()["fields"]["tts.provider"]["options"]
    assert "mycompany-voice" in options
    stt_options = resp.json()["fields"]["stt.provider"]["options"]
    assert "internal-stt" in stt_options


def test_legacy_bearer_header_also_counts_as_authenticated(client):
    from hermes_cli.web_server import _SESSION_TOKEN

    resp = client.get(
        "/api/config/schema",
        headers={"Authorization": f"Bearer {_SESSION_TOKEN}"},
    )
    assert resp.status_code == 200
    assert "mycompany-voice" in resp.json()["fields"]["tts.provider"]["options"]


def test_anonymous_schema_is_still_usable(client):
    """The pre-login SPA must keep rendering — this is not a 401."""
    resp = client.get("/api/config/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["fields"]) > 100
    assert "model" in data["fields"]
    assert "general" in data["category_order"]
    # The builtin voice options survive; only the config-derived ones are held back.
    assert data["fields"]["tts.provider"]["options"]


class TestRequestIsAuthenticated:
    """The helper must resolve the active scheme the same way _require_token does."""

    @staticmethod
    def _req(*, auth_required, session=None, token=None):
        from types import SimpleNamespace

        from hermes_cli.web_server import _SESSION_HEADER_NAME

        headers = {_SESSION_HEADER_NAME: token} if token else {}
        return SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(auth_required=auth_required)),
            state=SimpleNamespace(session=session),
            headers=headers,
        )

    def test_gated_mode_requires_a_verified_session(self):
        from hermes_cli.web_server import _request_is_authenticated

        assert not _request_is_authenticated(self._req(auth_required=True))
        assert _request_is_authenticated(
            self._req(auth_required=True, session=object())
        )

    def test_token_mode_requires_the_session_token(self):
        from hermes_cli.web_server import _SESSION_TOKEN, _request_is_authenticated

        assert not _request_is_authenticated(self._req(auth_required=False))
        assert not _request_is_authenticated(
            self._req(auth_required=False, token="wrong")
        )
        assert _request_is_authenticated(
            self._req(auth_required=False, token=_SESSION_TOKEN)
        )


# ---------------------------------------------------------------------------
# OAuth-gated deployment
# ---------------------------------------------------------------------------
#
# Public routes short-circuit in ``gated_auth_middleware`` BEFORE cookie
# verification, so ``request.state.session`` is never attached for them. Without
# the ``_OPTIONAL_SESSION_PUBLIC_PATHS`` opt-in a logged-in browser reads as
# anonymous and silently loses its own provider options — the enrichment would
# be dead code on exactly the deployment shape that needs the guard.


@pytest.fixture()
def gated_client(monkeypatch, tmp_path):
    starlette = pytest.importorskip("starlette.testclient")
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "tts:\n"
        "  providers:\n"
        "    mycompany-voice:\n"
        "      type: command\n"
        "      command: '/opt/acme/bin/tts'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import invalidate_env_cache

    invalidate_env_cache()

    clear_providers()
    register_provider(StubAuthProvider())
    prev = (
        getattr(web_server.app.state, "bound_host", None),
        getattr(web_server.app.state, "bound_port", None),
        getattr(web_server.app.state, "auth_required", None),
    )
    web_server.app.state.bound_host = "fly-app.fly.dev"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    client = starlette.TestClient(web_server.app, base_url="https://fly-app.fly.dev")
    yield client
    clear_providers()
    (
        web_server.app.state.bound_host,
        web_server.app.state.bound_port,
        web_server.app.state.auth_required,
    ) = prev


def _stub_login(client):
    """Walk the stub IdP round trip so the client holds session cookies."""
    r1 = client.get("/auth/login?provider=stub", follow_redirects=False)
    assert r1.status_code == 302
    state = r1.headers["location"].split("state=")[1]
    r2 = client.get(
        f"/auth/callback?code=stub_code&state={state}", follow_redirects=False
    )
    assert r2.status_code == 302
    assert any("hermes_session_at" in c for c in r2.headers.get_list("set-cookie"))


def test_gated_anonymous_caller_gets_no_config_derived_options(gated_client):
    resp = gated_client.get("/api/config/schema")
    assert resp.status_code == 200, "the route must stay public under the gate"
    assert "mycompany-voice" not in resp.text


def test_gated_caller_sees_custom_providers_after_login(gated_client):
    """The whole point of the guard: logged-in users keep their own options."""
    _stub_login(gated_client)
    resp = gated_client.get("/api/config/schema")
    assert resp.status_code == 200
    options = resp.json()["fields"]["tts.provider"]["options"]
    assert "mycompany-voice" in options, (
        "a logged-in browser was treated as anonymous — the public-route "
        "short-circuit skipped session attachment"
    )


def test_gated_invalid_cookie_still_answers_200_anonymously(gated_client):
    """A dead cookie must not turn a public route into a 401."""
    from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE

    gated_client.cookies.set(SESSION_AT_COOKIE, "garbage-not-a-real-token")
    resp = gated_client.get("/api/config/schema")
    assert resp.status_code == 200
    assert "mycompany-voice" not in resp.text
