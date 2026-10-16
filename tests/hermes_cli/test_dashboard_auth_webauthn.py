"""End-to-end tests for the dashboard WebAuthn (passkey) provider.

Covers the contracts from the plan's test list: config resolution (skip vs
fail-loud, store fail-closed), enrolment via a verified session AND via the
headless one-shot CLI token (store written 0600), assertion ceremony over the
real gated middleware (session mint -> authenticated request -> transparent
refresh - the 401_reauth harness), replay/origin/UV/signature rejection, the
shared login rate limiter, providers-list + login-page origin gating (the
LAN-http fail-inert contract), native broker tail parity, and CLI enrol/forget
behaviour with a temp HERMES_HOME.

Harness mirrors test_dashboard_auth_password_login.py: real ``gated_auth_middleware``
+ ``TestClient`` from ``hermes_cli.web_server``. The stub authenticator
(tests/hermes_cli/webauthn_helpers.py) speaks fido2's own response shapes - no
device, no network. Skips (never errors) without the ``fido2`` extra so lean CI
lanes still run the rest of the dashboard suites.
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import os
import stat
import time

import pytest
import hermes_yaml as yaml
from fastapi.testclient import TestClient

pytest.importorskip("fido2", reason="the webauthn provider needs the fido2 extra")

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth import routes as auth_routes
from hermes_cli.dashboard_auth import webauthn_state
from hermes_cli.dashboard_auth.cookies import SESSION_RT_COOKIE, WEBAUTHN_CHALLENGE_COOKIE
from hermes_cli.dashboard_auth.login_page import render_login_html
from hermes_cli.dashboard_auth.middleware import _GATE_PUBLIC_PREFIXES
from hermes_cli.dashboard_auth.routes import _PW_RATE_MAX_ATTEMPTS
from plugins.dashboard_auth._shared import SkipRegistration
from tests.hermes_cli.webauthn_helpers import StubWebAuthnAuthenticator, forwarded_headers

HTTPS_ORIGIN = "https://dash.example.com"
RP_ID = "dash.example.com"
LAN_ORIGIN = "http://192.168.1.50:9119"
_SECRET = "unit-test-secret-0123456789abcdef"


# ---------------------------------------------------------------------------
# Config writer + harness helpers (temp HERMES_HOME: resolution + store I/O)
# ---------------------------------------------------------------------------


def _write_config(home: Path, cfg: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET", _SECRET)
    return home


def _write_webauthn_config(home: Path, *, rp_id: str = RP_ID, origins=None, **extra) -> None:
    _write_config(home, {"dashboard": {"webauthn": {
        "rp_id": rp_id, "rp_name": "Hermes Test",
        "origins": origins if origins is not None else [HTTPS_ORIGIN],
        "credential_store": str(home / "dashboard_webauthn.json"),
        **extra}}})


def _settings_or_skip(home: Path):
    from plugins.dashboard_auth.webauthn import _settings

    try:
        return _settings()
    except SkipRegistration as skip:
        return skip


def _build_provider(home: Path, **kwargs_extra):
    """Resolve config, construct and register the provider."""
    from plugins.dashboard_auth.webauthn import WebauthnDashboardAuthProvider

    result = _settings_or_skip(home)
    if isinstance(result, SkipRegistration):
        raise AssertionError(f"valid config skipped: {result}")
    provider = WebauthnDashboardAuthProvider(**{**result, **kwargs_extra})
    clear_providers()
    register_provider(provider)
    return provider


@pytest.fixture()
def provider(hermes_home):
    _write_webauthn_config(hermes_home, rp_id=RP_ID, origins=[HTTPS_ORIGIN])
    built = _build_provider(hermes_home)
    webauthn_state._reset_for_tests()
    yield built
    clear_providers()
    webauthn_state._reset_for_tests()


@pytest.fixture()
def gated_client():
    app = web_server.app
    prev_required = getattr(app.state, "auth_required", None)
    app.state.auth_required = True
    client = TestClient(app, base_url=HTTPS_ORIGIN, follow_redirects=False)
    client.headers.update(forwarded_headers(HTTPS_ORIGIN))
    yield client
    app.state.auth_required = prev_required


def _lan_client() -> TestClient:
    """A request whose origin is a plain LAN http one (no secure context, no
    configured origin) - the http://<LAN-IP>:9119 direct case."""
    return TestClient(web_server.app, base_url=LAN_ORIGIN, follow_redirects=False)


def _cookie(client: TestClient, bare_name: str):
    """Read a cookie across the ``__Host-``/``__Secure-``/bare variants."""
    for variant in (f"__Host-{bare_name}", f"__Secure-{bare_name}", bare_name):
        value = client.cookies.get(variant)
        if value is not None:
            return value
    return None


def _challenge_bytes(source) -> bytes:
    """The parked challenge from the options response body (Response or dict)."""
    from fido2.utils import websafe_decode

    payload = source.get("publicKey") if isinstance(source, dict) else source.json()["publicKey"]
    return websafe_decode(payload["challenge"])


def _enrol(provider, auth: StubWebAuthnAuthenticator, label: str = "stub device") -> None:
    """Enrol the stub's credential server-side (like the CLI/headless path does)."""
    options, handle = provider.webauthn_register_options(client_ip="t")
    provider.complete_webauthn_registration(
        handle=handle, credential=auth.credential_json(
            auth.register(_challenge_bytes(options))), label=label)


def _verify_body(auth: StubWebAuthnAuthenticator, options: dict, **kw) -> dict:
    return {"credential": auth.credential_json(
        auth.assert_login(_challenge_bytes(options), **kw))}


# ---------------------------------------------------------------------------
# Config resolution contracts
# ---------------------------------------------------------------------------


class TestConfigResolution:
    def test_unconfigured_stays_off(self, hermes_home):
        (hermes_home / "config.yaml").write_text("", encoding="utf-8")
        result = _settings_or_skip(hermes_home)
        assert isinstance(result, SkipRegistration)
        assert "rp_id and rp_name are both required" in str(result)  # fail-loud message explains BOTH knobs

    def test_valid_config_resolves(self, hermes_home):
        _write_webauthn_config(hermes_home)
        settings = _settings_or_skip(hermes_home)
        assert not isinstance(settings, SkipRegistration), settings
        assert settings["origins"] == (HTTPS_ORIGIN,)
        assert settings["rp_id"] == RP_ID
        assert settings["user_verification"] == "preferred"
        assert settings["challenge_ttl_seconds"] == 120
        assert isinstance(settings["secret"], bytes) and len(settings["secret"]) >= 16

    def test_origin_rp_id_mismatch_fails_loud(self, hermes_home):
        _write_webauthn_config(hermes_home, origins=["https://other.example.org"])
        result = _settings_or_skip(hermes_home)
        assert isinstance(result, SkipRegistration)
        assert "does not pair" in str(result)

    def test_lan_http_origin_fails_loud_at_config_time(self, hermes_home):
        _write_webauthn_config(hermes_home, origins=[LAN_ORIGIN])
        assert isinstance(_settings_or_skip(hermes_home), SkipRegistration)

    def test_localhost_http_origin_is_valid(self, hermes_home):
        _write_webauthn_config(hermes_home, rp_id="localhost",
                               origins=["http://localhost:9119"])
        settings = _settings_or_skip(hermes_home)
        assert not isinstance(settings, SkipRegistration)
        assert settings["origins"] == ("http://localhost:9119",)

    def test_unknown_user_verification_fails_loud(self, hermes_home):
        _write_webauthn_config(hermes_home, user_verification="strict")
        assert isinstance(_settings_or_skip(hermes_home), SkipRegistration)

    def test_challenge_ttl_clamped(self, hermes_home):
        _write_webauthn_config(hermes_home, challenge_ttl_seconds=99999)
        assert _settings_or_skip(hermes_home)["challenge_ttl_seconds"] == 300

    def test_corrupt_store_fails_closed(self, hermes_home):
        _write_webauthn_config(hermes_home)
        (hermes_home / "dashboard_webauthn.json").write_text("not json{", encoding="utf-8")
        assert isinstance(_settings_or_skip(hermes_home), SkipRegistration)

    def test_unknown_store_version_fails_closed(self, hermes_home):
        _write_webauthn_config(hermes_home)
        (hermes_home / "dashboard_webauthn.json").write_text(
            json.dumps({"version": 99, "credentials": []}), encoding="utf-8")
        assert isinstance(_settings_or_skip(hermes_home), SkipRegistration)

    def test_store_rp_id_mismatch_fails_closed(self, hermes_home):
        _write_webauthn_config(hermes_home)
        (hermes_home / "dashboard_webauthn.json").write_text(json.dumps(
            {"version": 1, "rp_id": "frozen.example.com", "credentials": [
                {"credential_id": "AAAA", "public_key": "BBBB", "sign_count": 0,
                 "label": "x", "created_at": "now"}]}), encoding="utf-8")
        assert isinstance(_settings_or_skip(hermes_home), SkipRegistration)

    def test_secret_rides_the_basic_auth_secret(self, hermes_home):
        _write_webauthn_config(hermes_home)
        assert _settings_or_skip(hermes_home)["secret"] == _SECRET.encode()

    def test_secret_decode_order_matches_basic_contract(self, hermes_home, monkeypatch):
        """Same decoders as basic::_resolve_secret, same order: base64 first, then
        hex, then raw text (documented contract — a hex-shaped string that is
        also valid base64 decodes as base64)."""
        monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET")
        _write_webauthn_config(hermes_home, secret="0a1b2c3d4e5f60718293a4b5c6d7e8f9")
        assert _settings_or_skip(hermes_home)["secret"] == base64.b64decode(
            "0a1b2c3d4e5f60718293a4b5c6d7e8f9")

    def test_webauthn_secret_overrides_when_basic_unset(self, hermes_home, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_BASIC_AUTH_SECRET")
        _write_webauthn_config(
            hermes_home, secret="dashboard-webauthn-secret!notb64,16+len")
        assert _settings_or_skip(hermes_home)["secret"] == \
            b"dashboard-webauthn-secret!notb64,16+len"


# ---------------------------------------------------------------------------
# Store I/O: 0600, record shape, enrolment-token contract
# ---------------------------------------------------------------------------


class TestCredentialStore:
    def test_enrolment_writes_store_0600_posix(self, hermes_home, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth, label="headless")
        store_path = hermes_home / "dashboard_webauthn.json"
        assert store_path.exists()
        if os.name != "nt":  # POSIX-only 0600 assertion (CONTRIBUTING rule 8)
            assert stat.S_IMODE(store_path.stat().st_mode) & 0o077 == 0
        doc = json.loads(store_path.read_text(encoding="utf-8"))
        assert doc["version"] == 1 and doc["rp_id"] == RP_ID
        record = doc["credentials"][0]
        assert record["label"] == "headless" and record["created_at"].endswith("Z")
        assert set(record) == {"credential_id", "public_key", "transports",
                               "sign_count", "label", "created_at"}

    def test_duplicate_credential_rejected(self, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options, handle = provider.webauthn_register_options(client_ip="t")
        from hermes_cli.dashboard_auth import InvalidCredentialsError

        with pytest.raises(InvalidCredentialsError, match="already enrolled"):
            provider.complete_webauthn_registration(
                handle=handle, credential=auth.credential_json(
                    auth.register(_challenge_bytes(options))), label="dup")

    def test_enrolment_token_single_use_and_expiry(self, provider):
        token = provider.issue_enrolment_token(now=1000)
        assert provider.peek_enrolment_token(token, now=1000 + 599) is True
        assert provider.peek_enrolment_token(token, now=1000 + 601) is False
        provider.consume_enrolment_token(token)
        assert provider.peek_enrolment_token(token, now=1000) is False

    def test_expired_challenge_fails(self, provider):
        from fido2.utils import websafe_decode

        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options, handle = provider.webauthn_login_options(client_ip="t")
        entry = webauthn_state._challenges[handle]  # noqa: SLF001 - contract check
        object.__setattr__(entry, "expires_at", int(time.time()) - 1)
        with pytest.raises(Exception):
            provider.complete_webauthn_login(
                handle=handle, credential=auth.credential_json(
                    auth.assert_login(websafe_decode(
                        options["publicKey"]["challenge"]))))


# ---------------------------------------------------------------------------
# Login ceremony E2E over the real gated middleware
# ---------------------------------------------------------------------------


class TestLoginCeremony:
    def test_full_ceremony_mints_session_cookies(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        assert options.status_code == 200
        payload = options.json()["publicKey"]
        assert payload["rpId"] == RP_ID
        assert gated_client.cookies.get(WEBAUTHN_CHALLENGE_COOKIE) or \
            _cookie(gated_client, WEBAUTHN_CHALLENGE_COOKIE)
        assert _cookie(gated_client, WEBAUTHN_CHALLENGE_COOKIE)
        verify = gated_client.post("/auth/webauthn/verify", json=_verify_body(auth, options))
        assert verify.status_code == 200, verify.text
        assert verify.json() == {"ok": True, "next": "/"}
        cookie_names = " ".join(verify.headers.get_list("set-cookie"))
        assert "hermes_session_at" in cookie_names
        assert "hermes_session_rt" in cookie_names
        # the ceremony handle is cleared on completion
        assert _cookie(gated_client, WEBAUTHN_CHALLENGE_COOKIE) is None

    def test_gated_route_accepts_passkey_session(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        assert gated_client.post(
            "/auth/webauthn/verify", json=_verify_body(auth, options)).status_code == 200
        me = gated_client.get("/api/auth/me")
        assert me.status_code == 200, me.text
        assert me.json()["provider"] == "webauthn"
        assert me.json()["user_id"] == "hermes-operator"

    def test_challenge_replay_rejected(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        body = _verify_body(auth, options)
        assert gated_client.post("/auth/webauthn/verify", json=body).status_code == 200
        assert gated_client.post("/auth/webauthn/verify", json=body).status_code == 401

    def test_wrong_signature_rejected_generic_401(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        resp = gated_client.post("/auth/webauthn/verify", json=_verify_body(
            auth, options, wrong_key=True))
        assert resp.status_code == 401
        assert "detail" in resp.json()  # generic — no credential oracle

    def test_foreign_origin_rejected(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        resp = gated_client.post("/auth/webauthn/verify", json=_verify_body(
            auth, options, origin="https://evil.example"))
        assert resp.status_code == 401

    def test_unknown_credential_rejected_generic_401(self, gated_client, provider):
        """A credential that never enrolled: fido2's Unknown-credential-ID -> the
        generic 401 (no oracle)."""
        enrolled = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, enrolled)
        stranger = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID,
                                             credential_id=b"\xff" * 32)
        options = gated_client.get("/auth/webauthn/options")
        resp = gated_client.post("/auth/webauthn/verify", json=_verify_body(stranger, options))
        assert resp.status_code == 401
        assert "detail" in resp.json()

    def test_unknown_401_when_challenge_cookie_missing(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        for variant in (f"__Host-{WEBAUTHN_CHALLENGE_COOKIE}",
                        f"__Secure-{WEBAUTHN_CHALLENGE_COOKIE}", WEBAUTHN_CHALLENGE_COOKIE):
            gated_client.cookies.delete(variant)
        resp = gated_client.post("/auth/webauthn/verify", json=_verify_body(auth, options))
        assert resp.status_code == 401

    def test_uv_required_config_rejects_uv_absent(self, hermes_home, gated_client):
        _write_webauthn_config(hermes_home, user_verification="required")
        built = _build_provider(hermes_home)
        webauthn_state._reset_for_tests()
        try:
            auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
            _enrol(built, auth)
            options = gated_client.get("/auth/webauthn/options")
            assert options.json()["publicKey"]["userVerification"] == "required"
            resp = gated_client.post("/auth/webauthn/verify", json=_verify_body(
                auth, options, uv=False))
            assert resp.status_code == 401
            options2 = gated_client.get("/auth/webauthn/options")
            assert gated_client.post(
                "/auth/webauthn/verify", json=_verify_body(auth, options2)).status_code == 200
        finally:
            clear_providers()
            webauthn_state._reset_for_tests()

    def test_empty_store_options_explain(self, gated_client, provider):
        options = gated_client.get("/auth/webauthn/options")
        assert options.status_code == 503
        assert "no passkey enrolled" in options.json()["detail"]


# ---------------------------------------------------------------------------
# Shared login rate limiter + allowlist scoping
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_webauthn_and_password_share_one_budget(self, gated_client, provider):
        auth_routes._reset_login_rate_limit()
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        # _PW_RATE_MAX_ATTEMPTS failed attempts each consume the shared budget...
        statuses = []
        for _ in range(_PW_RATE_MAX_ATTEMPTS):
            statuses.append(gated_client.post(
                "/auth/webauthn/verify", json={"credential": {}}).status_code)
        assert set(statuses) == {401}
        # ...then the ceremony route 429s...
        blocked = gated_client.post("/auth/webauthn/verify", json={"credential": {}})
        assert blocked.status_code == 429
        # ...and the PASSWORD route rides the SAME per-IP bucket (no reset between)
        pw = gated_client.post("/auth/password-login", json={
            "provider": "basic", "username": "x", "password": "y"})
        assert pw.status_code == 429
        auth_routes._reset_login_rate_limit()

    def test_unknown_provider_password_404_unaffected(self, gated_client):
        auth_routes._reset_login_rate_limit()
        resp = gated_client.post("/auth/password-login", json={
            "provider": "basic", "username": "x", "password": "y"})
        assert resp.status_code == 404

    def test_enrolment_rejected_when_id_not_allowlisted(self, hermes_home):
        _write_webauthn_config(hermes_home, allowed_credential_ids=["c3VibWFya2luZS1pZA"])
        built = _build_provider(hermes_home)
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        options, handle = built.webauthn_register_options(client_ip="t")
        from hermes_cli.dashboard_auth import InvalidCredentialsError

        with pytest.raises(InvalidCredentialsError, match="allowed_credential_ids"):
            built.complete_webauthn_registration(
                handle=handle, credential=auth.credential_json(
                    auth.register(_challenge_bytes(options))), label="x")

    def test_allowlist_matching_the_enrolled_id_lets_login_through(self, hermes_home):
        from fido2.utils import websafe_encode

        stub_id = websafe_encode(bytes(range(1, 33)))  # the stub's fixed credential id
        _write_webauthn_config(hermes_home, allowed_credential_ids=[stub_id])
        built = _build_provider(hermes_home)
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(built, auth)
        options, handle = built.webauthn_login_options(client_ip="t")
        session = built.complete_webauthn_login(
            handle=handle, credential=auth.credential_json(
                auth.assert_login(_challenge_bytes(options))))
        assert session.provider == "webauthn"


# ---------------------------------------------------------------------------
# Origin gating: the LAN-http fail-inert contract
# ---------------------------------------------------------------------------


class TestOriginGating:
    def test_webauthn_only_lan_request_providers_fail_closed(self, gated_client, provider):
        body = _lan_client().get("/api/auth/providers")
        # only webauthn registered; a LAN origin is never offered -> fail closed
        assert body.status_code == 503, body.text

    def test_lan_ceremonies_answer_404(self, gated_client, provider):
        client = _lan_client()
        assert client.get("/auth/webauthn/options").status_code == 404
        assert client.post("/auth/webauthn/verify", json={}).status_code == 404
        assert client.get("/auth/webauthn/register/options").status_code == 404
        assert client.get("/auth/webauthn/enrol?token=x").status_code == 404

    def test_tls_origin_lists_webauthn(self, gated_client, provider):
        body = gated_client.get("/api/auth/providers")
        assert body.status_code == 200
        assert body.json()["providers"][0]["name"] == "webauthn"
        assert body.json()["providers"][0]["supports_password"] is False

    def test_register_options_json_shape(self, gated_client, provider):
        """The JSON the enrol page JS receives: a {"publicKey": …} wrapper whose
        challenge/user.id are base64url STRINGS (the browser must decode them —
        a page feeding this straight into parseCreationOptionsFromJSON gets
        'challenge is required and must be an instance of DOMString')."""
        import base64
        import re as re_mod

        token = provider.issue_enrolment_token()
        body = gated_client.get(
            "/auth/webauthn/register/options", params={"enrolment": token})
        assert body.status_code == 200
        payload = body.json()
        pk = payload["publicKey"]
        assert isinstance(pk, dict)
        for member in ("challenge", "user"):  # b64url strings, not ArrayBuffers
            value = pk[member] if member == "challenge" else pk[member]["id"]
            assert isinstance(value, str)
            base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        assert re_mod.search(r"attestation|authenticatorSelection|pubKeyCredParams", body.text)

    def test_enrol_page_js_unwraps_and_decodes(self, gated_client, provider):
        """The served page must unwrap `publicKey` and decode buffers itself —
        the parseCreationOptionsFromJSON shortcut is the bug shape."""
        token = provider.issue_enrolment_token()
        page = gated_client.get("/auth/webauthn/enrol",
                                params={"token": token, "label": "contract"})
        assert page.status_code == 200
        html = page.text
        assert "options.publicKey || options" in html
        assert "b64urlToBuffer(pk.challenge)" in html
        assert "parseCreationOptionsFromJSON" not in html

    def test_login_page_button_only_on_matching_origin(self, gated_client, provider):
        page = gated_client.get("/login")
        assert page.status_code == 200
        assert 'class="provider-form provider-webauthn"' in page.text
        assert "Sign in with a passkey" in page.text
        assert 'data-provider="webauthn"' in page.text
        assert "auth/webauthn/options" in page.text
        # no password provider registered -> the password script is NOT emitted
        assert "auth/password-login" not in page.text

    def test_password_script_never_attaches_to_the_passkey_form(
            self, gated_client, provider):
        """The UAT bug: the password form's script matched `form.provider-form` —
        which also matches the passkey form — so every passkey-button submit
        ALSO posted /auth/password-login with provider=webauthn (404) and
        painted 'Sign-in failed' on the passkey form while the real ceremony
        succeeded. The selector must exclude the webauthn form."""
        from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

        class _PW(StubAuthProvider):
            name = "pwlan"
            display_name = "PW LAN"
            supports_password = True

            def complete_password_login(self, *, username, password):
                raise NotImplementedError

        register_provider(_PW())
        try:
            page = gated_client.get("/login")
            assert page.status_code == 200
            html_text = page.text
            # the password script IS emitted now (both forms render), and it
            # must not attach to the passkey form
            assert "auth/password-login" in html_text
            assert "form.provider-form:not(.provider-webauthn)" in html_text
            assert html_text.count("querySelectorAll('form.provider-form')") == 0
        finally:
            clear_providers()  # restore the fixture's registry state exactly
            register_provider(provider)
        # LAN http: no button, no script, no webauthn HTML at all
        lan_page = _lan_client().get("/login")
        assert "provider-webauthn" not in lan_page.text
        assert "auth/webauthn" not in lan_page.text

    def test_gate_allowlist_covers_ceremony_prefix(self):
        assert "/auth/webauthn/" in _GATE_PUBLIC_PREFIXES

    def test_login_page_render_without_filter_still_renders_button(self, provider):
        page = render_login_html()
        assert 'data-provider="webauthn"' in page

    def test_basic_plus_webauthn_lan_shows_password_only(self, hermes_home, gated_client,
                                                         provider):
        """The live posture: basic + webauthn on LAN-http -> basic only."""
        from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

        class _PW(StubAuthProvider):
            name = "pwlan"
            display_name = "PW LAN"
            supports_password = True

            def complete_password_login(self, *, username, password):
                raise NotImplementedError

        register_provider(_PW())
        try:
            body = _lan_client().get("/api/auth/providers")
            assert body.status_code == 200
            names = [p["name"] for p in body.json()["providers"]]
            assert names == ["pwlan"]
        finally:
            clear_providers()


# ---------------------------------------------------------------------------
# Session lifecycle parity (shared basic seam)
# ---------------------------------------------------------------------------


class TestSessionParity:
    def test_tokens_are_basic_shaped_and_interoperable(self, provider):
        from plugins.dashboard_auth.basic import _unsign as basic_unsign

        session = provider._mint_session()  # noqa: SLF001
        payload = basic_unsign(  # noqa: SLF001
            session.access_token, provider._secret, "access")
        assert payload == {"sub": "hermes-operator", "kind": "access",
                           "exp": session.expires_at}
        refresh = basic_unsign(  # noqa: SLF001
            session.refresh_token, provider._secret, "refresh")
        assert refresh["kind"] == "refresh" and refresh["sub"] == "hermes-operator"

    def test_refresh_rotates_like_password_sessions(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        assert gated_client.post(
            "/auth/webauthn/verify", json=_verify_body(auth, options)).status_code == 200
        rt = _cookie(gated_client, SESSION_RT_COOKIE)
        assert rt, "refresh-token cookie should be minted via the shared seam"
        fresh = provider.refresh_session(refresh_token=rt)
        assert fresh.provider == "webauthn"
        assert fresh.access_token and fresh.refresh_token

    def test_expired_access_token_refreshes_transparently(self, gated_client, provider):
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        options = gated_client.get("/auth/webauthn/options")
        assert gated_client.post(
            "/auth/webauthn/verify", json=_verify_body(auth, options)).status_code == 200
        at = _cookie(gated_client, "hermes_session_at")
        rt = _cookie(gated_client, SESSION_RT_COOKIE)
        assert at and rt

        def _expired(token: str) -> str:
            blob = base64.urlsafe_b64decode(token.encode())
            raw, sig = blob[:-32], blob[-32:]
            payload = json.loads(raw)
            payload["exp"] = int(time.time()) - 5
            new_raw = json.dumps(payload, separators=(",", ":")).encode()
            new_sig = hmac_mod.new(  # noqa: SLF001 - test-only signing
                provider._secret, new_raw, hashlib.sha256).digest()
            return base64.urlsafe_b64encode(new_raw + new_sig).decode()

        gated_client.cookies.set("__Host-hermes_session_at", _expired(at),
                                 domain="dash.example.com")
        resp = gated_client.get("/api/auth/me")
        assert resp.status_code == 200, resp.text
        assert resp.json()["provider"] == "webauthn"


# ---------------------------------------------------------------------------
# Native (desktop) broker tail parity
# ---------------------------------------------------------------------------


def _make_pkce() -> tuple:
    def _b64u(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    verifier = _b64u(b"desktop-verifier-secret-material-0123456789abcd")
    return verifier, _b64u(hashlib.sha256(verifier.encode("ascii")).digest())


class TestNativeBrokerParity:
    def test_native_authorize_then_passkey_completes_with_loopback_code(
            self, gated_client, provider):
        verifier, challenge = _make_pkce()
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        start = gated_client.get("/auth/native/authorize", params={
            "code_challenge": challenge, "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb", "state": "desk"})
        assert start.status_code == 302
        assert start.headers["location"].endswith("/login")
        options = gated_client.get("/auth/webauthn/options")
        verify = gated_client.post("/auth/webauthn/verify", json=_verify_body(auth, options))
        assert verify.status_code == 200
        next_url = verify.json()["next"]
        assert next_url.startswith("http://127.0.0.1:53999/cb?")
        assert "code=" in next_url and "state=desk" in next_url
        gw_code = next_url.split("code=", 1)[1].split("&", 1)[0]
        token = gated_client.post("/auth/native/token", json={
            "code": gw_code, "code_verifier": verifier})
        assert token.status_code == 200, token.text
        body = token.json()
        assert body["provider"] == "webauthn"
        assert body["access_token"] and body["token_type"] == "Bearer"

    def test_native_wrong_provider_verify_is_rejected(self, gated_client, provider):
        verifier, challenge = _make_pkce()
        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        start = gated_client.get("/auth/native/authorize", params={
            "code_challenge": challenge, "code_challenge_method": "S256",
            "redirect_uri": "http://127.0.0.1:53999/cb", "state": "s"})
        assert start.status_code == 302
        # (provider mismatch for a webauthn completion would need a broker cookie
        # naming another provider — the webauthn ceremony itself holds no such
        # state; the check is the same parity as password-login's.)


# ---------------------------------------------------------------------------
# CLI enrol/forget (headless bootstrap + removal)
# ---------------------------------------------------------------------------


class TestCliEnrolAndForget:
    def test_enrol_prints_valid_origin_url_and_token_pending(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn
        from plugins.dashboard_auth.webauthn import store as webauthn_store

        _write_webauthn_config(hermes_home)
        args = type("A", (), {"origin": "", "label": "laptop"})()
        dashboard_webauthn.cmd_webauthn_enrol(args)
        out = capsys.readouterr().out
        assert "Open this URL" in out
        url = out.splitlines()[1].strip()
        assert url.startswith(f"{HTTPS_ORIGIN}/auth/webauthn/enrol?token=")
        token = url.split("token=", 1)[1].split("&", 1)[0]
        # the diagnostics block: pending path + explicit expiry + single-use semantics
        assert f"Token {token[:8]}" in out
        assert "pending in" in out and "expires" in out
        assert "already used" in out
        from plugins.dashboard_auth.webauthn import WebauthnDashboardAuthProvider

        settings = _settings_or_skip(hermes_home)
        live = WebauthnDashboardAuthProvider(**settings)
        assert live.peek_enrolment_token(token)
        doc = webauthn_store.load(
            webauthn_store.resolve_store_path(str(hermes_home / "dashboard_webauthn.json")))
        entry = list(doc["pending_enrolments"].values())[0]
        assert entry["expires_at"] > entry["created_at"]

    def test_enrol_origin_selection(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        _write_webauthn_config(hermes_home, rp_id=RP_ID,
                               origins=[HTTPS_ORIGIN, "https://login.dash.example.com"])
        args = type("A", (), {"origin": "", "label": ""})()
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_enrol(args)

    def test_forget_removes_by_label(self, hermes_home, capsys, provider):
        from hermes_cli import dashboard_webauthn

        auth = StubWebAuthnAuthenticator(HTTPS_ORIGIN, RP_ID)
        _enrol(provider, auth)
        args = type("A", (), {"all_credentials": False, "label": "stub device",
                              "credential_id": ""})()
        dashboard_webauthn.cmd_webauthn_forget(args)
        out = capsys.readouterr().out
        assert "forgot" in out
        doc = json.loads((hermes_home / "dashboard_webauthn.json").read_text(encoding="utf-8"))
        assert doc["credentials"] == []

    def test_forget_without_args_exits(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        args = type("A", (), {"all_credentials": False, "label": "", "credential_id": ""})()
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_forget(args)


# ---------------------------------------------------------------------------
# CLI setup — the pre-enrolment wizard for the dashboard.webauthn block
# ---------------------------------------------------------------------------


class TestCliSetup:
    @staticmethod
    def _args(**overrides):
        base = {"rp_id": "", "rp_name": "", "origin": None, "force": False,
                "skip_preflight": False}
        base.update(overrides)
        return type("A", (), base)()

    def test_setup_flags_write_config_and_resolve(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn
        from plugins.dashboard_auth._shared import load_config_section
        import logging

        args = self._args(rp_id="dash.example.com", rp_name="",  # rp_name default = rp_id
                          origin=["https://dash.example.com"])
        dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "✓ dashboard.webauthn written" in out
        assert "resolves cleanly" in out
        section = load_config_section(logging.getLogger("t"), "t",
                                      "dashboard", "webauthn")
        assert section["rp_id"] == "dash.example.com"
        assert section["rp_name"] == "dash.example.com"  # default applied
        assert section["origins"] == ["https://dash.example.com"]
        # the surrounding config keeps other sections: write and re-read round-trip
        from hermes_cli.config import cfg_get, load_config

        assert cfg_get(load_config(), "dashboard", "webauthn", "rp_id") == "dash.example.com"

    def test_setup_rejects_unpairable_origin(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        args = self._args(rp_id="dash.example.com", rp_name="Dash",
                          origin=["https://other.example.com"])
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "does not pair with rp_id" in out

    def test_setup_rejects_plain_http_origin(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        args = self._args(rp_id="dash.example.com", rp_name="Dash",
                          origin=["http://dash.example.com"])
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_setup(args)
        assert "plain http" in capsys.readouterr().out

    def test_setup_accepts_localhost_tunnel(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn
        from plugins.dashboard_auth._shared import load_config_section
        import logging
        from plugins.dashboard_auth.webauthn import _settings

        args = self._args(rp_id="localhost", rp_name="Local",
                          origin=["http://localhost:9119"])
        dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "resolves cleanly" in out
        section = load_config_section(logging.getLogger("t"), "t",
                                      "dashboard", "webauthn")
        assert section["rp_id"] == "localhost"
        assert _settings()["origins"] == ("http://localhost:9119",)

    def test_setup_rp_id_change_requires_force(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        _write_webauthn_config(hermes_home)  # rp_id = dash.example.com
        store = hermes_home / "dashboard_webauthn.json"
        store.write_text(json.dumps({
            "version": 1, "rp_id": RP_ID,
            "credentials": [{"credential_id": "abc", "label": "phone",
                             "created_at": 0}],
            "pending_enrolments": {}}), encoding="utf-8")
        # without --force: refuses to switch
        args = self._args(rp_id="elsewhere.example.com", rp_name="Dash",
                          origin=["https://elsewhere.example.com"], force=False)
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_setup(args)
        assert "unusable" in capsys.readouterr().out
        # with --force: proceeds and rewrites the block
        args = self._args(rp_id="elsewhere.example.com", rp_name="Dash",
                          origin=["https://elsewhere.example.com"], force=True)
        dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "proceeding (--force given)" in out
        assert "✓ dashboard.webauthn written" in out

    def test_setup_interactive_defaults(self, hermes_home, capsys, monkeypatch):
        import logging

        from hermes_cli import dashboard_webauthn
        from plugins.dashboard_auth._shared import load_config_section

        # pytest's stdin isn't a tty; fake one so the wizard runs its prompt path.
        class _FakeTty:
            @staticmethod
            def isatty():
                return True

        monkeypatch.setattr(dashboard_webauthn.sys, "stdin", _FakeTty())
        # rp_id, rp_name (blank -> default), confirm origin, decline www, no extras,
        # confirm the write
        answers = iter(["dash.example.com", "", "y", "n", "", "y"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        args = self._args()
        dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "✓ dashboard.webauthn written" in out
        section = load_config_section(logging.getLogger("t"), "t",
                                      "dashboard", "webauthn")
        assert section["origins"] == ["https://dash.example.com"]

    def test_enrol_hints_setup_when_unconfigured(self, hermes_home, capsys):
        from hermes_cli import dashboard_webauthn

        args = type("A", (), {"origin": "", "label": ""})()
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_enrol(args)
        assert "webauthn setup" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# setup preflight — fail fast BEFORE the operator types anything
# ---------------------------------------------------------------------------


class TestSetupPreflight:
    @staticmethod
    def _args(**overrides):
        base = {"rp_id": "", "rp_name": "", "origin": None, "force": False,
                "skip_preflight": False}
        base.update(overrides)
        return type("A", (), base)()

    def test_missing_fido2_stops_before_any_input(self, hermes_home, capsys, monkeypatch):
        """The ask: check fido2/other prerequisites BEFORE entering details.
        Non-interactive: prints the remedy and exits; config untouched."""
        import importlib.util

        from hermes_cli import dashboard_webauthn

        _write_webauthn_config(hermes_home)  # a readable config, so check 1 passes
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: None if name == "fido2" else importlib.util.find_spec(name))
        args = self._args(rp_id="dash.example.com", rp_name="Dash",
                          origin=["https://dash.example.com"])
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "fido2" in out and "hermes pm install --extra webauthn" in out
        # nothing asked, nothing written:
        import hermes_yaml as yaml

        cfg = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["dashboard"]["webauthn"]["rp_id"] == RP_ID  # the OLD value intact

    def test_missing_fido2_interactively_installs_the_extra(
            self, hermes_home, capsys, monkeypatch):
        """Interactive: one prompt -> PM unions the extra -> setup proceeds."""
        import importlib.util

        from hermes_cli import dashboard_webauthn

        class _FakeTty:
            @staticmethod
            def isatty():
                return True

        monkeypatch.setattr(dashboard_webauthn.sys, "stdin", _FakeTty())
        _write_webauthn_config(hermes_home)
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name: None if name == "fido2" else None)  # always None here
        import pm

        calls: list[list[str]] = []
        monkeypatch.setattr(pm, "sync_venv", lambda extras, **k: calls.append(list(extras)))
        # prompts: preflight-confirm Y, rp_name (blank = default), include-origin Y,
        # www N, extras blank, final write confirm Y
        answers = iter(["y", "", "y", "n", "", "y"])
        monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
        args = self._args(rp_id="dash.example.com", rp_name="")  # rp_name asked after preflight
        dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert calls == [["webauthn"]]
        assert "✓ webauthn extra installed" in out
        assert "✓ dashboard.webauthn written" in out

    def test_skip_preflight_stages_without_the_check(self, hermes_home, capsys, monkeypatch):
        """--skip-preflight = provisioning a config for another box; writes despite fido2 missing."""
        import importlib.util

        import hermes_yaml as yaml

        from hermes_cli import dashboard_webauthn

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        args = self._args(rp_id="other.example.com", rp_name="Other",
                          origin=["https://other.example.com"], skip_preflight=True)
        # no config on disk at all (check 1 must ALSO be skippable):
        cfg_path = hermes_home / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        dashboard_webauthn.cmd_webauthn_setup(args)

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["dashboard"]["webauthn"]["rp_id"] == "other.example.com"

    def test_unreadable_config_stops_before_input(self, hermes_home, capsys, monkeypatch):
        import hermes_cli.config as config

        from hermes_cli import dashboard_webauthn

        def _raise():
            raise RuntimeError("bad yaml")

        monkeypatch.setattr(config, "load_config_readonly", _raise)
        args = self._args(rp_id="dash.example.com", rp_name="Dash",
                          origin=["https://dash.example.com"])
        with pytest.raises(SystemExit):
            dashboard_webauthn.cmd_webauthn_setup(args)
        out = capsys.readouterr().out
        assert "unreadable" in out and "nothing was written" in out