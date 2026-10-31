"""The gateway relay: one Webapp reaching another gateway through a grant that gateway issued.

Contracts (``hermes_cli.webapp_relay``): a relayed request is signed by the hub's key and served by the
spoke as the identity that approved the grant, never replayed, never relayed onward, never past its
scopes; the grant ends when it is revoked or the spoke signs out; only the hub login that paired a
gateway reaches it. The pairing runs through the real routes with a gateway paired to itself (one
app plays hub and spoke; the hub's outbound client is an in-process ASGI transport).
"""
import re
import time
import urllib.parse
from html.parser import HTMLParser

import httpx
import pytest
from starlette.testclient import TestClient

from tests.hermes_cli.test_webapp_socket_tickets import serve_webapp

WEBAPP = "http://hermes.lan:9120"
ORIGIN = {"Origin": WEBAPP}
SPOKE = "http://mini.lan:9120"
USER, PASSWORD = "owner", "throwaway-test-password"


def _write_config(home, relay: bool) -> None:
    (home / "config.yaml").write_text(
        f"webapp:\n  enabled: true\n  relay:\n    enabled: {'true' if relay else 'false'}\n", encoding="utf-8")


@pytest.fixture(name="relay_app")
def _relay_app(tmp_path, monkeypatch):
    from hermes_cli import webapp_relay, webapp_relay_gateways, webapp_relay_grants, webapp_sign_in
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit
    from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests
    from hermes_cli.web_routers import webapp_relay_proxy
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    webapp = serve_webapp(tmp_path, monkeypatch)
    home = tmp_path / ".hermes"
    _write_config(home, relay=True)
    state = webapp.server.app.state
    monkeypatch.setattr(state, "bound_host", "hermes.lan")
    monkeypatch.setattr(state, "bound_port", 9120, raising=False)
    monkeypatch.setattr(state, "auth_required", True)
    monkeypatch.setattr(state, "webapp_home", home, raising=False)
    monkeypatch.setattr(state, "listener_public_url", WEBAPP, raising=False)
    monkeypatch.setattr(state, "trusted_public_hosts", frozenset({"hermes.lan"}), raising=False)
    for module in (webapp_relay, webapp_relay_grants, webapp_relay_gateways, webapp_sign_in):
        module._reset_for_tests()
    _reset_for_tests()
    _reset_password_rate_limit()
    clear_providers()
    register_provider(BasicAuthProvider(username=USER, password_hash=hash_password(PASSWORD), secret=b"k" * 32))
    register_provider(StubAuthProvider())
    # The hub's outbound requests reach the spoke (this same app) in process.
    monkeypatch.setattr(webapp_relay_proxy, "relay_client", lambda _state: httpx.AsyncClient(
        transport=httpx.ASGITransport(app=webapp.server.app), follow_redirects=False))
    webapp.home = home
    webapp.browser = lambda: TestClient(webapp.server.app, base_url=WEBAPP)
    try:
        yield webapp
    finally:
        clear_providers()
        _reset_for_tests()
        for module in (webapp_relay, webapp_relay_grants, webapp_relay_gateways, webapp_sign_in):
            module._reset_for_tests()


def _sign_in(client):
    started = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = started.headers["location"].split("state=")[1]
    assert client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False).status_code == 302


class _ConsentForm(HTMLParser):
    """What a browser submits from the consent page: the named inputs inside its form, checked
    checkboxes only."""

    def __init__(self):
        super().__init__()
        self.inside, self.fields, self.checkboxes = False, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.inside = True
        elif tag == "input" and self.inside and attrs.get("name"):
            if attrs.get("type") == "checkbox":
                self.checkboxes.append(attrs["value"])
            else:
                self.fields.append((attrs["name"], attrs.get("value", "")))

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def _approve(browser, *, scopes=("display", "files", "terminal", "voice"), label="Mini"):
    """Start pairing, approve it on the consent page as a browser submits it, leaving *scopes*
    checked; returns the callback path and query."""
    started = browser.post("/api/webapp/relay/gateways", json={"url": WEBAPP, "label": label}, headers=ORIGIN)
    assert started.status_code == 200, started.text
    authorize = urllib.parse.urlsplit(started.json()["authorize_url"])
    assert f"{authorize.scheme}://{authorize.netloc}" == WEBAPP and authorize.path == "/webapp/relay/authorize"
    page = browser.get(f"{authorize.path}?{authorize.query}")
    assert page.status_code == 200, page.text
    assert WEBAPP in page.text and "stub@example.test" in page.text
    assert page.headers["content-security-policy"].count(WEBAPP) == 1  # the form may return there only
    form = _ConsentForm()
    form.feed(page.text)
    assert sorted(form.checkboxes) == ["display", "files", "terminal", "voice"]  # every scope asked for
    data = form.fields + [("grant_scope", s) for s in form.checkboxes if s in scopes] + [("decision", "allow")]
    decided = browser.post("/webapp/relay/authorize", content=urllib.parse.urlencode(data),
                           headers={**ORIGIN, "content-type": "application/x-www-form-urlencoded"},
                           follow_redirects=False)
    assert decided.status_code == 303, decided.text
    callback = urllib.parse.urlsplit(decided.headers["location"])
    assert callback.path == "/webapp/relay/callback"
    return f"{callback.path}?{callback.query}"


def _pair(browser, **kwargs) -> dict:
    back = browser.get(_approve(browser, **kwargs), follow_redirects=False)
    assert back.status_code == 303 and "relay=connected" in back.headers["location"], back.headers
    rows = browser.get("/api/webapp/relay/gateways").json()["gateways"]
    assert len(rows) == 1 and rows[0]["url"] == WEBAPP
    assert "grant_id" not in rows[0]
    assert rows[0]["scopes"] == sorted(kwargs.get("scopes", ("display", "files", "terminal", "voice")))
    return rows[0]


def test_a_relay_signature_verifies_once_for_its_exact_request():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from hermes_cli import webapp_relay as relay

    relay._reset_for_tests()
    key, other = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    public = relay.public_key_text(key)
    signed = relay.sign_request(key, method="GET", target="/api/profiles?a=1", audience=SPOKE, grant_id="g",
                                subject="u")
    relay.verify_request(public, signed, method="GET", target="/api/profiles?a=1", audience=SPOKE)
    with pytest.raises(relay.RelayAuthError, match="nonce"):
        relay.verify_request(public, signed, method="GET", target="/api/profiles?a=1", audience=SPOKE)
    fresh = relay.sign_request(key, method="GET", target="/api/profiles?a=1", audience=SPOKE, grant_id="g",
                               subject="u")
    for method, target, audience, headers, public_key in (
            ("GET", "/api/profiles?a=2", SPOKE, fresh, public), ("POST", "/api/profiles?a=1", SPOKE, fresh, public),
            ("GET", "/api/profiles?a=1", "http://other.spoke", fresh, public),
            ("GET", "/api/profiles?a=1", SPOKE, {**fresh, relay.SUBJECT_HEADER: "someone-else"}, public),
            ("GET", "/api/profiles?a=1", SPOKE, fresh, relay.public_key_text(other))):
        with pytest.raises(relay.RelayAuthError, match="signature"):
            relay.verify_request(public_key, headers, method=method, target=target, audience=audience)
    stale = relay.sign_request(key, method="GET", target="/x", audience=SPOKE, grant_id="g", subject="u",
                               now=time.time() - 3600)
    with pytest.raises(relay.RelayAuthError, match="clock"):
        relay.verify_request(public, stale, method="GET", target="/x", audience=SPOKE)


def test_the_relay_is_off_until_its_config_turns_it_on(relay_app):
    from hermes_cli import webapp_relay

    _write_config(relay_app.home, relay=False)
    webapp_relay._reset_for_tests()
    browser = relay_app.browser()
    _sign_in(browser)
    assert browser.get("/api/webapp/relay/gateways").status_code == 404
    assert browser.post("/api/webapp/socket-ticket?socket=relay", headers=ORIGIN).status_code == 404
    forged = {webapp_relay.GRANT_HEADER: "g", webapp_relay.SUBJECT_HEADER: "stub-user-1",
              webapp_relay.TIME_HEADER: str(int(time.time())), webapp_relay.NONCE_HEADER: "n",
              webapp_relay.SIGNATURE_HEADER: "AAAA"}
    refused = TestClient(relay_app.server.app, base_url=WEBAPP).get("/api/profiles", headers=forged)
    assert refused.status_code == 401 and refused.json()["error"] == "relay_unauthorized"


def test_a_paired_gateway_serves_relayed_requests_as_the_approving_identity(relay_app):
    browser = relay_app.browser()
    _sign_in(browser)
    row = _pair(browser)
    profiles = browser.get(f"/api/relay/{row['id']}/api/profiles")
    assert profiles.status_code == 200, profiles.text
    assert any(p["name"] == "default" for p in profiles.json()["profiles"])
    assert "set-cookie" not in profiles.headers
    # The spoke lists what it granted, to the account that approved it.
    granted = browser.get("/api/webapp/relay/grants").json()["grants"]
    assert [g["hub_origin"] for g in granted] == [WEBAPP]
    # Sign-in, Web Access and the relay's own management are never reachable through it.
    for path in ("/api/auth/me", "/api/webapp/relay/grants", "/api/webapp-access/status"):
        refused = browser.get(f"/api/relay/{row['id']}{path}")
        assert refused.status_code == 403 and refused.json()["error"] == "relay_forbidden", (path, refused.text)
    # One hop: a relayed request is never relayed onward.
    assert browser.get(f"/api/relay/{row['id']}/api/relay/{row['id']}/api/profiles").status_code == 400


def test_a_grant_reaches_only_the_scopes_it_was_given(relay_app):
    from hermes_cli.dashboard_auth.ws_tickets import consume_ticket

    browser = relay_app.browser()
    _sign_in(browser)
    row = _pair(browser, scopes=())
    base = f"/api/relay/{row['id']}"
    refused = browser.post(f"{base}/api/webapp/socket-ticket?socket=host-terminal", headers=ORIGIN)
    assert refused.status_code == 403 and "terminal" in refused.json()["detail"]
    assert browser.get(f"{base}/api/fs/list?path=/").status_code == 403
    assert browser.post(f"{base}/api/webapp/socket-ticket?socket=relay", headers=ORIGIN).status_code == 403
    minted = browser.post(f"{base}/api/webapp/socket-ticket?socket=gateway", headers=ORIGIN)
    assert minted.status_code == 200, minted.text
    info = consume_ticket(minted.json()["ticket"])
    assert info["user_id"] == "stub-user-1" and info["relay_grant"]


def test_revoking_the_grant_ends_the_relay(relay_app):
    from hermes_cli import browser_credential, webapp_relay_grants

    browser = relay_app.browser()
    _sign_in(browser)
    row = _pair(browser)
    grant = browser.get("/api/webapp/relay/grants").json()["grants"][0]
    assert browser.delete(f"/api/webapp/relay/grants/{grant['id']}", headers=ORIGIN).status_code == 200
    ended = browser.get(f"/api/relay/{row['id']}/api/profiles")
    assert ended.status_code == 409 and ended.json()["error"] == "relay_reconnect"
    # A grant also stands only under the browser credential it was approved under: signing the spoke
    # out (hermes webapp --sign-out) ends every grant with every browser login.
    # Removing it here cannot confirm a revocation the spoke no longer recognises, so it stays until
    # the user removes it here only.
    removing = browser.delete(f"/api/webapp/relay/gateways/{row['id']}", headers=ORIGIN).json()
    assert removing["removed"] is False and "may already be revoked" in removing["detail"], removing
    forgot = browser.delete(f"/api/webapp/relay/gateways/{row['id']}?forget=1", headers=ORIGIN)
    assert forgot.json() == {"removed": True, "revoked": False}
    _pair(browser)
    standing = webapp_relay_grants.list_grants(relay_app.home, {"user_id": "stub-user-1", "provider": "stub"})
    assert len(standing) == 1
    browser_credential.rotate("x" * 43)
    assert webapp_relay_grants.grant_for(relay_app.home, standing[0]["id"]) is None


def test_another_account_on_the_hub_cannot_reach_your_gateways(relay_app):
    owner = relay_app.browser()
    _sign_in(owner)
    row = _pair(owner)
    other = relay_app.browser()
    signed_in = other.post("/auth/password-login", json={"provider": "basic", "username": USER, "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text
    assert other.get("/api/webapp/relay/gateways").json()["gateways"] == []
    assert other.get(f"/api/relay/{row['id']}/api/profiles").status_code == 404
    assert other.delete(f"/api/webapp/relay/gateways/{row['id']}", headers=ORIGIN).status_code == 404


def test_a_pairing_code_spends_once_and_only_with_its_verifier(relay_app):
    browser = relay_app.browser()
    _sign_in(browser)
    callback = _approve(browser)
    code = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(callback).query))["code"]
    stolen = TestClient(relay_app.server.app, base_url=WEBAPP).post("/api/webapp/relay/token", json={
        "code": code, "code_verifier": "v" * 64, "redirect_uri": f"{WEBAPP}/webapp/relay/callback", "audience": WEBAPP,
        "proof": "AAAA"})
    assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_grant"
    # The failed attempt spent the code: the real hub's exchange now fails too, and lists nothing.
    back = browser.get(callback, follow_redirects=False)
    assert "relay=failed" in back.headers["location"]
    assert browser.get("/api/webapp/relay/gateways").json()["gateways"] == []
    assert re.search(r"relay_detail=", back.headers["location"])
