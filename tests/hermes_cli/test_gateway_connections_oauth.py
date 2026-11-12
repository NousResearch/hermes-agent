"""Server-side OAuth sign-in to a saved gateway and the proxy's reach hook.

Contracts: a sign-in started here hands the browser the remote's native authorize URL with a
loopback redirect to this gateway's callback, and the callback redeems the code with the PKCE
verifier only for a state this process issued (once); the tokens land in the store's token file;
an expired access token is rotated through the remote's refresh route and a 401 there drops the
sign-in; ``resolve_target`` sends exactly Desktop's credentials per auth mode. The remote here is
an HTTP stand-in that enforces the native flow's PKCE rule; the live proof drives a real gateway.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_agent.hermes_cli import gateway_connections as store
from hermes_agent.hermes_cli import gateway_connections_oauth as oauth
from hermes_agent.hermes_cli import gateway_connections_reach as reach
from hermes_agent.hermes_cli import web_server

REMOTE = "https://remote.example:8443/hermes"


class FakeRemote:
    """The remote's /api/status, /auth/native/{token,refresh} and /api/auth/ws-ticket."""

    def __init__(self):
        self.challenges: dict[str, str] = {}
        self.refresh_status = 200
        self.calls: list[tuple[str, dict]] = []
        self.serial = 0

    def issue_code(self, authorize_url: str) -> tuple[str, str]:
        query = {k: v[0] for k, v in parse_qs(urlparse(authorize_url).query).items()}
        code = f"code-{len(self.challenges)}"
        self.challenges[code] = query["code_challenge"]
        return code, query["state"]

    def _tokens(self, expires_in: int) -> dict:
        self.serial += 1
        return {"access_token": f"at-{self.serial}", "refresh_token": f"rt-{self.serial}",
                "token_type": "Bearer", "expires_at": int(time.time()) + expires_in,
                "provider": "basic", "user_id": "u1"}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/hermes")
        body = json.loads(request.content or b"{}")
        self.calls.append((path, {**dict(request.headers), **body}))
        if path == "/api/status":
            return httpx.Response(200, json={"auth_flows": ["cookie", "native_pkce"]})
        if path == "/auth/native/token":
            challenge = self.challenges.pop(body.get("code"), None)
            digest = hashlib.sha256(str(body.get("code_verifier")).encode()).digest()
            if challenge != base64.urlsafe_b64encode(digest).rstrip(b"=").decode():
                return httpx.Response(400, json={"detail": "Invalid or expired authorization code."})
            return httpx.Response(200, json=self._tokens(3600))
        if path == "/auth/native/refresh":
            if self.refresh_status != 200:
                return httpx.Response(self.refresh_status, json={"error": "session_expired"})
            return httpx.Response(200, json=self._tokens(3600))
        if path == "/api/auth/ws-ticket":
            return httpx.Response(200, json={"ticket": "tk-1"})
        return httpx.Response(404)


@pytest.fixture
def remote(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    fake = FakeRemote()
    monkeypatch.setattr(oauth, "_client", lambda: httpx.Client(transport=httpx.MockTransport(fake.handle)))
    monkeypatch.setattr(oauth, "_flights", {})
    return fake


def _save(**fields) -> str:
    connection, _registry = store.save_connection({"kind": "remote", "label": fields.pop("label", "Remote"),
                                                   "url": REMOTE, **fields})
    return connection["id"]


def _sign_in(remote: FakeRemote, connection_id: str) -> dict:
    started = oauth.start_sign_in(connection_id, bound_host="127.0.0.1", bound_port=9120)
    code, state = remote.issue_code(started["authorizeUrl"])
    return oauth.complete_sign_in(code=code, state=state)


def test_a_sign_in_redirects_to_this_gateway_and_stores_the_tokens_for_the_connection(remote):
    cid = _save(authMode="oauth", headers={"CF-Access-Client-Id": "cf-id"})
    started = oauth.start_sign_in(cid, bound_host="0.0.0.0", bound_port=9120)
    authorize = urlparse(started["authorizeUrl"])
    query = {k: v[0] for k, v in parse_qs(authorize.query).items()}
    assert f"{authorize.scheme}://{authorize.netloc}{authorize.path}" == f"{REMOTE}/auth/native/authorize"
    assert query["redirect_uri"] == "http://127.0.0.1:9120/webapp/gateway-connections/oauth/callback"
    assert query["code_challenge_method"] == "S256"
    # A second start while one is open hands back the same flight (one PKCE cookie per browser).
    assert oauth.start_sign_in(cid, bound_host="0.0.0.0", bound_port=9120) == started

    code, state = remote.issue_code(started["authorizeUrl"])
    assert oauth.complete_sign_in(code=code, state=state)["connectionId"] == cid
    tokens = json.loads(store.tokens_path().read_text())["tokens"][cid]
    assert tokens["accessToken"] == "at-1" and tokens["refreshToken"] == "rt-1"
    token_call = next(h for p, h in remote.calls if p == "/auth/native/token")
    assert token_call["cf-access-client-id"] == "cf-id"
    assert oauth.sign_in_status(cid)["signedIn"] is True


def test_the_callback_refuses_a_state_it_did_not_issue_and_a_replay(remote):
    cid = _save(authMode="oauth")
    started = oauth.start_sign_in(cid, bound_host="127.0.0.1", bound_port=9120)
    code, state = remote.issue_code(started["authorizeUrl"])
    with pytest.raises(oauth.OAuthSignInError):
        oauth.complete_sign_in(code=code, state="forged")
    oauth.complete_sign_in(code=code, state=state)
    with pytest.raises(oauth.OAuthSignInError):
        oauth.complete_sign_in(code=code, state=state)
    assert sum(1 for p, _ in remote.calls if p == "/auth/native/token") == 1


def test_a_pasted_redirect_completes_the_same_flight(remote):
    cid = _save(authMode="oauth")
    started = oauth.start_sign_in(cid, bound_host="127.0.0.1", bound_port=9120)
    code, state = remote.issue_code(started["authorizeUrl"])
    pasted = f"http://127.0.0.1:9120/webapp/gateway-connections/oauth/callback?code={code}&state={state}"
    assert oauth.complete_from_url(pasted)["connectionId"] == cid
    with pytest.raises(oauth.OAuthSignInError):
        oauth.complete_from_url(f"https://evil.example/other?code={code}&state={state}")


def test_an_expiring_access_token_is_rotated_and_a_rejected_refresh_drops_the_sign_in(remote):
    cid = _save(authMode="oauth")
    _sign_in(remote, cid)
    assert reach.resolve_target(cid) == (REMOTE, {"Authorization": "Bearer at-1"})

    stored = store.read_tokens(cid)
    store.write_tokens(cid, {**stored, "expiresAt": int(time.time()) + 10})
    assert reach.resolve_target(cid)[1]["Authorization"] == "Bearer at-2"
    assert store.read_tokens(cid)["refreshToken"] == "rt-2"
    # The remote answered 401 to the current token: one forced rotation.
    assert reach.resolve_target(cid, rejected_token="at-2")[1]["Authorization"] == "Bearer at-3"

    remote.refresh_status = 503
    with pytest.raises(oauth.OAuthSignInError):
        reach.resolve_target(cid, rejected_token="at-3")
    assert store.read_tokens(cid)["accessToken"] == "at-3"  # kept through an outage
    remote.refresh_status = 401
    with pytest.raises(oauth.NotSignedIn):
        reach.resolve_target(cid, rejected_token="at-3")
    assert store.read_tokens(cid) is None


def test_token_mode_sends_the_session_token_header_and_the_entry_headers(remote):
    cid = _save(token="secret-token", headers={"CF-Access-Client-Secret": "cf-secret"})
    assert reach.resolve_target(cid) == (
        REMOTE, {"CF-Access-Client-Secret": "cf-secret", "X-Hermes-Session-Token": "secret-token"})
    assert reach.mint_ws_ticket(cid) is None
    with pytest.raises(store.ConnectionStoreError):
        reach.resolve_target("local")


def test_an_oauth_socket_ticket_is_minted_with_the_bearer(remote, monkeypatch):
    cid = _save(authMode="oauth")
    _sign_in(remote, cid)
    monkeypatch.setattr(reach, "_post_ticket", lambda base, headers: oauth._client().post(
        f"{base}/api/auth/ws-ticket", headers=headers))
    assert reach.mint_ws_ticket(cid) == "tk-1"
    ticket_call = next(h for p, h in remote.calls if p == "/api/auth/ws-ticket")
    assert ticket_call["authorization"] == "Bearer at-1"


def test_the_routes_serve_only_the_webapp_and_the_callback_never_shows_a_token(remote, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_port", 9120, raising=False)
    client = TestClient(web_server.app, raise_server_exceptions=False)
    cid = _save(authMode="oauth")
    assert client.post(f"/api/gateway-connections/{cid}/oauth/start").status_code == 401
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    started = client.post(f"/api/gateway-connections/{cid}/oauth/start").json()
    code, state = remote.issue_code(started["authorizeUrl"])

    page = client.get(f"/webapp/gateway-connections/oauth/callback?code={code}&state={state}")
    assert page.status_code == 200 and "at-1" not in page.text and "rt-1" not in page.text
    status = client.get(f"/api/gateway-connections/{cid}/oauth").json()
    assert status["signedIn"] is True and "accessToken" not in json.dumps(status)
    assert client.delete(f"/api/gateway-connections/{cid}/oauth").json() == {"ok": True}
    assert store.read_tokens(cid) is None

    monkeypatch.setattr(web_server.app.state, "ui_surface", "gateway", raising=False)
    assert client.get(f"/api/gateway-connections/{cid}/oauth").status_code == 404
