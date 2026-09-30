"""ChatGPT plan registrations stay bound to verified identities across login and refresh."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from hermes_cli import auth_chatgpt as chatgpt
from hermes_cli.auth_constants import AuthError


class ChatGPTIdP:
    """Real loopback OAuth/JWKS server; no signature, callback or PKCE mocks."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.authorizations = []
        self.token_requests = []
        self.revocations = []
        self.grants = {}
        self.codes = {}
        self.subject = "subject-one"
        self.client_id = "oaiapp_one"
        self.fault = None
        self.reject_code_once = False
        self.refresh_error = None
        self.scope = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
        idp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urlparse(self.path)
                if path.path == "/.well-known/openid-configuration":
                    return self.respond(200, {"issuer": idp.base, "jwks_uri": idp.base + "/jwks",
                        "revocation_endpoint": idp.base + "/revoke", "id_token_signing_alg_values_supported": ["RS256"]})
                if path.path == "/jwks":
                    key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(idp.key.public_key()))
                    return self.respond(200, {"keys": [{**key, "kid": "test-key", "alg": "RS256", "use": "sig"}]})
                if path.path != "/authorize":
                    return self.respond(404, {})
                params = {k: v[0] for k, v in parse_qs(path.query).items()}
                idp.authorizations.append(params)
                code = secrets.token_hex(12)
                idp.codes[code] = params
                result = {"code": code, "state": params["state"], "client_id": idp.client_id}
                if idp.fault == "state":
                    result["state"] = "wrong"
                elif idp.fault == "missing_client":
                    result.pop("client_id")
                elif idp.fault == "dynamic_client":
                    result["client_id"] = "dynamic_agent_client"
                elif idp.fault == "denied_state":
                    result = {"error": "access_denied", "state": "wrong"}
                elif idp.fault == "denied":
                    result = {"error": "access_denied", "state": params["state"]}
                self.send_response(302)
                self.send_header("Location", params["redirect_uri"] + "?" + urlencode(result))
                self.end_headers()

            def do_POST(self):
                form = {k: v[0] for k, v in parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()).items()}
                if self.path == "/revoke":
                    idp.revocations.append(form)
                    idp.grants.pop(form["token"], None)
                    self.send_response(200)
                    self.end_headers()
                    return
                idp.token_requests.append(form)
                if form["grant_type"] == "authorization_code":
                    request = idp.codes.pop(form["code"])
                    if idp.reject_code_once:
                        idp.reject_code_once = False
                        return self.respond(400, {"error": "invalid_grant"})
                    digest = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).decode().rstrip("=")
                    assert digest == request["code_challenge"]
                    assert form["redirect_uri"] == request["redirect_uri"]
                    identity = {"sub": idp.subject, "nonce": request["nonce"]}
                else:
                    if idp.refresh_error:
                        status = 503 if idp.refresh_error == "temporarily_unavailable" else 400
                        return self.respond(status, {"error": idp.refresh_error})
                    identity = idp.grants.pop(form["refresh_token"], None)
                    if identity is None:
                        return self.respond(400, {"error": "invalid_grant"})
                claims = {"iss": idp.base, "aud": form["client_id"], "iat": int(time.time()),
                          "exp": int(time.time()) + 3600, "email": "same@example.test", **identity}
                if idp.fault in {"nonce", "iss", "aud", "sub"}:
                    claims[idp.fault] = "wrong"
                if idp.fault == "expired":
                    claims["exp"] = int(time.time()) - 60
                if idp.fault == "missing_sub":
                    claims.pop("sub")
                key = rsa.generate_private_key(public_exponent=65537, key_size=2048) if idp.fault == "signature" else idp.key
                refresh = secrets.token_urlsafe(24)
                idp.grants[refresh] = identity
                tokens = {"access_token": secrets.token_urlsafe(24), "refresh_token": refresh,
                    "id_token": jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"}),
                    "token_type": "Bearer", "expires_in": 3600, "scope": idp.scope,
                    "earliest_refresh_at": int(time.time()) + 1800}
                if idp.fault == "omit_scope":
                    tokens.pop("scope")
                self.respond(200, tokens)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.browser_threads = []

    def open_browser(self, url):
        def visit():
            with urlopen(url, timeout=10) as response:
                response.read()
        worker = threading.Thread(target=visit)
        worker.start()
        self.browser_threads.append(worker)
        return True


@pytest.fixture
def idp(monkeypatch, tmp_path, request):
    import providers
    from agent.chatgpt_responses import classify_chatgpt_error
    from providers.base import ProviderProfile
    from hermes_cli import auth
    server = ChatGPTIdP()
    server.thread.start()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(chatgpt, "ISSUER", server.base)
    monkeypatch.setattr(chatgpt, "AUTHORIZE_URL", server.base + "/authorize")
    monkeypatch.setattr(chatgpt, "TOKEN_URL", server.base + "/token")
    monkeypatch.setattr(chatgpt.webbrowser, "open", server.open_browser)
    monkeypatch.setattr("hermes_cli.auth_device_flow._can_open_graphical_browser", lambda: True)
    class TestProfile(ProviderProfile):
        def credential_is_eligible(self, entry):
            return chatgpt.credential_is_eligible(entry)

    providers.register_provider(TestProfile(name=chatgpt.PROVIDER, auth_type="oauth_external",
        auth_handler=chatgpt.auth_handler, refresh_credential=chatgpt.refresh_credential,
        clear_credential=chatgpt.clear_credential, classify_api_error=classify_chatgpt_error))
    request.addfinalizer(lambda: (providers._REGISTRY.pop(chatgpt.PROVIDER, None), auth.PROVIDER_REGISTRY.pop(chatgpt.PROVIDER, None)))
    yield server
    server.server.shutdown()
    server.server.server_close()
    server.thread.join(timeout=2)
    for worker in server.browser_threads:
        worker.join(timeout=2)


def _args(label="personal"):
    return SimpleNamespace(provider=chatgpt.PROVIDER, no_browser=False, label=label, timeout=10)


def _rows():
    from hermes_cli.auth import _load_auth_store
    return _load_auth_store().get("credential_pool", {}).get(chatgpt.PROVIDER, [])


def test_registration_reauthorization_rotation_and_logout_preserve_account_binding(idp, capsys):
    from agent.credential_pool import load_pool
    assert chatgpt.auth_handler("add", _args()) is True
    first = _rows()[0]
    metadata = first["chatgpt"]
    assert metadata["client_id"] == idp.client_id
    assert metadata["subject"] == idp.subject
    assert metadata["scopes"] == idp.scope.split()
    authorize = idp.authorizations[-1]
    assert authorize["client_id"] == "dynamic_agent_client"
    assert authorize["agent_name_hint"] == "Hermes Agent"
    assert authorize["resource"] == "https://api.openai.com/v1"
    assert authorize["redirect_uri"].startswith("http://127.0.0.1:")
    assert authorize["redirect_uri"].endswith("/auth/callback")
    assert idp.token_requests[-1]["client_id"] == idp.client_id

    assert chatgpt.auth_handler("add", _args()) is True
    assert len(_rows()) == 1 and _rows()[0]["id"] == first["id"]
    assert idp.authorizations[-1]["client_id"] == idp.client_id
    assert "agent_name_hint" not in idp.authorizations[-1]
    assert idp.authorizations[-1]["ext_agent_host_id"] == authorize["ext_agent_host_id"]
    assert "id_token_hint" in idp.authorizations[-1]
    assert idp.authorizations[-1]["state"] != authorize["state"]
    assert idp.authorizations[-1]["nonce"] != authorize["nonce"]

    pool = load_pool(chatgpt.PROVIDER)
    stale = load_pool(chatgpt.PROVIDER)
    before = _rows()[0]["refresh_token"]
    rotated = pool.try_refresh_matching(credential_id=first["id"])
    assert rotated and rotated.refresh_token != before
    assert idp.token_requests[-1] == {"grant_type": "refresh_token", "client_id": idp.client_id,
        "refresh_token": before, "resource": "https://api.openai.com/v1"}
    posts = len(idp.token_requests)
    adopted = stale.try_refresh_matching(credential_id=first["id"])
    assert adopted and adopted.refresh_token == rotated.refresh_token
    assert len(idp.token_requests) == posts

    assert chatgpt.auth_handler("status", _args()) is True
    assert chatgpt.auth_handler("logout", _args()) is True
    logged_out = _rows()[0]
    assert not logged_out.get("access_token") and not logged_out.get("refresh_token")
    assert not logged_out["chatgpt"].get("id_token")
    assert logged_out["chatgpt"]["client_id"] == idp.client_id
    assert idp.revocations[-1]["token"] == rotated.refresh_token
    assert idp.revocations[-1]["client_id"] == idp.client_id
    assert chatgpt.auth_handler("add", _args()) is True
    assert idp.authorizations[-1]["client_id"] == idp.client_id
    assert "id_token_hint" not in idp.authorizations[-1]
    output = capsys.readouterr().out
    assert metadata["id_token"] not in output
    assert first["access_token"] not in output and first["refresh_token"] not in output


@pytest.mark.parametrize("fault", ["state", "denied_state", "denied", "missing_client", "dynamic_client", "nonce", "iss", "aud", "missing_sub", "expired", "signature"])
def test_invalid_authorization_never_replaces_a_saved_registration(idp, fault):
    idp.fault = fault
    with pytest.raises(AuthError):
        chatgpt.auth_handler("add", _args())
    assert _rows() == []
    if fault in {"state", "denied_state", "denied", "missing_client", "dynamic_client"}:
        assert not idp.token_requests


def test_returning_identity_mismatch_and_missing_permission_remain_distinct(idp):
    chatgpt.auth_handler("add", _args())
    before = _rows()[0]
    idp.subject = "different-person"
    with pytest.raises(AuthError):
        chatgpt.auth_handler("add", _args())
    assert _rows()[0] == before
    idp.subject = "subject-one"
    idp.client_id = "oaiapp_other"
    posts = len(idp.token_requests)
    with pytest.raises(AuthError):
        chatgpt.auth_handler("add", _args())
    assert len(idp.token_requests) == posts and _rows()[0] == before
    idp.scope = "openid email profile offline_access resource.invoke"
    chatgpt.auth_handler("add", _args("work"))
    rows = _rows()
    assert len(rows) == 2 and rows[0]["id"] != rows[1]["id"]
    assert rows[0]["chatgpt"]["email"] == rows[1]["chatgpt"]["email"]
    assert "chatgpt.tokens.use.direct" not in rows[1]["chatgpt"]["scopes"]


def test_expired_code_retries_with_the_issued_client_instead_of_registering_again(idp):
    idp.reject_code_once = True
    chatgpt.auth_handler("add", _args())
    assert [a["client_id"] for a in idp.authorizations] == ["dynamic_agent_client", idp.client_id]
    assert "agent_name_hint" not in idp.authorizations[1]
    assert _rows()[0]["chatgpt"]["client_id"] == idp.client_id


def test_expired_dead_credential_retains_the_registration_for_reauthorization(idp):
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
    chatgpt.auth_handler("add", _args())
    first = _rows()[0]
    with _auth_store_lock():
        store = _load_auth_store()
        row = store["credential_pool"][chatgpt.PROVIDER][0]
        row.update(last_status="dead", last_status_at=time.time() - 90000)
        _save_auth_store(store)
    load_pool(chatgpt.PROVIDER).select()
    assert _rows() == []
    registration = chatgpt._select_registration(_args(), adding=True)
    assert registration["id"] == first["id"]
    assert registration["chatgpt"]["client_id"] == idp.client_id
    assert not registration.get("access_token") and not registration.get("refresh_token")
    chatgpt.auth_handler("add", _args())
    assert idp.authorizations[-1]["client_id"] == idp.client_id
    assert _rows()[0]["id"] == first["id"]


def test_logout_retries_a_network_failure_before_clearing_local_session(idp, monkeypatch):
    from agent.credential_pool import load_pool
    chatgpt.auth_handler("add", _args())
    refresh = _rows()[0]["refresh_token"]
    active = load_pool(chatgpt.PROVIDER).select()
    post = chatgpt.httpx.post
    attempts = []

    def sometimes_unreachable(url, **kwargs):
        assert not chatgpt.credential_is_eligible(active)
        attempts.append(url)
        if len(attempts) == 1:
            raise chatgpt.httpx.ConnectError("offline")
        return post(url, **kwargs)

    monkeypatch.setattr(chatgpt.httpx, "post", sometimes_unreachable)
    chatgpt.auth_handler("logout", _args())
    assert len(attempts) == 2 and idp.revocations[-1]["token"] == refresh
    assert not _rows()[0].get("access_token") and not _rows()[0].get("refresh_token")


def test_refresh_missing_scope_retains_granted_scope_but_reauthorization_can_remove_it(idp):
    from agent.credential_pool import load_pool
    chatgpt.auth_handler("add", _args())
    pool = load_pool(chatgpt.PROVIDER)
    before = _rows()[0]
    stale = pool.select()
    idp.fault = "omit_scope"
    rotated = pool.try_refresh_matching(credential_id=before["id"])
    assert rotated.refresh_token != before["refresh_token"]
    assert rotated.extra["chatgpt"]["scopes"] == before["chatgpt"]["scopes"]
    idp.fault = None
    idp.scope = "openid email profile offline_access resource.invoke"
    chatgpt.auth_handler("add", _args())
    assert not chatgpt.credential_is_eligible(stale)
    assert not pool.select()


def test_terminal_session_cleanup_preserves_only_the_account_registration(idp):
    from agent.credential_pool import PooledCredential
    chatgpt.auth_handler("add", _args())
    before = _rows()[0]
    entry = PooledCredential.from_dict(chatgpt.PROVIDER, before)
    cleared = chatgpt.clear_credential(entry)
    assert not cleared["access_token"] and cleared["refresh_token"] is None
    assert cleared["expires_at_ms"] is None and "id_token" not in cleared["chatgpt"]
    assert cleared["chatgpt"]["client_id"] == idp.client_id
    assert cleared["chatgpt"]["subject"] == idp.subject
    assert entry.extra["chatgpt"]["id_token"] == before["chatgpt"]["id_token"]


@pytest.mark.parametrize("error", ["invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused", "invalid_client",
    "temporarily_unavailable"])
def test_terminal_refresh_clears_session_but_configuration_and_transient_errors_retain_it(idp, error):
    from agent.credential_pool import load_pool
    chatgpt.auth_handler("add", _args())
    before = _rows()[0]
    idp.refresh_error = error
    assert load_pool(chatgpt.PROVIDER).try_refresh_matching(credential_id=before["id"]) is None
    after = _rows()[0]
    if error in {"invalid_client", "temporarily_unavailable"}:
        assert after["access_token"] == before["access_token"]
        assert after["refresh_token"] == before["refresh_token"]
    else:
        assert not after.get("access_token") and not after.get("refresh_token")
        assert not after["chatgpt"].get("id_token")
        assert after["last_status"] == "dead"
        assert after["chatgpt"]["client_id"] == idp.client_id
        idp.refresh_error = None
        chatgpt.auth_handler("add", _args())
        assert idp.authorizations[-1]["client_id"] == idp.client_id


def test_cached_client_cannot_send_after_logout_or_account_change(idp):
    from unittest.mock import Mock
    from agent.codex_runtime import run_codex_stream
    from agent.error_classifier import FailoverReason, classify_api_error
    chatgpt.auth_handler("add", _args())
    old_token = _rows()[0]["access_token"]
    client = Mock(base_url="https://api.openai.com/v1", api_key=old_token)
    client.responses.create.side_effect = RuntimeError("A signed-out client must not reach the wire")
    chatgpt.assert_active_access_token(old_token)
    chatgpt.auth_handler("logout", _args())
    with pytest.raises(AuthError, match="session"):
        chatgpt.assert_active_access_token(old_token)
    with pytest.raises(AuthError, match="session") as rejected:
        run_codex_stream(SimpleNamespace(provider=chatgpt.PROVIDER, _interrupt_requested=False),
                         {"model": "test", "input": []}, client=client)
    client.responses.create.assert_not_called()
    classified = classify_api_error(rejected.value, provider=chatgpt.PROVIDER)
    assert classified.reason == FailoverReason.provider_policy_blocked
    assert not classified.retryable and not classified.should_fallback and not classified.should_rotate_credential
    chatgpt.auth_handler("add", _args())
    with pytest.raises(AuthError, match="session"):
        chatgpt.assert_active_access_token(old_token)
    chatgpt.assert_active_access_token(_rows()[0]["access_token"])
