"""``hermes auth upgrade``: the free tier signs into a Nous account, keeping its connectors.

Driven through a fake portal covering the device-code endpoints plus the promotion intent/status
surface, so the wire contract (both codes in the intent, status-driven outcome, token grant
persisted over the guest singleton) is exercised rather than mocked away.
"""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from hermes_cli import anon_auth
from hermes_cli.auth import _auth_file_path, _load_auth_store

WELCOME = "https://welcome-api.nousresearch.com/v1"
INFERENCE = "https://inference-api.nousresearch.com/v1"
PORTAL = "https://portal.example.test"
REFRESH_TOKEN = "rt-upgraded-1"
EMAIL = "sid@example.test"


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


class FakePortal:
    """NAS anonymous surface + device-code endpoints. Records every call; scenarios flip behaviour."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.minted = 0
        self.intent_bodies: list[dict] = []
        self.status_bodies: list[dict] = []
        self.device_code = "dev-code-123"
        self.user_code = "ABCD-EFGH"
        # Sequence of promotion-status payloads; the last one repeats.
        self.status_sequence: list[dict] = [{"status": "completed", "user_id": "nas_user:9", "account_email": EMAIL}]
        self.token_grants = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path.startswith("/api/anonymous/") and not request.headers.get("x-anonymous-api-secret"):
            return httpx.Response(401, json={"error": "invalid_shared_secret"})
        if path == "/api/anonymous/create":
            self.minted += 1
            return httpx.Response(201, json={"user_id": f"nas_user:{self.minted}", "org_id": "nas_org:1",
                                             "token": f"anon_{self.minted:04d}", "idle_ttl_days": 14})
        if path == "/api/anonymous/token":
            return httpx.Response(200, json={"access_token": _jwt(), "token_type": "Bearer", "expires_in": 900,
                                             "user_id": "nas_user:1", "org_id": "nas_org:1",
                                             "inference_base_url": WELCOME})
        if path == "/api/oauth/device/code":
            return httpx.Response(200, json={
                "device_code": self.device_code, "user_code": self.user_code,
                "verification_uri": f"{PORTAL}/device",
                "verification_uri_complete": f"{PORTAL}/device?user_code={self.user_code}",
                "expires_in": 600, "interval": 5})
        if path == "/api/anonymous/promotion-intent":
            body = json.loads(request.content)
            self.intent_bodies.append(body)
            if not (body.get("user_code") and body.get("device_code")):
                return httpx.Response(400, json={"error": "invalid_request"})
            return httpx.Response(200, json={"claim_code": "clm_1", "claim_url": f"{PORTAL}/device",
                                             "expires_in": 600, "interval": 0})
        if path == "/api/anonymous/promotion-status":
            body = json.loads(request.content)
            self.status_bodies.append(body)
            idx = min(len(self.status_bodies) - 1, len(self.status_sequence) - 1)
            return httpx.Response(200, json=self.status_sequence[idx])
        if path == "/api/oauth/token":
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            if form.get("grant_type") != "urn:ietf:params:oauth:grant-type:device_code":
                return httpx.Response(400, json={"error": "unsupported_grant_type"})
            if form.get("device_code") != self.device_code:
                return httpx.Response(400, json={"error": "invalid_grant"})
            self.token_grants += 1
            return httpx.Response(200, json={
                "access_token": _jwt(sub="nas_user:9", client_id="hermes-cli", account_tier="free"),
                "refresh_token": REFRESH_TOKEN, "token_type": "Bearer", "expires_in": 900,
                "scope": "inference:invoke tool:invoke", "inference_base_url": INFERENCE})
        return httpx.Response(500, json={"error": f"unexpected {path}"})


@pytest.fixture
def portal(monkeypatch, tmp_path):
    fake = FakePortal()
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", PORTAL)
    monkeypatch.setenv("HERMES_ANON_API_SECRET", "test-secret")
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.delenv("HERMES_FORCE_GUEST", raising=False)
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from hermes_cli import auth_nous

    def _client(timeout_seconds, verify):
        return httpx.Client(transport=httpx.MockTransport(fake.handler), base_url=PORTAL)
    monkeypatch.setattr(auth_nous, "_nous_http_client", _client)
    real_client = httpx.Client

    class _RoutedClient(real_client):
        def __init__(self, *a, **kw):
            kw.pop("verify", None)
            kw["transport"] = httpx.MockTransport(fake.handler)
            super().__init__(*a, **kw)
    monkeypatch.setattr(httpx, "Client", _RoutedClient)
    anon_auth._background_started = False
    anon_auth._mint_failed = False
    anon_auth._forced_new_done = False
    return fake


def _shared_store(tmp_path) -> dict:
    p = tmp_path / "shared-store" / "nous_auth.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _args():
    return SimpleNamespace(no_browser=True, timeout=None)


class TestUpgrade:
    def test_intent_carries_both_device_codes_from_the_code_request(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)
        anon_auth.upgrade_guest(_args())
        assert len(portal.intent_bodies) == 1
        body = portal.intent_bodies[0]
        assert body["user_code"] == portal.user_code
        assert body["device_code"] == portal.device_code
        assert body["token"].startswith("anon_")
        paths = [p for _, p in portal.calls]
        assert paths.index("/api/oauth/device/code") < paths.index("/api/anonymous/promotion-intent")
        assert paths.index("/api/anonymous/promotion-intent") < paths.index("/api/oauth/token")

    def test_declined_in_browser_prints_copy_and_leaves_auth_store_untouched(self, portal, capsys, tmp_path):
        anon_auth.ensure_portal_identity(blocking=True)
        before = _auth_file_path().read_bytes()
        shared_before = _shared_store(tmp_path)
        portal.status_sequence = [{"status": "pending"}, {"status": "voided", "reason": "user_declined"}]
        code = anon_auth.upgrade_guest(_args())
        out = capsys.readouterr().out
        assert code == 1
        assert "Sign-in was rejected in the browser." in out
        assert portal.token_grants == 0
        assert _auth_file_path().read_bytes() == before
        assert _shared_store(tmp_path) == shared_before

    def test_completed_promotion_signs_in_and_keeps_no_free_tier_fields(self, portal, capsys, tmp_path):
        guest = anon_auth.ensure_portal_identity(blocking=True)
        assert _shared_store(tmp_path).get("anon_token") == guest["anon_token"]
        code = anon_auth.upgrade_guest(_args())
        out = capsys.readouterr().out
        assert code == 0
        assert f"Signed in as {EMAIL}. Your connectors are kept." in out
        lowered = out.lower()
        for banned in ("guest", "anonymous", "claim"):
            assert banned not in lowered, f"{banned!r} leaked into user-facing output:\n{out}"
        store = _load_auth_store()
        state = store["providers"]["nous"]
        assert store["active_provider"] == "nous"
        assert "anon_token" not in state
        assert state.get("auth_method") != anon_auth.ANON_AUTH_METHOD
        assert not anon_auth.is_guest_state(state)
        assert state["refresh_token"] == REFRESH_TOKEN
        shared = _shared_store(tmp_path)
        assert shared.get("refresh_token") == REFRESH_TOKEN
        assert "anon_token" not in shared
