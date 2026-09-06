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


FREE_PICK = "upstage/solar-pro4:free"


@pytest.fixture
def free_account(monkeypatch):
    """The signed-in account is a $0 (free-plan) account: the Portal's tier read and its recommended
    free list are the only network egress the default pick has, stubbed at their seams."""
    from hermes_cli import models as m
    from hermes_cli import models_pricing as mp
    monkeypatch.setattr(m, "check_nous_free_tier", lambda **kw: True)
    monkeypatch.setattr(m, "fetch_nous_recommended_models", lambda *a, **kw: {
        "freeRecommendedModels": [{"modelName": FREE_PICK}]})
    monkeypatch.setattr(mp, "get_pricing_for_provider", lambda *a, **kw: {})
    monkeypatch.setattr(mp, "nous_policy_allowed_ids", lambda **kw: None)


def _write_model_config(model_cfg: dict) -> None:
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["model"] = model_cfg
    save_config(config)


def _model_config() -> dict:
    from hermes_cli.config import load_config_readonly
    return dict(load_config_readonly().get("model") or {})


class TestSignInCompletionSettlesTheModel:
    def test_config_on_the_free_tier_route_moves_to_the_account_host_and_the_recommended_free_model(
            self, portal, free_account, capsys):
        anon_auth.ensure_portal_identity(blocking=True)
        # What picking the free-tier row leaves behind: the welcome model pinned to the welcome host.
        _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})
        assert anon_auth.upgrade_guest(_args()) == 0
        model_cfg = _model_config()
        assert model_cfg["default"] == FREE_PICK
        assert model_cfg["base_url"] == INFERENCE.rstrip("/")
        assert not anon_auth.route_is_welcome_host(model_cfg["base_url"])
        assert f"Default model is now {FREE_PICK}." in capsys.readouterr().out

    def test_config_on_the_users_own_model_is_left_alone(self, portal, free_account, capsys):
        anon_auth.ensure_portal_identity(blocking=True)
        own = {"provider": "openrouter", "default": "anthropic/claude-sonnet-4"}
        _write_model_config(own)
        assert anon_auth.upgrade_guest(_args()) == 0
        assert {k: _model_config().get(k) for k in own} == own
        assert "Default model is now" not in capsys.readouterr().out

    def test_no_eligible_recommendation_leaves_no_default_rather_than_a_model_the_account_may_not_use(
            self, portal, free_account, monkeypatch, capsys):
        from hermes_cli import models as m
        anon_auth.ensure_portal_identity(blocking=True)
        _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})
        def _portal_down():
            raise RuntimeError("recommended models unavailable")
        monkeypatch.setattr(m, "recommended_nous_default_model", _portal_down)
        assert anon_auth.upgrade_guest(_args()) == 0
        model_cfg = _model_config()
        assert "default" not in model_cfg
        assert model_cfg["base_url"] == INFERENCE.rstrip("/")
        out = capsys.readouterr().out
        assert "Default model is now" not in out
        assert "run `hermes model` to pick one" in out
