"""Nous free tier core: identity lifecycle, token-acquisition seam, routing pin, opt-out.

Behaviour contracts on the public seams (``ensure_portal_identity``, ``resolve_provider``,
``resolve_runtime_provider``, ``normalize_model_for_provider``), driven through a fake portal so
the wire contract is exercised, never mocked away.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from hermes_cli import anon_auth
from hermes_cli.auth import _load_auth_store, resolve_provider

WELCOME = "https://welcome-api.nousresearch.com/v1"
PORTAL = "https://portal.example.test"


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


class FakePortal:
    """Minimal NAS anonymous surface. Records every call; scenarios flip its behaviour."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.dead_tokens: set[str] = set()
        self.gate_closed = False
        self.minted = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path.startswith("/api/anonymous/") and not request.headers.get("x-anonymous-api-secret"):
            return httpx.Response(401, json={"error": "invalid_shared_secret"})
        if self.gate_closed:
            return httpx.Response(401, json={"error": "invalid_shared_secret"})
        if path == "/api/anonymous/create":
            self.minted += 1
            return httpx.Response(201, json={"user_id": f"nas_user:{self.minted}", "org_id": "nas_org:1",
                                             "token": f"anon_{self.minted:04d}", "idle_ttl_days": 14})
        if path == "/api/anonymous/token":
            token = json.loads(request.content)["token"]
            if token in self.dead_tokens:
                return httpx.Response(404, json={"error": "unknown_token"})
            return httpx.Response(200, json={"access_token": _jwt(), "token_type": "Bearer", "expires_in": 900,
                                             "user_id": "nas_user:1", "org_id": "nas_org:1",
                                             "inference_base_url": WELCOME})
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
    # resolve_nous_access_token builds its own client; route it through the fake too.
    real_client = httpx.Client

    class _RoutedClient(real_client):
        def __init__(self, *a, **kw):
            kw.pop("verify", None)
            kw["transport"] = httpx.MockTransport(fake.handler)
            super().__init__(*a, **kw)
    monkeypatch.setattr(httpx, "Client", _RoutedClient)
    anon_auth._background_started = False
    return fake


def _write_config(monkeypatch, **nous):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("nous:\n" + "".join(f"  {k}: {str(v).lower()}\n" for k, v in nous.items()))
    from hermes_cli import config as cfg_mod
    for attr in ("_config_cache", "_cached_config"):
        if hasattr(cfg_mod, attr):
            monkeypatch.setattr(cfg_mod, attr, None, raising=False)


def _shared_store(tmp_path) -> dict:
    p = tmp_path / "shared-store" / "nous_auth.json"
    return json.loads(p.read_text()) if p.exists() else {}


class TestIdentityLifecycle:
    def test_fresh_install_mints_once_and_is_the_active_provider(self, portal, tmp_path):
        state = anon_auth.ensure_portal_identity(blocking=True)
        assert anon_auth.is_guest_state(state)
        assert "refresh_token" not in state
        store = _load_auth_store()
        assert store["active_provider"] == "nous"
        assert anon_auth.is_guest_state(store["providers"]["nous"])
        assert _shared_store(tmp_path).get("anon_token") == state["anon_token"]
        assert portal.minted == 1
        # Second call: identity exists, zero network.
        before = len(portal.calls)
        assert anon_auth.ensure_portal_identity(blocking=True)["anon_token"] == state["anon_token"]
        assert len(portal.calls) == before

    def test_second_profile_under_same_root_adopts_from_shared_store(self, portal, tmp_path, monkeypatch):
        first = anon_auth.ensure_portal_identity(blocking=True)
        other_home = tmp_path / "profiles" / "two"
        other_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(other_home))
        before = len(portal.calls)
        second = anon_auth.ensure_portal_identity(blocking=True)
        assert second["anon_token"] == first["anon_token"]
        assert len(portal.calls) == before, "adoption must not touch the network"
        assert portal.minted == 1

    def test_gate_closed_persists_nothing_and_raises_gate_code(self, portal):
        portal.gate_closed = True
        with pytest.raises(anon_auth.AuthError) as exc:
            anon_auth.ensure_portal_identity(blocking=True)
        assert exc.value.code == "anon_gate_closed"
        assert "nous" not in _load_auth_store().get("providers", {})

    def test_opt_out_bool_disables_everything(self, portal, monkeypatch):
        _write_config(monkeypatch, guest=False)
        assert anon_auth.ensure_portal_identity(blocking=True) is None
        assert portal.calls == []
        with pytest.raises(anon_auth.AuthError):
            resolve_provider("auto")

    def test_force_guest_overrides_opt_out_and_new_bypasses_shared_store(self, portal, monkeypatch):
        _write_config(monkeypatch, guest=False)
        monkeypatch.setenv("HERMES_FORCE_GUEST", "1")
        first = anon_auth.ensure_portal_identity(blocking=True)
        assert anon_auth.is_guest_state(first)
        monkeypatch.setenv("HERMES_FORCE_GUEST", "new")
        second = anon_auth.ensure_portal_identity(blocking=True)
        assert second["anon_token"] != first["anon_token"]
        assert portal.minted == 2


class TestResolverIsUnchanged:
    def test_guest_is_last_resort_and_explicit_key_wins(self, portal, monkeypatch):
        anon_auth.ensure_portal_identity(blocking=True)
        assert resolve_provider("auto") == "nous"
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        assert resolve_provider("auto") == "openrouter"

    def test_runtime_routes_to_welcome_host(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()
        assert runtime["provider"] == "nous"
        assert runtime["base_url"].rstrip("/") == WELCOME
        assert runtime["api_key"]


class TestTokenAcquisitionSeam:
    def test_expired_guest_jwt_reexchanges_and_never_hits_oauth_token(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.auth import _auth_store_lock, _save_auth_store
        with _auth_store_lock():
            store = _load_auth_store()
            store["providers"]["nous"]["access_token"] = _jwt(exp=int(time.time()) - 10)
            store["providers"]["nous"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            _save_auth_store(store)
        portal.calls.clear()
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials()
        paths = [p for _, p in portal.calls]
        assert paths == ["/api/anonymous/token"]
        assert "/api/oauth/token" not in paths
        assert creds["base_url"].rstrip("/") == WELCOME
        assert "quarantine" not in json.dumps(_load_auth_store())

    def test_dead_credential_is_replaced_by_a_fresh_identity(self, portal):
        first = anon_auth.ensure_portal_identity(blocking=True)
        portal.dead_tokens.add(first["anon_token"])
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials(force_refresh=True)
        assert creds["api_key"]
        state = _load_auth_store()["providers"]["nous"]
        assert state["anon_token"] != first["anon_token"]
        assert portal.minted == 2

    def test_tool_gateway_token_path_reexchanges(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.auth import _auth_store_lock, _save_auth_store, resolve_nous_access_token
        with _auth_store_lock():
            store = _load_auth_store()
            store["providers"]["nous"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            _save_auth_store(store)
        portal.calls.clear()
        token = resolve_nous_access_token()
        assert token
        assert [p for _, p in portal.calls] == ["/api/anonymous/token"]


class TestModelPin:
    def test_normalize_pins_welcome_model_only_while_guest_carries_inference(self, portal):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("openai/gpt-5", "nous") == "openai/gpt-5"
        anon_auth.ensure_portal_identity(blocking=True)
        assert normalize_model_for_provider("openai/gpt-5", "nous") == anon_auth.GUEST_MODEL
        assert normalize_model_for_provider("gpt-5", "openrouter") != anon_auth.GUEST_MODEL
