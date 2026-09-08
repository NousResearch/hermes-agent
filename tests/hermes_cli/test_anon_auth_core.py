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
        # What the token exchange names as the inference host; None = an older NAS that omits it.
        self.inference_base_url: str | None = WELCOME

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
            body = {"access_token": _jwt(), "token_type": "Bearer", "expires_in": 900,
                    "user_id": "nas_user:1", "org_id": "nas_org:1"}
            if self.inference_base_url:
                body["inference_base_url"] = self.inference_base_url
            return httpx.Response(200, json=body)
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
    anon_auth._mint_failed = False
    anon_auth._forced_new_done = False
    # resolve_nous_access_token memoises the last token for 5 s across the process; a token minted
    # by an earlier test must not be served to this one.
    from hermes_cli import auth as auth_mod
    monkeypatch.setattr(auth_mod, "_RESOLVE_TOKEN_CACHE", None)
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
        # A process tries once: later bootstrap sites must not hit the portal again.
        assert anon_auth.ensure_portal_identity(blocking=True) is None
        assert [p for _, p in portal.calls].count("/api/anonymous/create") == 1

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


class TestSetupPolicy:
    """``nous.guest_setup: explicit``: nothing creates the identity on its own — not the resolver's
    last rung, not a ``nous/welcome`` request, not a sign-in. Only :func:`provision_free_tier` (the
    guided setup's ``free_tier.provision``) does; every implicit caller then adopts it."""

    def test_explicit_implicit_callers_never_mint(self, portal, monkeypatch):
        _write_config(monkeypatch, guest_setup="explicit")
        assert anon_auth.guest_setup_policy() == anon_auth.GUEST_SETUP_EXPLICIT
        assert anon_auth.ensure_portal_identity(blocking=True) is None
        with pytest.raises(anon_auth.AuthError):
            resolve_provider("auto")               # the resolver's last rung is implicit
        assert portal.minted == 0 and portal.calls == []

    def test_explicit_welcome_model_on_the_nous_route_does_not_mint(self, portal, monkeypatch):
        _write_config(monkeypatch, guest_setup="explicit")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        with pytest.raises(anon_auth.AuthError):
            resolve_runtime_provider(requested="nous", target_model=anon_auth.GUEST_MODEL)
        assert portal.minted == 0

    def test_explicit_sign_in_has_nothing_to_sign_in_from(self, portal, monkeypatch):
        _write_config(monkeypatch, guest_setup="explicit")
        states = list(anon_auth.run_sign_in(timeout_seconds=1.0))
        assert len(states) == 1 and states[0].kind == "unavailable"
        assert portal.minted == 0

    def test_explicit_provision_mints_once_then_everything_adopts(self, portal, monkeypatch, tmp_path):
        _write_config(monkeypatch, guest_setup="explicit")
        assert anon_auth.is_guest_state(anon_auth.provision_free_tier())
        assert portal.minted == 1
        token = _load_auth_store()["providers"]["nous"]["anon_token"]
        # A second provision is idempotent, and the runtime now serves nous/welcome on the welcome host.
        anon_auth.provision_free_tier()
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested="nous", target_model=anon_auth.GUEST_MODEL)
        assert runtime["base_url"].rstrip("/") == WELCOME
        assert resolve_provider("auto") == "nous"
        # A sibling profile adopts the same identity implicitly; no second create call.
        sibling = tmp_path / "sibling-profile"
        sibling.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(sibling))
        _write_config(monkeypatch, guest_setup="explicit")
        adopted = anon_auth.ensure_portal_identity(blocking=True)
        assert adopted and adopted["anon_token"] == token
        assert portal.minted == 1

    def test_explicit_retired_identity_is_replaced(self, portal, monkeypatch):
        _write_config(monkeypatch, guest_setup="explicit")
        anon_auth.provision_free_tier()
        first = _load_auth_store()["providers"]["nous"]["anon_token"]
        portal.dead_tokens.add(first)
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        assert resolve_nous_runtime_credentials(force_refresh=True)["api_key"]   # replaced, not refused
        assert _load_auth_store()["providers"]["nous"]["anon_token"] != first
        assert portal.minted == 2

    def test_auto_policy_keeps_todays_first_use_mint(self, portal, monkeypatch):
        _write_config(monkeypatch, guest_setup="auto")
        assert resolve_provider("auto") == "nous"
        assert portal.minted == 1

    def test_guest_off_beats_an_explicit_provision(self, portal, monkeypatch):
        _write_config(monkeypatch, guest=False, guest_setup="explicit")
        assert anon_auth.provision_free_tier() is None
        assert portal.minted == 0


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


class TestRouteFallback:
    """A guest never falls back to the paid host: the gateway cross-refuses an anonymous JWT there."""

    def test_exchange_without_inference_url_routes_to_welcome_literal(self, portal):
        portal.inference_base_url = None
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()
        assert runtime["base_url"].rstrip("/") == WELCOME
        state = _load_auth_store()["providers"]["nous"]
        assert state["inference_base_url"].rstrip("/") == WELCOME

    def test_disallowed_inference_host_heals_to_welcome_literal(self, portal):
        portal.inference_base_url = "https://welcome-api.staging-nousresearch.com/v1"
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider()
        assert runtime["base_url"].rstrip("/") == WELCOME

    def test_guest_state_without_url_never_resolves_to_the_paid_host(self, portal):
        from hermes_cli.auth_nous import _nous_effective_routing
        guest = {"auth_method": "anonymous", "anon_token": "anon_x"}
        _portal, stored, effective, _client = _nous_effective_routing(guest)
        assert stored.rstrip("/") == WELCOME and effective.rstrip("/") == WELCOME
        _portal, stored, _effective, _client = _nous_effective_routing({"refresh_token": "r"})
        assert stored.rstrip("/") == "https://inference-api.nousresearch.com/v1"

    def test_shared_store_shape_keeps_a_guest_on_the_welcome_host(self, portal):
        from hermes_cli.auth_nous import _nous_shared_shape
        shape = _nous_shared_shape({"auth_method": "anonymous", "anon_token": "anon_x"})
        assert shape["inference_base_url"].rstrip("/") == WELCOME


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
    """The pin is a property of the selected ROUTE (welcome host), never of profile state: a paid
    pool credential routed to the portal host keeps its model even beside a guest singleton."""

    def test_pin_keys_on_the_welcome_host_not_on_guest_state(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)  # guest singleton exists
        assert anon_auth.route_is_welcome_host(WELCOME)
        assert not anon_auth.route_is_welcome_host("https://inference-api.nousresearch.com/v1")
        assert not anon_auth.route_is_welcome_host("")

    def test_agent_init_pins_only_on_welcome_route(self, portal):
        anon_auth.ensure_portal_identity(blocking=True)
        from run_agent import AIAgent
        welcome = AIAgent(provider="nous", base_url=WELCOME, api_key="k", model="openai/gpt-5",
                          quiet_mode=True, skip_context_files=True, skip_memory=True)
        paid = AIAgent(provider="nous", base_url="https://inference-api.nousresearch.com/v1", api_key="k",
                       model="nous/paid-model", quiet_mode=True, skip_context_files=True, skip_memory=True)
        assert welcome.model == anon_auth.GUEST_MODEL
        assert paid.model == "nous/paid-model"


class TestLogout:
    def test_logout_with_only_free_tier_is_a_true_noop(self, portal, capsys):
        from types import SimpleNamespace
        from hermes_cli.auth import _auth_file_path, logout_command
        anon_auth.ensure_portal_identity(blocking=True)
        before = _auth_file_path().read_bytes()
        logout_command(SimpleNamespace(provider=None))
        out = capsys.readouterr().out.lower()
        assert "not signed in" in out
        assert "guest" not in out and "anonymous" not in out
        assert _auth_file_path().read_bytes() == before

    def test_logout_of_real_account_clears_shared_store(self, portal, tmp_path):
        from types import SimpleNamespace
        from hermes_cli.auth import logout_command
        from hermes_cli.auth_nous import persist_nous_credentials
        persist_nous_credentials({"access_token": _jwt(client_id="hermes-cli", account_tier="free"),
                                  "refresh_token": "rt-1", "expires_at": "2030-01-01T00:00:00+00:00",
                                  "auth_method": "oauth_device_code"})
        assert _shared_store(tmp_path).get("refresh_token") == "rt-1"
        logout_command(SimpleNamespace(provider="nous"))
        assert _shared_store(tmp_path) == {}
        assert "nous" not in _load_auth_store().get("providers", {})


class TestModelSwitchCopy:
    def test_switching_away_from_welcome_names_the_account_path_not_another_provider(self, portal, monkeypatch):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli import model_switch
        monkeypatch.setattr(model_switch, "list_provider_models", lambda *a, **k: [], raising=False)
        result = model_switch.switch_model("gpt-5", "nous", anon_auth.GUEST_MODEL, WELCOME)
        assert not result.success
        msg = (result.error_message or "").lower()
        assert "/login" in msg
        assert "openrouter" not in msg and "switching" not in msg


class TestRotationNeverRewritesTheConversationModel:
    """A credential rotation adopts an entry only if its route can serve the conversation's model.
    The model is never changed by a swap; an ineligible entry is refused (swap returns False)."""

    def _agent(self, api_mode="chat_completions", model="nous/paid-model"):
        from types import SimpleNamespace
        return SimpleNamespace(provider="nous", api_mode=api_mode, base_url="https://inference-api.nousresearch.com/v1",
                               api_key="k", model=model, _client_kwargs={}, _credential_pool_entry_id="p",
                               _reapply_route_client_config=lambda **kw: None, _replace_primary_openai_client=lambda **kw: None,
                               _anthropic_client=SimpleNamespace(close=lambda: None),
                               _build_direct_anthropic_client=lambda key, url: object(), _anthropic_oauth_flag=lambda key: False)

    def test_paid_conversation_refuses_a_welcome_route_on_every_wire_mode(self, portal):
        from types import SimpleNamespace
        from agent.client_lifecycle import ClientLifecycleMixin
        for mode, model in (("chat_completions", "nous/paid-model"), ("anthropic_messages", "anthropic/claude-sonnet")):
            agent = self._agent(mode, model)
            ok = ClientLifecycleMixin._swap_credential(agent, SimpleNamespace(id="g", runtime_api_key="jwt", runtime_base_url=WELCOME))
            assert ok is False
            assert agent.model == model and agent.api_key == "k" and agent._credential_pool_entry_id == "p"

    def test_welcome_conversation_may_move_to_the_portal_host(self, portal):
        from types import SimpleNamespace
        from agent.client_lifecycle import ClientLifecycleMixin
        agent = self._agent(model=anon_auth.GUEST_MODEL); agent.base_url = WELCOME
        ok = ClientLifecycleMixin._swap_credential(agent, SimpleNamespace(id="p2", runtime_api_key="key", runtime_base_url="https://inference-api.nousresearch.com/v1"))
        assert ok is True and agent.model == anon_auth.GUEST_MODEL


class TestBackgroundRetry:
    def test_background_failure_releases_the_latch(self, portal):
        import time as _t
        portal.gate_closed = True
        assert anon_auth.ensure_portal_identity(blocking=False) is None
        for _ in range(50):
            if not anon_auth._background_started:
                break
            _t.sleep(0.05)
        assert anon_auth._background_started is False, "a failed background attempt must not consume the latch"
        portal.gate_closed = False
        anon_auth.ensure_portal_identity(blocking=False)
        for _ in range(50):
            if anon_auth.has_guest():
                break
            _t.sleep(0.05)
        assert anon_auth.has_guest()


class TestBackgroundLatchOnThreadFailure:
    def test_thread_start_failure_releases_latch(self, portal, monkeypatch):
        import threading
        class Boom(threading.Thread):
            def start(self): raise RuntimeError("can't start new thread")
        monkeypatch.setattr(anon_auth.threading, "Thread", Boom)
        assert anon_auth.ensure_portal_identity(blocking=False) is None
        assert anon_auth._background_started is False


class TestIdentityOfRecordIsTheSharedStore:
    def test_stale_profile_guest_adopts_a_newer_shared_account(self, portal, tmp_path):
        anon_auth.ensure_portal_identity(blocking=True)
        # A sibling profile signed in: the shared store now holds a real account.
        from hermes_cli.auth_nous import _write_shared_nous_state
        _write_shared_nous_state({"access_token": _jwt(client_id="hermes-cli", account_tier="free"),
                                  "refresh_token": "rt-sibling", "expires_at": "2030-01-01T00:00:00+00:00",
                                  "auth_method": "oauth_device_code"})
        state = anon_auth.ensure_portal_identity(blocking=True)
        assert not anon_auth.is_guest_state(state)
        assert state["refresh_token"] == "rt-sibling"
        assert _load_auth_store()["providers"]["nous"]["refresh_token"] == "rt-sibling"
        assert _shared_store(tmp_path)["refresh_token"] == "rt-sibling", "the profile must never overwrite the shared account"

    def test_mint_persists_before_any_exchange_and_first_use_exchanges_once(self, portal):
        first = anon_auth.ensure_portal_identity(blocking=True)
        assert anon_auth.is_guest_state(first) and "access_token" not in first
        assert [p for _, p in portal.calls] == ["/api/anonymous/create"], "mint alone; exchange is lazy"
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        creds = resolve_nous_runtime_credentials()
        assert creds["api_key"]
        assert portal.minted == 1, "a stored credential is exchanged, never re-minted"
        assert [p for _, p in portal.calls].count("/api/anonymous/token") == 1

    def test_clearing_a_dead_guest_leaves_a_sibling_identity_alone(self, portal, tmp_path):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.auth_nous import _write_shared_nous_state
        _write_shared_nous_state({"access_token": _jwt(client_id="hermes-cli"), "refresh_token": "rt-sibling",
                                  "expires_at": "2030-01-01T00:00:00+00:00", "auth_method": "oauth_device_code"})
        anon_auth.clear_dead_guest("test")
        assert "nous" not in _load_auth_store().get("providers", {})
        assert _shared_store(tmp_path)["refresh_token"] == "rt-sibling"

    def test_lock_order_is_profile_then_shared(self, portal, monkeypatch):
        order = []
        from hermes_cli import auth as auth_mod, auth_nous
        real_profile, real_shared = auth_mod._auth_store_lock, auth_nous._nous_shared_store_lock
        from contextlib import contextmanager

        @contextmanager
        def profile(*a, **k):
            order.append("profile")
            with real_profile(*a, **k):
                yield

        @contextmanager
        def shared(*a, **k):
            order.append("shared")
            with real_shared(*a, **k):
                yield
        monkeypatch.setattr(auth_mod, "_auth_store_lock", profile)
        monkeypatch.setattr(auth_nous, "_nous_shared_store_lock", shared)
        anon_auth.ensure_portal_identity(blocking=True)
        assert order[:2] == ["profile", "shared"]


class TestConnectorTokenPath:
    def test_opt_out_hides_the_free_tier_from_connectors_including_cached_tokens(self, portal, monkeypatch):
        anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.auth_nous import resolve_nous_runtime_credentials
        resolve_nous_runtime_credentials()  # now a cached, valid JWT exists
        from tools import managed_tool_gateway as mtg
        assert mtg.read_nous_access_token()
        _write_config(monkeypatch, guest=False)
        assert mtg.peek_nous_access_token() is None
        assert mtg.read_nous_access_token() is None

    def test_connector_path_replaces_a_dead_credential_once(self, portal):
        first = anon_auth.ensure_portal_identity(blocking=True)
        from hermes_cli.auth import _auth_store_lock, _save_auth_store
        with _auth_store_lock():
            store = _load_auth_store()
            store["providers"]["nous"]["expires_at"] = "2000-01-01T00:00:00+00:00"
            store["providers"]["nous"]["access_token"] = _jwt(exp=1)
            _save_auth_store(store)
        portal.dead_tokens.add(first["anon_token"])
        from tools import managed_tool_gateway as mtg
        token = mtg.read_nous_access_token()
        assert token and token != _jwt(exp=1)
        assert _load_auth_store()["providers"]["nous"]["anon_token"] != first["anon_token"]
        assert portal.minted == 2
