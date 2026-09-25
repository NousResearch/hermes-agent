"""A Codex credential is only ever sent to the host it belongs to (#121486).

Adversarial regressions for the catalog picker, the context probe, the auxiliary Codex client and
the image plugin behind a custom Codex gateway. Every case records each outbound request and
asserts that no ``Authorization`` header reaches a host other than the credential's own route —
including the ``model.base_url``-only gateway shape (``HERMES_CODEX_BASE_URL`` unset), an env/route
mismatch, opaque and JWT-shaped gateway keys, a pool-selected credential, and the direct-ChatGPT
positive control.
"""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

GW = "https://codex-gw.example/backend-api/codex"
OTHER_GW = "https://other-gw.example/backend-api/codex"
CHATGPT = "https://chatgpt.com/backend-api/codex"
OPAQUE = "dummy-gateway-pool-key"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _jwt(account: str = "acct") -> str:
    claims = {"sub": account, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    header = _b64(json.dumps({"alg": "RS256"}).encode())
    return f"{header}.{_b64(json.dumps(claims).encode())}.sig"


JWT = _jwt()
_REPO = Path(__file__).resolve().parents[2]


def _home(monkeypatch) -> Path:
    import os
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.delenv("HERMES_CODEX_BASE_URL", raising=False)
    return home


def _write_config(home: Path, *, base_url: str = "") -> None:
    model = {"provider": "openai-codex", "default": "gpt-5.5"}
    if base_url:
        model["base_url"] = base_url
    import yaml
    (home / "config.yaml").write_text(yaml.safe_dump({"model": model}))  # load cache keys on stat


def _write_pool(home: Path, key: str, *, row_base: str = CHATGPT) -> None:
    """A pool-only setup: the gateway key lives in ``credential_pool.openai-codex`` (no singleton)."""
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "gw", "label": "gateway", "auth_type": "api_key", "priority": 0,
            "source": "manual", "access_token": key, "base_url": row_base,
        }]},
    }))


def _write_singleton(home: Path, token: str) -> None:
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "active_provider": "openai-codex",
        "providers": {"openai-codex": {
            "tokens": {"access_token": token, "refresh_token": "rt"},
            "last_refresh": "2026-09-24T00:00:00Z", "auth_mode": "chatgpt"}},
    }))


def _authorized_hosts(seen) -> set:
    return {urlparse(url).hostname for url, auth in seen if auth}


def _catalog_recorder(seen):
    def get(url, headers=None, **_kw):
        seen.append((url, (headers or {}).get("Authorization", "")))
        return SimpleNamespace(status_code=200, json=lambda: {"models": [
            {"slug": "gpt-5.5", "visibility": "list", "priority": 1, "context_window": 272000}]})
    return get


@pytest.fixture
def picker_http(monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(get=_catalog_recorder(seen)))
    return seen


# ── /model picker ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("key", [OPAQUE, JWT], ids=["opaque", "jwt"])
def test_picker_model_base_url_only_gateway_keeps_pool_key_on_gateway(monkeypatch, picker_http, key):
    """``model.base_url`` gateway, env unset, pool-only key: the picker asks the gateway (the host
    the chat route uses for that key), never chatgpt.com."""
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, key)
    from hermes_cli.models import _codex_catalog

    _codex_catalog("openai-codex", force_refresh=True)

    assert _authorized_hosts(picker_http) == {"codex-gw.example"}


def test_picker_pool_row_with_its_own_gateway_url(monkeypatch, picker_http):
    """A pool row that carries its gateway URL is authoritative for its own key."""
    home = _home(monkeypatch)
    _write_config(home)
    _write_pool(home, JWT, row_base=OTHER_GW)
    from hermes_cli.models import _codex_catalog

    _codex_catalog("openai-codex", force_refresh=True)

    assert _authorized_hosts(picker_http) == {"other-gw.example"}


def test_picker_uses_the_route_base_not_a_mismatched_env(monkeypatch, picker_http):
    """env/route mismatch: the caller's resolved route wins over a stale process env — the
    credential is composed with the base it was resolved with, not an ambient re-read."""
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", OTHER_GW)
    from hermes_cli.codex_models import get_codex_model_ids

    get_codex_model_ids(access_token=JWT, base_url=GW)

    assert _authorized_hosts(picker_http) == {"codex-gw.example"}


def test_setup_flow_status_carries_the_pool_entry_route(monkeypatch):
    """``hermes model`` → Codex reads the token from the auth status; the status must carry the
    host that token routes to (model.base_url gateway), not leave the caller to assume chatgpt.com."""
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, OPAQUE)
    from hermes_cli.auth import get_codex_auth_status

    status = get_codex_auth_status()

    assert status["api_key"] == OPAQUE
    assert status["base_url"] == GW


def test_setup_flow_status_carries_a_pool_row_own_gateway_url(monkeypatch):
    home = _home(monkeypatch)
    _write_config(home)
    _write_pool(home, OPAQUE, row_base=OTHER_GW)
    from hermes_cli.auth import get_codex_auth_status

    assert get_codex_auth_status()["base_url"] == OTHER_GW


def test_model_setup_flow_catalog_goes_to_the_pool_entry_gateway(monkeypatch, picker_http):
    """``hermes model`` → OpenAI Codex with a pooled gateway key: the model list is fetched from
    the gateway that key routes to, not chatgpt.com."""
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, OPAQUE)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")  # reuse existing credentials
    confirm = {}
    monkeypatch.setattr("hermes_cli.auth._prompt_model_selection", lambda *a, **kw: confirm.update(kw))
    from hermes_cli.model_setup_flows import _model_flow_openai_codex

    _model_flow_openai_codex({}, current_model="gpt-5.5")

    assert _authorized_hosts(picker_http) == {"codex-gw.example"}
    # The confirm guards see the key together with its own route, not the chatgpt.com default.
    assert (confirm["confirm_api_key"], confirm["confirm_base_url"]) == (OPAQUE, GW)


def test_cli_default_model_swap_asks_the_session_route(monkeypatch, picker_http):
    """The CLI's untouched-default swap queries the catalog with the session's own
    ``(api_key, base_url)`` route, even when the process env names another gateway."""
    from types import MethodType

    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

    monkeypatch.setenv("HERMES_CODEX_BASE_URL", OTHER_GW)
    cli = SimpleNamespace(model="", api_key=JWT, base_url=GW, _model_is_default=True,
                          _console_print=lambda *a, **kw: None)
    cli._normalize_model_for_provider = MethodType(CLIModelSwitchMixin._normalize_model_for_provider, cli)

    cli._normalize_model_for_provider("openai-codex")

    assert _authorized_hosts(picker_http) == {"codex-gw.example"}


def test_picker_direct_chatgpt_positive_control(monkeypatch, picker_http):
    """No gateway anywhere: a ChatGPT OAuth JWT is still sent to chatgpt.com (allowed)."""
    home = _home(monkeypatch)
    _write_config(home)
    _write_singleton(home, JWT)
    from hermes_cli.models import _codex_catalog

    _codex_catalog("openai-codex", force_refresh=True)

    assert _authorized_hosts(picker_http) == {"chatgpt.com"}


def test_picker_refuses_opaque_key_aimed_at_chatgpt(picker_http):
    """Defense in depth: a non-JWT key composed with chatgpt.com is never sent there."""
    from hermes_cli.codex_models import get_codex_model_ids

    get_codex_model_ids(access_token=OPAQUE, base_url=CHATGPT)

    assert picker_http == []


# ── context probe (every chat turn without model.context_length) ───────────────────────────


@pytest.fixture
def probe_http(monkeypatch):
    from agent import model_metadata as mm
    seen = []
    monkeypatch.setattr(mm.model_metadata_http, "get", _catalog_recorder(seen))
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    return seen


@pytest.mark.parametrize("key", [OPAQUE, JWT], ids=["opaque", "jwt"])
def test_context_probe_asks_the_gateway_with_its_own_key(probe_http, key):
    """The route's base is bound to its key: a gateway key (opaque or JWT) probes that gateway's
    catalog, never chatgpt.com."""
    from agent import model_metadata as mm

    live, fresh = mm._fetch_codex_oauth_context_lengths_with_source(key, base_url=GW)

    assert _authorized_hosts(probe_http) == {"codex-gw.example"}
    assert fresh and live.get("gpt-5.5") == 272000


def test_context_probe_refuses_opaque_key_on_the_chatgpt_default(probe_http):
    from agent import model_metadata as mm

    assert mm._fetch_codex_oauth_context_lengths_with_source(OPAQUE, base_url="") == ({}, False)
    assert probe_http == []


def test_context_probe_direct_chatgpt_positive_control(probe_http):
    from agent import model_metadata as mm

    mm._fetch_codex_oauth_context_lengths_with_source(JWT, base_url=CHATGPT)

    assert _authorized_hosts(probe_http) == {"chatgpt.com"}


# ── auxiliary Codex client + image plugin (pool-selected credential) ───────────────────────


def test_aux_codex_client_binds_pool_key_to_model_base_url_gateway(monkeypatch):
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, OPAQUE)
    from agent import auxiliary_client as aux

    token, base_url = aux._resolve_codex_credential_and_base()

    assert (token, base_url) == (OPAQUE, GW)


def test_aux_codex_client_singleton_positive_control(monkeypatch):
    home = _home(monkeypatch)
    _write_config(home)
    _write_singleton(home, JWT)
    from agent import auxiliary_client as aux

    assert aux._resolve_codex_credential_and_base() == (JWT, CHATGPT)


def _load_image_plugin():
    spec = importlib.util.spec_from_file_location(
        "codex_img_binding_test", _REPO / "plugins/image_gen/openai-codex/__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", [OPAQUE, JWT], ids=["opaque", "jwt"])
def test_image_request_goes_to_the_pool_entry_gateway(monkeypatch, key):
    """Image generation with a pool-selected gateway key and a ``model.base_url``-only gateway:
    zero Authorization-bearing traffic to chatgpt.com."""
    import httpx

    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, key)
    img = _load_image_plugin()
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers.get("authorization", "")))
        return httpx.Response(200, json={"data": [{"b64_json": "aGk="}]}, request=request)

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(
        transport=httpx.MockTransport(handler), headers=kw.get("headers"), timeout=kw.get("timeout")))
    monkeypatch.setattr(img, "save_b64_image", lambda b64, prefix="": home / "img.png")

    result = img.OpenAICodexImageGenProvider().generate("a cat")

    assert result["success"] is True, result
    assert _authorized_hosts(seen) == {"codex-gw.example"}
    assert seen[0][1] == f"Bearer {key}"


# ── quota-restored probe + /usage (pool rows keep the canonical ChatGPT URL) ──────────────


class _UsageRecorder:
    """``httpx.Client`` stand-in recording every Authorization-bearing GET."""

    def __init__(self, seen):
        self.seen = seen

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None, **_kw):
        self.seen.append((str(url), (headers or {}).get("Authorization", "")))
        return SimpleNamespace(status_code=200, json=lambda: {
            "rate_limit": {"primary_window": {"used_percent": 10}}}, headers={})


def _exhausted_jwt_pool(home: Path) -> None:
    """A JWT-shaped gateway key in a rate-limited pool row that still carries the canonical URL."""
    import time
    now = time.time()
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {},
        "credential_pool": {"openai-codex": [{
            "id": "gw", "label": "gateway", "auth_type": "oauth", "priority": 0,
            "source": "device_code", "access_token": JWT, "base_url": CHATGPT,
            "last_status": "exhausted", "last_status_at": now, "last_error_code": 429,
            "last_error_reason": "usage_limit_reached", "last_error_message": "The usage limit has been reached",
            "last_error_reset_at": now + 3 * 24 * 3600,
        }]},
    }))


@pytest.fixture
def usage_probe_http(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import auth_codex
    seen = []
    auth_mod._codex_quota_probe_cache.clear()
    monkeypatch.setattr(auth_codex, "_codex_http_client", lambda **kw: _UsageRecorder(seen))
    yield seen
    auth_mod._codex_quota_probe_cache.clear()


def test_quota_restored_probe_of_a_persisted_entry_asks_the_gateway(monkeypatch, usage_probe_http):
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _exhausted_jwt_pool(home)
    from hermes_cli.auth_codex import _probe_codex_pool_entry_quota_restored

    entry = json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    assert _probe_codex_pool_entry_quota_restored(entry) is True

    assert _authorized_hosts(usage_probe_http) == {"codex-gw.example"}


def test_pool_selection_quota_probe_asks_the_gateway(monkeypatch, usage_probe_http):
    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _exhausted_jwt_pool(home)
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    (entry,) = pool.entries()
    assert pool._codex_quota_restored_upstream(entry) is True

    assert _authorized_hosts(usage_probe_http) == {"codex-gw.example"}


def test_quota_restored_probe_direct_chatgpt_positive_control(monkeypatch, usage_probe_http):
    home = _home(monkeypatch)
    _write_config(home)
    _exhausted_jwt_pool(home)
    from hermes_cli.auth_codex import _probe_codex_pool_entry_quota_restored

    entry = json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    _probe_codex_pool_entry_quota_restored(entry)

    assert _authorized_hosts(usage_probe_http) == {"chatgpt.com"}


def test_usage_pool_fallback_tier_sends_gateway_key_to_the_gateway(monkeypatch):
    """``/usage`` tier 3 (runtime resolver raised): the pooled key is paired with its route."""
    from agent import account_usage

    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    _write_pool(home, OPAQUE)
    seen = []

    def _no_runtime(**_kw):
        raise account_usage.AuthError("no creds", provider="openai-codex", code="codex_auth_missing")

    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials", _no_runtime)
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: _UsageRecorder(seen))

    account_usage.fetch_account_usage("openai-codex")

    assert _authorized_hosts(seen) == {"codex-gw.example"}
    assert seen[0][1] == f"Bearer {OPAQUE}"


def test_usage_forced_refresh_keeps_the_refreshed_pool_key_on_the_gateway(monkeypatch):
    """The 401 retry refreshes the live agent's own pool entry; the row's canonical URL must not
    pull the refreshed gateway key over to chatgpt.com."""
    from agent import account_usage

    home = _home(monkeypatch)
    _write_config(home, base_url=GW)
    monkeypatch.setattr(account_usage, "_read_codex_tokens", lambda: {"tokens": {}})

    class Pool:
        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            return SimpleNamespace(runtime_api_key="fresh-gw-key", runtime_base_url=CHATGPT)

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: Pool())

    token, base_url, _acct = account_usage._resolve_codex_usage_credentials(
        GW, "stale-gw-key", force_refresh=True)

    assert (token, base_url) == ("fresh-gw-key", GW)


def test_route_fallback_reads_the_profile_scoped_override_not_a_sibling_process_env(monkeypatch):
    """If route resolution itself fails, the fallback still honours only the routed profile's
    ``HERMES_CODEX_BASE_URL`` — never a multiplexed sibling's process env."""
    from agent import secret_scope
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import runtime_provider
    from hermes_cli.auth_codex import _codex_pool_route_base_url

    def _boom(*_a, **_kw):
        raise RuntimeError("route resolution failed")

    monkeypatch.setattr(runtime_provider, "_pool_entry_mode_and_url", _boom)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", OTHER_GW)
    for scope, expected in [({}, GW), ({"HERMES_CODEX_BASE_URL": OTHER_GW + "/"}, OTHER_GW)]:
        token = set_secret_scope(scope)
        try:
            assert _codex_pool_route_base_url(GW) == expected
        finally:
            reset_secret_scope(token)
