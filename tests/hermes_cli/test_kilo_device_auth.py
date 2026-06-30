"""Tests for Kilo Gateway device authorization (hermes_cli/kilo_auth.py).

Covers the custom device-auth flow (POST initiate + GET poll with HTTP status
codes), profile/defaults fetching, organization selection, credential-pool
storage, the org header mirror into ``model.default_headers``, and logout
cleanup. No live network calls — httpx is stubbed.
"""

import json
import types
from unittest.mock import patch

import pytest

from hermes_cli import kilo_auth
from hermes_cli.kilo_auth import (
    DEFAULT_KILO_API_BASE,
    KILO_ORG_HEADER,
    _initiate_device_auth,
    _poll_device_auth,
    _strip_api_segment,
    fetch_kilo_default_model,
    fetch_kilo_profile,
    kilo_api_base,
    kilo_device_auth_login,
    set_kilo_org_header,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code, json_data=None, headers=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._json


class _FakeClient:
    """Drop-in for httpx.Client with a queue of responses per verb."""

    def __init__(self, post_responses=None, get_responses=None):
        self._post = list(post_responses or [])
        self._get = list(get_responses or [])
        self.post_calls = []
        self.get_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.post_calls.append((args, kwargs))
        return self._post.pop(0) if self._post else _FakeResp(500)

    def get(self, *args, **kwargs):
        self.get_calls.append((args, kwargs))
        return self._get.pop(0) if self._get else _FakeResp(500)


def _patch_httpx_client(monkeypatch, client):
    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.Client", lambda *a, **k: client)


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------

def test_kilo_api_base_default(monkeypatch):
    monkeypatch.delenv("KILO_API_URL", raising=False)
    monkeypatch.delenv("KILOCODE_BASE_URL", raising=False)
    assert kilo_api_base() == DEFAULT_KILO_API_BASE


def test_kilo_api_base_kilo_api_url_override(monkeypatch):
    monkeypatch.setenv("KILO_API_URL", "https://staging.kilo.ai")
    assert kilo_api_base() == "https://staging.kilo.ai"


def test_kilo_api_base_derived_from_gateway_override(monkeypatch):
    monkeypatch.delenv("KILO_API_URL", raising=False)
    monkeypatch.setenv("KILOCODE_BASE_URL", "https://api.kilo.ai/api/gateway")
    assert kilo_api_base() == "https://api.kilo.ai"


def test_strip_api_segment_handles_root():
    assert _strip_api_segment("https://api.kilo.ai") == "https://api.kilo.ai"
    assert _strip_api_segment("https://api.kilo.ai/api/gateway") == "https://api.kilo.ai"
    assert _strip_api_segment("https://api.kilo.ai/api/openrouter/v1") == "https://api.kilo.ai"
    assert _strip_api_segment("https://host.example/base/api/gateway") == "https://host.example/base"


# ---------------------------------------------------------------------------
# _initiate_device_auth
# ---------------------------------------------------------------------------

def test_initiate_device_auth_posts_and_returns_code():
    client = _FakeClient(post_responses=[
        _FakeResp(200, {"code": "ABCD-EFGH", "verificationUrl": "https://app.kilo.ai/device-auth?code=ABCD-EFGH", "expiresIn": 599}),
    ])
    data = _initiate_device_auth(client, DEFAULT_KILO_API_BASE)
    assert data["code"] == "ABCD-EFGH"
    assert data["verificationUrl"].startswith("https://app.kilo.ai/device-auth")
    assert data["expiresIn"] == 599

    args, kwargs = client.post_calls[0]
    assert args[0] == f"{DEFAULT_KILO_API_BASE}/api/device-auth/codes"


def test_initiate_device_auth_429_raises(monkeypatch):
    client = _FakeClient(post_responses=[_FakeResp(429)])
    with pytest.raises(RuntimeError, match="Too many pending"):
        _initiate_device_auth(client, DEFAULT_KILO_API_BASE)


def test_initiate_device_auth_missing_fields_raises():
    client = _FakeClient(post_responses=[_FakeResp(200, {"code": "X"})])
    with pytest.raises(RuntimeError, match="missing fields"):
        _initiate_device_auth(client, DEFAULT_KILO_API_BASE)


# ---------------------------------------------------------------------------
# _poll_device_auth
# ---------------------------------------------------------------------------

def test_poll_pending_then_approved(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = _FakeClient(get_responses=[
        _FakeResp(202),                          # pending
        _FakeResp(202),                          # pending
        _FakeResp(200, {"status": "approved", "token": "tok-123", "userEmail": "u@x.io"}),
    ])
    result = _poll_device_auth(client, DEFAULT_KILO_API_BASE, "ABCD-EFGH", expires_in=60)
    assert result["token"] == "tok-123"
    assert result["userEmail"] == "u@x.io"
    # The poll URL embeds the code.
    assert client.get_calls[0][0][0] == f"{DEFAULT_KILO_API_BASE}/api/device-auth/codes/ABCD-EFGH"


def test_poll_denied_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = _FakeClient(get_responses=[_FakeResp(403)])
    with pytest.raises(RuntimeError, match="denied"):
        _poll_device_auth(client, DEFAULT_KILO_API_BASE, "CODE", expires_in=60)


def test_poll_expired_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = _FakeClient(get_responses=[_FakeResp(410)])
    with pytest.raises(RuntimeError, match="expired"):
        _poll_device_auth(client, DEFAULT_KILO_API_BASE, "CODE", expires_in=60)


def test_poll_timeout_raises(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    # Drive time.monotonic past the deadline after the first pending poll so
    # the loop exits with TimeoutError instead of spinning forever.
    ticks = iter([1000.0, 1000.0, 2000.0])
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    client = _FakeClient(get_responses=[_FakeResp(202), _FakeResp(202)])
    with pytest.raises(TimeoutError, match="Timed out"):
        _poll_device_auth(client, DEFAULT_KILO_API_BASE, "CODE", expires_in=60)


# ---------------------------------------------------------------------------
# kilo_device_auth_login
# ---------------------------------------------------------------------------

def test_device_auth_login_returns_token(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr("hermes_cli.kilo_auth._is_remote_session", lambda: False)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)

    client = _FakeClient(
        post_responses=[_FakeResp(200, {"code": "C1", "verificationUrl": "https://app.kilo.ai/device-auth?code=C1", "expiresIn": 60})],
        get_responses=[_FakeResp(200, {"status": "approved", "token": "tok", "userEmail": "a@b.c"})],
    )
    _patch_httpx_client(monkeypatch, client)

    creds = kilo_device_auth_login()
    assert creds["token"] == "tok"
    assert creds["user_email"] == "a@b.c"
    assert "obtained_at" in creds
    assert opened == ["https://app.kilo.ai/device-auth?code=C1"]


def test_device_auth_login_respects_remote_session(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr("hermes_cli.kilo_auth._is_remote_session", lambda: True)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)

    client = _FakeClient(
        post_responses=[_FakeResp(200, {"code": "C2", "verificationUrl": "https://app.kilo.ai/device-auth?code=C2", "expiresIn": 60})],
        get_responses=[_FakeResp(200, {"status": "approved", "token": "tok2", "userEmail": ""})],
    )
    _patch_httpx_client(monkeypatch, client)

    creds = kilo_device_auth_login()
    assert creds["token"] == "tok2"
    # Browser must NOT be opened on a remote session.
    assert opened == []


def test_device_auth_login_invokes_on_verification(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr("hermes_cli.kilo_auth._is_remote_session", lambda: True)
    monkeypatch.setattr("webbrowser.open", lambda url: True)

    client = _FakeClient(
        post_responses=[_FakeResp(200, {"code": "C3", "verificationUrl": "https://app.kilo.ai/device-auth?code=C3", "expiresIn": 60})],
        get_responses=[_FakeResp(200, {"status": "approved", "token": "t3", "userEmail": ""})],
    )
    _patch_httpx_client(monkeypatch, client)

    captured = []
    kilo_device_auth_login(on_verification=lambda url, code: captured.append((url, code)))
    assert captured == [("https://app.kilo.ai/device-auth?code=C3", "C3")]


# ---------------------------------------------------------------------------
# fetch_kilo_profile / fetch_kilo_default_model
# ---------------------------------------------------------------------------

def test_fetch_profile_parses_organizations(monkeypatch):
    payload = {
        "user": {"email": "me@x.io", "name": "Me"},
        "organizations": [{"id": "org-1", "name": "My Org", "role": "owner"}],
    }
    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.get", lambda *a, **k: _FakeResp(200, payload))
    profile = fetch_kilo_profile("tok")
    assert profile["email"] == "me@x.io"
    assert profile["name"] == "Me"
    assert profile["organizations"][0]["id"] == "org-1"


def test_fetch_profile_returns_none_on_auth_failure(monkeypatch):
    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.get", lambda *a, **k: _FakeResp(401))
    assert fetch_kilo_profile("bad") is None


def test_fetch_default_model_uses_org_path(monkeypatch):
    captured = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        return _FakeResp(200, {"defaultModel": "anthropic/claude-sonnet-4.6"})

    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.get", _fake_get)
    model = fetch_kilo_default_model("tok", "org-9")
    assert model == "anthropic/claude-sonnet-4.6"
    assert "/api/organizations/org-9/defaults" in captured["url"]


def test_fetch_default_model_personal_path(monkeypatch):
    captured = {}

    def _fake_get(url, **kwargs):
        captured["url"] = url
        return _FakeResp(200, {"defaultModel": "kilo-auto/free"})

    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.get", _fake_get)
    model = fetch_kilo_default_model("tok", None)
    assert model == "kilo-auto/free"
    assert captured["url"].endswith("/api/defaults")


def test_fetch_default_model_anonymous_uses_free_field(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.kilo_auth.httpx.get",
        lambda *a, **k: _FakeResp(200, {"defaultFreeModel": "kilo-auto/free"}),
    )
    assert fetch_kilo_default_model(None, None) == "kilo-auto/free"


def test_fetch_default_model_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr("hermes_cli.kilo_auth.httpx.get", lambda *a, **k: _FakeResp(500))
    assert fetch_kilo_default_model("tok", None) is None


# ---------------------------------------------------------------------------
# set_kilo_org_header (model.default_headers)
# ---------------------------------------------------------------------------

def test_set_kilo_org_header_writes_org(monkeypatch):
    from hermes_cli.config import load_config

    set_kilo_org_header("org-42")
    cfg = load_config()
    assert cfg["model"]["default_headers"][KILO_ORG_HEADER] == "org-42"


def test_set_kilo_org_header_clears_org(monkeypatch):
    from hermes_cli.config import load_config, save_config

    # Seed the header first.
    set_kilo_org_header("org-42")
    assert load_config()["model"]["default_headers"][KILO_ORG_HEADER] == "org-42"

    # Clear it.
    set_kilo_org_header(None)
    cfg = load_config()
    headers = cfg.get("model", {}).get("default_headers", {})
    assert KILO_ORG_HEADER not in headers


# ---------------------------------------------------------------------------
# credential pool round-trip of organization_id
# ---------------------------------------------------------------------------

def test_credential_pool_roundtrips_organization_id():
    from agent.credential_pool import PooledCredential

    entry = PooledCredential(
        provider="kilocode",
        id="abc123",
        label="test",
        auth_type="api_key",
        priority=0,
        source="manual:device_code",
        access_token="tok",
        base_url="https://api.kilo.ai/api/gateway",
    )
    entry.extra["organization_id"] = "org-7"

    dumped = entry.to_dict()
    assert dumped["organization_id"] == "org-7"

    restored = PooledCredential.from_dict("kilocode", dumped)
    assert restored.extra.get("organization_id") == "org-7"


# ---------------------------------------------------------------------------
# auth_add_command (hermes auth add kilocode)
# ---------------------------------------------------------------------------

def _kilo_args(**overrides):
    base = {"provider": "kilocode", "auth_type": "", "label": "", "no_browser": False, "timeout": 15.0}
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_auth_add_kilocode_creates_pool_entry(monkeypatch):
    from hermes_cli import auth_commands

    # Stub the device-auth flow so no network/browser is touched.
    # acquire_and_store_kilo_credential calls these module-level functions,
    # so patching the module attributes is sufficient.
    monkeypatch.setattr(
        "hermes_cli.kilo_auth.kilo_device_auth_login",
        lambda **kw: {"token": "tok-abc", "user_email": "u@x.io", "obtained_at": "now"},
    )
    monkeypatch.setattr(
        "hermes_cli.kilo_auth.fetch_kilo_profile",
        lambda token, **kw: {"email": "u@x.io", "name": None, "organizations": [{"id": "org-1", "name": "Org1", "role": "owner"}]},
    )
    monkeypatch.setattr("hermes_cli.kilo_auth.prompt_kilo_organization", lambda orgs: "org-1")

    auth_commands.auth_add_command(_kilo_args())

    from agent.credential_pool import load_pool
    pool = load_pool("kilocode")
    entries = pool.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.access_token == "tok-abc"
    assert entry.auth_type == "api_key"
    assert entry.source == "manual:device_code"
    assert entry.base_url == "https://api.kilo.ai/api/gateway"
    assert entry.extra.get("organization_id") == "org-1"


def test_auth_add_kilocode_personal_account_leaves_org_none(monkeypatch):
    from hermes_cli import auth_commands

    monkeypatch.setattr(
        "hermes_cli.kilo_auth.kilo_device_auth_login",
        lambda **kw: {"token": "tok-personal", "user_email": "u@x.io", "obtained_at": "now"},
    )
    monkeypatch.setattr(
        "hermes_cli.kilo_auth.fetch_kilo_profile",
        lambda token, **kw: {"email": "u@x.io", "name": None, "organizations": [{"id": "org-1", "name": "Org1", "role": "owner"}]},
    )
    # User picks "Personal Account" → None.
    monkeypatch.setattr("hermes_cli.kilo_auth.prompt_kilo_organization", lambda orgs: None)

    auth_commands.auth_add_command(_kilo_args())

    from agent.credential_pool import load_pool
    entry = load_pool("kilocode").entries()[0]
    assert entry.access_token == "tok-personal"
    assert entry.extra.get("organization_id") is None


# ---------------------------------------------------------------------------
# logout cleanup
# ---------------------------------------------------------------------------

def test_logout_clears_org_header(monkeypatch):
    from hermes_cli.auth import clear_provider_auth
    from hermes_cli.config import load_config

    # Seed an org header and a kilocode pool entry so clear_provider_auth has
    # something to clear.
    set_kilo_org_header("org-99")
    from agent.credential_pool import load_pool, PooledCredential
    pool = load_pool("kilocode")
    pool.add_entry(PooledCredential(
        provider="kilocode", id="x1", label="t", auth_type="api_key",
        priority=0, source="manual:device_code", access_token="tok",
        base_url="https://api.kilo.ai/api/gateway",
    ))

    assert load_config()["model"]["default_headers"][KILO_ORG_HEADER] == "org-99"

    cleared = clear_provider_auth("kilocode")
    assert cleared is True

    cfg = load_config()
    headers = cfg.get("model", {}).get("default_headers", {})
    assert KILO_ORG_HEADER not in headers


def test_deactivate_provider_clears_kilo_org_header():
    """Switching away from Kilo (deactivate_provider) must clear the org header
    so it doesn't leak to the newly-selected provider."""
    from hermes_cli.auth import deactivate_provider
    from hermes_cli.config import load_config

    set_kilo_org_header("org-leak")
    assert load_config()["model"]["default_headers"][KILO_ORG_HEADER] == "org-leak"

    deactivate_provider()

    headers = load_config().get("model", {}).get("default_headers", {})
    assert KILO_ORG_HEADER not in headers


# ---------------------------------------------------------------------------
# acquire_and_store_kilo_credential (shared helper)
# ---------------------------------------------------------------------------

def test_acquire_and_store_credential_creates_entry(monkeypatch):
    from hermes_cli.kilo_auth import acquire_and_store_kilo_credential
    from agent.credential_pool import load_pool

    monkeypatch.setattr(
        "hermes_cli.kilo_auth.kilo_device_auth_login",
        lambda **kw: {"token": "helper-tok", "user_email": "h@x.io", "obtained_at": "now"},
    )
    monkeypatch.setattr(
        "hermes_cli.kilo_auth.fetch_kilo_profile",
        lambda token, **kw: {"email": "h@x.io", "name": None, "organizations": []},
    )

    pool = load_pool("kilocode")
    entry = acquire_and_store_kilo_credential(pool, open_browser=False, timeout_seconds=1.0)

    assert entry.access_token == "helper-tok"
    assert entry.auth_type == "api_key"
    assert entry.source == "manual:device_code"
    assert entry.base_url == "https://api.kilo.ai/api/gateway"
    # No orgs → organization_id is None.
    assert entry.extra.get("organization_id") is None
    # Entry was added to the pool.
    assert pool.entries()[-1].id == entry.id


# ---------------------------------------------------------------------------
# provider profile sanity (invariants, not snapshots)
# ---------------------------------------------------------------------------

def test_kilocode_provider_is_api_key():
    """Kilo tokens are long-lived without refresh → auth_type stays api_key."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY["kilocode"]
    assert pconfig.auth_type == "api_key"
    assert pconfig.inference_base_url == "https://api.kilo.ai/api/gateway"
    assert "KILOCODE_API_KEY" in pconfig.api_key_env_vars


def test_kilocode_in_oauth_capable_providers():
    """`hermes auth add kilocode` (no --type) routes to the device-auth path."""
    from hermes_cli.auth_commands import _OAUTH_CAPABLE_PROVIDERS

    assert "kilocode" in _OAUTH_CAPABLE_PROVIDERS
