"""Issue #119553: cloud wizard with a stale local baseUrl must use cloud OAuth endpoints.

Repro: honcho.json on disk holds a local deployment (top-level and
hosts.<host> baseUrl). The user picks cloud + oauth in the setup wizard.
_setup_cloud_auth clears baseUrl only in memory; authorize_via_loopback /
authorize_via_device_code re-resolve endpoints from disk, land on
localhost:3000, and browser sign-in is unreachable.
"""
import json

import pytest

from plugins.memory.honcho import oauth_flow
from plugins.memory.honcho.oauth import OAuthCredential

_STALE = "http://localhost:8000"


def _write_stale_disk(path):
    path.write_text(json.dumps({
        "enabled": True, "baseUrl": _STALE, "workspace": "hermes", "aiPeer": "hermes",
        "hosts": {"hermes": {"baseUrl": _STALE, "workspace": "hermes"}},
    }))


@pytest.fixture
def stale_disk(monkeypatch, tmp_path):
    """Point the ambient honcho config at a file holding a stale local deployment."""
    cfg_path = tmp_path / "honcho.json"
    _write_stale_disk(cfg_path)
    import plugins.memory.honcho.client as client_mod
    monkeypatch.setattr(client_mod, "resolve_config_path", lambda: cfg_path)
    monkeypatch.setattr(client_mod, "resolve_active_host", lambda: "hermes")
    for var in ("HONCHO_OAUTH_DASHBOARD", "HONCHO_OAUTH_TOKEN_URL", "HONCHO_OAUTH_AUTHORIZE_URL",
                "HONCHO_OAUTH_DEVICE_AUTH_URL", "HONCHO_OAUTH_CLIENT_ID", "HONCHO_OAUTH_SCOPE",
                "HONCHO_BASE_URL", "HONCHO_ENVIRONMENT", "HONCHO_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return cfg_path


def test_stale_disk_poisons_bare_resolution(stale_disk):
    """Premise check: with a stale local baseUrl on disk, a bare resolve goes local."""
    endpoints = oauth_flow.resolve_endpoints()
    assert endpoints.authorize_url.startswith("http://localhost:3000"), endpoints


def _run_cloud_auth(monkeypatch, stale_disk, method):
    """Drive _setup_cloud_auth with stubbed I/O; capture the authorize URL the flow would open."""
    import plugins.memory.honcho.cli as honcho_cli
    monkeypatch.setattr(honcho_cli, "_headless", lambda: (False, True))
    monkeypatch.setattr(honcho_cli, "_device_login_available", lambda *a, **k: method == "device")
    monkeypatch.setattr(honcho_cli, "_prompt", lambda label, default=None, secret=False: method)

    urls: list[str] = []
    cred = OAuthCredential(
        access_token="hch-at-x", refresh_token="hch-rt-x", expires_at=9_999_999_999,
        client_id="hermes-agent", token_endpoint="https://api.honcho.dev/oauth/token",
        consent_peer_name="lyra",
    )

    def _capture(endpoints, **kw):
        url, _ = oauth_flow.begin_authorization(endpoints, source="hermes-cli", config_path="honcho.json")
        urls.append(url)
        return cred

    def fake_loopback(*, endpoints=None, **kw):
        # Mirror the real authorize_via_loopback: no explicit endpoints means
        # re-resolve from disk — the stale read this issue is about.
        return _capture(endpoints if endpoints is not None else oauth_flow.resolve_endpoints())

    def fake_device(*, endpoints=None, **kw):
        return _capture(endpoints if endpoints is not None else oauth_flow.resolve_endpoints())

    monkeypatch.setattr(oauth_flow, "authorize_via_loopback", fake_loopback)
    monkeypatch.setattr(oauth_flow, "authorize_via_device_code", fake_device)

    cfg = {"enabled": True, "baseUrl": _STALE, "workspace": "hermes", "aiPeer": "hermes",
           "hosts": {"hermes": {"baseUrl": _STALE, "workspace": "hermes"}}}
    host = cfg["hosts"]["hermes"]
    assert honcho_cli._setup_cloud_auth(cfg, host, stale_disk) is True
    return cfg, host, urls


@pytest.mark.parametrize("method", ["oauth", "device"])
def test_cloud_wizard_uses_cloud_endpoints_despite_stale_disk(monkeypatch, stale_disk, method):
    """The cloud wizard must not let a stale on-disk baseUrl drag OAuth to localhost."""
    _, _, urls = _run_cloud_auth(monkeypatch, stale_disk, method)
    assert len(urls) == 1
    assert urls[0].startswith("https://app.honcho.dev/authorize"), urls[0]


@pytest.mark.parametrize("method", ["oauth", "device"])
def test_cloud_auth_clears_host_block_baseurl(monkeypatch, stale_disk, method):
    """Switching to cloud drops the per-host local baseUrl too (client prefers it over top-level)."""
    cfg, host, _ = _run_cloud_auth(monkeypatch, stale_disk, method)
    assert "baseUrl" not in cfg and "base_url" not in cfg
    assert "baseUrl" not in host and "base_url" not in host


def test_cloud_auth_probes_device_support_on_cloud_endpoints(monkeypatch, stale_disk):
    """The device-grant probe must target cloud, not the stale on-disk localhost."""
    import plugins.memory.honcho.cli as honcho_cli
    monkeypatch.setattr(honcho_cli, "_headless", lambda: (False, True))
    monkeypatch.setattr(honcho_cli, "_prompt", lambda label, default=None, secret=False: "")
    seen: dict = {}
    monkeypatch.setattr(honcho_cli, "_device_login_available",
                        lambda endpoints=None, **k: seen.setdefault("endpoints", endpoints) or False)
    cfg = {"enabled": True, "baseUrl": _STALE,
           "hosts": {"hermes": {"baseUrl": _STALE, "workspace": "hermes"}}}
    honcho_cli._setup_cloud_auth(cfg, cfg["hosts"]["hermes"], stale_disk)
    assert seen["endpoints"].authorize_url.startswith("https://app.honcho.dev"), seen["endpoints"]
