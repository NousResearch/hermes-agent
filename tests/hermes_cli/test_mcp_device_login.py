"""Tests for the device-code login path (``hermes mcp login --flow device``).

Mocks the OAuth wire and the MCP probe so nothing touches the network or a
real server; ``HERMES_HOME`` points at a temp dir.
"""

import io
import json
import urllib.request
from unittest.mock import MagicMock

import pytest

import hermes_cli.mcp_config as mc
from tools.mcp_oauth_device import DeviceAuthorization


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _server_cfg(**overrides):
    cfg = {"url": "https://mcp.example.com/mcp", "auth": "oauth", "oauth": {}}
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Flow selection
# ---------------------------------------------------------------------------

class TestUseDeviceFlow:
    def test_flag_device_wins(self):
        assert mc._use_device_flow(_server_cfg(), "device") is True

    def test_flag_browser_overrides_config(self):
        cfg = _server_cfg(oauth={"flow": "device"})
        assert mc._use_device_flow(cfg, "browser") is False

    def test_config_device_without_flag(self):
        assert mc._use_device_flow(_server_cfg(oauth={"flow": "device"}), None) is True

    def test_default_is_browser(self):
        assert mc._use_device_flow(_server_cfg(), None) is False

    def test_non_dict_oauth_is_browser(self):
        assert mc._use_device_flow({"url": "https://x", "auth": "oauth", "oauth": None}, None) is False


# ---------------------------------------------------------------------------
# _reauth_device_flow orchestration (all wire calls stubbed)
# ---------------------------------------------------------------------------

def _stub_device_flow(monkeypatch, *, tokens_present=True, poll_error=None):
    import tools.mcp_oauth_device as dev

    endpoints = {
        "device_authorization_endpoint": "https://auth.example/device",
        "token_endpoint": "https://auth.example/token",
        "registration_endpoint": "https://auth.example/register",
        "resource": "https://mcp.example.com/mcp",
    }
    authorization = DeviceAuthorization(
        device_code="dc_1", user_code="USER-1",
        verification_uri="https://auth.example/activate",
        expires_in=900, interval=5)
    persisted = {}

    monkeypatch.setattr(dev, "discover_device_endpoints", lambda url: endpoints)
    monkeypatch.setattr(
        dev, "register_device_client", lambda *a, **k: {"client_id": "c1"})
    monkeypatch.setattr(
        dev, "request_device_authorization", lambda *a, **k: authorization)
    monkeypatch.setattr(dev, "announce_device_authorization", lambda auth: None)
    if poll_error is not None:
        def _raise(*a, **k):
            raise poll_error
        monkeypatch.setattr(dev, "poll_device_token", _raise)
    else:
        monkeypatch.setattr(
            dev, "poll_device_token",
            lambda *a, **k: {"access_token": "at_1", "token_type": "Bearer"})
    monkeypatch.setattr(
        dev, "persist_device_state",
        lambda name, client_info, payload, **k: persisted.update(payload))
    monkeypatch.setattr(
        "tools.mcp_oauth_manager.get_manager", lambda: MagicMock())
    monkeypatch.setattr(
        mc, "_probe_single_server", lambda name, cfg: [("tool_a", "desc")] if tokens_present else [])
    monkeypatch.setattr(mc, "_oauth_tokens_present", lambda name: tokens_present)
    return persisted


class TestReauthDeviceFlow:
    def test_success_persists_and_probes(self, monkeypatch):
        persisted = _stub_device_flow(monkeypatch)
        assert mc._reauth_device_flow("srv", _server_cfg()) is True
        assert persisted.get("access_token") == "at_1"

    def test_poll_failure_returns_false(self, monkeypatch):
        from tools.mcp_oauth_device import DeviceFlowError
        _stub_device_flow(monkeypatch, poll_error=DeviceFlowError("denied"))
        assert mc._reauth_device_flow("srv", _server_cfg()) is False

    def test_missing_tokens_returns_false(self, monkeypatch):
        _stub_device_flow(monkeypatch, tokens_present=False)
        assert mc._reauth_device_flow("srv", _server_cfg()) is False

    def test_rejects_non_oauth_server(self, monkeypatch):
        cfg = {"url": "https://mcp.example.com/mcp", "auth": "header"}
        assert mc._reauth_oauth_server("srv", cfg, flow="device") is False


# ---------------------------------------------------------------------------
# discover_device_endpoints against a stubbed wire (real SDK handlers)
# ---------------------------------------------------------------------------

def _fake_urlopen_factory(routes):
    def _fake_urlopen(request, timeout=None):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        for prefix, payload in routes:
            if prefix in url:
                body = json.dumps(payload).encode()

                class _Resp:
                    status = 200

                    def read(self):
                        return body

                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                return _Resp()
        raise AssertionError(f"unexpected discovery URL: {url}")

    return _fake_urlopen


class TestDiscoverDeviceEndpoints:
    def test_resolves_all_endpoints(self, monkeypatch):
        import tools.mcp_oauth_device as dev

        routes = [
            ("mcp.example.com/.well-known/oauth-protected-resource", {
                "resource": "https://mcp.example.com/mcp",
                "authorization_servers": ["https://auth.example/oauth"],
            }),
            ("auth.example", {
                "issuer": "https://auth.example/oauth",
                "authorization_endpoint": "https://auth.example/oauth/authorize",
                "token_endpoint": "https://auth.example/oauth/token",
                "registration_endpoint": "https://auth.example/oauth/register",
                "device_authorization_endpoint": "https://auth.example/oauth/device",
                "response_types_supported": ["code"],
            }),
        ]
        monkeypatch.setattr(
            urllib.request, "urlopen", _fake_urlopen_factory(routes))
        endpoints = dev.discover_device_endpoints("https://mcp.example.com/mcp")
        assert endpoints["device_authorization_endpoint"] == "https://auth.example/oauth/device"
        assert endpoints["token_endpoint"] == "https://auth.example/oauth/token"
        assert endpoints["registration_endpoint"] == "https://auth.example/oauth/register"
        assert endpoints["resource"] == "https://mcp.example.com/mcp"

    def test_no_device_endpoint_fails(self, monkeypatch):
        import tools.mcp_oauth_device as dev
        from tools.mcp_oauth_device import DeviceFlowError

        routes = [
            ("mcp.example.com/.well-known/oauth-protected-resource", {
                "resource": "https://mcp.example.com/mcp",
                "authorization_servers": ["https://auth.example/oauth"],
            }),
            ("auth.example", {
                "issuer": "https://auth.example/oauth",
                "authorization_endpoint": "https://auth.example/oauth/authorize",
                "token_endpoint": "https://auth.example/oauth/token",
                "registration_endpoint": "https://auth.example/oauth/register",
                "response_types_supported": ["code"],
            }),
        ]
        monkeypatch.setattr(
            urllib.request, "urlopen", _fake_urlopen_factory(routes))
        with pytest.raises(DeviceFlowError, match="device"):
            dev.discover_device_endpoints("https://mcp.example.com/mcp")
