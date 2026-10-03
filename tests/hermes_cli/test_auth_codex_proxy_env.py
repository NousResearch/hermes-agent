"""Codex OAuth HTTP clients must not let malformed NO_PROXY mounts break auth."""

import base64
import json
import time

import hermes_cli.auth as auth_mod
import hermes_cli.auth_codex as auth_codex
from hermes_cli.auth_codex import _codex_http_client


def test_codex_http_client_ignores_bracketed_ipv6_no_proxy(monkeypatch):
    """Regression for #118159: httpx rejects ``NO_PROXY=[::1]`` before a request."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1,[::1]")
    monkeypatch.delenv("no_proxy", raising=False)

    with _codex_http_client() as client:
        assert client is not None


def test_codex_http_client_uses_shared_explicit_proxy_policy(monkeypatch):
    """Ordinary proxies and bracketed IPv6 bypasses retain the chat policy."""
    captured = []
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("NO_PROXY", "[::1]")
    monkeypatch.setattr(auth_codex.httpx, "Client", lambda **kwargs: captured.append(kwargs) or object())

    _codex_http_client(target_url="https://auth.openai.com/oauth/token")
    _codex_http_client(target_url="http://[::1]/callback")

    remote, loopback = captured
    assert remote["proxy"] == "http://proxy.example:3128"
    assert remote["trust_env"] is False
    assert loopback["proxy"] is None
    assert loopback["trust_env"] is False


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self.response

    def get(self, *args, **kwargs):
        return self.response


def _jwt() -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + 3600}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_device_refresh_and_quota_requests_supply_their_target_url(monkeypatch):
    """All Codex OAuth/probe call sites use the explicit proxy policy."""
    targets = []
    response = _Response({"access_token": "fresh", "rate_limit": {"primary_window": {"used_percent": 0}}})
    monkeypatch.setattr(
        auth_codex, "_codex_http_client",
        lambda **kwargs: targets.append(kwargs["target_url"]) or _Client(response),
    )
    auth_mod._codex_quota_probe_cache.clear()

    auth_codex._codex_login_post("https://issuer.example/device", failure=("failed", "failed"))
    auth_codex.refresh_codex_oauth_pure("old", "refresh")
    assert auth_codex._probe_codex_quota_restored(_jwt()) is True

    assert targets == [
        "https://issuer.example/device",
        auth_codex.CODEX_OAUTH_TOKEN_URL,
        auth_codex._codex_usage_probe_url(None),
    ]
