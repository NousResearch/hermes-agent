"""Codex OAuth/probe clients never derive proxy mounts from the environment.

A bracketed IPv6 entry in NO_PROXY — ``[::1]``, the form Clash Verge / mihomo export on
Linux — makes httpx derive an unparseable ``all://*[::1]`` mount from the environment, so
every bare ``httpx.Client()`` (trust_env) raises ``InvalidURL: Invalid port: ':1]'`` the
moment it is constructed. The Codex auth paths (device login, token refresh, usage probes)
used exactly that shape while the chat transport — which resolves the proxy itself and
mounts it explicitly — kept working: login/refresh died on hosts where chat was fine
(#118159).

``_codex_http_client`` now resolves the proxy for its target URL through the same matcher
as the chat transport and hands the result to httpx explicitly, so httpx never reparses
NO_PROXY. NO_PROXY semantics are kept: bypassed targets go direct behind a trust_env SSL
context (``SSL_CERT_FILE`` keeps working behind corporate proxies).
"""

import httpx

from hermes_cli import auth_codex

_BAD_NO_PROXY = (
    "127.0.0.1,localhost,::1,[::1]"  # both bare and bracketed ::1, as mihomo writes it
)


def _env(monkeypatch, *, proxy=None, no_proxy=None):
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    if proxy is not None:
        monkeypatch.setenv("HTTPS_PROXY", proxy)
        monkeypatch.setenv("https_proxy", proxy)
    if no_proxy is not None:
        monkeypatch.setenv("NO_PROXY", no_proxy)
        monkeypatch.setenv("no_proxy", no_proxy)


def _mount_patterns(client):
    return sorted(pattern.pattern for pattern in client._mounts)


def test_bracketed_ipv6_no_proxy_no_longer_kills_client_construction(monkeypatch):
    # Before the fix the bare trust_env client raised InvalidURL("Invalid port: ':1]'") at
    # construction — this exact env is the #118159 report (Clash Verge mixed port).
    _env(monkeypatch, proxy="http://127.0.0.1:7897", no_proxy=_BAD_NO_PROXY)
    client = auth_codex._codex_http_client(url="https://auth.openai.com/oauth/token")
    try:
        # One explicit proxy mount; nothing was derived from the broken NO_PROXY.
        assert _mount_patterns(client) == ["all://"]
    finally:
        client.close()


def test_broken_no_proxy_alone_is_not_fatal_when_going_direct(monkeypatch):
    _env(monkeypatch, no_proxy=_BAD_NO_PROXY)
    client = auth_codex._codex_http_client(url="https://auth.openai.com/oauth/token")
    try:
        assert (
            _mount_patterns(client) == []
        )  # direct: nothing derived from the environment
        assert client.trust_env is False
    finally:
        client.close()


def test_no_proxy_bypass_still_goes_direct(monkeypatch):
    _env(monkeypatch, proxy="http://127.0.0.1:7897", no_proxy="auth.openai.com")
    client = auth_codex._codex_http_client(url="https://auth.openai.com/oauth/token")
    try:
        assert _mount_patterns(client) == []
        assert client.trust_env is False
    finally:
        client.close()


def test_unbypassed_target_still_uses_env_proxy(monkeypatch):
    _env(monkeypatch, proxy="http://127.0.0.1:7897", no_proxy="localhost,::1,[::1]")
    client = auth_codex._codex_http_client(url="https://auth.openai.com/oauth/token")
    try:
        assert _mount_patterns(client) == ["all://"]
    finally:
        client.close()
