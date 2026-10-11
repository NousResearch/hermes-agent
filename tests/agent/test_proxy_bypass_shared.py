"""Invariant: the LLM transport (``agent/process_bootstrap``) and the platform adapters
(``gateway/platforms/base``) answer "is this host in NO_PROXY" with the same matcher, so a
corporate ``NO_PROXY=10.0.0.0/8`` bypasses the proxy for a self-hosted ``10.x`` model endpoint
exactly as it does for Telegram/Discord/Slack.
"""

import pytest

from agent.process_bootstrap import _get_proxy_for_base_url
from agent.proxy_bypass import should_bypass_proxy
from gateway.platforms.base import is_host_excluded_by_no_proxy, resolve_proxy_url

_PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
               "NO_PROXY", "no_proxy")


@pytest.fixture
def proxy_env(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    monkeypatch.setattr("gateway.platforms.base.gateway_trust_env", lambda: True)
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)
    return monkeypatch


@pytest.mark.parametrize("no_proxy, host", [
    ("10.0.0.0/8", "10.1.2.3"),
    ("*.internal", "svc.internal"),
    ("*.slack.com", "slack.com"),  # ``*.`` covers the apex, as the adapter docstring promised
    (".slack.com", "slack.com"),
    ("localhost,.corp.example", "llm.corp.example"),
    ("api.example.com:8443", "api.example.com:8443"),
])
def test_llm_and_adapter_paths_bypass_the_same_entries(proxy_env, no_proxy, host):
    proxy_env.setenv("NO_PROXY", no_proxy)
    assert should_bypass_proxy(host)
    assert _get_proxy_for_base_url(f"https://{host}/v1") is None
    assert resolve_proxy_url(target_hosts=host) is None
    assert is_host_excluded_by_no_proxy(host.split(":")[0]) or ":" in host  # Slack passes bare hosts


def test_non_matching_host_keeps_the_proxy_on_both_paths(proxy_env):
    proxy_env.setenv("NO_PROXY", "10.0.0.0/8,*.internal")
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://proxy.corp:3128"
    assert resolve_proxy_url(target_hosts="api.telegram.org") == "http://proxy.corp:3128"
    assert not is_host_excluded_by_no_proxy("slack.com")
    assert is_host_excluded_by_no_proxy("files.slack.com", "slack.com")  # explicit value wins
    assert not should_bypass_proxy("notslack.com", no_proxy_value="*.slack.com")


def test_malformed_port_in_base_url_keeps_the_proxy_instead_of_raising(proxy_env):
    """A ``host:notaport`` base_url must not raise out of the bypass check: the caller's
    blanket ``except`` would otherwise drop the shared keepalive transport entirely."""
    proxy_env.setenv("NO_PROXY", "other.example")
    assert _get_proxy_for_base_url("http://host:notaport/v1") == "http://proxy.corp:3128"
    assert _get_proxy_for_base_url("http://host:99999/v1") == "http://proxy.corp:3128"
    proxy_env.setenv("NO_PROXY", "host")
    assert _get_proxy_for_base_url("http://host:notaport/v1") is None  # host still matched


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:11434/v1",   # Ollama default
    "http://localhost:11434/v1",
    "http://[::1]:11434/v1",
    "http://127.8.8.7:8080/v1",    # any 127.x.x.x literal, not just .0.0.1
    "127.0.0.1:11434",             # scheme-less form callers may pass
])
def test_loopback_base_url_never_dials_through_the_proxy(proxy_env, base_url):
    """A local model endpoint must bypass the proxy unconditionally — the same rule the
    CDP/websockets path applies (#110565). With a bare ``HTTPS_PROXY`` and no loopback entry in
    NO_PROXY, the request used to be tunneled into the proxy and the SSE stream hung with no
    error callback until the stale watchdog fired (~200 s, #135208)."""
    assert _get_proxy_for_base_url(base_url) is None
    # and the exclusion is loopback-only: remote endpoints keep the proxy
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://proxy.corp:3128"
