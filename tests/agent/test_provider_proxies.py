"""``provider_proxies`` — per-provider/route HTTP proxy overrides.

Shape (see ``agent/provider_proxy.py``):

* ``{"opencode-zen": "http://..."}`` — provider-id key: the MAIN agent's client build
  (which knows the provider id) routes through the proxy.
* ``{"opencode.ai": "http://..."}`` — hostname key: covers auxiliary calls (no provider
  id at build time) and keyless-healed traffic whose provider id changes at runtime
  (OpenCode Zen → opencode-free) while the endpoint stays put.
* Explicit entries win over ``HTTPS_PROXY`` env and ignore ``NO_PROXY``.
* Malformed URLs raise ``RuntimeError`` (fail loud, never silently direct).
"""

from unittest.mock import patch

import httpx
import pytest

from agent.auxiliary_client import _create_openai_client, _openai_http_client_kwargs
from agent.provider_proxy import provider_proxy_override, validate_provider_proxy_urls
from agent.process_bootstrap import build_keepalive_http_client


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)


def _pool_types(http_client) -> list:
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


def test_provider_id_key_resolves_for_matching_provider():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"opencode-zen": "http://127.0.0.1:3128"}}):
        assert provider_proxy_override("opencode-zen", "https://opencode.ai/zen/v1") == "http://127.0.0.1:3128"
        # a different provider must not match
        assert provider_proxy_override("openrouter", "https://openrouter.ai/api/v1") is None


def test_hostname_key_resolves_by_base_url():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"opencode.ai": "http://127.0.0.1:3128"}}):
        # no provider id (auxiliary path) still matches by host
        assert provider_proxy_override(None, "https://opencode.ai/zen/v1") == "http://127.0.0.1:3128"
        # keyless-healed provider id (opencode-free) + same endpoint matches by host
        assert provider_proxy_override("opencode-free", "https://opencode.ai/zen/v1") == "http://127.0.0.1:3128"
        # unrelated host does not
        assert provider_proxy_override(None, "https://api.firecrawl.dev/v1") is None


def test_provider_id_wins_over_hostname():
    mapping = {"opencode.ai": "http://host-route:1", "opencode-zen": "http://provider-route:2"}
    with patch("hermes_cli.config.load_config_readonly", return_value={"provider_proxies": mapping}):
        assert provider_proxy_override("opencode-zen", "https://opencode.ai/zen/v1") == "http://provider-route:2"


def test_malformed_url_raises_runtime_error():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"opencode-zen": "127.0.0.1:3128"}}):
        with pytest.raises(RuntimeError, match="missing scheme"):
            provider_proxy_override("opencode-zen", "https://opencode.ai/zen/v1")
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"opencode-zen": "http://127.0.0.1:3128x"}}):
        with pytest.raises(RuntimeError, match="Malformed"):
            provider_proxy_override("opencode-zen", "https://opencode.ai/zen/v1")


def test_validate_provider_proxy_urls_fail_fast():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"x": "ftp://ok:1", "y": "no-scheme"}}):
        with pytest.raises(RuntimeError, match="missing scheme"):
            validate_provider_proxy_urls()
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"x": "http://ok:1", "y": ""}}):
        validate_provider_proxy_urls()  # empty value tolerated


def test_build_keepalive_http_client_explicit_proxy_wins(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "opencode.ai")
    client = build_keepalive_http_client(
        "https://opencode.ai/zen/v1", proxy="http://127.0.0.1:3128")
    try:
        assert isinstance(client, httpx.Client)
        # explicit override ignores NO_PROXY
        assert "HTTPProxy" in _pool_types(client)
    finally:
        client.close()


def test_main_client_build_routes_via_provider_proxy():
    """Main-path chokepoint: create_openai_client resolves provider-id + host keys."""
    from agent.agent_runtime_helpers import create_openai_client

    class _Agent:
        provider = "opencode-zen"
        base_url = "https://opencode.ai/zen/v1"

        @staticmethod
        def _build_keepalive_http_client(base_url="", *, verify=True, proxy=None):
            from agent.process_bootstrap import build_keepalive_http_client
            return build_keepalive_http_client(base_url, verify=verify, proxy=proxy)

        def _client_log_context(self):
            return ""

    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"opencode-zen": "http://127.0.0.1:3128"}}):
        client = create_openai_client(
            _Agent(), {"api_key": "test-key", "base_url": "https://opencode.ai/zen/v1"},
            reason="test", shared=False)
    inner = getattr(client, "_client", None) or client
    http_client = getattr(inner, "_client", None)
    if http_client is None:
        http_client = inner
    try:
        assert "HTTPProxy" in _pool_types(http_client)
    finally:
        close = getattr(client, "close", None)
        if close:
            close()


def test_aux_client_build_routes_via_hostname_proxy():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"litellm.internal.example.com": "http://127.0.0.1:7897"}}):
        kwargs = _openai_http_client_kwargs("https://litellm.internal.example.com/v1")
    http_client = kwargs.get("http_client")
    try:
        assert isinstance(http_client, httpx.Client)
        assert "HTTPProxy" in _pool_types(http_client)
    finally:
        http_client.close()


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_routes_via_hostname_proxy(mock_openai):
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"provider_proxies": {"litellm.internal.example.com": "http://127.0.0.1:7897"}}):
        _create_openai_client(
            api_key="test-key",
            base_url="https://litellm.internal.example.com/v1",
        )
    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" in _pool_types(http_client)
    http_client.close()


def test_env_proxy_still_applies_when_no_override_configured(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    kwargs = _openai_http_client_kwargs("https://litellm.internal.example.com/v1")
    http_client = kwargs.get("http_client")
    try:
        assert "HTTPProxy" in _pool_types(http_client)
    finally:
        http_client.close()
