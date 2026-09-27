"""Regression guard: auxiliary OpenAI clients must use env-only proxy policy.

On macOS, httpx with default ``trust_env=True`` reads system proxy settings
via ``urllib.request.getproxies()`` but not the macOS proxy exception list.
Auxiliary clients (vision, title generation, etc.) must mirror the main
agent: explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars only, via a custom
keepalive transport that suppresses automatic system-proxy detection.
"""
from unittest.mock import patch

import httpx

from agent.auxiliary_client import _create_openai_client, _openai_http_client_kwargs
from agent.process_bootstrap import _get_proxy_for_base_url


def _pool_types(http_client) -> list:
    return [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]


@patch("agent.auxiliary_client.OpenAI")
def test_create_openai_client_routes_via_env_proxy(mock_openai, monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    _create_openai_client(
        api_key="test-key",
        base_url="https://litellm.internal.example.com/v1",
    )

    http_client = mock_openai.call_args.kwargs.get("http_client")
    assert isinstance(http_client, httpx.Client)
    assert "HTTPProxy" in _pool_types(http_client)
    http_client.close()






def test_get_proxy_for_base_url_respects_no_proxy(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "internal.example.com")

    assert _get_proxy_for_base_url("https://litellm.internal.example.com/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


# ---------------------------------------------------------------------------
# Windows OS-proxy consultation (#124773)
# ---------------------------------------------------------------------------

def test_os_proxy_for_url_reads_wininet_proxy(monkeypatch):
    """With no proxy ENV vars set, the OS proxy (WinINET registry) still governs
    Windows egress and is consulted via urllib (#124773)."""
    import urllib.request as _urllib
    from agent.process_bootstrap import _os_proxy_for_url

    monkeypatch.setattr(_urllib, "getproxies",
                        lambda: {"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"})
    monkeypatch.setattr(_urllib, "proxy_bypass", lambda _host: False)

    assert _os_proxy_for_url("https://openrouter.ai/api/v1", is_windows=True) == "http://127.0.0.1:7897"


def test_os_proxy_bypass_list_excludes_host(monkeypatch):
    """ProxyOverride excludes this host → direct egress (None), not the proxy."""
    import urllib.request as _urllib
    from agent.process_bootstrap import _os_proxy_for_url

    monkeypatch.setattr(_urllib, "getproxies",
                        lambda: {"https": "http://127.0.0.1:7897"})
    monkeypatch.setattr(_urllib, "proxy_bypass", lambda _host: True)

    assert _os_proxy_for_url("https://localhost/v1", is_windows=True) is None


def test_os_proxy_not_consulted_off_windows(monkeypatch):
    """macOS keeps the env-only policy: its system proxies can omit the ExceptionsList."""
    import urllib.request as _urllib
    from agent.process_bootstrap import _os_proxy_for_url

    monkeypatch.setattr(_urllib, "getproxies",
                        lambda: {"https": "http://127.0.0.1:7897"})
    assert _os_proxy_for_url("https://openrouter.ai/api/v1", is_windows=False) is None


def test_aux_client_uses_windows_os_proxy_when_env_unset(monkeypatch):
    """Red-to-green: the auxiliary client egressed DIRECTLY (explicit mounts) with no
    env proxy; on Windows it must route through the OS proxy instead."""
    import urllib.request as _urllib
    from agent import process_bootstrap

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(_urllib, "getproxies",
                        lambda: {"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"})
    monkeypatch.setattr(_urllib, "proxy_bypass", lambda _host: False)

    captured = {}
    real_client = httpx.Client

    class SpyClient(real_client):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(httpx, "Client", SpyClient)
    monkeypatch.setattr(process_bootstrap, "close_shared_transports", lambda: None)
    client = process_bootstrap.build_keepalive_http_client(
        "https://openrouter.ai/api/v1", is_windows=True)
    assert captured.get("proxy") == "http://127.0.0.1:7897"
    client.close()


