"""``_codex_http_client`` survives a bracketed-IPv6 NO_PROXY environment (#118159).

Clash Verge / mihomo write ``[::1]`` into NO_PROXY; httpx 0.28.1 turns that into an
unparseable ``all://*[::1]`` proxy mount, so every bare ``httpx.Client()`` raised
``InvalidURL: Invalid port`` — codex device login died before its first request while chat
(explicit mounts) kept working. The client builder now sanitizes the env before construction.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")

# The exact environment Clash Verge / mihomo exports (issue #118159).
_CLASH_NO_PROXY = "127.0.0.1,localhost,::1,[::1]"


@pytest.fixture
def clash_env(monkeypatch):
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
    monkeypatch.setenv("no_proxy", _CLASH_NO_PROXY)


def test_codex_http_client_constructs_under_clash_ipv6_no_proxy(clash_env):
    from hermes_cli.auth import _codex_http_client

    # Red on base: InvalidURL('Invalid port: :1]') is raised at httpx.Client construction.
    client = _codex_http_client(timeout=5.0)
    try:
        assert client._mounts is not None  # constructed; the bypass mount is a real entry
    finally:
        client.close()
