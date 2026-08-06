"""Regression tests for the in-container WebSocket client host resolution.

Issue #58993: when the dashboard binds to a wildcard (``0.0.0.0`` / ``::``),
the in-container WS clients built by ``_build_gateway_ws_url`` and
``_build_sidecar_url`` used the bind host verbatim, so the child TUI
dialed ``ws://0.0.0.0:9119/api/ws``. Behind a forward proxy whose
``NO_PROXY`` does not list ``0.0.0.0`` that wildcard dial is routed through
the proxy and fails the handshake.

The contract these tests pin down:

  * Wildcard bind (``0.0.0.0`` / ``::``) → client dials ``127.0.0.1``.
  * Loopback bind (``127.0.0.1``) → client dials ``127.0.0.1`` (unchanged).
  * LAN / non-wildcard bind (``192.168.1.5``) → client dials that exact
    address (no rewrite to loopback — the bind was deliberate).
  * Explicit ``HERMES_DASHBOARD_WS_HOST`` env var → wins always, regardless
    of the bind host.
  * ``app.state.bound_host`` is left untouched — the bind address used by
    the listener doesn't change.
"""

from __future__ import annotations

import os

import pytest

from hermes_cli import web_server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_app_state():
    """Snapshot and restore the bits of ``app.state`` the tests mutate.

    The dashboard's WS URL builders read three things off ``app.state``:
    ``bound_host``, ``bound_port``, ``auth_required``. We capture them so
    the suite doesn't leak state into other tests, then yield control.
    """
    saved = {
        "bound_host": getattr(web_server.app.state, "bound_host", None),
        "bound_port": getattr(web_server.app.state, "bound_port", None),
        "auth_required": getattr(web_server.app.state, "auth_required", None),
    }
    yield saved
    for key, value in saved.items():
        setattr(web_server.app.state, key, value)


@pytest.fixture
def clear_ws_host_env(monkeypatch):
    """Ensure no ``HERMES_DASHBOARD_WS_HOST`` leaks in from the test shell."""
    monkeypatch.delenv("HERMES_DASHBOARD_WS_HOST", raising=False)
    yield monkeypatch


def _set_bound(saved_app_state, host: str, port: int = 9119):
    web_server.app.state.bound_host = host
    web_server.app.state.bound_port = port
    web_server.app.state.auth_required = False


def _netloc(ws_url: str) -> str:
    """Pull the ``host:port`` (or ``[host]:port``) segment out of a ws URL."""
    assert ws_url is not None, "expected a URL, got None"
    # ws://host:port/path?qs — strip the scheme, then take netloc up to "/".
    after_scheme = ws_url.split("://", 1)[1]
    netloc = after_scheme.split("/", 1)[0]
    return netloc


# ---------------------------------------------------------------------------
# _resolve_client_ws_host — direct unit tests
# ---------------------------------------------------------------------------


class TestResolveClientWsHost:
    def test_wildcard_ipv4_uses_loopback(self, saved_app_state, clear_ws_host_env):
        _set_bound(saved_app_state, "0.0.0.0")
        assert web_server._resolve_client_ws_host() == "127.0.0.1"

    def test_wildcard_ipv6_uses_loopback(self, saved_app_state, clear_ws_host_env):
        _set_bound(saved_app_state, "::")
        assert web_server._resolve_client_ws_host() == "127.0.0.1"

    def test_loopback_bind_unchanged(self, saved_app_state, clear_ws_host_env):
        _set_bound(saved_app_state, "127.0.0.1")
        assert web_server._resolve_client_ws_host() == "127.0.0.1"

    def test_lan_bind_preserved(self, saved_app_state, clear_ws_host_env):
        """A non-loopback, non-wildcard bind must NOT be rewritten — the
        operator chose that address deliberately (e.g. bridge networking in
        a sidecar topology) and rewriting it to 127.0.0.1 would break their
        setup."""
        _set_bound(saved_app_state, "192.168.1.5")
        assert web_server._resolve_client_ws_host() == "192.168.1.5"

    def test_public_dns_bind_preserved(self, saved_app_state, clear_ws_host_env):
        _set_bound(saved_app_state, "fly-app.example.dev")
        assert web_server._resolve_client_ws_host() == "fly-app.example.dev"

    def test_explicit_env_wins_over_wildcard(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "0.0.0.0")
        assert web_server._resolve_client_ws_host() == "10.0.0.7"

    def test_explicit_env_wins_over_lan_bind(
        self, saved_app_state, monkeypatch
    ):
        """Even when the bind is a routable address, the explicit override
        still wins — operators may want to bypass the bind address
        altogether (e.g. to dial a different sidecar replica)."""
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "192.168.1.5")
        assert web_server._resolve_client_ws_host() == "10.0.0.7"

    def test_explicit_env_wins_over_loopback(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "127.0.0.1")
        assert web_server._resolve_client_ws_host() == "10.0.0.7"

    def test_blank_env_falls_back_to_bind(
        self, saved_app_state, monkeypatch
    ):
        """An explicitly empty override (e.g. ``HERMES_DASHBOARD_WS_HOST=``)
        must NOT silently pin to loopback — it's an unset-by-accident, not
        an intent. Treat whitespace-only as absent and fall through."""
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "   ")
        _set_bound(saved_app_state, "0.0.0.0")
        assert web_server._resolve_client_ws_host() == "127.0.0.1"

    def test_no_bound_host_returns_none(
        self, saved_app_state, clear_ws_host_env
    ):
        web_server.app.state.bound_host = None
        web_server.app.state.bound_port = None
        assert web_server._resolve_client_ws_host() is None

    def test_bind_host_unchanged_after_wildcard_resolution(
        self, saved_app_state, clear_ws_host_env
    ):
        """Resolution only affects the client netloc — ``bound_host`` on
        ``app.state`` (used by the listener and host-header middleware) is
        NOT mutated."""
        _set_bound(saved_app_state, "0.0.0.0")
        web_server._resolve_client_ws_host()
        assert web_server.app.state.bound_host == "0.0.0.0"


# ---------------------------------------------------------------------------
# _build_gateway_ws_url — end-to-end URL contract
# ---------------------------------------------------------------------------


class TestGatewayWsUrlHost:
    def test_wildcard_bind_dials_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "0.0.0.0", port=9119)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        # ws://127.0.0.1:9119/api/ws?token=…
        assert url.startswith("ws://127.0.0.1:9119/api/ws")
        assert "0.0.0.0" not in url

    def test_ipv6_wildcard_bind_dials_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "::", port=9119)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert url.startswith("ws://127.0.0.1:9119/api/ws")
        # The ``::`` must not leak into the client URL.
        assert "::" not in url

    def test_loopback_bind_uses_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "127.0.0.1", port=8080)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert url.startswith("ws://127.0.0.1:8080/api/ws")

    def test_lan_bind_preserved(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "192.168.1.5", port=9120)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert url.startswith("ws://192.168.1.5:9120/api/ws")

    def test_explicit_env_overrides_wildcard(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "0.0.0.0", port=9119)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert url.startswith("ws://10.0.0.7:9119/api/ws")
        assert "0.0.0.0" not in url

    def test_explicit_env_overrides_lan(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "192.168.1.5", port=9120)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert url.startswith("ws://10.0.0.7:9120/api/ws")

    def test_wildcard_keeps_query_string(
        self, saved_app_state, clear_ws_host_env
    ):
        """Regression-guard: rewriting the host must not drop the
        ``?token=`` or ``?internal=`` credential."""
        _set_bound(saved_app_state, "0.0.0.0", port=9119)
        url = web_server._build_gateway_ws_url()
        assert url is not None
        assert "?" in url
        # Loopback / ``--insecure`` path uses the session token.
        assert f"token={web_server._SESSION_TOKEN}" in url

    def test_no_bound_host_returns_none(
        self, saved_app_state, clear_ws_host_env
    ):
        web_server.app.state.bound_host = None
        web_server.app.state.bound_port = None
        assert web_server._build_gateway_ws_url() is None


# ---------------------------------------------------------------------------
# _build_sidecar_url — end-to-end URL contract
# ---------------------------------------------------------------------------


class TestSidecarUrlHost:
    def test_wildcard_bind_dials_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "0.0.0.0", port=9119)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://127.0.0.1:9119/api/pub")
        assert "0.0.0.0" not in url
        assert "channel=ch-1" in url

    def test_ipv6_wildcard_bind_dials_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "::", port=9119)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://127.0.0.1:9119/api/pub")
        assert "::" not in url

    def test_loopback_bind_uses_loopback(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "127.0.0.1", port=8080)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://127.0.0.1:8080/api/pub")

    def test_lan_bind_preserved(
        self, saved_app_state, clear_ws_host_env
    ):
        _set_bound(saved_app_state, "192.168.1.5", port=9120)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://192.168.1.5:9120/api/pub")

    def test_explicit_env_overrides_wildcard(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "0.0.0.0", port=9119)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://10.0.0.7:9119/api/pub")
        assert "0.0.0.0" not in url

    def test_explicit_env_overrides_lan(
        self, saved_app_state, monkeypatch
    ):
        monkeypatch.setenv("HERMES_DASHBOARD_WS_HOST", "10.0.0.7")
        _set_bound(saved_app_state, "192.168.1.5", port=9120)
        url = web_server._build_sidecar_url("ch-1")
        assert url is not None
        assert url.startswith("ws://10.0.0.7:9120/api/pub")

    def test_no_bound_host_returns_none(
        self, saved_app_state, clear_ws_host_env
    ):
        web_server.app.state.bound_host = None
        web_server.app.state.bound_port = None
        assert web_server._build_sidecar_url("ch-1") is None


# ---------------------------------------------------------------------------
# _netloc helper is exposed only because it's useful for tests; if the
# production code ever changes the URL shape the tests catch the regression
# above without needing to assert on the full string.
# ---------------------------------------------------------------------------


def test_netloc_helper_handles_ipv6_bracket_form():
    """The IPv6 netloc path is exercised by the production ``[host]:port``
    branch when ``HERMES_DASHBOARD_WS_HOST`` points at an IPv6 address.
    Verify the helper doesn't choke on the bracket form."""
    assert _netloc("ws://[::1]:9119/api/ws?x=1") == "[::1]:9119"
    assert _netloc("ws://127.0.0.1:9119/api/ws?x=1") == "127.0.0.1:9119"