"""Regression tests for the WebSocket Origin guard's loopback-Origin exemption.

Issue #131271: Hermes Desktop in remote-only mode serves its renderer from a
random loopback port, so the renderer's WebSocket upgrade to a backend bound
to a non-loopback interface (e.g. a Tailscale IP) carries
``Origin: http://127.0.0.1:<random>``. ``_ws_host_origin_reason`` compared that
Origin against the bound host, never matched, and rejected the upgrade with
``origin_mismatch`` (surfaced as HTTP 403) — while the main-process HTTP probe
succeeded, leaving Desktop stuck on "Could not connect to Hermes gateway
(WebSocket error before open)".

The contract these tests pin down:

  * A loopback http(s) Origin (127.0.0.1 / ::1 / localhost, any port) passes
    the Origin half of the guard regardless of the bound host.
  * A non-loopback Origin that does not match the bound host is still
    rejected — the exemption must not weaken the DNS-rebinding defence for
    remote web pages.
  * A non-loopback Origin matching the bound host or a trusted public host
    still passes.
  * The Host header check is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_cli.web_server_chat as wsc
from hermes_cli import web_server


@pytest.fixture
def app_state():
    """Set bound_host / trusted_public_hosts on the dashboard app and restore after."""
    saved_bound = getattr(web_server.app.state, "bound_host", None)
    saved_trusted = getattr(web_server.app.state, "trusted_public_hosts", None)
    web_server.app.state.bound_host = "100.64.0.5"
    web_server.app.state.trusted_public_hosts = frozenset()

    def _ws(origin="", host="100.64.0.5:9119"):
        return SimpleNamespace(headers={"origin": origin, "host": host} if origin else {"host": host})

    yield _ws
    web_server.app.state.bound_host = saved_bound
    if saved_trusted is None:
        if hasattr(web_server.app.state, "trusted_public_hosts"):
            del web_server.app.state.trusted_public_hosts
    else:
        web_server.app.state.trusted_public_hosts = saved_trusted


class TestLoopbackOriginExemption:
    def test_desktop_loopback_origin_passes_on_non_loopback_bind(self, app_state):
        """The #131271 repro: Desktop renderer origin vs a Tailscale bind."""
        assert wsc._ws_host_origin_reason(app_state(origin="http://127.0.0.1:41235")) is None

    def test_localhost_origin_passes(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="http://localhost:3000")) is None

    def test_ipv6_loopback_origin_passes(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="http://[::1]:8080")) is None

    def test_https_loopback_origin_passes(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="https://127.0.0.1:9119")) is None


class TestNonLoopbackOriginsStillGuarded:
    def test_foreign_origin_still_rejected(self, app_state):
        reason = wsc._ws_host_origin_reason(app_state(origin="http://evil.example.com:8080"))
        assert reason is not None and reason.startswith("origin_mismatch")

    def test_matching_origin_still_passes(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="http://100.64.0.5:9119")) is None

    def test_trusted_public_origin_still_passes(self, app_state):
        web_server.app.state.trusted_public_hosts = frozenset({"dash.example.dev"})
        assert wsc._ws_host_origin_reason(app_state(origin="http://dash.example.dev")) is None

    def test_missing_origin_still_passes(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="")) is None

    def test_null_origin_still_passes(self, app_state):
        # file://-style origins ride the pre-existing non-http trust rule.
        assert wsc._ws_host_origin_reason(app_state(origin="null")) is None


class TestHostHeaderCheckUntouched:
    def test_bad_host_header_still_rejected(self, app_state):
        reason = wsc._ws_host_origin_reason(app_state(origin="http://127.0.0.1:41235", host="evil.example.com"))
        assert reason is not None and reason.startswith("host_mismatch")

    def test_good_host_header_accepted(self, app_state):
        assert wsc._ws_host_origin_reason(app_state(origin="http://127.0.0.1:41235", host="100.64.0.5:9119")) is None
