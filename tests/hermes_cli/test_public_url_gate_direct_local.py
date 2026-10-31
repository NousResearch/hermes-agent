"""A loopback listener that only a public URL put behind sign-in decides its gate per request.

Tailscale Serve (or any proxy) reaches the listener on 127.0.0.1 too, so the peer cannot tell a phone on
the tailnet from this machine; what Serve forwards carries the public Host and forwarding headers (measured
on Tailscale 1.9x: X-Forwarded-For/-Host/-Proto and Tailscale-User-*). A forwarded request keeps sign-in;
one made directly on this machine gets the private listener it had before the public URL.
"""
from __future__ import annotations

import pytest

from hermes_cli.web_server_boundary import (
    MarkDirectLocal, PublicUrlGate, direct_local_context, scope_is_direct_local)
from hermes_cli.web_server_surface import private_launch


def _scope(host=b"127.0.0.1:9120", peer="127.0.0.1", scheme="http", kind="http", extra=()):
    return {"type": kind, "scheme": scheme, "client": (peer, 50000),
            "headers": [(b"host", host), (b"user-agent", b"x"), *extra]}


TAILSCALE_SERVE = [(b"x-forwarded-for", b"100.101.102.103"), (b"x-forwarded-host", b"box.tailnet.ts.net"),
                   (b"x-forwarded-proto", b"https"), (b"tailscale-user-login", b"someone@example.com")]


@pytest.mark.parametrize("scope", [
    _scope(), _scope(host=b"localhost:9120"), _scope(host=b"[::1]:9120", peer="::1"), _scope(kind="websocket", scheme="ws")])
def test_a_request_made_directly_on_this_machine_is_local(scope):
    assert scope_is_direct_local(scope) is True


@pytest.mark.parametrize("scope", [
    pytest.param(_scope(host=b"box.tailnet.ts.net", extra=TAILSCALE_SERVE), id="tailscale-serve"),
    pytest.param(_scope(host=b"box.tailnet.ts.net"), id="public-host-without-headers"),
    pytest.param(_scope(extra=[(b"x-forwarded-for", b"100.1.2.3")]), id="loopback-host-forwarded"),
    pytest.param(_scope(extra=[(b"forwarded", b"for=100.1.2.3")]), id="rfc7239-forwarded"),
    pytest.param(_scope(extra=[(b"tailscale-user-login", b"a@b")]), id="tailscale-header-alone"),
    pytest.param(_scope(scheme="https"), id="https-scheme"),
    pytest.param(_scope(peer="100.101.102.103"), id="tailnet-peer"),
    pytest.param(_scope(peer="192.168.1.20"), id="lan-peer"),
    pytest.param(_scope(host=b"evil.example"), id="dns-rebinding-host"),
    pytest.param({"type": "lifespan"}, id="lifespan"),
])
def test_anything_forwarded_or_not_plainly_local_keeps_sign_in(scope):
    assert scope_is_direct_local(scope) is False


@pytest.mark.asyncio
async def test_the_gate_is_open_only_inside_a_direct_local_request():
    gate = PublicUrlGate()
    seen = {}

    async def app(scope, receive, send):
        seen[scope["tag"]] = (bool(gate), direct_local_context())

    marked = MarkDirectLocal(app)
    await marked({**_scope(), "tag": "direct"}, None, None)
    await marked({**_scope(host=b"box.tailnet.ts.net", extra=TAILSCALE_SERVE), "tag": "phone"}, None, None)
    assert seen == {"direct": (False, True), "phone": (True, False)}
    assert bool(gate) is True and direct_local_context() is None, "outside any request the gate stays on"


def test_tickets_minted_outside_a_request_open_a_private_webapp_but_a_forwarded_redeem_does_not():
    class State:
        ui_surface = "webapp"
        auth_required = PublicUrlGate()

    assert private_launch(State()) is True, "minted on the gateway's own loopback routes"

    class Plain:
        ui_surface = "webapp"
        auth_required = True

    assert private_launch(Plain()) is False, "an ordinary gated listener is unchanged"
