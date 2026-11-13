"""A test client that reaches the dashboard app as native Desktop reaches the gateway's own listener.

Native Desktop's local connection is the gateway's listener, and Electron's main process mints one
``native-http`` grant per request over the gateway's private control socket and sends it as
``X-Hermes-Gateway-Ticket`` from a loopback peer, with no Origin
(``apps/desktop/electron/local-gateway.ts``, ``nativeGatewayHttpHeaders``). Here the grant comes from
the same ``TicketStore`` the control socket mints from, bound to a runner serving the test's home,
and the peer is recorded as ``GatewayRuntimeAPI`` records it, so the real
``hermes_agent.hermes_cli.dashboard_auth.native_http`` gate admits each request.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

from fastapi.testclient import TestClient


def native_desktop_client(monkeypatch, profiles: dict[str, Path] | None = None) -> TestClient:
    """A client for ``web_server.app`` bound to a gateway runner serving ``get_hermes_home()``.

    The runner also serves *profiles* (name → home), as one multiplexing gateway does: a request
    naming one in ``?profile=`` carries a grant for that home, as Desktop mints one per profile."""
    from hermes_agent.gateway.runtime_bootstrap import TicketStore
    from hermes_agent.gateway.session_authorities import SessionAuthorities
    from hermes_agent.hermes_cli import web_server
    from hermes_agent.hermes_constants import get_hermes_home

    home = get_hermes_home().resolve()
    homes = {name: Path(path).resolve() for name, path in (profiles or {}).items()}
    authorities = SessionAuthorities(home)
    for served in (home, *homes.values()):
        authorities.add(served, SimpleNamespace(profile_id=str(served), instance_id="daemon"))
    store = TicketStore("daemon", authorities.profile_ids())
    runner = SimpleNamespace(
        session_authorities=authorities, session_ticket_store=store, _draining=False,
        session_runtime_descriptor={"state": "ready", "instance_id": "daemon",
                                    "served_profiles": authorities.served_profiles()})
    monkeypatch.setattr(web_server.app.state, "gateway_runner", runner, raising=False)

    async def native_desktop(scope, receive, send):
        if scope["type"] == "http":
            named = parse_qs(scope.get("query_string", b"").decode()).get("profile", [""])[0]
            ticket = store.mint(profile_id=str(homes.get(named, home)), subject="desktop", purpose="native-http")
            scope = {**scope, "headers": [*scope["headers"], (b"x-hermes-gateway-ticket", ticket.encode())],
                     "hermes.gateway_socket_peer": ("127.0.0.1", 50000)}
        await web_server.app(scope, receive, send)

    return TestClient(native_desktop, raise_server_exceptions=False)
