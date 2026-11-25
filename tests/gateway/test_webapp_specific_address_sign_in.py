"""A Webapp bound to one IP address of this machine requires sign-in exactly like a wildcard bind.

Each check runs the same requests against a gateway dashboard mount bound to ``0.0.0.0`` and one bound
to ``192.168.1.20`` (with its loopback socket), from a remote peer at the address and from this
machine's own browser at ``127.0.0.1``, and asserts the two answer alike: no API route without
sign-in, the browser credential (session-token header, bearer or ``?token=``) never a way around it,
the page never carrying it, cookie writes only from the Webapp's own page, the same refusal when no
sign-in provider is registered, and ``webapp.public_url`` handled the same way. The one intended
difference: a wildcard bind cannot check the Host header, a specific address can, so a foreign Host
is refused there (the DNS-rebinding guard).
"""
import asyncio
import types

import httpx
import pytest

from gateway.run_api import GatewayDashboardMount
from gateway.webapp_listener import WebappMount, loopback_companion

ADDRESS = "192.168.1.20"
PORT = 9120
REMOTE_PEER = ("192.168.1.50", 50000)
LOCAL_PEER = ("127.0.0.1", 50000)
BINDS = ("0.0.0.0", ADDRESS)


@pytest.fixture
def stub_sign_in(monkeypatch):
    """The stub identity provider, the only sign-in a mount finds; the shared app state restored after."""
    from hermes_cli import web_server
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    for key in ("bound_host", "bound_port", "auth_required", "trusted_public_hosts", "trusted_public_origins",
                "loopback_companion"):
        monkeypatch.setattr(web_server.app.state, key, getattr(web_server.app.state, key, None), raising=False)
    clear_providers()
    register_provider(StubAuthProvider())
    yield
    clear_providers()


def _mount(tmp_path, bind: str, public_url: str = "") -> GatewayDashboardMount:
    """The gateway's dashboard mount for a Webapp bound to *bind*, as ``start_gateway_api`` builds it."""
    from hermes_constants import get_hermes_home

    dist = tmp_path / f"dist-{bind.replace(':', '_')}"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP</body></html>", encoding="utf-8")
    state = types.SimpleNamespace(bound_host=bind, bound_port=PORT, loopback_companion=loopback_companion(bind),
                                  webapp=WebappMount(dist, get_hermes_home(), public_url))
    return GatewayDashboardMount(types.SimpleNamespace(session_authority=None), state)


async def _gates(mount, base: str, peer) -> dict:
    """What a browser at *base* (peer *peer*) gets from the mounted Webapp, signed out then signed in."""
    web = await mount.ensure()
    token = web._SESSION_TOKEN  # the browser credential, as a tab holds it
    transport = httpx.ASGITransport(app=mount, client=peer)
    own_page = {"Origin": base, "Sec-Fetch-Site": "same-origin"}
    foreign_page = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    async with httpx.AsyncClient(transport=transport, base_url=base) as browser:
        page = await browser.get("/")
        answers = {
            "auth_required": web.app.state.auth_required,
            "status": (await browser.get("/api/status")).status_code,
            "signed_out": (await browser.get("/api/sessions")).status_code,
            "session_token_header": (await browser.get(
                "/api/sessions", headers={"X-Hermes-Session-Token": token})).status_code,
            "session_token_bearer": (await browser.get(
                "/api/sessions", headers={"Authorization": "Bearer " + token})).status_code,
            "session_token_query": (await browser.get(f"/api/sessions?token={token}")).status_code,
            "page": (page.status_code, page.headers.get("location", "").split("?")[0], token in page.text),
            "signed_out_write": (await browser.put("/api/config", headers=own_page, json={})).status_code,
        }
        await browser.get("/auth/login?provider=stub", follow_redirects=True)
        answers["signed_in"] = (await browser.get("/api/config")).status_code
        answers["own_page_write"] = (await browser.post(
            "/api/webapp/socket-ticket?socket=gateway", headers=own_page)).status_code
        answers["foreign_page_write"] = (await browser.post(
            "/api/webapp/socket-ticket?socket=gateway", headers=foreign_page)).status_code
    return answers


def _answers(tmp_path, bind: str, base: str, peer, public_url: str = "") -> dict:
    mount = _mount(tmp_path, bind, public_url)

    async def scenario():
        try:
            return await _gates(mount, base, peer)
        finally:
            await mount.close()

    return asyncio.run(scenario())


@pytest.mark.parametrize(("base", "peer"), [
    (f"http://{ADDRESS}:{PORT}", REMOTE_PEER),  # another machine, at the address
    (f"http://127.0.0.1:{PORT}", LOCAL_PEER),  # this machine's own browser, at the loopback socket
], ids=["at-the-address", "at-loopback"])
def test_a_specific_address_gates_every_request_like_a_wildcard_bind(tmp_path, stub_sign_in, base, peer):
    wildcard, specific = (_answers(tmp_path / bind, bind, base, peer) for bind in BINDS)
    assert specific == wildcard
    assert specific["auth_required"] is True and specific["status"] == 200
    for key in ("signed_out", "session_token_header", "session_token_bearer", "session_token_query"):
        assert specific[key] == 401, key
    assert specific["page"] == (302, "/auth/login", False), "the page redirects to sign-in and never holds the token"
    assert specific["signed_out_write"] == 401
    assert (specific["signed_in"], specific["own_page_write"], specific["foreign_page_write"]) == (200, 200, 403)


def test_behind_its_own_public_url_a_specific_address_gates_like_a_wildcard_bind(tmp_path, stub_sign_in):
    public = "https://hermes.example.net"
    wildcard, specific = (_answers(tmp_path / bind, bind, public, REMOTE_PEER, public) for bind in BINDS)
    assert specific == wildcard
    assert specific["auth_required"] is True
    for key in ("signed_out", "session_token_header", "session_token_bearer", "session_token_query"):
        assert specific[key] == 401, key


def test_a_specific_address_checks_the_host_a_wildcard_bind_cannot(tmp_path, stub_sign_in):
    """The intended difference: a DNS-rebinding Host reaches a wildcard bind's routes, never a
    specific address's (``/api/status`` is public, so only the Host check answers). The Webapp's own
    public hostname still passes."""
    async def host_answers(bind: str, public_url: str = "") -> dict:
        mount = _mount(tmp_path / f"{bind}-{bool(public_url)}", bind, public_url)
        await mount.ensure()
        transport = httpx.ASGITransport(app=mount, client=REMOTE_PEER)
        try:
            async with httpx.AsyncClient(transport=transport, base_url=f"http://{ADDRESS}:{PORT}") as client:
                return {host: (await client.get("/api/status", headers={"Host": host})).status_code
                        for host in ("rebound.example.net", "hermes.example.net", f"{ADDRESS}:{PORT}",
                                     f"127.0.0.1:{PORT}", f"localhost:{PORT}", "192.168.1.21")}
        finally:
            await mount.close()

    assert asyncio.run(host_answers("0.0.0.0")) == dict.fromkeys(
        ("rebound.example.net", "hermes.example.net", f"{ADDRESS}:{PORT}", f"127.0.0.1:{PORT}",
         f"localhost:{PORT}", "192.168.1.21"), 200)
    assert asyncio.run(host_answers(ADDRESS)) == {
        "rebound.example.net": 400, "hermes.example.net": 400, f"{ADDRESS}:{PORT}": 200,
        f"127.0.0.1:{PORT}": 200, f"localhost:{PORT}": 200, "192.168.1.21": 400}
    assert asyncio.run(host_answers(ADDRESS, "https://hermes.example.net"))["hermes.example.net"] == 200


def test_a_dashboard_bound_to_the_address_after_the_mount_closed_refuses_a_loopback_host(tmp_path, stub_sign_in):
    """The loopback socket goes with the mount: a standalone dashboard bound to the same address later
    in the process (no loopback socket) refuses a loopback Host through the real Host check, as main does."""
    from hermes_cli import web_server

    async def scenario() -> dict:
        mount = _mount(tmp_path, ADDRESS)
        await mount.ensure()
        await mount.close()
        web_server.app.state.bound_host = ADDRESS  # as ``hermes dashboard --host <address>`` binds it
        transport = httpx.ASGITransport(app=web_server.app, client=REMOTE_PEER)
        async with httpx.AsyncClient(transport=transport, base_url=f"http://{ADDRESS}:{PORT}") as client:
            return {host: (await client.get("/api/status", headers={"Host": host})).status_code
                    for host in (f"localhost:{PORT}", f"127.0.0.1:{PORT}", f"{ADDRESS}:{PORT}")}

    answers = asyncio.run(scenario())
    assert answers[f"localhost:{PORT}"] == answers[f"127.0.0.1:{PORT}"] == 400
    assert answers[f"{ADDRESS}:{PORT}"] != 400


def test_without_a_sign_in_provider_a_specific_address_is_refused_like_a_wildcard_bind(tmp_path, stub_sign_in):
    from hermes_cli.dashboard_auth import clear_providers

    clear_providers()

    def refusal(bind: str) -> str:
        mount = _mount(tmp_path / bind, bind)
        with pytest.raises(SystemExit) as refused:
            asyncio.run(mount.ensure())
        assert not mount.loaded
        return str(refused.value.code)

    wildcard, specific = refusal("0.0.0.0"), refusal(ADDRESS)
    assert specific == wildcard.replace("0.0.0.0", ADDRESS)
    assert f"Refusing to bind dashboard to {ADDRESS}" in specific


@pytest.mark.parametrize("origin", [f"http://{ADDRESS}:{PORT}", f"http://127.0.0.1:{PORT}", None])
def test_the_socket_host_and_origin_checks_admit_the_webapps_own_pages_like_a_wildcard_bind(origin):
    from hermes_cli.web_server_boundary import ws_client_reason, ws_host_origin_reason

    host = (origin or f"http://{ADDRESS}:{PORT}").split("//")[1]
    headers = {"host": host, **({"origin": origin} if origin else {})}
    for bind in BINDS:
        state = types.SimpleNamespace(bound_host=bind, bound_port=PORT, auth_required=True,
                                      trusted_public_hosts=frozenset(), loopback_companion=loopback_companion(bind))
        for peer in (REMOTE_PEER, LOCAL_PEER):
            ws = types.SimpleNamespace(headers=headers, client=types.SimpleNamespace(host=peer[0]))
            assert (ws_host_origin_reason(ws, state), ws_client_reason(ws, state)) == (None, None), (bind, peer)


def test_a_page_on_another_site_or_port_cannot_open_the_socket_at_a_specific_address():
    """A page on another loopback port is refused on both binds; a page on another site is refused
    by a specific address's Origin check, where a wildcard bind leaves it to the credential check."""
    from hermes_cli.web_server_boundary import ws_host_origin_reason

    def reason(bind: str, origin: str):
        state = types.SimpleNamespace(bound_host=bind, bound_port=PORT, auth_required=True,
                                      trusted_public_hosts=frozenset(), loopback_companion=loopback_companion(bind))
        ws = types.SimpleNamespace(headers={"host": f"127.0.0.1:{PORT}", "origin": origin}, client=None)
        return (ws_host_origin_reason(ws, state) or "").split(" ")[0]

    other_port = f"http://127.0.0.1:{PORT + 1}"
    assert reason("0.0.0.0", other_port) == reason(ADDRESS, other_port) == "origin_mismatch"
    assert (reason("0.0.0.0", "https://evil.example"), reason(ADDRESS, "https://evil.example")) == (
        "", "origin_mismatch")
