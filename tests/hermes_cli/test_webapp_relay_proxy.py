"""The Webapp's forwarding proxy reaches a saved gateway connection as an ordinary client.

``/api/relay/<connection id>/...`` on the serving gateway carries a signed-in browser's requests and
sockets to a gateway saved in this machine's one connection list (``hermes_agent.hermes_cli.gateway_connections``),
keyed by that registry id, with the credential saved for it (``hermes_agent.hermes_cli.webapp_relay_target``). The
other gateway sees what native Desktop sends it, so it needs nothing of its own. What it sends back
never acts on the serving gateway's origin, and the browser's own credentials never reach it.
"""
import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from hermes_agent.hermes_cli.web_routers import webapp_relay_proxy
from tests.hermes_cli.test_webapp_socket_tickets import _SESSION, serve_webapp
from tests.waiting import wait_until

_REAL_RELAY_CLIENT = webapp_relay_proxy.relay_client
OTHER = "http://other.lan:8000"
TOKEN = "saved-session-token-0123456789"
_ORIGIN = {"Origin": "http://localhost"}


def _other_gateway(seen: dict) -> Starlette:
    """Another gateway: records what each request brought, and answers the hardening cases."""

    async def echo(request: Request):
        seen.setdefault("requests", []).append({k: v for k, v in request.headers.items()})
        if request.headers.get("x-hermes-session-token") != seen.get("accepts", TOKEN) and \
                request.headers.get("authorization") != f"Bearer {seen.get('bearer')}":
            return JSONResponse({"error": "unauthenticated"}, status_code=401)
        return JSONResponse({"profiles": [{"name": "default"}]}, headers={"Set-Cookie": "edge=sticky; Path=/"})

    async def page(_request: Request):
        return HTMLResponse("<script>fetch('/api/env/reveal',{method:'POST'})</script>", headers={
            "Service-Worker-Allowed": "/", "Clear-Site-Data": '"cookies"', "Strict-Transport-Security": "max-age=1",
            "Access-Control-Allow-Origin": "https://attacker.example", "Access-Control-Allow-Credentials": "true",
            "Content-Security-Policy": "default-src *", "Refresh": "0; url=https://attacker.example/"})

    async def cached(_request: Request):
        return JSONResponse({"profiles": []}, headers={
            "Cache-Control": "public, max-age=31536000, immutable", "ETag": '"v1"',
            "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT", "Expires": "Fri, 01 Jan 2100 00:00:00 GMT"})

    async def redirect(request: Request):
        return Response(status_code=307, headers={"Location": request.query_params["to"]})

    async def offsite(_request: Request):
        return Response(status_code=302, headers={"Location": "https://attacker.example/phish"})

    async def refresh(request: Request):
        body = await request.json()
        seen["refreshed_with"] = body
        return JSONResponse({"access_token": "fresh-access", "refresh_token": "fresh-refresh",
                             "expires_at": int(time.time()) + 900, "provider": "nous", "user_id": "u1"})

    return Starlette(routes=[
        Route("/api/profiles", echo), Route("/page", page), Route("/cached", cached),
        Route("/redirect", redirect, methods=["GET", "POST"]), Route("/offsite", offsite),
        Route("/auth/native/refresh", refresh, methods=["POST"])])


@pytest.fixture
def webapp(tmp_path, monkeypatch):
    """The loopback Webapp; its machine's list saves the other gateway as ``other`` (token mode)."""
    from hermes_agent.hermes_cli import gateway_connections

    served = serve_webapp(tmp_path, monkeypatch)
    seen: dict = {}
    other = httpx.ASGITransport(app=_other_gateway(seen))
    monkeypatch.setattr(webapp_relay_proxy, "relay_client",
                        lambda _state: httpx.AsyncClient(transport=other, follow_redirects=False))
    saved, _registry = gateway_connections.save_connection(
        {"kind": "remote", "label": "Other", "url": OTHER, "authMode": "token", "token": TOKEN,
         "headers": {"CF-Access-Client-Id": "edge-id"}})
    served.seen, served.id = seen, saved["id"]
    served.browser = TestClient(served.server.app, base_url="http://localhost", headers=_SESSION)
    return served


def test_a_saved_connection_is_reached_with_its_saved_credential_and_never_the_browsers(webapp):
    reached = webapp.browser.get(f"/api/relay/{webapp.id}/api/profiles",
                                 headers={"Cookie": "hermes_session_at=browser-login", "Authorization": "Bearer mine",
                                          "X-Forwarded-For": "10.0.0.9"})
    assert reached.status_code == 200, reached.text
    sent = webapp.seen["requests"][-1]
    assert sent["x-hermes-session-token"] == TOKEN and sent["cf-access-client-id"] == "edge-id"
    for header in ("cookie", "authorization", "x-forwarded-for", "origin", "referer"):
        assert header not in sent, header
    assert "set-cookie" not in reached.headers


def test_an_unknown_connection_or_this_machines_own_is_not_relayed(webapp):
    assert webapp.browser.get("/api/relay/nope/api/profiles").status_code == 404
    assert webapp.browser.get("/api/relay/local/api/profiles").status_code == 404


def test_a_credential_the_gateway_refuses_asks_for_an_update_instead_of_this_gateways_sign_in(webapp):
    webapp.seen["accepts"] = "rotated-token"
    refused = webapp.browser.get(f"/api/relay/{webapp.id}/api/profiles")
    assert refused.status_code == 409 and refused.json()["error"] == "relay_reconnect"


def test_a_saved_gateway_cannot_run_content_or_leave_a_copy_on_this_origin(webapp):
    base = f"/api/relay/{webapp.id}"
    page = webapp.browser.get(f"{base}/page")
    assert page.status_code == 200 and page.headers.get_list("content-security-policy") == ["sandbox"]
    assert page.headers["x-content-type-options"] == "nosniff"
    for header in ("service-worker-allowed", "clear-site-data", "strict-transport-security", "refresh",
                   "access-control-allow-origin", "access-control-allow-credentials"):
        assert header not in page.headers, header
    cached = webapp.browser.get(f"{base}/cached")
    assert cached.headers.get_list("cache-control") == ["no-store"]
    for header in ("etag", "last-modified", "expires"):
        assert header not in cached.headers, header
    offsite = webapp.browser.get(f"{base}/offsite", follow_redirects=False)
    assert offsite.status_code == 502 and "location" not in offsite.headers


@pytest.mark.parametrize("escape", [
    "/../../../api/gateway/restart",
    "/%2e%2e/%2E%2E/%2e./api/gateway/restart",
    "/.%2E/%2e./api/x/../gateway/restart",
    "/a/../../../../api/gateway/restart",
    f"{OTHER}/../../../api/gateway/restart",
])
def test_a_saved_gateway_redirect_cannot_climb_out_of_its_relay(webapp, escape):
    # A browser resolves dot segments in the Location against the relay prefix: left in, any of
    # these sends a POST to this gateway's own /api/gateway/restart.
    base = f"/api/relay/{webapp.id}"
    sent = webapp.browser.post(f"{base}/redirect", params={"to": escape}, follow_redirects=False)
    assert (sent.status_code, sent.headers.get("location")) == (307, f"{base}/api/gateway/restart"), escape


@pytest.mark.parametrize("hidden", ["/.\t./.\t./../api/gateway/restart", "/..\\..\\api/gateway/restart",
                                    "/ok\n/x", "/a b", "../../api/gateway/restart"])
def test_a_saved_gateway_redirect_a_browser_would_reparse_is_refused(webapp, hidden):
    sent = webapp.browser.post(f"/api/relay/{webapp.id}/redirect", params={"to": hidden}, follow_redirects=False)
    assert sent.status_code == 502 and "location" not in sent.headers, (hidden, sent.status_code)


def test_a_saved_gateway_redirect_within_itself_resolves_under_its_relay(webapp):
    base = f"/api/relay/{webapp.id}"
    for to, expected in (("/a/../b?x=1", f"{base}/b?x=1"), ("/a/%2E/b/.", f"{base}/a/b/"), ("/a/b/..", f"{base}/a/")):
        sent = webapp.browser.get(f"{base}/redirect", params={"to": to}, follow_redirects=False)
        assert sent.status_code == 307 and sent.headers["location"] == expected, (to, sent.headers)


# ---- On real sockets ------------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def live_other(webapp, monkeypatch):
    """The other gateway on a real loopback port: sets a cookie, echoes voice frames on a socket
    that opens only with the saved token, and closes it with 4401 when told to."""
    import uvicorn

    from hermes_agent.hermes_cli import gateway_connections

    seen = {"cookies": [], "dials": []}
    stop = threading.Event()

    async def set_cookie(request: Request):
        seen["cookies"].append(request.headers.get("cookie"))
        return JSONResponse({}, headers={"Set-Cookie": "edge=sticky; Path=/"})

    async def speak(ws: WebSocket):
        seen["dials"].append(dict(ws.query_params))
        if ws.query_params.get("token") != TOKEN:
            await ws.close(code=4401)
            return
        await ws.accept()
        text = await ws.receive_text()
        await ws.send_text(json.dumps({"echo": json.loads(text)}))
        await asyncio.to_thread(stop.wait, 30)
        await ws.close(code=4401, reason="Signed out there")

    async def profiles(request: Request):
        seen.setdefault("bearers", []).append(request.headers.get("authorization"))
        if request.headers.get("authorization") != "Bearer fresh-access":
            return JSONResponse({"error": "unauthenticated", "reason": "invalid_or_expired_session"}, status_code=401)
        return JSONResponse({"profiles": [{"name": "default"}]})

    async def refresh(request: Request):
        seen["refreshed_with"] = await request.json()
        return JSONResponse({"access_token": "fresh-access", "refresh_token": "fresh-refresh",
                             "expires_at": int(time.time()) + 900, "provider": "nous", "user_id": "u1"})

    app = Starlette(routes=[Route("/set", set_cookie), Route("/echo", set_cookie), Route("/api/profiles", profiles),
                            Route("/auth/native/refresh", refresh, methods=["POST"]),
                            WebSocketRoute("/api/audio/speak-stream", speak)])
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning",
                                           ws="websockets"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until(lambda: server.started or not thread.is_alive(), timeout=15, what="the other gateway starting")
    monkeypatch.setattr(webapp_relay_proxy, "relay_client", _REAL_RELAY_CLIENT)
    saved, _registry = gateway_connections.save_connection(
        {"kind": "remote", "label": "Live", "url": f"http://127.0.0.1:{port}", "authMode": "token", "token": TOKEN})
    webapp.live_url = f"http://127.0.0.1:{port}"
    try:
        yield webapp, saved["id"], seen, stop
    finally:
        stop.set()
        server.should_exit = True
        thread.join(10)


def _relay_ticket(browser) -> str:
    minted = browser.post("/api/webapp/socket-ticket?socket=relay")
    assert minted.status_code == 200, minted.text
    return minted.json()["ticket"]


def test_an_oauth_connection_sends_its_bearer_and_a_refused_one_earns_one_forced_refresh(live_other):
    from hermes_agent.hermes_cli import gateway_connections

    webapp, _connection_id, seen, _stop = live_other
    saved, _registry = gateway_connections.save_connection(
        {"kind": "remote", "label": "Signed", "url": webapp.live_url.replace("127.0.0.1", "localhost"), "authMode": "oauth"})
    # Not near expiry: only the remote's 401 can say it is no longer good.
    gateway_connections.write_tokens(saved["id"], {"accessToken": "revoked-access", "refreshToken": "old-refresh",
                                                   "expiresAt": int(time.time()) + 900, "provider": "nous", "userId": "u1"})
    reached = webapp.browser.get(f"/api/relay/{saved['id']}/api/profiles")
    assert reached.status_code == 200, reached.text
    assert seen["bearers"] == ["Bearer revoked-access", "Bearer fresh-access"]
    assert seen["refreshed_with"] == {"refresh_token": "old-refresh", "provider": "nous"}
    assert gateway_connections.read_tokens(saved["id"])["accessToken"] == "fresh-access"


def test_an_oauth_connection_with_no_sign_in_asks_for_one(webapp):
    from hermes_agent.hermes_cli import gateway_connections

    saved, _registry = gateway_connections.save_connection(
        {"kind": "remote", "label": "Unsigned", "url": "http://unsigned.lan:8000", "authMode": "oauth"})
    refused = webapp.browser.get(f"/api/relay/{saved['id']}/api/profiles")
    assert refused.status_code == 409 and refused.json()["error"] == "relay_reconnect", refused.text


def test_the_proxy_carries_no_cookie_from_one_relayed_request_to_the_next(live_other):
    webapp, connection_id, seen, _stop = live_other

    async def browse():
        transport = httpx.ASGITransport(app=webapp.server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost", headers=_SESSION) as tab:
            assert (await tab.get(f"/api/relay/{connection_id}/set")).status_code == 200
            followed = await asyncio.gather(*(tab.get(f"/api/relay/{connection_id}/echo") for _ in range(4)))
            assert [r.status_code for r in followed] == [200] * 4

    asyncio.run(browse())
    assert seen["cookies"] == [None] * 5


def test_a_relayed_socket_dials_with_the_saved_token_and_carries_frames_and_the_far_close(live_other):
    webapp, connection_id, seen, stop = live_other
    url = f"ws://localhost/api/relay/{connection_id}/api/audio/speak-stream?relay_ticket={_relay_ticket(webapp.browser)}"
    with webapp.browser.websocket_connect(url, headers=_ORIGIN) as ws:
        ws.send_text(json.dumps({"text": "hi"}))
        assert ws.receive_json() == {"echo": {"text": "hi"}}
        stop.set()
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        assert (closed.value.code, closed.value.reason) == (4401, "Signed out there")
    # The browser's ticket never crossed; the saved token did.
    assert seen["dials"] == [{"token": TOKEN}]


def test_a_relayed_socket_opens_only_with_this_gateways_relay_ticket(live_other):
    webapp, connection_id, seen, _stop = live_other
    for query in ("", "?relay_ticket=forged", f"?ticket={TOKEN}"):
        with pytest.raises(WebSocketDisconnect) as refused:
            with webapp.browser.websocket_connect(
                    f"ws://localhost/api/relay/{connection_id}/api/audio/speak-stream{query}", headers=_ORIGIN) as ws:
                ws.receive_text()
        assert refused.value.code == 4401, query
    assert seen["dials"] == []


def test_removing_a_connection_closes_its_relayed_socket(live_other):
    webapp, connection_id, _seen, _stop = live_other
    url = f"ws://localhost/api/relay/{connection_id}/api/audio/speak-stream?relay_ticket={_relay_ticket(webapp.browser)}"
    with webapp.browser.websocket_connect(url, headers=_ORIGIN) as ws:
        ws.send_text(json.dumps({"text": "hi"}))
        ws.receive_json()
        assert webapp.browser.delete(f"/api/gateway-connections/{connection_id}").status_code == 200
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        assert (closed.value.code, closed.value.reason) == (4404, "Gateway removed")


def test_a_connection_removed_while_its_socket_dials_does_not_keep_that_socket(live_other, monkeypatch):
    import websockets.asyncio.client

    from hermes_agent.hermes_cli import gateway_connections

    webapp, connection_id, _seen, _stop = live_other
    url = f"ws://localhost/api/relay/{connection_id}/api/audio/speak-stream?relay_ticket={_relay_ticket(webapp.browser)}"
    dial = websockets.asyncio.client.connect

    def removed_mid_dial(*args, **kwargs):
        gateway_connections.remove_connection(connection_id)
        webapp_relay_proxy.close_gateway_sockets(connection_id)  # what the remove route does next
        return dial(*args, **kwargs)

    monkeypatch.setattr(websockets.asyncio.client, "connect", removed_mid_dial)
    with pytest.raises(WebSocketDisconnect) as closed:
        with webapp.browser.websocket_connect(url, headers=_ORIGIN) as ws:
            ws.send_text(json.dumps({"text": "hi"}))
            ws.receive_json()
    assert (closed.value.code, closed.value.reason) == (4404, "Gateway removed")
