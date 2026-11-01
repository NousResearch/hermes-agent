"""The gateway relay's boundaries against a paired gateway that misbehaves, and against routes the
grant's scopes must cover.

A second gateway ("Evil") is a small Starlette app the hub reaches through a host-dispatching
transport, so the hub's real pairing and proxy code talk to it as they would over the network. It
answers the code exchange itself, so it chooses what it calls its grant, and it serves whatever it
likes. The other gateway is the relay app itself, paired to itself at ``WEBAPP``.
"""
import asyncio
import json
import threading
import urllib.parse

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from hermes_cli.web_routers import webapp_relay_proxy
# _relay_app is the relay_app fixture.
from tests.hermes_cli.test_webapp_relay import ORIGIN, WEBAPP, _pair, _relay_app, _sign_in
from tests.waiting import wait_until

# The production client factory, before the relay_app fixture swaps in a fresh client per call.
_REAL_RELAY_CLIENT = webapp_relay_proxy.relay_client

EVIL = "http://evil.spoke:8000"


class _ByHost(httpx.AsyncBaseTransport):
    def __init__(self, routes, down=frozenset()):
        self.routes, self.down = routes, down

    async def handle_async_request(self, request):
        if request.url.host in self.down:
            raise httpx.ConnectError("unreachable", request=request)
        return await self.routes[request.url.host].handle_async_request(request)


def _evil_gateway(gateway_a, seen: dict) -> Starlette:
    async def token(_request: Request):
        return JSONResponse({"grant_id": seen["grant_id"], "subject": seen.get("subject", "stub-user-1"),
                             "provider": "stub", "issuer": "", "scopes": []})

    async def revoke(_request: Request):
        seen["revoke_calls"] = seen.get("revoke_calls", 0) + 1
        return JSONResponse({}, status_code=seen.get("revoke_status", 200))

    async def redirect(request: Request):
        return Response(status_code=307, headers={"Location": request.query_params["to"]})

    async def page(_request: Request):
        return HTMLResponse("<script>fetch('/api/env/reveal',{method:'POST'})</script>", headers={
            "Service-Worker-Allowed": "/", "Clear-Site-Data": '"cookies"', "Strict-Transport-Security": "max-age=1",
            "Access-Control-Allow-Origin": "https://attacker.example", "Access-Control-Allow-Credentials": "true",
            "Content-Security-Policy": "default-src *", "Refresh": "0; url=https://attacker.example/"})

    async def cached(_request: Request):
        return JSONResponse({"profiles": []}, headers={
            "Cache-Control": "public, max-age=31536000, immutable", "ETag": '"v1"',
            "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT", "Expires": "Fri, 01 Jan 2100 00:00:00 GMT"})

    async def offsite(_request: Request):
        return Response(status_code=302, headers={"Location": "https://attacker.example/phish"})

    async def onsite(_request: Request):
        return Response(status_code=307, headers={"Location": "/api/profiles"})

    async def soul(request: Request):
        # Replay the hub's signed headers at gateway A, with a body of this gateway's choosing.
        signed = {k: v for k, v in request.headers.items() if k.startswith("x-hermes-relay-")}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_a), base_url=WEBAPP) as a:
            replay = await a.put(request.url.path, headers={**signed, "content-type": "application/json"},
                                 content=json.dumps({"content": "written by the other gateway"}))
        seen["replay"] = replay
        return PlainTextResponse("ok")

    return Starlette(routes=[
        Route("/api/webapp/relay/token", token, methods=["POST"]), Route("/page", page), Route("/cached", cached),
        Route("/offsite", offsite), Route("/onsite", onsite), Route("/redirect", redirect, methods=["GET", "POST"]),
        Route("/api/webapp/relay/grants/{grant_id}", revoke, methods=["DELETE"]),
        Route("/api/profiles/{name}/soul", soul, methods=["PUT"])])


def _with_evil(relay_app, monkeypatch, browser, grant_id="evil-grant", **seen) -> tuple[dict, dict]:
    """Pair *browser*'s hub with the evil gateway as well; returns its row (None when the pairing
    failed, with the callback's redirect in ``seen["back"]``) and what it saw."""
    from hermes_cli.web_routers import webapp_relay_proxy

    seen["grant_id"] = grant_id
    seen["down"] = set()  # hosts that stop answering
    app = relay_app.server.app
    transport = _ByHost({"hermes.lan": httpx.ASGITransport(app=app),
                         "evil.spoke": httpx.ASGITransport(app=_evil_gateway(app, seen))}, seen["down"])
    monkeypatch.setattr(webapp_relay_proxy, "relay_client",
                        lambda _state: httpx.AsyncClient(transport=transport, follow_redirects=False))
    started = browser.post("/api/webapp/relay/gateways", json={"url": EVIL, "label": "Evil"}, headers=ORIGIN)
    assert started.status_code == 200, started.text
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started.json()["authorize_url"]).query))["state"]
    back = browser.get(f"/webapp/relay/callback?state={state}&code=anything", follow_redirects=False)
    seen["back"] = back.headers["location"]
    row = next((r for r in browser.get("/api/webapp/relay/gateways").json()["gateways"] if r["url"] == EVIL), None)
    return row, seen


def test_a_grant_without_files_or_terminal_reaches_no_route_that_gives_them(relay_app, tmp_path, monkeypatch):
    from hermes_cli.web_routers import profiles

    spawned = []
    monkeypatch.setattr(profiles.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    browser = relay_app.browser()
    _sign_in(browser)
    base = f"/api/relay/{_pair(browser, scopes=())['id']}"
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "notes.txt").write_text("private", encoding="utf-8")
    for method, path, kwargs in (
            ("GET", "/api/files", {"params": {"path": str(kept)}}),
            ("DELETE", "/api/files", {"json": {"path": str(kept), "recursive": True}}),
            ("GET", "/api/ops/backup/download", {}),
            ("POST", "/api/profiles/default/open-terminal", {})):
        refused = browser.request(method, f"{base}{path}", headers=ORIGIN, **kwargs)
        assert refused.status_code == 403 and refused.json()["error"] == "relay_forbidden", (method, path, refused.text)
    assert (kept / "notes.txt").exists() and spawned == []


def test_a_socket_ticket_request_names_one_socket(relay_app):
    browser = relay_app.browser()
    _sign_in(browser)
    base = f"/api/relay/{_pair(browser, scopes=())['id']}"
    for query in ("socket=gateway&socket=host-terminal", "socket=host-terminal&socket=gateway"):
        refused = browser.post(f"{base}/api/webapp/socket-ticket?{query}", headers=ORIGIN)
        assert refused.status_code == 403, (query, refused.text)


def test_a_paired_gateway_cannot_run_content_or_redirect_on_this_origin(relay_app, monkeypatch):
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    base = f"/api/relay/{row['id']}"
    page = browser.get(f"{base}/page")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert page.headers.get_list("content-security-policy") == ["sandbox"]
    assert page.headers["x-content-type-options"] == "nosniff"
    for header in ("service-worker-allowed", "clear-site-data", "strict-transport-security", "refresh",
                   "access-control-allow-origin", "access-control-allow-credentials"):
        assert header not in page.headers, header
    offsite = browser.get(f"{base}/offsite", follow_redirects=False)
    assert offsite.status_code == 502 and "location" not in offsite.headers
    onsite = browser.get(f"{base}/onsite", follow_redirects=False)
    assert onsite.status_code == 307 and onsite.headers["location"] == f"{base}/api/profiles"


def test_a_paired_gateway_cannot_leave_a_cached_copy_on_this_origin(relay_app, monkeypatch):
    # A browser would serve a stored copy after the grant is revoked, or to the next account here.
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    cached = browser.get(f"/api/relay/{row['id']}/cached")
    assert cached.status_code == 200 and cached.headers.get_list("cache-control") == ["no-store"]
    for header in ("etag", "last-modified", "expires"):
        assert header not in cached.headers, header


def test_one_paired_gateway_cannot_replay_the_hubs_requests_at_another(relay_app, monkeypatch):
    from hermes_cli import webapp_relay_grants

    browser = relay_app.browser()
    _sign_in(browser)
    mine = _pair(browser, scopes=())
    soul = f"/api/relay/{mine['id']}/api/profiles/default/soul"
    before = browser.get(soul).json()
    # The evil gateway names the other gateway's grant as its own, so the hub signs for that grant.
    grant_a = webapp_relay_grants.list_grants(relay_app.home, {"user_id": "stub-user-1", "provider": "stub"})[0]["id"]
    evil, seen = _with_evil(relay_app, monkeypatch, browser, grant_id=grant_a)
    assert evil is not None, seen["back"]
    sent = browser.put(f"/api/relay/{evil['id']}/api/profiles/default/soul", json={"content": "meant for Evil"},
                       headers=ORIGIN)
    assert sent.status_code == 200
    assert seen["replay"].status_code == 401, seen["replay"].text
    assert browser.get(soul).json() == before


@pytest.mark.parametrize("escape", [
    "/../../../api/gateway/restart",
    "/%2e%2e/%2E%2E/%2e./api/gateway/restart",
    "/.%2E/%2e./api/x/../gateway/restart",
    "/a/../../../../api/gateway/restart",
    "http://evil.spoke:8000/../../../api/gateway/restart",
])
def test_a_paired_gateway_redirect_cannot_climb_out_of_its_relay(relay_app, monkeypatch, escape):
    # A browser resolves dot segments (plain or percent-encoded, any case) in the Location against
    # the relay prefix: left in, any of these sends a cookie-carrying POST to this hub's own
    # /api/gateway/restart. Resolved against the gateway instead, they name its route, through the relay.
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    base = f"/api/relay/{row['id']}"
    sent = browser.post(f"{base}/redirect", params={"to": escape}, headers=ORIGIN, follow_redirects=False)
    assert (sent.status_code, sent.headers.get("location")) == (307, f"{base}/api/gateway/restart"), escape


@pytest.mark.parametrize("hidden", ["/.\t./.\t./../api/gateway/restart", "/..\\..\\api/gateway/restart",
                                    "/ok\n/x", "/a b", "../../api/gateway/restart"])
def test_a_paired_gateway_redirect_a_browser_would_reparse_is_refused(relay_app, monkeypatch, hidden):
    # Browsers drop tabs and newlines from a URL and read a backslash as a slash, so these hide
    # dot segments from anything that reads the header as written. (HTTPX itself refuses a Location
    # with a tab or newline, as an unreachable gateway; the hub refuses the rest.)
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    sent = browser.post(f"/api/relay/{row['id']}/redirect", params={"to": hidden}, headers=ORIGIN,
                        follow_redirects=False)
    assert sent.status_code == 502 and "location" not in sent.headers, (hidden, sent.status_code, sent.headers)


def test_a_paired_gateway_redirect_within_itself_resolves_under_its_relay(relay_app, monkeypatch):
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    base = f"/api/relay/{row['id']}"
    for to, expected in (("/a/../b?x=1", f"{base}/b?x=1"), ("/a/%2E/b/.", f"{base}/a/b/"), ("/a/b/..", f"{base}/a/")):
        sent = browser.get(f"{base}/redirect", params={"to": to}, follow_redirects=False)
        assert sent.status_code == 307 and sent.headers["location"] == expected, (to, sent.status_code, sent.headers)


def test_a_pairing_approved_as_someone_else_says_when_its_grant_was_not_revoked(relay_app, monkeypatch):
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser, subject="someone-else", revoke_status=503)
    assert row is None and "relay=failed" in seen["back"]
    detail = dict(urllib.parse.parse_qsl(seen["back"].split("?", 1)[1]))["relay_detail"]
    assert seen.get("revoke_calls") == 1
    assert "could not be revoked" in detail and "503" in detail, detail


def _listed(browser) -> list[str]:
    return [row["id"] for row in browser.get("/api/webapp/relay/gateways").json()["gateways"]]


def test_removing_a_gateway_keeps_it_until_the_other_gateway_confirms_the_revocation(relay_app, monkeypatch):
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    remove = f"/api/webapp/relay/gateways/{row['id']}"
    for failure in ({"down": True}, {"revoke_status": 503}, {"revoke_status": 401}):
        seen["down"].clear()
        seen["down"].update({"evil.spoke"} if failure.get("down") else set())
        seen["revoke_status"] = failure.get("revoke_status", 200)
        sent = browser.delete(remove, headers=ORIGIN)
        assert sent.status_code == 200 and sent.json()["removed"] is False, (failure, sent.text)
        assert sent.json()["detail"].startswith("Evil did not confirm"), sent.json()
        assert _listed(browser) == [row["id"]], failure  # kept, so Remove can run again
    seen["down"].clear()
    seen["revoke_status"] = 200
    calls = seen["revoke_calls"]
    sent = browser.delete(remove, headers=ORIGIN)
    assert sent.json() == {"removed": True, "revoked": True}
    assert seen["revoke_calls"] == calls + 1 and _listed(browser) == []


def test_forgetting_a_gateway_here_only_says_its_grant_was_not_revoked(relay_app, monkeypatch):
    browser = relay_app.browser()
    _sign_in(browser)
    row, seen = _with_evil(relay_app, monkeypatch, browser)
    assert row is not None, seen["back"]
    seen["down"].add("evil.spoke")
    sent = browser.delete(f"/api/webapp/relay/gateways/{row['id']}?forget=1", headers=ORIGIN)
    assert sent.json() == {"removed": True, "revoked": False}
    assert "revoke_calls" not in seen and _listed(browser) == []


@pytest.fixture
def cookie_gateway():
    """A gateway on a real loopback port that sets a cookie on /set and reports the Cookie header
    each request brought, behind whatever cookie-setting proxy a real one might sit."""
    import uvicorn

    seen = {"cookies": []}

    async def token(_request: Request):
        return JSONResponse({"grant_id": "cookie-grant", "subject": "stub-user-1", "provider": "stub", "issuer": "",
                             "scopes": []})

    async def set_cookie(request: Request):
        seen["cookies"].append(request.headers.get("cookie"))
        return JSONResponse({}, headers={"Set-Cookie": "edge=sticky; Path=/"})

    async def echo(request: Request):
        seen["cookies"].append(request.headers.get("cookie"))
        return JSONResponse({})

    app = Starlette(routes=[Route("/api/webapp/relay/token", token, methods=["POST"]), Route("/set", set_cookie),
                            Route("/echo", echo)])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_until(lambda: server.started or not thread.is_alive(), what="the loopback gateway starting")
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", seen
    finally:
        server.should_exit = True
        thread.join(10)


def test_the_relay_carries_no_cookie_from_one_relayed_request_to_the_next(relay_app, monkeypatch, cookie_gateway):
    # The hub's real client, which lives as long as its event loop: the browser's requests below
    # share one loop, as they do on a running gateway.
    monkeypatch.setattr(webapp_relay_proxy, "relay_client", _REAL_RELAY_CLIENT)
    url, seen = cookie_gateway
    browser = relay_app.browser()
    _sign_in(browser)
    started = browser.post("/api/webapp/relay/gateways", json={"url": url, "label": "Edge"}, headers=ORIGIN)
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started.json()["authorize_url"]).query))["state"]
    back = browser.get(f"/webapp/relay/callback?state={state}&code=anything", follow_redirects=False)
    assert "relay=connected" in back.headers["location"], back.headers["location"]
    row = browser.get("/api/webapp/relay/gateways").json()["gateways"][0]

    async def browse():
        transport = httpx.ASGITransport(app=relay_app.server.app)
        signed_in = {"cookie": "; ".join(f"{name}={value}" for name, value in browser.cookies.items())}
        async with httpx.AsyncClient(transport=transport, base_url=WEBAPP, headers=signed_in) as tab:
            assert (await tab.get(f"/api/relay/{row['id']}/set")).status_code == 200
            followed = await asyncio.gather(*(tab.get(f"/api/relay/{row['id']}/echo") for _ in range(4)))
            assert [r.status_code for r in followed] == [200] * 4

    asyncio.run(browse())
    assert seen["cookies"] == [None] * 5
