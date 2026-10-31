"""The gateway relay's boundaries against a paired gateway that misbehaves, and against routes the
grant's scopes must cover.

A second gateway ("Evil") is a small Starlette app the hub reaches through a host-dispatching
transport, so the hub's real pairing and proxy code talk to it as they would over the network. It
answers the code exchange itself, so it chooses what it calls its grant, and it serves whatever it
likes. The other gateway is the relay app itself, paired to itself at ``WEBAPP``.
"""
import json
import urllib.parse

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

# _relay_app is the relay_app fixture.
from tests.hermes_cli.test_webapp_relay import ORIGIN, WEBAPP, _pair, _relay_app, _sign_in

EVIL = "http://evil.spoke:8000"


class _ByHost(httpx.AsyncBaseTransport):
    def __init__(self, routes):
        self.routes = routes

    async def handle_async_request(self, request):
        return await self.routes[request.url.host].handle_async_request(request)


def _evil_gateway(gateway_a, seen: dict) -> Starlette:
    async def token(_request: Request):
        return JSONResponse({"grant_id": seen["grant_id"], "subject": "stub-user-1", "provider": "stub",
                             "issuer": "", "scopes": []})

    async def page(_request: Request):
        return HTMLResponse("<script>fetch('/api/env/reveal',{method:'POST'})</script>", headers={
            "Service-Worker-Allowed": "/", "Clear-Site-Data": '"cookies"', "Strict-Transport-Security": "max-age=1",
            "Access-Control-Allow-Origin": "https://attacker.example", "Access-Control-Allow-Credentials": "true",
            "Content-Security-Policy": "default-src *", "Refresh": "0; url=https://attacker.example/"})

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
        Route("/api/webapp/relay/token", token, methods=["POST"]), Route("/page", page),
        Route("/offsite", offsite), Route("/onsite", onsite),
        Route("/api/profiles/{name}/soul", soul, methods=["PUT"])])


def _with_evil(relay_app, monkeypatch, browser, grant_id="evil-grant") -> tuple[dict, dict]:
    """Pair *browser*'s hub with the evil gateway as well; returns its row and what it saw."""
    from hermes_cli.web_routers import webapp_relay_proxy

    seen = {"grant_id": grant_id}
    app = relay_app.server.app
    transport = _ByHost({"hermes.lan": httpx.ASGITransport(app=app),
                         "evil.spoke": httpx.ASGITransport(app=_evil_gateway(app, seen))})
    monkeypatch.setattr(webapp_relay_proxy, "relay_client",
                        lambda _state: httpx.AsyncClient(transport=transport, follow_redirects=False))
    started = browser.post("/api/webapp/relay/gateways", json={"url": EVIL, "label": "Evil"}, headers=ORIGIN)
    assert started.status_code == 200, started.text
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started.json()["authorize_url"]).query))["state"]
    back = browser.get(f"/webapp/relay/callback?state={state}&code=anything", follow_redirects=False)
    assert "relay=connected" in back.headers["location"], back.headers["location"]
    row = next(r for r in browser.get("/api/webapp/relay/gateways").json()["gateways"] if r["url"] == EVIL)
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
    row, _seen = _with_evil(relay_app, monkeypatch, browser)
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
    sent = browser.put(f"/api/relay/{evil['id']}/api/profiles/default/soul", json={"content": "meant for Evil"},
                       headers=ORIGIN)
    assert sent.status_code == 200
    assert seen["replay"].status_code == 401, seen["replay"].text
    assert browser.get(soul).json() == before
