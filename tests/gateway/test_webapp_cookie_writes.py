"""The gateway's Webapp on a wildcard host takes cookie-authenticated writes only from its own page.

``webapp.host: 0.0.0.0`` gates the gateway's dashboard mount behind dashboard sign-in, and the browser
keeps that sign-in in a cookie. A page on another site, or on another port of the same host, that makes
the browser POST, PUT or DELETE with the cookie is refused before any route runs. The Webapp's own page
writes normally, and native clients keep working: a bearer token from their own sign-in, or Electron's
cookie requests, which carry no Origin.
"""
import asyncio
import base64
import hashlib
import types
from urllib.parse import parse_qs, urlsplit

import httpx

from gateway.run_api import GatewayDashboardMount
from gateway.webapp_listener import WebappMount

WEBAPP = "http://hermes.lan:9120"
SKIN = {"config": {"display": {"skin": "mono"}}}
OWN_PAGE = {"Origin": WEBAPP, "Sec-Fetch-Site": "same-origin"}
FOREIGN_PAGES = (
    {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
    {"Origin": "http://hermes.lan:8080", "Sec-Fetch-Site": "same-site"},
    {"Origin": "https://evil.example"},  # a browser without Fetch Metadata
    {"Origin": "null", "Sec-Fetch-Site": "cross-site"},
)


async def _write_statuses(client: httpx.AsyncClient, headers: dict) -> tuple[int, int, int]:
    """Status codes of a POST (socket ticket), a PUT (config) and a DELETE (cron job) sent with *headers*."""
    ticket = await client.post("/api/webapp/socket-ticket?socket=gateway", headers=headers)
    config = await client.put("/api/config", headers=headers, json=SKIN)
    job = await client.delete("/api/cron/jobs/no-such-job", headers=headers)
    return ticket.status_code, config.status_code, job.status_code


async def _native_app_token(client: httpx.AsyncClient) -> str:
    """The access token a native app gets from its own sign-in (RFC 8252) on the Webapp's listener."""
    verifier = "native-app-verifier-" + "v" * 40
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    started = await client.get("/auth/native/authorize", params={
        "provider": "stub", "code_challenge": challenge, "code_challenge_method": "S256",
        "redirect_uri": "http://127.0.0.1:53999/cb", "state": "desk"})
    callback = parse_qs(urlsplit(started.headers["location"]).query)
    landed = await client.get("/auth/callback", params={"code": callback["code"][0], "state": callback["state"][0]})
    code = parse_qs(urlsplit(landed.headers["location"]).query)["code"][0]
    tokens = await client.post("/auth/native/token", json={"code": code, "code_verifier": verifier})
    return tokens.json()["access_token"]


def test_a_wildcard_webapp_takes_cookie_writes_only_from_its_own_page(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.config import read_raw_config
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from hermes_constants import get_hermes_home
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider

    for key in ("bound_host", "bound_port", "auth_required", "trusted_public_hosts"):
        monkeypatch.setattr(web_server.app.state, key, getattr(web_server.app.state, key, None), raising=False)
    dist = tmp_path / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP</body></html>", encoding="utf-8")
    state = types.SimpleNamespace(bound_host="0.0.0.0", bound_port=9120,
                                  webapp=WebappMount(dist, get_hermes_home()))
    mount = GatewayDashboardMount(types.SimpleNamespace(session_authority=None), state)
    clear_providers()
    register_provider(StubAuthProvider())

    async def scenario():
        web = await mount.ensure()
        try:
            assert (web.app.state.auth_required, web.app.state.ui_surface) == (True, "webapp")
            transport = httpx.ASGITransport(app=mount)
            async with httpx.AsyncClient(transport=transport, base_url=WEBAPP) as browser:
                await browser.get("/auth/login?provider=stub", follow_redirects=True)
                for page in FOREIGN_PAGES:
                    assert await _write_statuses(browser, page) == (403, 403, 403), page
                assert read_raw_config().get("display", {}).get("skin") != "mono"

                ticket, config, job = await _write_statuses(browser, OWN_PAGE)
                assert (ticket, config) == (200, 200)
                assert job not in {401, 403}
                assert read_raw_config()["display"]["skin"] == "mono"
                # Electron's net.request sends the cookie with no Origin and no page Fetch Metadata.
                for native in ({}, {"Sec-Fetch-Site": "none"}):
                    native_ticket = await browser.post("/api/webapp/socket-ticket?socket=gateway", headers=native)
                    assert native_ticket.status_code == 200, native_ticket.text
            async with httpx.AsyncClient(transport=transport, base_url=WEBAPP) as native_app:
                bearer = {"Authorization": f"Bearer {await _native_app_token(native_app)}", "Origin": "null"}
                ticket, config, job = await _write_statuses(native_app, bearer)
                assert (ticket, config) == (200, 200)
                assert job not in {401, 403}
        finally:
            await mount.close()

    try:
        asyncio.run(scenario())
    finally:
        clear_providers()
