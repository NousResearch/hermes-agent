"""A real gateway serving the Webapp behind the sign-in gate; a browser signs in, then the owner signs out.

Run as ``python webapp_gated_sign_out_peer.py <port>`` with an isolated ``HERMES_HOME``. The listener binds
loopback only; a public URL engages the sign-in gate (``engage_sign_in_gate``), and the bundled password
provider (``plugins/dashboard_auth/basic``) is configured with a throwaway password. Writes ``receipt.json``
there (the gateway redirects stdout). No external inference: the model points at a closed port.
"""
import asyncio
import json
import os
from pathlib import Path
import secrets
import sys
import traceback

USER = "owner"
PASSWORD = "throwaway-" + secrets.token_urlsafe(12)


def engage_sign_in_gate(config: dict, public_url: str) -> None:
    """The setting that puts the gateway's loopback Webapp behind the sign-in gate: the URL browsers
    reach it at. The one place this test names that setting."""
    config.setdefault("webapp", {})["public_url"] = public_url


def write_config(home: Path, port: int) -> None:
    from plugins.dashboard_auth.basic import hash_password

    config = {
        "gateway": {"multiplex_profiles": False},
        "model": {"provider": "custom", "default": "no-inference", "base_url": "http://127.0.0.1:1/v1"},
        "auxiliary": {"title_generation": {"enabled": False}},
        "webapp": {"enabled": True, "port": port},
        "dashboard": {"basic_auth": {"username": USER, "password_hash": hash_password(PASSWORD),
                                     "secret": secrets.token_hex(32)}},
    }
    engage_sign_in_gate(config, f"http://hermes-webapp.test:{port}")
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")  # JSON is YAML


async def _close_code(ws, seconds=10):
    """The code the server closed *ws* with, reading past any frames it sends first."""
    from websockets.exceptions import ConnectionClosed

    try:
        async with asyncio.timeout(seconds):
            while True:
                await ws.recv()
    except ConnectionClosed as exc:
        return exc.rcvd.code if exc.rcvd else None
    except TimeoutError:
        return "open"


async def _upgrade(url, **kwargs):
    """``"accepted"``, or the HTTP status of a refused upgrade."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    try:
        async with connect(url, **kwargs):
            return "accepted"
    except InvalidStatus as exc:
        return exc.response.status_code


def _refusal(response):
    """Status and error kind of a refused request, so the report never carries a credential."""
    try:
        error = response.json().get("error")
    except ValueError:
        error = None
    return [response.status_code, error, response.headers.get("location", "").split("?", 1)[0]]


async def across_sign_out(runner, origin, ask):
    import httpx
    from websockets.asyncio.client import connect

    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_host_terminal
    from tests.gateway.fixtures.local_recovery_probe import rpc

    # A harmless program stands in for the user's login shell.
    web_host_terminal.shell_spec = lambda: ([sys.executable, "-c", "import time; time.sleep(120)"], "python")
    sockets = origin.replace("http:", "ws:")
    page = {"Origin": origin, "Sec-Fetch-Site": "same-origin"}  # the Webapp's own page
    report = {}

    def chat_protocols(value):
        return ["hermes-gateway-v1", f"hermes-gateway-ticket.{value}"]

    async def sign_in(browser):
        signed_in = await browser.post("/auth/password-login", headers=page,
                                       json={"provider": "basic", "username": USER, "password": PASSWORD})
        return signed_in.status_code

    async def ticket(browser, socket):
        minted = await browser.post(f"/api/webapp/socket-ticket?socket={socket}", headers=page)
        return minted.json()["ticket"] if minted.status_code == 200 else None

    authority = runner.session_authority
    runner.session_ticket_store = TicketStore("webapp-gated", frozenset({authority.profile_id}))
    native_ticket = runner.session_ticket_store.mint(
        profile_id=authority.profile_id, subject="fixture-same-user", purpose="interactive")
    async with httpx.AsyncClient(base_url=origin, timeout=20) as browser:
        report["before_sign_in"] = _refusal(await browser.get("/api/sessions"))
        report["sign_in_status"] = await sign_in(browser)
        report["sessions_before"] = (await browser.get("/api/sessions")).status_code
        report["me_before"] = (await browser.get("/api/auth/me")).status_code
        stolen = dict(browser.cookies.items())
        unused_terminal_ticket = await ticket(browser, "host-terminal")
        async with connect(f"{sockets}/api/ws?native_dial=browser", origin=origin,
                           subprotocols=chat_protocols(await ticket(browser, "gateway"))) as chat, \
                connect(f"{sockets}/api/host-terminal?ticket={await ticket(browser, 'host-terminal')}",
                        origin=origin) as terminal, \
                connect(f"{sockets}/api/ws", subprotocols=chat_protocols(native_ticket)) as native:
            report["terminal_open"] = (await terminal.recv()).startswith(web_host_terminal.META_PREFIX)
            report["chat_open"] = "result" in await rpc(chat, "session.list")
            answer = ask("webapp-sign-out")
            report["sign_out_ok"] = answer["ok"]
            report["sign_out_result"] = answer.get("result")
            report["chat_close"] = await _close_code(chat)
            report["terminal_close"] = await _close_code(terminal)
            report["native_answers"] = "result" in await rpc(native, "session.list")
        report["sessions_after"] = _refusal(await browser.get("/api/sessions"))
        report["me_after"] = _refusal(await browser.get("/api/auth/me"))
        report["page_after"] = _refusal(await browser.get("/", follow_redirects=False))
        report["terminal_ticket_after"] = _refusal(
            await browser.post("/api/webapp/socket-ticket?socket=host-terminal", headers=page))
        report["old_terminal_ticket_after"] = await _upgrade(
            f"{sockets}/api/host-terminal?ticket={unused_terminal_ticket}", origin=origin)
        async with httpx.AsyncClient(base_url=origin, timeout=20, cookies=stolen) as replay:
            report["replayed_cookies_after"] = _refusal(await replay.get("/api/sessions"))
        # The provider tokens those cookies held, presented on their own as a native app would.
        # The Webapp's listener names its cookies for the gateway (``hermes_session_at_<gateway>``).
        access, refresh = (next(value for key, value in stolen.items() if key.startswith(f"{name}_")).strip('"')
                           for name in ("hermes_session_at", "hermes_session_rt"))
        async with httpx.AsyncClient(base_url=origin, timeout=20) as thief:
            report["stolen_access_token_after"] = _refusal(
                await thief.get("/api/sessions", headers={"Authorization": "Bearer " + access}))
            report["stolen_refresh_token_after"] = _refusal(
                await thief.post("/auth/native/refresh", json={"refresh_token": refresh}))
        report["sign_in_again_status"] = await sign_in(browser)
        report["sessions_after_sign_in_again"] = (await browser.get("/api/sessions")).status_code
    return report


async def probe(port):
    from gateway.control_socket import GatewayControlServer
    from gateway.run import GatewayRunner
    from gateway.run_api import stop_gateway_api
    from gateway.run_runtime import start_gateway_runtime_api
    from gateway.session_authority import initialize_session_authority
    from hermes_cli.dashboard_auth import list_providers
    from hermes_cli.plugins import discover_plugins
    import hermes_cli.webapp

    home = Path(os.environ["HERMES_HOME"])
    write_config(home, port)
    dist = home / "dist-webapp"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP_BUNDLE</body></html>", encoding="utf-8")
    hermes_cli.webapp.webapp_dist_dir = lambda _root: dist  # the built bundle, without a build
    discover_plugins()  # the gateway loads the bundled sign-in providers at startup
    runner = GatewayRunner()
    runner.session_control_server = GatewayControlServer(home)  # verbs only; not listening
    # Set by the gateway bootstrap, which this fixture skips; the gateway is up.
    runner.session_runtime_descriptor = {"runtime_protocol": 1, "state": "ready", "capabilities": [],
                                         "served_profiles": []}
    await initialize_session_authority(runner, profile_id=str(home), instance_id="webapp-gated")
    await start_gateway_runtime_api(runner)
    api = runner.session_api
    from hermes_cli import web_server

    def ask(verb):
        raw = json.dumps({"protocol": 1, "verb": verb, "id": 1, "params": {}}).encode()
        return json.loads(runner.session_control_server.handle_request_line(raw, "fixture-same-user"))

    report = {"api_origin": api.api_origin, "status": ask("webapp-status").get("result"),
              "providers": sorted(p.name for p in list_providers()),
              "gate_engaged": bool(getattr(web_server.app.state, "auth_required", False)),
              "surface": getattr(web_server.app.state, "ui_surface", None)}
    try:
        report.update(await across_sign_out(runner, api.api_origin, ask))
    finally:
        await stop_gateway_api(api)
    return report


def main():
    receipt = Path(os.environ["HERMES_HOME"]) / "receipt.json"
    try:
        receipt.write_text(json.dumps(asyncio.run(probe(int(sys.argv[1])))))
    except BaseException:
        receipt.write_text(json.dumps({"error": traceback.format_exc()}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
