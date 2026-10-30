"""Start the gateway listener with the Webapp on or off; report what a browser and a native client get.

Run as ``python webapp_listener_peer.py on|off <port> <dist> [macos]`` with an isolated ``HERMES_HOME``;
``macos`` takes the launch path of macOS and the BSDs (the printed link) on any host. Writes
``receipt.json`` there (the gateway redirects stdout). No external inference: the model points at a closed
port and no turn is submitted.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import traceback

SEEDED_TOKEN = "seeded-browser-credential-from-an-earlier-0"  # 43 URL-safe characters


def follow_launch_path(runner, home, platform: str) -> bool:
    """Have this gateway take *platform*'s launch path and say whether its launch links are checked.
    ``macos`` takes the path of macOS and the BSDs, where nothing identifies who owns the browser's
    connection: the gateway is told so through ``register_control_verbs(links_checked=False)``, and its
    link is printed for the user to open by hand. Anything else keeps the host's own path."""
    from gateway import webapp_listener

    if platform == "macos":
        webapp_listener.register_control_verbs(runner, home, links_checked=False)
        return False
    return webapp_listener._one_use_links_work()


async def _upgrade_refusal(url, **kwargs):
    """The HTTP status a refused upgrade answers, or ``"accepted"``."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    try:
        async with connect(url, **kwargs):
            return "accepted"
    except InvalidStatus as exc:
        return exc.response.status_code


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


async def sockets_across_sign_out(runner, client, origin, ask):
    """A tab's chat socket and host terminal and a native client's session socket, open across a
    sign-out; also what each browser socket ticket refuses to open."""
    from websockets.asyncio.client import connect

    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_host_terminal, web_server
    from tests.gateway.fixtures.local_recovery_probe import rpc

    # A harmless program stands in for the user's login shell.
    web_host_terminal.shell_spec = lambda: ([sys.executable, "-c", "import time; time.sleep(120)"], "python")
    sockets = origin.replace("http:", "ws:")
    session = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}

    async def ticket(socket):
        return (await client.post(f"/api/webapp/socket-ticket?socket={socket}", headers=session)).json()["ticket"]

    def chat_protocols(value):
        return ["hermes-gateway-v1", f"hermes-gateway-ticket.{value}"]

    report = {
        "terminal_refuses_session_token": await _upgrade_refusal(
            f"{sockets}/api/host-terminal?token={web_server._SESSION_TOKEN}", origin=origin),
        "terminal_refuses_chat_ticket": await _upgrade_refusal(
            f"{sockets}/api/host-terminal?ticket={await ticket('gateway')}", origin=origin),
        "chat_refuses_terminal_ticket": await _upgrade_refusal(
            f"{sockets}/api/ws?native_dial=browser", origin=origin,
            subprotocols=chat_protocols(await ticket("host-terminal"))),
    }
    authority = runner.session_authority
    runner.session_ticket_store = TicketStore("webapp-listener", frozenset({authority.profile_id}))
    native_ticket = runner.session_ticket_store.mint(
        profile_id=authority.profile_id, subject="fixture-same-user", purpose="interactive")
    async with connect(f"{sockets}/api/ws?native_dial=browser", origin=origin,
                       subprotocols=chat_protocols(await ticket("gateway"))) as chat, \
            connect(f"{sockets}/api/host-terminal?ticket={await ticket('host-terminal')}", origin=origin) as terminal, \
            connect(f"{sockets}/api/ws", subprotocols=chat_protocols(native_ticket)) as native:
        report["terminal_opens_with_its_ticket"] = (await terminal.recv()).startswith(web_host_terminal.META_PREFIX)
        report["chat_opens_before_sign_out"] = "result" in await rpc(chat, "session.list")
        report["sign_out_ok"] = ask("webapp-sign-out")["ok"]
        report["chat_close_after_sign_out"] = await _close_code(chat)
        report["terminal_close_after_sign_out"] = await _close_code(terminal)
        report["native_answers_after_sign_out"] = "result" in await rpc(native, "session.list")
    return report


async def probe(mode, port, dist, platform):
    import httpx
    from websockets.asyncio.client import connect

    from gateway.control_socket import GatewayControlServer
    from gateway.run import GatewayRunner
    from gateway.run_api import stop_gateway_api
    from gateway.run_runtime import start_gateway_runtime_api
    from gateway.session_authority import initialize_session_authority
    from gateway.webapp_listener import TOKEN_PATH
    import hermes_cli.webapp
    from tests.gateway.fixtures.local_recovery_probe import rpc

    home = Path(os.environ["HERMES_HOME"])
    webapp_block = f"webapp:\n  enabled: true\n  port: {port}\n" if mode == "on" else ""
    (home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: false\nmodel:\n  provider: custom\n  default: no-inference\n"
        "  base_url: http://127.0.0.1:1/v1\nauxiliary:\n  title_generation:\n    enabled: false\n" + webapp_block)
    if mode == "on":
        # Seeded user-only, as the gateway writes it: a stored token other users can read is replaced.
        (home / TOKEN_PATH).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / TOKEN_PATH).write_text(SEEDED_TOKEN + "\n", encoding="utf-8")
        (home / TOKEN_PATH).chmod(0o600)
    hermes_cli.webapp.webapp_dist_dir = lambda _root: Path(dist)  # the built bundle, without a build
    runner = GatewayRunner()
    runner.session_control_server = GatewayControlServer(home)  # verbs only; not listening
    # Set by the gateway bootstrap, which this fixture skips; the gateway is up.
    runner.session_runtime_descriptor = {"runtime_protocol": 1, "state": "ready", "capabilities": [],
                                         "served_profiles": []}
    await initialize_session_authority(runner, profile_id=str(home), instance_id="webapp-listener")
    await start_gateway_runtime_api(runner)
    links_checked = follow_launch_path(runner, home, platform)
    api = runner.session_api
    from hermes_cli import web_server

    def ask(verb):
        raw = json.dumps({"protocol": 1, "verb": verb, "id": 1, "params": {}}).encode()
        return json.loads(runner.session_control_server.handle_request_line(raw, "fixture-same-user"))

    report = {"api_origin": api.api_origin, "status": ask("webapp-status").get("result")}
    origin = api.api_origin
    try:
        async with httpx.AsyncClient(base_url=origin, timeout=20) as client:
            index = await client.get("/")
            report["index_status"] = index.status_code
            report["index_is_webapp"] = "WEBAPP_BUNDLE" in index.text
            report["index_carries_token"] = web_server._SESSION_TOKEN in index.text
            report["surface"] = getattr(web_server.app.state, "ui_surface", None)
            report["host_terminals"] = hasattr(web_server.app.state, "host_terminals")
            report["session_token_is_seeded"] = web_server._SESSION_TOKEN == SEEDED_TOKEN
            report["token_file"] = (home / TOKEN_PATH).exists()
            minted = await client.post("/api/webapp/socket-ticket?socket=gateway",
                                       headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN})
            report["ticket_status"] = minted.status_code
            if minted.status_code == 200:
                ticket = minted.json()["ticket"]
                url = origin.replace("http:", "ws:") + "/api/ws?native_dial=browser"
                async with connect(url, origin=origin,
                                   subprotocols=["hermes-gateway-v1", f"hermes-gateway-ticket.{ticket}"]) as ws:
                    created = await rpc(ws, "session.create", request_id="browser-create", source="gui",
                                        toolsets=[])
                    report["browser_session"] = "result" in created
            launched = ask("webapp-launch")
            report["launch_ok"] = launched["ok"]
            if launched["ok"]:
                url = launched["result"]["url"]
                report["launch_page"] = url.split("#", 1)[0] == origin + "/webapp/launch"
                report["launch_private"] = launched["result"]["private"]
                report["launch_printed_for_the_user"] = not links_checked
                ticket = url.split("#ticket=", 1)[1]
                handoff = {"X-Hermes-Window-Ticket": ticket, "Origin": origin}
                redeemed = await client.post("/webapp/window-session", headers=handoff)
                report["handoff_status"] = redeemed.status_code
                report["handoff_gives_credential"] = redeemed.status_code == 200 and (
                    redeemed.json()["token"] == SEEDED_TOKEN)
                report["handoff_reuse_status"] = (
                    await client.post("/webapp/window-session", headers=handoff)).status_code
                report.update(await sockets_across_sign_out(runner, client, origin, ask))
                stale = await client.post("/api/webapp/socket-ticket?socket=gateway",
                                          headers={"X-Hermes-Session-Token": SEEDED_TOKEN})
                report["old_credential_status"] = stale.status_code
                report["token_file_rotated"] = (home / TOKEN_PATH).read_text().strip() not in {"", SEEDED_TOKEN}
    finally:
        await stop_gateway_api(api)
    report["host_terminals_after_stop"] = hasattr(web_server.app.state, "host_terminals")
    return report


def main():
    receipt = Path(os.environ["HERMES_HOME"]) / "receipt.json"
    try:
        receipt.write_text(json.dumps(asyncio.run(probe(
            sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "this platform"))))
    except BaseException:
        receipt.write_text(json.dumps({"error": traceback.format_exc()}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
