"""The gateway's session listener serves local sessions without importing the dashboard app.

``start_gateway_api`` imported ``hermes_cli.web_server`` (about 690 routes, ~0.6 s on the event loop)
before the listener bound, stalling every cold start and the control socket's identify. The native
ticketed ``/api/ws`` must be served — with the same Host and peer boundary — before the dashboard is
imported, and the dashboard routes must still answer on the same origin (mounted on first request).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_PROBE = textwrap.dedent("""
    import asyncio, json, sys, types
    import httpx
    from websockets.asyncio.client import connect

    async def main():
        from gateway.run_api import start_gateway_api, stop_gateway_api
        from gateway.runtime_bootstrap import TicketStore
        import tui_gateway.ws

        async def fake_handle_ws(ws, **kwargs):
            await ws.accept(subprotocol="hermes-gateway-v1")
            await ws.send_json({"native": True, "dashboard_loaded": "hermes_cli.web_server" in sys.modules})
            await ws.close()
        tui_gateway.ws.handle_ws = fake_handle_ws

        home = sys.argv[1]
        tickets = TicketStore("owner", [home])
        authority = types.SimpleNamespace(db=types.SimpleNamespace(db_path=home + "/state.db"), profile_id=home)
        runner = types.SimpleNamespace(_draining=False, session_runtime_descriptor={"state": "ready"},
                                       session_authority=authority, session_ticket_store=tickets)
        out = {}
        handle = await start_gateway_api(runner)
        try:
            out["after_bind"] = sorted(m for m in ("hermes_cli.web_server", "fastapi") if m in sys.modules)
            url = handle.api_origin.replace("http:", "ws:") + "/api/ws"
            def protocols():
                return ["hermes-gateway-v1", "hermes-gateway-ticket." + tickets.mint(
                    profile_id=home, subject="human", purpose="interactive")]
            async with connect(url, subprotocols=protocols()) as ws:
                out["native"] = json.loads(await ws.recv())
            # DNS-rebinding guard applies before the dashboard is loaded: a raw upgrade carrying a
            # valid ticket but a foreign Host is refused (403) at the handshake.
            port = int(handle.api_origin.rsplit(":", 1)[1])
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write((
                "GET /api/ws HTTP/1.1\\r\\nHost: attacker.invalid\\r\\nUpgrade: websocket\\r\\n"
                "Connection: Upgrade\\r\\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\\r\\n"
                "Sec-WebSocket-Version: 13\\r\\nSec-WebSocket-Protocol: " + ", ".join(protocols()) + "\\r\\n\\r\\n"
            ).encode())
            await writer.drain()
            out["rebind"] = int((await asyncio.wait_for(reader.readline(), 10)).split()[1])
            writer.close()
            out["before_http"] = "hermes_cli.web_server" in sys.modules
            async with httpx.AsyncClient(base_url=handle.api_origin, trust_env=False) as client:
                out["status"] = (await client.get("/api/status")).status_code
                out["rebind_http"] = (await client.get("/api/status", headers={"Host": "attacker.invalid"})).status_code
            out["after_http"] = "hermes_cli.web_server" in sys.modules
            from hermes_cli import web_server
            out["runner_bound"] = web_server.app.state.gateway_runner is runner
            out["token_withheld"] = bool(web_server.app.state.withhold_session_token)
        finally:
            await stop_gateway_api(handle)
        from hermes_cli import web_server
        out["unbound"] = web_server.app.state.gateway_runner is None
        print("RESULT " + json.dumps(out), file=sys.stderr, flush=True)

    asyncio.run(main())
""")


def test_session_listener_serves_native_ws_before_importing_the_dashboard(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "TZ", "HERMES_RUNTIME_DIR", "TMPDIR"}}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(REPO))
    proc = subprocess.run([sys.executable, "-c", _PROBE, str(home)], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=120, check=False)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    out = json.loads(lines[-1][len("RESULT "):])
    assert out["after_bind"] == [], f"listener start imported {out['after_bind']}"
    assert out["native"] == {"native": True, "dashboard_loaded": False}
    assert out["rebind"] == 403
    assert out["before_http"] is False
    # Dashboard routes still answer on the same origin, behind the same Host guard.
    assert out["status"] == 200 and out["rebind_http"] == 400
    assert out["after_http"] and out["runner_bound"] and out["token_withheld"] and out["unbound"]
