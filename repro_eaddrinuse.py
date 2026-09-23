"""Live repro: bind the port, then drive the REAL api_server connect() + the REAL
startup no-connections gate, and report what the outside world can observe."""
import asyncio, os, socket, sys, tempfile

home = tempfile.mkdtemp(prefix="hermes-repro-")
os.environ["HERMES_HOME"] = home
os.environ["API_SERVER_KEY"] = "9f3c1d7b52a84e0c6b1f49d2a7c30e85"

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner


async def main():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(5)
    print(f"[repro] squatter holds 127.0.0.1:{port}")

    ad = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port}))
    ok = await ad.connect()
    print(f"[repro] api_server connect() -> {ok}")
    print(f"[repro] has_fatal_error={ad.has_fatal_error} code={ad.fatal_error_code!r} "
          f"retryable={ad.fatal_error_retryable}")

    r = GatewayRunner.__new__(GatewayRunner)
    r._startup_parked_platforms = False
    nonretry = [f"api_server: {ad.fatal_error_message}"]
    must_exit = GatewayRunner._start_handle_no_connections(
        r, connected_count=1, enabled_platform_count=2,
        startup_retryable_errors=[], startup_nonretryable_errors=nonretry)
    print(f"[repro] _start_handle_no_connections -> must_exit={must_exit}")
    print(f"[repro] _serving_state() -> {GatewayRunner._serving_state(r)!r}")
    print(f"[repro] queued for retry: {getattr(r, '_failed_platforms', {})}")

    # What an outside health probe sees:
    from gateway.readiness import collect_runtime_readiness
    rt = {"gateway_state": GatewayRunner._serving_state(r),
          "platforms": {"telegram": {"state": "connected"},
                        "api_server": {"state": "fatal"}}}
    rd = collect_runtime_readiness(configured_model="x", runtime_status=rt)
    print(f"[repro] readiness.status={rd['status']} gateway_check={rd['checks']['gateway']}")

    sock_probe = socket.socket()
    sock_probe.settimeout(1)
    try:
        sock_probe.connect(("127.0.0.1", port))
        print("[repro] port answers -- but by the SQUATTER, not the gateway")
    except OSError as e:
        print(f"[repro] port dead: {e}")
    s.close()
    print("[repro] squatter released the port; nothing in the gateway retries the bind.")


asyncio.run(main())
