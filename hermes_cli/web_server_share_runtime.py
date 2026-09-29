"""Start and stop sharing a running backend over tailcat.

One share per backend process: a loopback listener (``web_server_share.ShareGate``
over the same app) plus a supervised ``tailcat serve`` for that listener's port.
The port is remembered in ``share.json`` so the connection-code address stays
valid across restarts, the same way the saved node key keeps the tailcat
address stable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Optional

from hermes_cli import tailcat_share_store as store
from hermes_cli.tailcat_share import TailcatSupervisor, TailcatUnavailable, ensure_server_key, find_tailcat

log = logging.getLogger(__name__)

SHARE_MODES = ("off", "tailcat")


class _Share:
    def __init__(self, server, supervisor: TailcatSupervisor, port: int):
        self.server = server
        self.supervisor = supervisor
        self.port = port
        self.main_loop_task: Optional[asyncio.Task] = None


_ACTIVE: Optional[_Share] = None


def configured_mode(cfg: Optional[dict] = None) -> str:
    """``dashboard.share`` — ``off`` unless set to a supported transport."""
    if cfg is None:
        from hermes_cli.config import load_config

        cfg = load_config()
    mode = str(((cfg.get("dashboard") or {}).get("share")) or "off").strip().lower()
    return mode if mode in SHARE_MODES else "off"


def _port_free(port: int) -> bool:
    # SO_REUSEADDR like uvicorn's own bind: connections from the previous run
    # linger in TIME_WAIT, and without it the saved port always looks taken.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_port() -> int:
    saved = int(store.load_share_state().get("port") or 0)
    return saved if saved and _port_free(saved) else 0


def status() -> dict:
    state = store.load_share_state()
    if _ACTIVE is None:
        return {"state": "stopped", "address": state.get("address", ""), "port": state.get("port", 0),
                "error": "", "restarts": 0}
    return _ACTIVE.supervisor.status.as_dict()


async def start(*, install: bool = False) -> dict:
    """Bind the share listener and launch tailcat. Idempotent while running."""
    global _ACTIVE
    if _ACTIVE is not None:
        return status()

    binary = await asyncio.to_thread(find_tailcat, install=install)
    if binary is None:
        raise TailcatUnavailable("tailcat is not installed; run `hermes pm install tailcat`")
    key = await asyncio.to_thread(ensure_server_key, binary)

    import uvicorn

    from hermes_cli import web_server
    from hermes_cli.web_server_idle_exit import TUNNEL_WS_PING_INTERVAL_S, TUNNEL_WS_PING_TIMEOUT_S
    from hermes_cli.web_server_share import ShareGate

    gate = ShareGate(
        web_server.app,
        session_token=lambda: web_server._SESSION_TOKEN,
        upstream_host=lambda: f"127.0.0.1:{getattr(web_server.app.state, 'bound_port', 0)}",
    )
    # lifespan off: the main server already ran the app's lifespan. log_config
    # None: a second dictConfig at runtime would tear down Hermes's live log
    # handlers. A tailcat relay link can go half-open while the laptop sleeps,
    # so keep the tunnel ping.
    config = uvicorn.Config(
        gate, host="127.0.0.1", port=_pick_port(), log_config=None, lifespan="off",
        proxy_headers=False, ws_ping_interval=TUNNEL_WS_PING_INTERVAL_S,
        ws_ping_timeout=TUNNEL_WS_PING_TIMEOUT_S, ws_max_size=web_server._DESKTOP_ATTACHMENT_WS_MAX_BYTES,
    )
    server = uvicorn.Server(config)
    config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup()
    port = web_server._read_bound_port(server, fallback=0)

    announced = []

    def _record(address: str) -> None:
        state = store.load_share_state()
        if state.get("address") != address or state.get("port") != port:
            store.save_share_state({**state, "address": address, "port": port})
        # tailcat re-reports the address after every restart; tell the operator once.
        if not announced:
            announced.append(address)
            print(f"  Shared over tailcat, address {store.address_fingerprint(address)}\n"
                  "  Pair a device: hermes share code", flush=True)

    store.save_share_state({**store.load_share_state(), "port": port})
    supervisor = TailcatSupervisor(binary=binary, key=key, port=port, on_address=_record)
    supervisor.start()
    share = _Share(server, supervisor, port)
    share.main_loop_task = asyncio.ensure_future(server.main_loop())
    _ACTIVE = share
    log.info("tailcat share: listener on 127.0.0.1:%d; starting tailcat", port)
    return status()


async def stop() -> None:
    global _ACTIVE
    share, _ACTIVE = _ACTIVE, None
    if share is None:
        return
    await asyncio.to_thread(share.supervisor.stop)
    share.server.should_exit = True
    if share.main_loop_task is not None:
        try:
            await asyncio.wait_for(share.main_loop_task, timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            share.main_loop_task.cancel()
    await share.server.shutdown()
    log.info("tailcat share: stopped")


async def start_if_configured(mode: str) -> None:
    """Boot hook: a failure is logged and reported by /api/share/status, never fatal."""
    if mode != "tailcat":
        return
    try:
        await start(install=True)
    except Exception as exc:  # noqa: BLE001 — the backend must still serve locally
        log.error("tailcat share did not start: %s", exc)
