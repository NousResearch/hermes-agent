"""Discover and attach clients to the live Desktop ``hermes serve`` backend.

Discovery is evidence based: a canonical Hermes subcommand match, a real
LISTEN socket owned by that process, and a root-page marker plus session token.
When any proof fails callers retain their stock in-process behavior.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)

ATTACH_ENV_VAR = "HERMES_ATTACH_DESKTOP_BACKEND"
_TOKEN_RE = re.compile(r'__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"')
_HEADLESS_MARKER = "Headless backend"
_PROBE_TIMEOUT_S = 3.0
_CONNECT_TIMEOUT_S = 10.0
_MAX_FRAME_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DesktopBackend:
    """A proven-live Desktop ``hermes serve`` backend."""

    pid: int
    port: int
    token: str

    def ws_url(self, host: str = "127.0.0.1") -> str:
        token = urllib.parse.quote(self.token, safe="")
        return f"ws://{host}:{self.port}/api/ws?token={token}"


def _listening_ports(proc: Any) -> list[int]:
    """Return process-owned inet LISTEN ports; never infer or guess."""
    try:
        return sorted(
            {
                int(connection.laddr.port)
                for connection in proc.net_connections(kind="inet")
                if connection.status == "LISTEN"
                and connection.laddr
                and connection.laddr.port
            }
        )
    except Exception:
        return []


def _probe(port: int, host: str = "127.0.0.1") -> Optional[str]:
    """Return a token only when the endpoint proves it is a headless backend."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/", timeout=_PROBE_TIMEOUT_S
        ) as response:
            if response.status != 200:
                return None
            body = response.read(4096).decode("utf-8", "replace")
    except Exception:
        return None
    if _HEADLESS_MARKER not in body:
        return None
    match = _TOKEN_RE.search(body)
    return match.group(1) if match else None


def find_desktop_backend() -> Optional[DesktopBackend]:
    """Locate a live Desktop backend, returning None unless every proof passes."""
    try:
        import psutil
        from hermes_constants import get_default_hermes_root, hermes_home_key
        from hermes_cli.process_identity import ledger_entries
        from hermes_cli.profiles import (
            _argv_profile_selectors,
            get_active_profile,
            normalize_profile_name,
        )
        from hermes_cli.update_cmd import _hermes_holder_subcommand
    except Exception:
        logger.debug("Desktop-backend discovery dependencies unavailable", exc_info=True)
        return None

    try:
        intended_home = hermes_home_key(get_default_hermes_root())
        unselected_profile_is_default = (
            normalize_profile_name(get_active_profile()) == "default"
        )
        ledger = {
            int(entry["pid"]): entry
            for entry in ledger_entries()
            if entry.get("purpose") == "serve"
            and entry.get("profile") in ("", "default")
            and isinstance(entry.get("pid"), int)
            and isinstance(entry.get("port"), int)
        }
    except Exception:
        logger.debug("Desktop-backend identity ledger unavailable", exc_info=True)
        return None

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            argv = list(proc.info.get("cmdline") or [])
            cmdline = " ".join(argv)
            if not cmdline or _hermes_holder_subcommand(cmdline) != "serve":
                continue
            selectors = tuple(
                normalize_profile_name(value)
                for value in _argv_profile_selectors(argv)
            )
            if (selectors and any(value != "default" for value in selectors)) or (
                not selectors and not unselected_profile_is_default
            ):
                continue
            pid = int(proc.info["pid"])
            entry = ledger.get(pid)
            if entry is None:
                continue
            environment = proc.environ() or {}
            candidate_home = str(environment.get("HERMES_HOME") or "").strip()
            if (
                environment.get("HERMES_DESKTOP") != "1"
                or not candidate_home
                or hermes_home_key(candidate_home) != intended_home
            ):
                continue
            port = int(entry["port"])
            if port not in _listening_ports(proc):
                continue
            token = _probe(port)
            if token:
                backend = DesktopBackend(pid=pid, port=port, token=token)
                logger.info(
                    "Discovered Desktop backend pid=%s port=%s",
                    backend.pid,
                    backend.port,
                )
                return backend
        except Exception:
            continue
    logger.debug("No proven live Desktop backend found")
    return None


def sidecar_attach_enabled() -> bool:
    return os.environ.get(ATTACH_ENV_VAR) == "1"


async def _pump(read: Any, write: Any, label: str) -> None:
    """Copy frames in one direction until either endpoint closes."""
    try:
        while True:
            frame = await read()
            if frame is None:
                return
            await write(frame)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Sidecar relay %s ended: %s", label, exc)


async def proxy_sidecar_to_desktop(
    ws: Any, subprotocol: Optional[str] = None
) -> bool:
    """Relay a dashboard ``/api/ws`` client to Desktop when attach is proven.

    False means the client socket has not been accepted and the caller can run
    the stock in-process handler. True means this function owned the socket.
    """
    if not sidecar_attach_enabled():
        return False
    backend = find_desktop_backend()
    if backend is None:
        logger.warning(
            "%s=1 but no live Desktop backend was proven; using in-process sidecar",
            ATTACH_ENV_VAR,
        )
        return False

    try:
        import websockets
    except ImportError:
        logger.warning("websockets unavailable; using in-process sidecar", exc_info=True)
        return False

    url = backend.ws_url()
    try:
        upstream = await asyncio.wait_for(
            websockets.connect(
                url,
                subprotocols=[cast(Any, subprotocol)] if subprotocol else None,
                max_size=_MAX_FRAME_BYTES,
                open_timeout=_CONNECT_TIMEOUT_S,
            ),
            timeout=_CONNECT_TIMEOUT_S,
        )
    except Exception:
        # The downstream remains unaccepted, so the caller has a clean stock
        # fallback when discovery raced a Desktop restart.
        logger.warning(
            "Could not dial Desktop backend pid=%s port=%s; using in-process sidecar",
            backend.pid,
            backend.port,
            exc_info=True,
        )
        return False

    try:
        await (ws.accept(subprotocol=subprotocol) if subprotocol else ws.accept())
    except Exception:
        # A downstream disconnect during accept is already terminal. Do not let
        # the upstream connection leak or ask the caller to accept it again.
        await upstream.close()
        return True

    logger.info(
        "Sidecar attached to Desktop backend pid=%s port=%s",
        backend.pid,
        backend.port,
    )

    async def client_read() -> Optional[Any]:
        message = await ws.receive()
        if message.get("type") == "websocket.disconnect":
            return None
        text = message.get("text")
        return text if text is not None else message.get("bytes")

    async def client_write(frame: Any) -> None:
        if isinstance(frame, bytes):
            await ws.send_bytes(frame)
        else:
            await ws.send_text(frame)

    up = asyncio.create_task(_pump(client_read, upstream.send, "client->desktop"))
    down = asyncio.create_task(_pump(upstream.recv, client_write, "desktop->client"))
    try:
        await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (up, down):
            task.cancel()
        await asyncio.gather(up, down, return_exceptions=True)
        await upstream.close()
    return True
