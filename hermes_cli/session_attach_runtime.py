"""Advertise a same-machine cooperative attach origin without minting authority.

The shipped TUI client (`shared_session_attach.discover_attach_url`) will only
handshake an explicit loopback HTTP origin. A registry row is discovery data;
the handshake still has to prove the live lease before returning a WS URL.
"""
from __future__ import annotations

import ipaddress
from typing import Optional

SESSION_ATTACH_PATH = "/api/session-attach"


def advertised_shared_runtime_origin() -> Optional[str]:
    """Return the loopback HTTP origin a second local TUI may handshake.

    None when the process is unbound, gated, or not dialable as loopback — the
    existing exclusive lease remains the only ownership signal in those cases.
    """
    try:
        from hermes_cli.web_server import app
        from hermes_cli.web_server_chat import _resolve_client_ws_host
    except (ImportError, SystemExit):
        return None
    if getattr(app.state, "auth_required", False):
        return None
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)
    try:
        port = int(port)
    except (TypeError, ValueError):
        return None
    if not host or not 1 <= port <= 65535 or not _host_is_loopback(host):
        return None
    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    return f"http://{netloc}"


def peer_is_loopback(host: str | None) -> bool:
    """True only for a concrete loopback address, never a hostname alias."""
    return _host_is_loopback(host, allow_localhost=False)


def _host_is_loopback(host: str | None, *, allow_localhost: bool = True) -> bool:
    raw = (host or "").strip()
    if not raw:
        return False
    if allow_localhost and raw.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(raw.split("%", 1)[0]).is_loopback
    except ValueError:
        return False
