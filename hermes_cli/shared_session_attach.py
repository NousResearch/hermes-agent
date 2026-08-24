"""Discover a cooperative local runtime without taking its session lease.

The owner's handshake supplies the existing authenticated WebSocket URL. A
registry entry is discovery information, not authority to mint a credential.
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx

from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import active_session_registry_snapshot, session_already_owned_message


def _local_origin(url: str, scheme: str) -> tuple[str, int]:
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if (parts.scheme != scheme or not local or parts.username is not None
            or parts.password is not None or parts.fragment or not parts.port):
        raise ValueError("Shared runtime endpoint must be an explicit loopback address and port.")
    return host, parts.port


def discover_attach_url(session_id: str, *, registry_home: str | Path | None = None) -> str | None:
    """Return a fenced authenticated URL, None for no owner, or refuse safely.

    This deliberately does not scan ports or read another profile. The runtime
    must advertise ``metadata.shared_runtime_url`` and implement the local
    ``/api/session-attach`` handshake. Unsupported owners keep their lease.
    """
    home = Path(registry_home if registry_home is not None else get_hermes_home()).resolve()
    owners = [entry for entry in active_session_registry_snapshot(home, strict=True)
              if entry.get("session_id") == session_id]
    if not owners:
        return None
    if len(owners) != 1:
        raise ValueError("Session owner identity is ambiguous; no attachment was attempted.")
    owner = owners[0]
    endpoint = (owner.get("metadata") or {}).get("shared_runtime_url")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("The live owner does not advertise cooperative attachment. "
                         + session_already_owned_message(session_id, owner))
    origin = _local_origin(endpoint, "http")
    parts = urlsplit(endpoint)
    if parts.path not in ("", "/") or parts.query:
        raise ValueError("Shared runtime endpoint must be an origin without a path or query.")
    query = urlencode({"session_id": session_id, "lease_id": owner["lease_id"],
                       "profile_home": str(home)})
    try:
        # Ignore proxy env and redirects: local discovery must stay on the
        # advertised endpoint, including on machines with corporate proxies.
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=3.0) as client:
            with client.stream("GET", endpoint.rstrip("/") + "/api/session-attach?" + query) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise ValueError("Shared runtime handshake response is too large.")
                reply = json.loads(body)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        # Never include a remote body or authenticated URL in diagnostics.
        raise ValueError("The live owner could not authorize cooperative attachment; "
                         "its lease was left intact.") from exc
    if not isinstance(reply, dict) or any(reply.get(key) != value for key, value in {
        "session_id": session_id, "lease_id": owner["lease_id"], "profile_home": str(home),
    }.items()):
        raise ValueError("Shared runtime handshake identity does not match the requested owner.")
    websocket_url = reply.get("websocket_url")
    if (not isinstance(websocket_url, str) or _local_origin(websocket_url, "ws") != origin
            or urlsplit(websocket_url).path != "/api/ws"):
        raise ValueError("Shared runtime handshake returned a different endpoint.")
    return websocket_url


def configure_tui_attachment(env: dict[str, str], session_id: str | None, *,
                             registry_home: str | Path | None = None) -> None:
    """Retain an explicit transport, otherwise attach a resumed owner's runtime."""
    if not session_id or env.get("HERMES_TUI_GATEWAY_URL", "").strip():
        return
    url = discover_attach_url(session_id, registry_home=registry_home)
    if url is not None:
        env["HERMES_TUI_GATEWAY_URL"] = url
