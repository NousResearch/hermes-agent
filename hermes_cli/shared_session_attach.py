"""Discover a cooperative local runtime without taking its session lease.

The owner's handshake supplies the existing authenticated WebSocket URL. A
registry entry is discovery information, not authority to mint a credential.
The handshake is a mutual proof of the registry's lease id, which never travels
in the request: the client sends ``nonce`` plus
``client_proof = hex(HMAC_SHA256(lease_id, "hermes-session-attach-request:<session_id>:<nonce>"))``
and the reply must echo the request fields and return
``attach_proof = hex(HMAC_SHA256(lease_id, "hermes-session-attach:<session_id>:<nonce>"))``.
A listener that only echoes request parameters cannot produce either proof.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx

from hermes_constants import get_hermes_home
from hermes_cli.active_sessions import active_session_registry_snapshot, session_owner_details


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
        raise ValueError("This chat is open in another Hermes window/terminal, and attaching "
                         "this terminal to it is not available in this build. Close the chat "
                         "there and run hermes --resume " + session_id + " here to take it over.\n"
                         + session_owner_details(session_id, owner))
    origin = _local_origin(endpoint, "http")
    parts = urlsplit(endpoint)
    if parts.path not in ("", "/") or parts.query:
        raise ValueError("Shared runtime endpoint must be an origin without a path or query.")
    nonce = secrets.token_urlsafe(24)
    # Proof the requester read the registry, without disclosing the lease id it keys on; the
    # responder checks this instead of a lease_id parameter so the secret is never transmitted.
    client_proof = hmac.new(
        str(owner["lease_id"]).encode("utf-8"),
        f"hermes-session-attach-request:{session_id}:{nonce}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    query = urlencode({"session_id": session_id, "nonce": nonce,
                       "profile_home": str(home), "client_proof": client_proof})
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
        raise ValueError("This chat is open in another Hermes window/terminal, and attaching "
                         "this terminal to it just failed. Use the chat where it is open, or "
                         "close it there and run hermes --resume " + session_id + " here.\n"
                         + session_owner_details(session_id, owner)) from exc
    # Echoed fields only prove the listener heard the request; the proof is what ties the
    # reply to the registry's lease id, which a rogue listener never sees.
    expected_proof = hmac.new(
        str(owner["lease_id"]).encode("utf-8"),
        f"hermes-session-attach:{session_id}:{nonce}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if (not isinstance(reply, dict)
            or any(reply.get(key) != value for key, value in {
                "session_id": session_id, "nonce": nonce, "profile_home": str(home),
            }.items())
            or not hmac.compare_digest(
                str(reply.get("attach_proof") or "").encode("utf-8", "replace"),
                expected_proof.encode("ascii"))):
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
