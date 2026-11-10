"""The Webapp proxy's async view of :mod:`hermes_cli.gateway_connections_reach`.

That module answers, per saved connection, where to dial and with which credentials, exactly as
native Desktop sends them (session token, refreshed native bearer, SSH tunnel). It blocks (token
refresh, tunnel start), so every call here runs off the event loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import httpx

from hermes_cli import gateway_connections as store
from hermes_cli import gateway_connections_oauth as oauth
from hermes_cli import gateway_connections_reach as reach
from hermes_cli.gateway_connections_ssh_transport import SshReachError

#: The socket that opens only with a ticket of its own (``web_session_sockets.HOST_TERMINAL_SOCKET``).
_OWN_TICKET_PATHS = {"/api/host-terminal": "host-terminal"}
#: What resolving a saved connection can refuse with: a reason the browser is shown.
_REACH_ERRORS = (store.ConnectionStoreError, oauth.NotSignedIn, oauth.OAuthSignInError, SshReachError)


class Unreachable(LookupError):
    """The connection exists but cannot be reached now (not signed in, no tunnel, no credential)."""


@dataclass(frozen=True)
class Target:
    connection_id: str
    label: str
    base_url: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def bearer(self) -> str:
        return self.headers.get("Authorization", "").removeprefix("Bearer ")


def _entry(connection_id: str) -> Optional[dict]:
    return next((c for c in store.read_registry()["connections"] if c["id"] == connection_id), None)


def entry_url(connection_id: str) -> Optional[str]:
    """What *connection_id* names now (its URL, or its SSH host), or None once it is removed."""
    entry = _entry(connection_id)
    return (entry.get("url") or entry.get("host") or entry["id"]) if entry else None


def _resolve(connection_id: str, rejected_token: str) -> Optional[Target]:
    entry = _entry(connection_id)
    if entry is None or entry["kind"] == "local":
        return None
    try:
        base_url, headers = reach.resolve_target(connection_id, rejected_token=rejected_token)
    except store.NoSuchConnection:
        return None
    except _REACH_ERRORS as exc:
        raise Unreachable(str(exc)) from exc
    if not any(k in headers for k in ("Authorization", reach.SESSION_TOKEN_HEADER)):
        raise Unreachable(f"{entry['label']} has no saved credential. Edit it in Settings → Gateways.")
    return Target(connection_id=connection_id, label=entry["label"], base_url=base_url, headers=headers)


async def resolve_target(connection_id: str, *, rejected: Optional[Target] = None) -> Optional[Target]:
    """*connection_id*'s target; None for no such saved connection or this machine's own (``local``).
    With *rejected* (the target the remote just answered 401), one forced sign-in refresh."""
    return await asyncio.to_thread(_resolve, connection_id, rejected.bearer if rejected else "")


def _socket_credential(target: Target, path: str) -> tuple[str, str]:
    own = _OWN_TICKET_PATHS.get(path)
    if own is not None:
        with httpx.Client(timeout=8.0, trust_env=False, follow_redirects=False) as client:
            response = client.post(f"{target.base_url}/api/webapp/socket-ticket?socket={own}", headers=target.headers)
        ticket = response.json().get("ticket") if response.status_code == 200 else None
        if not isinstance(ticket, str) or not ticket:
            raise Unreachable(f"{target.label} did not issue a {own} ticket ({response.status_code}).")
        return "ticket", ticket
    try:
        ticket = reach.mint_ws_ticket(target.connection_id, for_base_url=target.base_url)
    except _REACH_ERRORS as exc:
        raise Unreachable(str(exc)) from exc
    if ticket:
        return "ticket", ticket
    return "token", target.headers.get(reach.SESSION_TOKEN_HEADER, "")


async def socket_credential(target: Target, path: str) -> tuple[str, str]:
    """The ``(query name, value)`` that opens *path*'s socket on *target* for one dial: the session
    token (token mode, SSH), or a one-use ticket (oauth mode, and the host terminal everywhere)."""
    return await asyncio.to_thread(_socket_credential, target, path)
