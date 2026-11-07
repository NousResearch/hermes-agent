"""How the Webapp's forwarding proxy reaches a saved gateway connection as a normal client.

Every gateway this machine saved (``hermes_cli.gateway_connections``, the one list native Desktop
and the Webapp share) is reached with the credential saved beside it, exactly as Desktop sends it:

- token mode: the saved session token in ``X-Hermes-Session-Token`` on REST (``fetchJson`` in
  ``apps/desktop/electron/main.ts``) and as ``?token=`` on a socket (``buildGatewayWsUrl``);
- oauth mode (and Hermes Cloud): ``Authorization: Bearer <access token>`` from the saved native
  sign-in, refreshed at ``/auth/native/refresh`` when it is near expiry, and a one-use
  ``/api/auth/ws-ticket`` ticket for each socket (``oauth-rest-request.ts::mintGatewayWsTicket``);
- the entry's extra headers (an access proxy's) on both.

So the remote gateway needs no code or config of its own: it sees the client Desktop already is.
"""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import httpx

#: Refresh a native access token this many seconds before it expires (Electron ``tokenNeedsRefresh``).
_REFRESH_SKEW_SECONDS = 60
#: The socket that opens only with a ticket of its own (``web_session_sockets.HOST_TERMINAL_SOCKET``).
_OWN_TICKET_PATHS = {"/api/host-terminal": "host-terminal"}


class Unreachable(LookupError):
    """The connection exists but this proxy cannot reach it (its kind, or no saved sign-in)."""


@dataclass(frozen=True)
class Target:
    connection_id: str
    label: str
    base_url: str
    auth_mode: str
    headers: dict[str, str] = field(default_factory=dict)
    token: Optional[str] = None


def _entry(connection_id: str) -> Optional[dict]:
    from hermes_cli import gateway_connections

    return next((c for c in gateway_connections.read_registry()["connections"] if c["id"] == connection_id), None)


def entry_url(connection_id: str) -> Optional[str]:
    """The URL *connection_id* dials now, or None once it is removed (a live socket's re-check)."""
    entry = _entry(connection_id)
    return entry.get("url") if entry else None


def _needs_refresh(tokens: dict) -> bool:
    expires = tokens.get("expiresAt")
    if not isinstance(expires, (int, float)) or expires <= 0:
        return True
    return time.time() >= expires - _REFRESH_SKEW_SECONDS


def native_refresh_url(base_url: str) -> str:
    """``electron/native-oauth.ts::nativeRefreshUrl``."""
    parts = urllib.parse.urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}/auth/native/refresh"


async def _access_token(client: httpx.AsyncClient, connection_id: str, base_url: str) -> str:
    from hermes_cli import gateway_connections

    tokens = gateway_connections.read_tokens(connection_id)
    if not tokens:
        raise Unreachable("Sign in to this gateway from Settings → Gateways first.")
    if not _needs_refresh(tokens) or not tokens.get("refreshToken"):
        return str(tokens["accessToken"])
    response = await client.post(native_refresh_url(base_url), timeout=10.0,
                                 json={"refresh_token": tokens["refreshToken"], "provider": tokens.get("provider", "")})
    body = response.json() if response.status_code == 200 else {}
    access = str(body.get("access_token") or "")
    if not access:
        # Refused: the saved sign-in is over, so the next request asks for a new one.
        raise Unreachable("This gateway's sign-in expired. Sign in again from Settings → Gateways.")
    expires = body.get("expires_at")
    refreshed = {"accessToken": access, "refreshToken": str(body.get("refresh_token") or tokens["refreshToken"]),
                 "expiresAt": expires if isinstance(expires, (int, float)) else 0,
                 "provider": str(body.get("provider") or tokens.get("provider") or ""),
                 "userId": str(body.get("user_id") or tokens.get("userId") or "")}
    gateway_connections.write_tokens(connection_id, refreshed)
    return access


async def resolve_target(client: httpx.AsyncClient, connection_id: str) -> Optional[Target]:
    """*connection_id*'s base URL and the headers that authenticate a request there; None when no
    saved connection has that id or it is this machine's own (``local``, served directly)."""
    from hermes_cli import gateway_connections

    entry = _entry(connection_id)
    if entry is None or entry["kind"] == "local":
        return None
    if entry["kind"] not in {"remote", "cloud"} or not entry.get("url"):
        raise Unreachable(f"{entry['label']} is an {entry['kind'].upper()} connection, which the Webapp does not reach yet.")
    secret = gateway_connections.connection_secret(connection_id)
    base = str(entry["url"]).rstrip("/")
    auth_mode = "oauth" if entry["kind"] == "cloud" else entry.get("authMode") or "token"
    headers = dict(secret["headers"])
    token = None
    if auth_mode == "oauth":
        headers["Authorization"] = f"Bearer {await _access_token(client, connection_id, base)}"
    else:
        token = secret["token"]
        if not token:
            raise Unreachable(f"{entry['label']} has no saved session token.")
        headers["X-Hermes-Session-Token"] = token
    return Target(connection_id=connection_id, label=entry["label"], base_url=base, auth_mode=auth_mode,
                  headers=headers, token=token)


async def socket_credential(client: httpx.AsyncClient, target: Target, path: str) -> tuple[str, str]:
    """The ``(query name, value)`` that opens *path*'s socket on *target* for one dial."""
    own = _OWN_TICKET_PATHS.get(path)
    if own is None and target.auth_mode == "token":
        return "token", target.token or ""
    mint = f"/api/webapp/socket-ticket?socket={own}" if own else "/api/auth/ws-ticket"
    response = await client.post(f"{target.base_url}{mint}", headers=target.headers, timeout=8.0)
    ticket = response.json().get("ticket") if response.status_code == 200 else None
    if not isinstance(ticket, str) or not ticket:
        raise Unreachable(f"{target.label} did not issue a socket ticket ({response.status_code}).")
    return "ticket", ticket
