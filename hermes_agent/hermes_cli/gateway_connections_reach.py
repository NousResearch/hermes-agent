"""Where the relay proxy dials a saved gateway connection, and with which credentials.

:func:`resolve_target` is the one hook the proxy calls per request: ``(base_url, headers)`` for the
registry entry, exactly the credentials native Desktop sends for that auth mode:

- token-mode remote/cloud: ``X-Hermes-Session-Token`` (``electron/main.ts`` fetchJson), plus the
  entry's extra headers (Cloudflare Access and the like);
- oauth-mode remote/cloud: ``Authorization: Bearer <access token>``, refreshed through the remote's
  ``/auth/native/refresh`` when at/near expiry (:mod:`hermes_agent.hermes_cli.gateway_connections_oauth`), plus
  the extra headers. After the remote answers 401, call again with ``rejected_token`` to force one
  rotation (``oauth-rest-request.ts``: only a structured 401 earns it);
- ssh: the tunnel's forwarded ``http://127.0.0.1:<port>`` and the owned serve's session token
  (:mod:`hermes_agent.hermes_cli.gateway_connections_ssh` ensures the tunnel; it blocks, so call it off the loop).

:func:`mint_ws_ticket` mints the remote's one-use socket ticket for an oauth-mode entry the way
Desktop does (``POST /api/auth/ws-ticket`` with the bearer). Token-mode and ssh sockets carry the
session token as ``?token=`` (``connection-config.ts::buildGatewayWsUrl``) and need no ticket.
"""

from __future__ import annotations

from typing import Optional

import httpx

from hermes_agent.hermes_cli import gateway_connections as store
from hermes_agent.hermes_cli import gateway_connections_oauth as oauth

SESSION_TOKEN_HEADER = "X-Hermes-Session-Token"
_TICKET_TIMEOUT_S = 8.0


def _entry(connection_id: str) -> dict:
    entry = next((c for c in store.read_registry()["connections"] if c["id"] == connection_id), None)
    if entry is None:
        raise store.NoSuchConnection(f'No connection with id "{connection_id}".')
    return entry


def _remote_target(entry: dict, rejected_token: str) -> tuple[str, dict[str, str]]:
    if entry.get("authMode") == "oauth":
        # The token comes back with the address it was issued by and that read's extra headers:
        # one snapshot, never this entry's (possibly older) URL paired with a newer sign-in.
        grant = oauth.ensure_access_token(entry["id"], rejected_token=rejected_token)
        return grant.base_url, {**grant.headers, "Authorization": f"Bearer {grant.access_token}"}
    snap = store.dial_snapshot(entry["id"])
    if snap["url"] is None or snap["entry"].get("kind") not in ("remote", "cloud") \
            or snap["entry"].get("authMode") == "oauth":
        raise store.ConnectionStoreError(f'"{entry.get("label")}" changed while connecting; try again.')
    headers = dict(snap["headers"])
    if snap["token"]:
        headers[SESSION_TOKEN_HEADER] = snap["token"]
    return snap["url"], headers


def _ssh_target(entry: dict, _rejected_token: str) -> tuple[str, dict[str, str]]:
    from hermes_agent.hermes_cli import gateway_connections_ssh as ssh

    tunnel = ssh.ensure_tunnel(entry)
    return tunnel.base_url, {SESSION_TOKEN_HEADER: tunnel.token}


_BY_KIND = {"remote": _remote_target, "cloud": _remote_target, "ssh": _ssh_target}


def resolve_target(connection_id: str, *, rejected_token: str = "") -> tuple[str, dict[str, str]]:
    """``(base_url, headers)`` to reach *connection_id* as a normal client of that gateway.

    Raises :class:`store.NoSuchConnection` (404), :class:`store.ConnectionStoreError` (400: the
    local entry, or a remote with no URL), :class:`oauth.NotSignedIn` (the user must sign in),
    :class:`oauth.OAuthSignInError` (the remote could not renew it) and
    ``gateway_connections_ssh.SshReachError`` (the tunnel could not be brought up)."""
    entry = _entry(connection_id)
    target = _BY_KIND.get(str(entry.get("kind") or ""))
    if target is None:
        raise store.ConnectionStoreError("The local connection is this gateway itself; it is not relayed.")
    if entry.get("kind") != "ssh" and not entry.get("url"):
        raise store.ConnectionStoreError(f'"{entry.get("label")}" has no gateway address.')
    return target(entry, rejected_token)


def _post_ticket(base_url: str, headers: dict[str, str]) -> httpx.Response:
    with httpx.Client(timeout=_TICKET_TIMEOUT_S, trust_env=False, follow_redirects=False) as client:
        return client.post(f"{base_url}/api/auth/ws-ticket", headers=headers)


def mint_ws_ticket(connection_id: str, *, for_base_url: Optional[str] = None) -> Optional[str]:
    """The remote's one-use socket ticket for an oauth-mode *connection_id*; ``None`` for the auth
    modes whose socket carries the session token instead. One forced rotation on a 401.

    With *for_base_url* (the address the socket will dial), a ticket is minted only by that
    gateway: an entry re-pointed meanwhile is refused rather than handing one host's ticket to
    another."""
    entry = _entry(connection_id)
    if entry.get("kind") not in ("remote", "cloud") or entry.get("authMode") != "oauth":
        return None
    base_url, headers = _ticket_target(connection_id, "", for_base_url)
    response = _post_ticket(base_url, headers)
    if response.status_code == 401:
        rejected = headers["Authorization"].removeprefix("Bearer ")
        base_url, headers = _ticket_target(connection_id, rejected, for_base_url or base_url)
        response = _post_ticket(base_url, headers)
    if response.status_code != 200:
        raise oauth.OAuthSignInError(f"{base_url} refused a socket ticket ({response.status_code}).")
    ticket = (response.json() or {}).get("ticket")
    if not isinstance(ticket, str) or not ticket:
        raise oauth.OAuthSignInError(f"{base_url} did not return a socket ticket.")
    return ticket


def _ticket_target(connection_id: str, rejected_token: str, for_base_url: Optional[str]) -> tuple[str, dict[str, str]]:
    base_url, headers = resolve_target(connection_id, rejected_token=rejected_token)
    if for_base_url is not None and base_url != for_base_url.rstrip("/"):
        raise oauth.OAuthSignInError("The connection changed address while connecting; try again.")
    return base_url, headers


def forget(connection_id: str) -> None:
    """Drop what reaching *connection_id* holds on this machine: its SSH tunnel (and the owned
    remote serve) and its sign-in. For the store's remove path."""
    from hermes_agent.hermes_cli import gateway_connections_ssh as ssh

    ssh.disconnect(connection_id)
    oauth.cancel_sign_in(connection_id)
    store.drop_tokens([connection_id])
