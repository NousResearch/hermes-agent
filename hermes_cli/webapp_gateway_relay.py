"""Relay a Webapp browser's live sockets to the profile's gateway, as the login the Webapp verified.

A browser cannot reach the owner (loopback, Origin-less, control-socket tickets), so the Webapp dials
for it: the browser's one-use Webapp ticket (``POST /api/webapp/gateway-ticket``) pins the gateway
home the route already ensured, and the upstream socket is an ``exposure`` grant naming the browser's
own login (``gateway/runtime_bootstrap.py::delegated_identity``), never the host's native grant
("An exposure proxy must not replace remote callers' credentials with its own native grant",
gateway-internals.md). Frames pass through unchanged: the renderer already speaks the canonical
contract when its URL carries ``native_dial`` (``apps/desktop/src/api/client.ts``).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

_log = logging.getLogger("hermes_cli.web_server")

#: Close code for a socket that never carried a gateway ticket: a tab served before this Webapp
#: attached to the gateway (or a remote Desktop speaking the old contract) must reload.
CLOSE_RELOAD = 4426
#: Codes a peer receives but no endpoint may send; mapped before mirroring an owner close.
_UNSENDABLE = {1005: 1000, 1006: 1012, 1015: 1012, 1004: 1011}


def principal_identity(request) -> dict:
    """``{user_id, provider, issuer}`` of the Webapp caller: its verified login, or on an
    unauthenticated loopback bind the session-token owner (the same two principals
    ``web_server_sessions._session_mutation_context`` forwards REST mutations as)."""
    from fastapi import HTTPException
    session = getattr(request.state, "session", None)
    if session is not None:
        return {"user_id": session.user_id, "provider": session.provider, "issuer": getattr(session, "issuer", "") or ""}
    if not getattr(request.app.state, "auth_required", False):
        from hermes_cli.web_server import _has_valid_session_token
        if _has_valid_session_token(request):
            from gateway.session_identity import SESSION_TOKEN_OWNER
            return {**SESSION_TOKEN_OWNER, "issuer": ""}
    raise HTTPException(status_code=401, detail="Unauthorized")


def principal_subject(identity: dict) -> str:
    """The ``auth:v1`` subject the gateway records for this login (``authenticated_subject``)."""
    from gateway.session_identity import authenticated_subject
    return authenticated_subject({"user_id": identity.get("user_id"), "provider": identity.get("provider"),
                                  "issuer": identity.get("issuer")})


async def _mint_exposure_ticket(home: Path, subject: str):
    """``(endpoint, ticket)``; a stale control socket (owner replaced between ensure and mint) is
    re-ensured once, as Electron's ``redialLocalGateway`` does."""
    from hermes_cli.gateway_client import GatewayClientError, _session_ticket
    from hermes_cli.webapp_gateway_owner import GatewayAttachError, owner_endpoint
    for attempt in (1, 2):
        endpoint = await owner_endpoint(home)
        try:
            ticket = await asyncio.to_thread(_session_ticket, home, endpoint, purpose="exposure", scope="host",
                                             subject=subject)
            return endpoint, ticket
        except (GatewayClientError, OSError) as exc:
            if attempt == 2:
                raise GatewayAttachError("starting", "ticket_unavailable", str(exc)) from exc
    raise AssertionError("unreachable")


async def _dial_owner(home: Path, subject: str):
    from websockets.asyncio.client import connect
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, gateway_ws_target
    from hermes_cli.web_server_boundary import _DESKTOP_ATTACHMENT_WS_MAX_BYTES
    from hermes_cli.webapp_gateway_owner import GatewayAttachError
    endpoint, ticket = await _mint_exposure_ticket(home, subject)
    url, protocols = gateway_ws_target(endpoint, ticket)
    # Loopback and Origin-less (the owner refuses a ticket socket that carries an Origin); never via
    # the user's HTTP(S) proxy; attachments ride the same 384 MiB frames the Webapp accepts.
    upstream = await connect(url, subprotocols=protocols, open_timeout=10, max_size=_DESKTOP_ATTACHMENT_WS_MAX_BYTES,
                             compression=None, ping_interval=None, proxy=None)
    if upstream.subprotocol != GATEWAY_WS_PROTOCOL:
        await upstream.close()
        raise GatewayAttachError("incompatible", "protocol_mismatch", "update or restart the gateway")
    return upstream


async def relay_gateway_ws(ws) -> None:
    """``/api/ws`` on a gateway-attached Webapp: the browser socket, already past the Webapp's own
    upgrade gate, becomes a pipe to the owner's canonical socket for that browser's login."""
    from websockets.exceptions import InvalidHandshake
    from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL
    from hermes_cli.webapp_gateway_owner import GatewayAttachError
    home = getattr(ws, "_hermes_gateway_home", None)
    if getattr(ws, "_hermes_ws_subprotocol", None) != GATEWAY_WS_PROTOCOL or not home:
        await ws.close(code=CLOSE_RELOAD, reason="Reload: this Webapp now attaches to the gateway")
        return
    identity = ws._hermes_auth_identity  # stamped with the gateway home checked above (_consume_ws_ticket)
    # Every refusal is reported AFTER accept, so its code and reason reach the browser in a close
    # frame instead of an opaque failed handshake: 1013 retries (the renderer re-dials and the
    # ticket route explains a lasting cause), 4403 means the owner would not take this login.
    try:
        upstream = await _dial_owner(Path(home), principal_subject(identity))
    except (GatewayAttachError, OSError, TimeoutError) as exc:
        _log.warning("webapp gateway relay for %s could not reach its owner: %s", home, exc)
        upstream, code, reason = None, 1013, str(exc)[:120]
    except InvalidHandshake as exc:
        _log.warning("webapp gateway relay for %s was refused: %s", home, exc)
        upstream, code, reason = None, 4403, "the gateway refused this login"
    await ws.accept(subprotocol=GATEWAY_WS_PROTOCOL)
    if upstream is None:
        await ws.close(code=code, reason=reason)
        return
    _log.info("webapp gateway relay attached peer=%s home=%s", getattr(ws.client, "host", "?"), home)
    await pump(ws, upstream)


async def relay_display_ws(ws, profile: str | None) -> None:
    """``/api/display/ws`` on a gateway-attached Webapp: the Bot Desktop ticket was minted by the
    owner's ``display.observe``, so only the owner's own route can redeem it; pass the RFB bytes.

    A real ticket therefore arrives while its owner runs. The relay only looks that owner up: a
    socket the owner has not yet authenticated never starts or restarts a gateway."""
    from urllib.parse import urlencode
    from fastapi import HTTPException
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidHandshake
    from hermes_cli.gateway_runtime import attachable, discover_gateway_endpoint
    from hermes_cli.web_server_cron import _cron_profile_home
    from hermes_cli.webapp_gateway_owner import GatewayAttachError
    await ws.accept()
    ticket = ws.query_params.get("display_ticket", "")
    if not ticket:
        await ws.close(code=4401, reason="display ticket missing, expired or used")
        return
    query = urlencode({"display_ticket": ticket})
    try:
        found = await asyncio.to_thread(discover_gateway_endpoint, _cron_profile_home(profile)[1])
        if not attachable(found):
            raise GatewayAttachError(found.state, found.reason_code, found.detail)
        upstream = await connect(found.endpoint.api_origin.replace("http:", "ws:") + "/api/display/ws?" + query,
                                 open_timeout=10, max_size=None, compression=None, ping_interval=None, proxy=None)
    except (GatewayAttachError, HTTPException, InvalidHandshake, OSError, TimeoutError) as exc:
        _log.warning("webapp display relay could not reach its owner: %s", exc)
        await ws.close(code=1013, reason="the gateway screen is unavailable")
        return
    # The owner's own route redeems the ticket and answers 4401/4001 in a close frame; mirrored as-is.
    await pump(ws, upstream)


async def pump(ws, upstream) -> None:
    """Pass frames both ways until either side closes, then close the other with the same code."""
    from websockets.exceptions import ConnectionClosed

    async def browser_to_owner():
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                return "browser", message.get("code") or 1000, ""
            data = message.get("text") if message.get("text") is not None else message.get("bytes")
            if data is not None:
                try:
                    await upstream.send(data)
                except ConnectionClosed:
                    return "owner", upstream.close_code, upstream.close_reason or ""

    async def owner_to_browser():
        try:
            async for frame in upstream:
                await (ws.send_bytes(frame) if isinstance(frame, bytes) else ws.send_text(frame))
        except ConnectionClosed:
            pass
        return "owner", upstream.close_code, upstream.close_reason or ""

    tasks = [asyncio.create_task(browser_to_owner()), asyncio.create_task(owner_to_browser())]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    first = next(iter(done))
    # A direction that died on its own error (the browser vanished mid-send) ends both sides.
    side, code, reason = ("browser", 1011, "") if first.exception() is not None else first.result()
    if side == "owner":
        code = _UNSENDABLE.get(code or 1006, code or 1006)
        try:
            await ws.close(code=code, reason=(reason or "")[:120])
        except RuntimeError:
            pass  # the browser already went away
    else:
        await upstream.close()
