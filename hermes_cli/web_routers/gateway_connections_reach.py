"""OAuth sign-in and SSH reach for the Webapp's saved gateway connections.

Beside ``/api/gateway-connections`` (lane store's router), on the same surface and behind the same
sign-in gate:

- ``GET|DELETE /api/gateway-connections/{id}/oauth``: sign-in status (never a token) / sign out;
- ``POST /api/gateway-connections/{id}/oauth/start``: the remote's authorize URL for the browser
  to open; its redirect comes back to ``/webapp/gateway-connections/oauth/callback`` here;
- ``POST /api/gateway-connections/{id}/oauth/complete {url}``: finish from a pasted redirect
  address (the browser was on another machine, so the redirect reached that machine's loopback);
- ``GET|POST|DELETE /api/gateway-connections/{id}/ssh``: tunnel status / bring it up / tear it
  down together with the owned remote serve.
"""

from __future__ import annotations

import html
from typing import Any, Callable

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from hermes_cli import gateway_connections as store
from hermes_cli import gateway_connections_oauth as oauth
from hermes_cli import gateway_connections_ssh as ssh

router = APIRouter()

_STATUS: tuple[tuple[type, int], ...] = (
    (store.NoSuchConnection, 404),
    (store.ConnectionStoreError, 400),
    (store.ConnectionStoreBusy, 503),
    (oauth.NotSignedIn, 401),
    (oauth.OAuthSignInError, 502),
    (ssh.SshReachError, 502),
)
_NO_STORE = {"Cache-Control": "no-store"}


def _http_error(exc: Exception) -> HTTPException | None:
    for kind, status in _STATUS:
        if isinstance(exc, kind):
            detail: Any = str(exc)
            if isinstance(exc, ssh.SshReachError):
                detail = {"message": str(exc), "kind": exc.kind}
            return HTTPException(status_code=status, detail=detail)
    return None


async def _call(fn: Callable[[], Any]) -> Any:
    try:
        return await run_in_threadpool(fn)
    except Exception as exc:
        mapped = _http_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _ssh_entry(connection_id: str) -> dict:
    entry = next((c for c in store.read_registry()["connections"] if c["id"] == connection_id), None)
    if entry is None:
        raise store.NoSuchConnection(f'No connection with id "{connection_id}".')
    if entry.get("kind") != "ssh":
        raise store.ConnectionStoreError(f'"{entry.get("label")}" is not an SSH connection.')
    return entry


# ── OAuth ────────────────────────────────────────────────────────────────────


@router.get("/api/gateway-connections/{connection_id}/oauth")
async def oauth_status(connection_id: str) -> dict:
    return {"ok": True, **await _call(lambda: oauth.sign_in_status(connection_id))}


@router.post("/api/gateway-connections/{connection_id}/oauth/start")
async def oauth_start(connection_id: str, request: Request, body: dict = Body(default={})) -> dict:
    state = request.app.state
    port = getattr(state, "bound_port", None)
    if not port:
        raise HTTPException(status_code=503, detail="This gateway's listener has not bound a port yet.")
    provider = str(body.get("provider") or "") if isinstance(body, dict) else ""
    started = await _call(lambda: oauth.start_sign_in(
        connection_id, bound_host=str(getattr(state, "bound_host", "") or ""), bound_port=int(port),
        provider=provider))
    return {"ok": True, **started}


@router.post("/api/gateway-connections/{connection_id}/oauth/complete")
async def oauth_complete(connection_id: str, body: dict = Body(...)) -> dict:
    url = str(body.get("url") or "") if isinstance(body, dict) else ""
    # The flight is matched to this connection before its code is redeemed (a mismatch is a 400
    # that leaves the other connection's flight usable).
    return {"ok": True, **await _call(lambda: oauth.complete_from_url(url, connection_id=connection_id))}


@router.delete("/api/gateway-connections/{connection_id}/oauth")
async def oauth_sign_out(connection_id: str) -> dict:
    await _call(lambda: oauth.sign_out(connection_id))
    return {"ok": True}


_PAGE = ('<!doctype html><meta charset="utf-8"><title>{title}</title>'
         '<body style="font:15px system-ui;margin:3rem;text-align:center">'
         '<h2>{title}</h2><p>{body}</p>')


@router.get(oauth.CALLBACK_PATH)
async def oauth_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    """The loopback redirect the remote sends the browser to. Answers a page only: the tokens stay
    on this server."""
    try:
        await run_in_threadpool(lambda: oauth.complete_sign_in(
            code=code, state=state, error=error, error_description=error_description))
    except (oauth.OAuthSignInError, store.NoSuchConnection, store.ConnectionStoreError,
            store.ConnectionStoreBusy) as exc:
        page = _PAGE.format(title="Sign-in failed", body=html.escape(str(exc)))
        return HTMLResponse(page, status_code=400, headers=_NO_STORE)
    page = _PAGE.format(title="&#10003; Signed in", body="You can close this tab and return to Hermes.")
    return HTMLResponse(page, headers=_NO_STORE)


# ── SSH ──────────────────────────────────────────────────────────────────────


def _tunnel_view(connection_id: str) -> dict:
    status = ssh.tunnel_status(connection_id)
    return {"ok": True, "connected": status is not None, **(status or {})}


@router.get("/api/gateway-connections/{connection_id}/ssh")
async def ssh_status(connection_id: str) -> dict:
    return await _call(lambda: (_ssh_entry(connection_id), _tunnel_view(connection_id))[1])


@router.post("/api/gateway-connections/{connection_id}/ssh")
async def ssh_connect(connection_id: str) -> dict:
    return await _call(lambda: (ssh.ensure_tunnel(_ssh_entry(connection_id)), _tunnel_view(connection_id))[1])


@router.delete("/api/gateway-connections/{connection_id}/ssh")
async def ssh_disconnect(connection_id: str) -> dict:
    return {"ok": True, "disconnected": await _call(lambda: ssh.disconnect(connection_id))}
