"""Loopback cooperative-session handshake used by ``hermes --tui --resume``.

The route is intentionally not on ``PUBLIC_API_PATHS``: a gated public dashboard
must still require a session. Loopback token middleware lets this one path
through so a second local TUI can bootstrap before it has the dashboard token.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.session_attach_runtime import advertised_shared_runtime_origin, peer_is_loopback

router = APIRouter()


@router.get("/api/session-attach")
def session_attach(request: Request, session_id: str, lease_id: str, profile_home: str):
    """Return the live gateway WebSocket URL only for the exact owner lease."""
    client_host = request.client.host if request.client else ""
    if not peer_is_loopback(client_host):
        raise HTTPException(status_code=403, detail="Cooperative attachment is loopback-only")
    origin = advertised_shared_runtime_origin()
    if not origin:
        raise HTTPException(status_code=403, detail="Session owner does not support cooperative attachment")
    try:
        home = Path(profile_home).expanduser().resolve(strict=False)
        from hermes_cli.active_sessions import active_session_registry_snapshot

        owners = [
            entry for entry in active_session_registry_snapshot(home, strict=True)
            if entry.get("session_id") == session_id and entry.get("lease_id") == lease_id
        ]
    except OSError as exc:
        raise HTTPException(status_code=503, detail="Active-session ownership is unavailable") from exc
    advertised = (owners[0].get("metadata") or {}).get("shared_runtime_url") if len(owners) == 1 else None
    if advertised != origin:
        raise HTTPException(status_code=403, detail="Session owner does not support cooperative attachment")
    from hermes_cli.web_server_chat import _build_gateway_ws_url

    websocket_url = _build_gateway_ws_url()
    if not websocket_url:
        raise HTTPException(status_code=503, detail="Shared runtime is not ready")
    return {
        "session_id": session_id,
        "lease_id": lease_id,
        "profile_home": str(home),
        "websocket_url": websocket_url,
    }
