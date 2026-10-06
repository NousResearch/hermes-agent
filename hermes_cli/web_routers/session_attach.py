"""Cooperative local attachment to the existing serve WebSocket dispatcher."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/api/session-attach")
def session_attach(request: Request, session_id: str, lease_id: str, profile_home: str) -> dict:
    from hermes_cli import web_server
    from tui_gateway.shared_runtime import _attach_reply, attach_origin

    origin = attach_origin()
    if (not origin or request.headers.get("origin") or request.client is None
            or not ipaddress.ip_address(request.client.host).is_loopback
            or request.headers.get("host") != urlsplit(origin).netloc):
        raise HTTPException(status_code=403, detail="Local attachment only.")
    reply = _attach_reply(session_id, lease_id, profile_home, "")
    if getattr(web_server.app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import INTERNAL_PROVIDER, INTERNAL_USER_ID, mint_ticket
        credential = {"ticket": mint_ticket(user_id=INTERNAL_USER_ID, provider=INTERNAL_PROVIDER)}
    else:
        credential = {"token": web_server._SESSION_TOKEN}
    reply["websocket_url"] = origin.replace("http:", "ws:", 1) + "/api/ws?" + urlencode(credential)
    return reply
