"""Team Gateway account linking for the Channels surfaces (dashboard and Desktop).

On a relay-fronted instance the shared bot is the only way to reach Telegram, Discord and Slack;
these routes let the user link their chat account to it from the instance (#141546). Codes are
minted by the running gateway, which alone holds the per-gateway relay secret, through its
``relay-link`` control-socket verb (``gateway/relay/link.py``).
"""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Connector answers that mean "the instance is not ready yet" rather than "this cannot work".
_NOT_READY = frozenset({"relay_not_provisioned", "connector_unreachable"})


def _ask_gateway(params: dict[str, Any]) -> dict[str, Any]:
    from gateway.control_socket import query_gateway_control
    from gateway.relay.link import CONTROL_TIMEOUT_S
    from hermes_constants import get_process_hermes_home

    answer = query_gateway_control(get_process_hermes_home(), "relay-link", params=params,
                                   timeout=CONTROL_TIMEOUT_S)
    if answer is None:
        raise HTTPException(status_code=503, detail={
            "code": "gateway_not_running",
            "message": "Start the gateway first: it holds the Team Gateway connection."})
    if not answer.get("ok"):
        code = str(answer.get("error") or "connector_refused")
        raise HTTPException(status_code=503 if code in _NOT_READY else 409,
                            detail={"code": code, "message": answer.get("message") or code})
    return {k: v for k, v in answer.items() if k != "ok"}


async def _call(params: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(_ask_gateway, params)


@router.get("/api/messaging/relay")
async def get_relay_links():
    """Accounts linked to this instance per shared-bot platform, and each bot's DM link."""
    return await _call({"action": "status"})


@router.post("/api/messaging/relay/{platform}/link")
async def request_relay_link(platform: str):
    """A one-time ``/link <code>`` for the account that sends it to the shared bot."""
    return await _call({"action": "request", "platform": platform})


@router.get("/api/messaging/relay/link/{code}")
async def get_relay_link_state(code: str):
    return await _call({"action": "state", "code": code})


@router.delete("/api/messaging/relay/{platform}/accounts/{platform_user_id}")
async def unlink_relay_account(platform: str, platform_user_id: str):
    return await _call({"action": "unlink", "platform": platform, "platform_user_id": platform_user_id})
