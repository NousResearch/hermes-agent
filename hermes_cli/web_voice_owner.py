"""Forward GPT-Live negotiation to the authenticated profile's serving gateway."""

import asyncio
from dataclasses import replace

from fastapi import HTTPException

from gateway.owner_identity import host_owner_subject
from gateway.session_voice_capture import VOICE_NOT_NEGOTIATED
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, connect_gateway
from hermes_cli.gateway_runtime import discover_gateway_endpoint
from hermes_cli.gateway_runtime_discovery import DiscoveryError


async def start_via_voice_owner(home, actor, session_id, sdp, history):
    """A missing/newer-incompatible owner can offer an explicitly unsaved call, never a second writer."""
    discovery = await asyncio.to_thread(discover_gateway_endpoint, home, timeout=1.5)
    endpoint = discovery.endpoint
    if endpoint is None:
        if discovery.state == 'absent' and not discovery.reason_code:
            return None
        status = 403 if discovery.reason_code in {'authorization', 'profile_mismatch'} else 503
        raise HTTPException(status_code=status, detail='owner_unreachable')
    if not endpoint.control_home:
        endpoint = replace(endpoint, control_home=str(home.expanduser().resolve()))
    subject = None if actor.subject == host_owner_subject() else actor.subject
    try:
        async with connect_gateway(endpoint=endpoint, subject=subject) as client:
            return await client.rpc('voice.live.start', session_id=session_id, sdp=sdp, history=history, _timeout=45)
    except GatewayRPCError as exc:
        reason = str(exc)
        # Authorization can also fail after negotiation. Only the owner's
        # explicit pre-negotiation refusal can authorize an unsaved fallback.
        if exc.unknown_method or (reason == 'not_found' and exc.data.get('detail') == VOICE_NOT_NEGOTIATED):
            return None
        from hermes_cli.web_server_sessions import _MUTATION_REFUSAL_STATUS
        status = {'voice_provider_unavailable': 502, 'voice_capture_unavailable': 503}.get(
            reason, _MUTATION_REFUSAL_STATUS.get(reason, 409),
        )
        raise HTTPException(status_code=status, detail=reason) from None
    except (GatewayClientError, DiscoveryError, OSError, TimeoutError):
        # A timeout may mean the vendor session exists. Do not create a second paid call.
        raise HTTPException(status_code=503, detail='owner_unreachable') from None
