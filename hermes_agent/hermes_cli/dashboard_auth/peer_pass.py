"""Bearer check for ``/api/peer/...``: only a pass minted for this agent opens it.

The route owns its prefix. A provider with ``supports_peer_pass`` verifies the bearer; a pass
attaches ``request.state.peer_principal`` (also as ``token_principal``, with
``token_authenticated``, so the cookie gates stand aside) and the route then checks the pass's
permissions per path (``hermes_agent.hermes_cli.web_routers.peer``).
Cookies, the session token and native tickets never open it, and nothing else reaches the shared
token-auth seam's providers: a pass never authorises their routes, nor they this one. No pass
provider registered (an agent not signed in through a provider that mints passes) is a 404.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from hermes_agent.hermes_cli.dashboard_auth import TokenPrincipal, list_providers
from hermes_agent.hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
from hermes_agent.hermes_cli.dashboard_auth.base import ProviderError
from hermes_agent.hermes_cli.dashboard_auth.request_utils import client_ip, extract_bearer, unreachable_response

_log = logging.getLogger(__name__)

PEER_PREFIX = "/api/peer/"
#: The pass contract this agent implements, reported as ``peer_access`` in ``/api/status`` while a
#: provider that mints passes is registered. The issuer mints passes only for agents reporting it,
#: because an agent without this route reads a pass as a dashboard sign-in. Raise it when what a
#: pass may carry changes, so agents on the older contract stop getting the new passes.
PEER_ACCESS_CONTRACT = 1


def is_peer_path(path: str) -> bool:
    return path.startswith(PEER_PREFIX)


def peer_access_contract() -> int:
    """:data:`PEER_ACCESS_CONTRACT` when a registered provider takes peer passes, else 0 (the route
    answers 404 then)."""
    return PEER_ACCESS_CONTRACT if any(getattr(p, "supports_peer_pass", False) for p in list_providers()) else 0


async def _verify(providers: list, token: str) -> tuple[Optional[TokenPrincipal], Optional[str]]:
    """``(principal, None)`` for a pass; ``(None, provider name)`` when none accepted it and one
    was unreachable (503); ``(None, None)`` otherwise (401)."""
    unreachable: Optional[str] = None
    for provider in providers:
        try:
            principal = await run_in_threadpool(provider.verify_peer_pass, token=token)
        except ProviderError as exc:
            _log.warning("dashboard-auth: peer pass provider %r unreachable: %s", provider.name, exc)
            unreachable = unreachable or provider.name
            continue
        if principal is not None:
            return principal, None
    return None, unreachable


async def peer_pass_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    providers = [p for p in list_providers() if getattr(p, "supports_peer_pass", False)]
    if not providers:
        return JSONResponse({"error": "peer_access_unavailable",
                             "detail": "This agent does not accept peer passes."}, status_code=404)
    token = extract_bearer(request)
    principal, unreachable = await _verify(providers, token) if token else (None, None)
    if principal is not None:
        request.state.token_principal = request.state.peer_principal = principal
        request.state.token_authenticated = True
        return await call_next(request)
    if unreachable:
        audit_log(AuditEvent.TOKEN_AUTH_FAILURE, provider=unreachable, reason="provider_unreachable",
                  path=request.url.path, ip=client_ip(request))
        return unreachable_response(unreachable)
    audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason="no_peer_pass" if not token else "peer_pass_rejected",
              path=request.url.path, ip=client_ip(request))
    return JSONResponse({"error": "unauthenticated", "detail": "A peer pass for this agent is required."},
                        status_code=401)
