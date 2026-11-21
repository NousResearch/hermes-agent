"""``GET /api/egress/requests`` — outbound requests a session's sandbox made through the egress proxy.

Service-gated on ``proxy.enabled``: with egress off the route answers ``enabled: false`` and no
rows, and the dashboard renders nothing. Rows carry method/host/path/status/duration only — see
:mod:`agent.proxy_sources.iron_proxy_requests` for the privacy contract and how rows are attributed.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from agent.proxy_sources.iron_proxy_requests import read_requests
from hermes_cli.web_routers._common import config_scoped_to_thread

router = APIRouter()


@router.get("/api/egress/requests")
async def egress_requests(session: Optional[str] = None, profile: Optional[str] = None,
                          limit: int = Query(200, ge=1, le=1000)):
    return await config_scoped_to_thread(profile, lambda: read_requests(session_id=session, limit=limit))
