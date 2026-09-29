"""``GET /api/connections/status``: which connections are in place (presence only, no network)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


@router.get("/api/connections/status")
async def connections_status(profile: Optional[str] = None):
    from agent import connection_status
    return await scoped_to_thread(profile, connection_status.snapshot)
