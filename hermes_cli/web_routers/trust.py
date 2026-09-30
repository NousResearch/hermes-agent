"""Integration trust levels and the record of what Czesiek did: ``/api/trust/*``.

Thin REST over :mod:`agent.integration_trust`; the approval gate enforces the levels, this only edits them.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class LevelBody(BaseModel):
    integration: str
    level: str
    profile: Optional[str] = None


def _state():
    from agent import integration_trust as trust
    return {"levels": trust.all_levels(), "log": trust.read_log(12)}


@router.get("/api/trust")
async def get_trust(profile: Optional[str] = None, limit: int = Query(default=12, ge=1, le=100)):
    from agent import integration_trust as trust

    def read():
        return {"levels": trust.all_levels(), "log": trust.read_log(limit)}
    return await scoped_to_thread(profile, read)


@router.post("/api/trust/level")
async def set_trust_level(body: LevelBody):
    from agent import integration_trust as trust
    try:
        await scoped_to_thread(body.profile, lambda: trust.set_level(body.integration, body.level))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return await scoped_to_thread(body.profile, _state)
