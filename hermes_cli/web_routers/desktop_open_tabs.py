"""``/api/desktop/open-tabs`` — the tab strip that follows the backend.

Any Desktop client of this profile home reads and writes the same ordered list
of open stored sessions. The client must GET before it PUTs: a fresh device
with an empty local strip must not wipe the other device. ``base_revision``
makes a stale PUT a 409 carrying the current document instead of a silent
overwrite.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from hermes_cli.desktop_open_tabs import RevisionConflict, load, save
from hermes_cli.web_routers._common import destructive_profile, http_failure, scoped_to_thread

router = APIRouter()


class OpenTabsPut(BaseModel):
    tiles: list[Any] = Field(default_factory=list)
    base_revision: int = Field(ge=0)


def _home_document():
    from hermes_constants import get_hermes_home

    return load(get_hermes_home())


def _write(body: OpenTabsPut):
    from hermes_constants import get_hermes_home

    try:
        return save(get_hermes_home(), body.tiles, body.base_revision)
    except RevisionConflict as conflict:
        raise HTTPException(status_code=409, detail=conflict.current) from conflict


@router.get("/api/desktop/open-tabs")
async def get_open_tabs(profile: Optional[str] = None):
    """Open tab strip for this profile home. Missing file is revision 0, no tiles."""
    with http_failure("GET /api/desktop/open-tabs failed", 500, "Failed to read open tabs"):
        return await scoped_to_thread(profile, _home_document)


@router.put("/api/desktop/open-tabs")
async def put_open_tabs(body: OpenTabsPut, profile: Optional[str] = None):
    """Replace the open tab strip. 409 + current document when ``base_revision`` is stale."""
    profile = destructive_profile(profile, "PUT /api/desktop/open-tabs")
    with http_failure("PUT /api/desktop/open-tabs failed", 500, "Failed to save open tabs"):
        return await scoped_to_thread(profile, lambda: _write(body))
