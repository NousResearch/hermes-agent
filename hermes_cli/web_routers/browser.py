"""Browser modes for the desktop: ``/api/browser/*``. Thin REST over :mod:`agent.browser_modes`."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class ModeBody(BaseModel):
    mode: str
    profile: Optional[str] = None


class ProfileBody(BaseModel):
    profile: Optional[str] = None


async def _run(profile: Optional[str], fn):
    try:
        return await scoped_to_thread(profile, fn)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/browser/status")
async def browser_status(profile: Optional[str] = None):
    from agent import browser_modes
    return await _run(profile, browser_modes.status)


@router.post("/api/browser/mode")
async def browser_mode(body: ModeBody):
    from agent import browser_modes
    return await _run(body.profile, lambda: browser_modes.set_mode(body.mode))


@router.post("/api/browser/open-signin")
async def browser_open_signin(body: ProfileBody):
    from agent import browser_modes
    return await _run(body.profile, browser_modes.open_sign_in)


@router.post("/api/browser/import")
async def browser_import(body: ProfileBody):
    from agent import browser_modes
    return await _run(body.profile, browser_modes.import_now)


@router.post("/api/browser/clear-copy")
async def browser_clear_copy(body: ProfileBody):
    from agent import browser_modes
    return await _run(body.profile, browser_modes.clear_copy)


@router.post("/api/browser/clear-own")
async def browser_clear_own(body: ProfileBody):
    from agent import browser_modes
    return await _run(body.profile, browser_modes.clear_own_profile)
