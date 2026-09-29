"""Google connection wizard for the desktop: ``/api/google/*``.

Thin REST over :mod:`agent.google_connect`; the skill owns the OAuth flow, this only drives it.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class ClientSecretBody(BaseModel):
    content: str
    profile: Optional[str] = None


class AuthCodeBody(BaseModel):
    code: str
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


@router.get("/api/google/status")
async def google_status(profile: Optional[str] = None):
    from agent import google_connect
    return await _run(profile, google_connect.status)


@router.post("/api/google/client-secret")
async def google_client_secret(body: ClientSecretBody):
    from agent import google_connect
    return await _run(body.profile, lambda: google_connect.save_client_secret(body.content))


@router.post("/api/google/auth-url")
async def google_auth_url(body: ProfileBody):
    from agent import google_connect
    return await _run(body.profile, lambda: {"url": google_connect.auth_url()})


@router.post("/api/google/auth-code")
async def google_auth_code(body: AuthCodeBody):
    from agent import google_connect
    return await _run(body.profile, lambda: google_connect.exchange_code(body.code))


@router.post("/api/google/verify")
async def google_verify(body: ProfileBody):
    from agent import google_connect
    return await _run(body.profile, google_connect.verify)


@router.post("/api/google/revoke")
async def google_revoke(body: ProfileBody):
    from agent import google_connect
    return await _run(body.profile, google_connect.revoke)
