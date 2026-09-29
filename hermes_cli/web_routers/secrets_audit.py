"""Credential storage audit for the desktop: ``/api/secrets/*``. Presence-only, never returns values."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class ProfileBody(BaseModel):
    profile: Optional[str] = None


@router.get("/api/secrets/audit")
async def secrets_audit(profile: Optional[str] = None):
    from agent import secret_audit
    return await scoped_to_thread(profile, secret_audit.audit)


@router.post("/api/secrets/fix")
async def secrets_fix(body: ProfileBody):
    from agent import secret_audit
    return await scoped_to_thread(body.profile, secret_audit.fix_permissions)
