"""Wisdom REST for the Desktop page (mounted at /api/plugins/wisdom/).

Two-step consent: ``/plan`` returns exactly what the user must see; ``/install`` applies only
when the echoed ``content_hash`` matches the plan the server computes now. Same ``Wisdom``
service as the CLI and tools; only the confirmation surface differs.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from plugins.wisdom import notices, state
from plugins.wisdom.client import WisdomAuthError, WisdomError, entitlement
from plugins.wisdom.package import PackageError

router = APIRouter()


class Target(BaseModel):
    skill_id: str
    version: int | None = None


class Apply(BaseModel):
    skill_id: str
    version: int
    content_hash: str


def _run(fn):
    from plugins.wisdom.service import Wisdom
    try:
        return fn(Wisdom(state()))
    except WisdomAuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    except (WisdomError, PackageError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/overview")
def overview():
    if not entitlement():
        return {"entitled": False, "skills": [], "status": notices.summary(state())}
    return _run(lambda svc: {"entitled": True, "skills": svc.browse(), "status": svc.status()})


@router.post("/plan")
def plan(body: Target):
    return _run(lambda svc: svc.plan(body.skill_id, version=body.version))


@router.post("/install")
def install(body: Apply):
    # The dialog showed this exact hash; a republished version between plan and click fails closed.
    return _run(lambda svc: svc.install(body.skill_id, version=body.version,
                                        confirm=lambda _t, detail: f"content_hash: {body.content_hash}" in detail))


@router.post("/uninstall")
def uninstall(body: Target):
    return _run(lambda svc: svc.uninstall(body.skill_id, confirm=lambda *_: True))


@router.post("/notices/dismiss")
def dismiss(body: Target | None = None):
    notices.dismiss(state(), body.skill_id if body else None)
    return notices.summary(state())
