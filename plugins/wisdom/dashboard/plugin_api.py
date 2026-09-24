"""Wisdom REST for the Desktop page (mounted at /api/plugins/wisdom/).

Two-step consent: ``/plan`` returns exactly what the user must see; ``/install`` applies only
when the echoed ``content_hash`` matches the plan the server computes now. Same ``Wisdom``
service as the CLI and tools; only the confirmation surface differs. Sharing is the same shape:
``/share/prepare`` packages locally (no upload) and ``/share`` echoes the reviewed hash; the
Gateway's review then publishes only on a passing security AND professionalism verdict, anything
else withdraws the draft and returns the verdict for the user to read.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plugins.wisdom import candidates, notices, state, updates
from plugins.wisdom.client import WisdomAuthError, WisdomError, entitlement
from plugins.wisdom.package import PackageError

router = APIRouter()
_HASH = r"^sha256:[0-9a-f]{64}$"


class Target(BaseModel):
    skill_id: str
    version: int | None = None


class Apply(BaseModel):
    skill_id: str
    version: int
    content_hash: str = Field(pattern=_HASH)


class Keep(BaseModel):
    skill_id: str
    version: int


class Candidate(BaseModel):
    skill: str


class SharePrepare(BaseModel):
    skill_name: str
    description: str | None = None


class Share(SharePrepare):
    content_hash: str = Field(pattern=_HASH)


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
        return {"entitled": False, "skills": [], "status": notices.summary(state()), "candidates": []}
    return _run(lambda svc: {"entitled": True, "skills": svc.browse(),
                             "status": dict(svc.status(), updates=updates.pending(svc)),
                             "candidates": candidates.qualify(state())})


@router.post("/plan")
def plan(body: Target):
    return _run(lambda svc: svc.plan(body.skill_id, version=body.version))


@router.post("/install")
def install(body: Apply):
    # The dialog showed this exact hash (schema-validated full sha256, so an empty echo cannot match);
    # a republished version between plan and click fails closed.
    return _run(lambda svc: svc.install(body.skill_id, version=body.version,
                                        confirm=lambda _t, detail: f"content_hash: {body.content_hash}\n" in detail + "\n"))


@router.post("/update/keep")
def keep(body: Keep):
    updates.defer(state(), body.skill_id, body.version)
    return {"kept": body.skill_id, "version": body.version}


@router.post("/uninstall")
def uninstall(body: Target):
    return _run(lambda svc: svc.uninstall(body.skill_id, confirm=lambda *_: True))


@router.post("/notices/dismiss")
def dismiss(body: Target | None = None):
    notices.dismiss(state(), body.skill_id if body else None)
    return notices.summary(state())


@router.post("/candidates/not-now")
def not_now(body: Candidate):
    return {"skill": body.skill, "until": candidates.defer(state(), body.skill)}


def _description(body: SharePrepare) -> str:
    return body.description or candidates.default_description(body.skill_name)


@router.post("/share/prepare")
def share_prepare(body: SharePrepare):
    """Package locally and describe what would be uploaded; nothing leaves the machine."""
    import tempfile
    from pathlib import Path

    from hermes_constants import get_skills_dir
    from plugins.wisdom.package import prepare, slug_for
    from tools.skill_usage import _find_skill_dir

    def go(svc):
        source = _find_skill_dir(body.skill_name)
        if source is None:
            raise WisdomError(f"local skill {body.skill_name!r} not found")
        if source.resolve().is_relative_to((get_skills_dir() / "_wisdom").resolve()):
            raise WisdomError("a Wisdom-managed installation cannot be re-shared; fork it first")
        description = _description(body)
        with tempfile.TemporaryDirectory(prefix="wisdom-share-") as staging:
            prepared = prepare(source, description=description, owner=svc.client.owner,
                               installation_id=svc.installation_id(), staging=Path(staging))
            return {"skill_name": body.skill_name, "slug": slug_for(body.skill_name), "description": description,
                    "content_hash": prepared.content_hash,
                    "files": [{"path": p, "bytes": len(b)} for p, _, b in prepared.files]}
    return _run(go)


@router.post("/share")
def share(body: Share):
    """Upload + publish the package the user reviewed. Confirm 1 is the echoed package hash; confirm 2
    (the Gateway's verdicts) passes only when both checks read ``pass`` — otherwise the draft is
    withdrawn and the verdict is returned as the error detail."""
    def confirm(title: str, detail: str) -> bool:
        if title.startswith("Share "):
            return f"content_hash: {body.content_hash}\n" in detail + "\n"
        lines = dict(line.split(": ", 1) for line in detail.splitlines() if ": " in line)
        return all(lines.get(k, "").startswith("pass") for k in ("security", "professionalism"))
    return _run(lambda svc: svc.share(body.skill_name, description=_description(body), confirm=confirm))
