"""Drive → vault memory for the desktop: ``/api/drive-memory/*``. Thin REST over :mod:`agent.drive_memory`."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class FolderBody(BaseModel):
    link: str = ""
    id: str = ""
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


@router.get("/api/drive-memory/status")
async def drive_status(profile: Optional[str] = None):
    from agent import drive_memory
    return await _run(profile, drive_memory.status)


@router.post("/api/drive-memory/folders")
async def drive_add_folder(body: FolderBody):
    from agent import drive_memory
    return await _run(body.profile, lambda: drive_memory.add_folder(body.link))


@router.post("/api/drive-memory/folders/remove")
async def drive_remove_folder(body: FolderBody):
    from agent import drive_memory
    return await _run(body.profile, lambda: drive_memory.remove_folder(body.id))


@router.post("/api/drive-memory/sync")
async def drive_sync(body: ProfileBody):
    from agent import drive_memory
    return await _run(body.profile, drive_memory.sync_step)
