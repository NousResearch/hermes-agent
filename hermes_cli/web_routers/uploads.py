"""Authenticated browser-to-host attachment staging for Hermes Webapp."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, File, HTTPException, UploadFile

from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES
from hermes_cli.staged_uploads import (
    publish_staged_upload,
    resolve_upload_home,
    staged_upload_descriptor,
)


router = APIRouter()
_MAX_UPLOAD_BYTES = WEBAPP_ATTACHMENT_MAX_BYTES


@router.post("/api/chat/file-upload")
async def upload_chat_file(
    file: UploadFile = File(...),
    profile: str | None = None,
):
    """Stage one user-selected browser file under the active Hermes profile.

    The client never chooses a server path. A unique 0600 file under
    ``$HERMES_HOME/uploads`` is returned for the existing ``file.attach`` flow.
    """
    profile_home = await asyncio.to_thread(resolve_upload_home, profile)
    try:
        # FastAPI has already spooled the complete upload outside the profile.
        # Check the actual bytes, then publish that spool into the resolved home.
        total = await asyncio.to_thread(file.file.seek, 0, os.SEEK_END)
        if total > _MAX_UPLOAD_BYTES:
            cap_mib = _MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File is too large; cap is {cap_mib} MiB",
            )
        target = await asyncio.to_thread(
            publish_staged_upload,
            file.file,
            profile_home,
            file.filename or "attachment",
        )
    finally:
        await file.close()

    result = {"ok": True, "path": str(target), "size": total}
    if descriptor := await asyncio.to_thread(staged_upload_descriptor, target, profile_home):
        result["staged_upload"] = descriptor
    return result
