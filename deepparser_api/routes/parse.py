from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

logger = logging.getLogger(__name__)

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile

from .. import db
from ..auth import require_api_key
from ..config import (
    DISK_RESERVE_BYTES,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    SYNC_WAIT_SECS,
    UPLOAD_DIR,
)
from ..models import ParseJobStatus, ParseSubmitResponse, SyncParseResponse
from ..tasks.parse_task import run_parse_task

router = APIRouter()

_DOC_URL = "https://github.com/ysh145/hermes-agent/tree/main/deepparser"


def _check_disk() -> None:
    try:
        usage = shutil.disk_usage(UPLOAD_DIR)
        if usage.free < DISK_RESERVE_BYTES:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "DISK_FULL",
                    "message": "Server storage is critically low. Try again later.",
                    "doc_url": _DOC_URL,
                },
            )
    except FileNotFoundError:
        pass  # UPLOAD_DIR not yet created — will be created on first write


@router.post("/parse", tags=["parse"])
async def submit_parse(
    request: Request,
    file: UploadFile,
    mode: Annotated[str | None, Query()] = None,
    key_row: dict = Depends(require_api_key),
) -> ParseSubmitResponse | SyncParseResponse:
    _check_disk()

    # Validate extension
    ext = Path(file.filename or "").suffix.lstrip(".").lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_FORMAT",
                "message": (
                    f"File type '.{ext}' is not supported. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
                ),
                "doc_url": _DOC_URL,
            },
        )

    # Read and size-check
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": f"File exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.",
                "doc_url": _DOC_URL,
            },
        )

    # Save with UUID filename (security: original filename never reaches subprocess)
    job_id = str(uuid.uuid4())
    stored_name = f"{uuid.uuid4()}.{ext}"
    await asyncio.to_thread(os.makedirs, UPLOAD_DIR, exist_ok=True)
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    await asyncio.to_thread(Path(stored_path).write_bytes, content)

    async with db.connect() as conn:
        await conn.execute(
            """INSERT INTO parse_jobs
               (id, api_key_id, status, filename_original, filename_stored)
               VALUES (?, ?, 'QUEUED', ?, ?)""",
            (job_id, key_row["id"], file.filename, stored_name),
        )
        await conn.commit()

    task = asyncio.create_task(
        run_parse_task(job_id, key_row["id"], stored_path)
    )

    def _log_task_error(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception() is not None:
            logger.error("parse task failed job=%s", job_id, exc_info=t.exception())

    task.add_done_callback(_log_task_error)

    if mode == "sync" and len(content) < 5 * 1024 * 1024:
        # Wait briefly; if done return inline, otherwise fall through to QUEUED
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=SYNC_WAIT_SECS)
            # _log_task_error already attached above handles post-timeout exceptions.
            async with db.connect() as conn:
                conn.row_factory = aiosqlite.Row
                row = await db.fetchone(conn, "SELECT * FROM parse_jobs WHERE id=?", (job_id,))
            if row and row["status"] == "READY":
                result = json.loads(row["result_json"]) if row["result_json"] else None
                return SyncParseResponse(job_id=job_id, status="READY", result=result)
        except asyncio.TimeoutError:
            pass  # fall through: return QUEUED below

    return ParseSubmitResponse(job_id=job_id, status="QUEUED")


@router.get("/parse/{job_id}", response_model=ParseJobStatus, tags=["parse"])
async def get_parse_status(
    job_id: str,
    key_row: dict = Depends(require_api_key),
) -> ParseJobStatus:
    async with db.connect() as conn:
        conn.row_factory = aiosqlite.Row
        row = await db.fetchone(
            conn, "SELECT * FROM parse_jobs WHERE id=? AND api_key_id=?",
            (job_id, key_row["id"]),
        )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NOT_FOUND",
                "message": "Job not found.",
                "doc_url": _DOC_URL,
            },
        )

    result = json.loads(row["result_json"]) if row["result_json"] else None
    return ParseJobStatus(
        job_id=row["id"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        result=result,
        error_detail=row["error_detail"],
    )
