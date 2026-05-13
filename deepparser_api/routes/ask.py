from __future__ import annotations

import json
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import db
from ..auth import require_api_key
from ..dp_cli_wrapper import run_dp_ask
from ..models import AskRequest, AskResponse

router = APIRouter()

_DOC_URL = "https://github.com/ysh145/hermes-agent/tree/main/deepparser"


@router.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(
    body: AskRequest,
    mode: Annotated[str | None, Query()] = None,
    key_row: dict = Depends(require_api_key),
) -> AskResponse:
    async with db.connect() as conn:
        conn.row_factory = aiosqlite.Row
        row = await db.fetchone(
            conn, "SELECT * FROM parse_jobs WHERE id=? AND api_key_id=?",
            (body.job_id, key_row["id"]),
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

    if row["status"] != "READY":
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NOT_READY",
                "message": f"Job is not READY yet (current status: {row['status']}). "
                           "Poll GET /parse/{job_id} until status is READY.",
                "detail": row["status"],
                "doc_url": _DOC_URL,
            },
        )

    dp_file_id = row["dp_file_id"]
    if not dp_file_id:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": "Parse result is missing file_id.",
                "doc_url": _DOC_URL,
            },
        )

    result = await run_dp_ask(dp_file_id, body.question)
    return AskResponse(
        job_id=body.job_id,
        answer=result.answer,
        citations=result.citations,
    )
