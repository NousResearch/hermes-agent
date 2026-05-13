from __future__ import annotations

import secrets
import uuid

import aiosqlite
from fastapi import APIRouter, Request

from .. import db
from ..auth import check_keys_rate_limit
from ..models import KeyRegistrationRequest, KeyRegistrationResponse

router = APIRouter()

_DOC_URL = "https://github.com/ysh145/hermes-agent/tree/main/deepparser"


@router.post("/keys", response_model=KeyRegistrationResponse, tags=["auth"])
async def register_key(
    request: Request,
    body: KeyRegistrationRequest,
) -> KeyRegistrationResponse:
    check_keys_rate_limit(request)

    key_id = str(uuid.uuid4())
    api_key = "dp_live_" + secrets.token_hex(16)

    async with db.connect() as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(
            "INSERT INTO api_keys (id, key, email, intended_use) VALUES (?, ?, ?, ?)",
            (key_id, api_key, body.email, body.intended_use),
        )
        await conn.commit()
        row = await db.fetchone(conn, "SELECT created_at FROM api_keys WHERE id=?", (key_id,))

    created_at = row["created_at"] if row else ""
    return KeyRegistrationResponse(api_key=api_key, created_at=created_at)
