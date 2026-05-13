from __future__ import annotations

import asyncio
import os

import aiosqlite
from fastapi import APIRouter, HTTPException, Request

from .. import db
from ..config import ADMIN_PASSWORD, UPLOAD_DIR
from ..models import AdminStatsResponse

router = APIRouter()


@router.get("/admin/stats", response_model=AdminStatsResponse, tags=["admin"])
async def admin_stats(request: Request) -> AdminStatsResponse:
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=500, detail="ADMIN_PASSWORD not configured")

    provided = request.headers.get("X-Admin-Password", "")
    import hmac
    if not provided or not hmac.compare_digest(provided, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Invalid admin password")

    async with db.connect() as conn:
        conn.row_factory = aiosqlite.Row

        (keys_registered,) = (await db.fetchone(
            conn, "SELECT COUNT(*) FROM api_keys WHERE revoked=0"
        ) or (0,))

        (keys_activated,) = (await db.fetchone(
            conn, "SELECT COUNT(*) FROM api_keys WHERE revoked=0 AND first_parse_at IS NOT NULL"
        ) or (0,))

        (requests_today,) = (await db.fetchone(
            conn, "SELECT COUNT(*) FROM request_log WHERE ts >= datetime('now', '-1 day')"
        ) or (0,))

        (parse_jobs,) = (await db.fetchone(
            conn, "SELECT COUNT(*) FROM parse_jobs"
        ) or (0,))

        (errors_today,) = (await db.fetchone(
            conn, "SELECT COUNT(*) FROM request_log WHERE status_code >= 500 AND ts >= datetime('now', '-1 day')"
        ) or (0,))

    # Disk usage of upload dir (run synchronous os.walk in a thread to avoid blocking event loop)
    def _disk_usage() -> int:
        total = 0
        if os.path.isdir(UPLOAD_DIR):
            for dirpath, _, filenames in os.walk(UPLOAD_DIR):
                for fname in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, fname))
                    except OSError:
                        pass
        return total

    storage_bytes = await asyncio.to_thread(_disk_usage)

    activation_rate = (
        round(keys_activated / keys_registered, 4) if keys_registered else 0.0
    )

    return AdminStatsResponse(
        keys_registered=keys_registered,
        keys_activated=keys_activated,
        activation_rate=activation_rate,
        requests_today=requests_today,
        parse_jobs=parse_jobs,
        storage_bytes=storage_bytes,
        errors_today=errors_today,
    )
