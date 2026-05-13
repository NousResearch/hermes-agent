from __future__ import annotations

import asyncio
import json
import logging
import os

from .. import db
from ..dp_cli_wrapper import run_dp_parse

logger = logging.getLogger(__name__)

# Global semaphore — caps concurrent dp_cli subprocesses to protect the
# DeepParser backend API (dp_cli is an HTTP client; this is not about CPU).
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        from ..config import SEMAPHORE_SIZE
        _semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)
    return _semaphore


async def run_parse_task(
    job_id: str,
    api_key_id: str,
    file_path: str,
) -> None:
    """Detached background task: upload + parse via dp_cli, update job status."""
    from ..config import PARSE_TIMEOUT_SECS, SEMAPHORE_TIMEOUT_SECS

    sem = get_semaphore()
    try:
        # Acquire semaphore with timeout → 503 if all slots busy
        try:
            await asyncio.wait_for(sem.acquire(), timeout=SEMAPHORE_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            async with db.connect() as conn:
                await conn.execute(
                    "UPDATE parse_jobs SET status='PARSE_FAILED', "
                    "error_detail=?, completed_at=datetime('now') WHERE id=?",
                    ("Semaphore timeout: server busy, try again shortly", job_id),
                )
                await conn.commit()
            return

        async with db.connect() as conn:
            await conn.execute(
                "UPDATE parse_jobs SET status='PARSING' WHERE id=?", (job_id,)
            )
            await conn.commit()

        try:
            result = await asyncio.wait_for(
                run_dp_parse(file_path),
                timeout=PARSE_TIMEOUT_SECS,
            )
            result_json = json.dumps(result.model_dump())
            async with db.connect() as conn:
                await conn.execute(
                    "UPDATE parse_jobs SET status='READY', result_json=?, "
                    "dp_file_id=?, dp_folder_id=?, completed_at=datetime('now') WHERE id=?",
                    (result_json, result.file_id, result.folder_id, job_id),
                )
                # Record first parse time for activation funnel
                await conn.execute(
                    "UPDATE api_keys SET first_parse_at=COALESCE(first_parse_at, datetime('now')) "
                    "WHERE id=?",
                    (api_key_id,),
                )
                await conn.commit()
            logger.info("parse READY job=%s file_id=%s", job_id, result.file_id)

        except asyncio.TimeoutError:
            async with db.connect() as conn:
                await conn.execute(
                    "UPDATE parse_jobs SET status='TIMEOUT', "
                    "error_detail=?, completed_at=datetime('now') WHERE id=?",
                    (f"Parse exceeded {PARSE_TIMEOUT_SECS}s timeout", job_id),
                )
                await conn.commit()
            logger.warning("parse TIMEOUT job=%s", job_id)

        except Exception as exc:
            detail = str(exc)[:1000]
            async with db.connect() as conn:
                await conn.execute(
                    "UPDATE parse_jobs SET status='PARSE_FAILED', "
                    "error_detail=?, completed_at=datetime('now') WHERE id=?",
                    (detail, job_id),
                )
                await conn.commit()
            logger.error("parse FAILED job=%s err=%s", job_id, detail)

    finally:
        sem.release()
        # Always delete the uploaded file after parse attempt
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except OSError as exc:
            logger.warning("file delete failed path=%s err=%s", file_path, exc)
