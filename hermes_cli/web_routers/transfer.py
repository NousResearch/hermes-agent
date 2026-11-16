"""Hermes Cloud transfer: carry a backup from one dashboard to another over JSON, then restore it.

A client with JSON-only transport (the Desktop's OAuth connections, the Webapp's relayed ones)
reads a backup off the source in base64 slices and appends each slice to an upload on the
receiving agent, holding one slice at a time. The receiving side restores the finished upload
with ``hermes import --keep-host`` (``hermes_cli.backup_keep_host``).

Every request body is bounded before it is buffered (:func:`_bounded_json`): FastAPI would otherwise
read a JSON body of any size into memory before a model sees it.
"""

import asyncio
import base64
import binascii
import contextlib
import json
import os
import secrets
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.web_routers._common import destructive_profile

router = APIRouter()

CHUNK_MAX = 8 * 1024 * 1024
# A whole Hermes home (state.db, media) can be many GB; the managed-file upload cap is 100 MB.
TRANSFER_MAX_BYTES = 32 * 1024 ** 3
# One chunk in base64 plus the JSON around it.
_BODY_MAX = CHUNK_MAX * 4 // 3 + 4096
_UPLOAD_PREFIX = "transfer-upload-"
_UPLOAD_ID_LEN = 24
_UPLOAD_STALE_SECONDS = 3600


def _backups(profile: Optional[str]) -> Path:
    from hermes_cli.web_routers.ops import _dashboard_backup_dir
    return _dashboard_backup_dir(profile)


async def _bounded_json(request: Request) -> dict[str, Any]:
    """The request's JSON object, refused with 413 past :data:`_BODY_MAX` before it is buffered."""
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Content-Length")
    if declared > _BODY_MAX:
        raise HTTPException(status_code=413, detail="Chunk is too large")
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > _BODY_MAX:
            raise HTTPException(status_code=413, detail="Chunk is too large")
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Body is not JSON")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Body is not a JSON object")
    return data


def _upload_path(profile: Optional[str], upload_id: str) -> Path:
    if len(upload_id) != _UPLOAD_ID_LEN or not all(c in "0123456789abcdef" for c in upload_id):
        raise HTTPException(status_code=404, detail="Unknown upload")
    return _backups(profile) / f"{_UPLOAD_PREFIX}{upload_id}.zip"


def _read_chunk(target: Path, offset: int, length: int) -> dict[str, Any]:
    size = target.stat().st_size
    with target.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    return {"data": base64.b64encode(data).decode("ascii"), "offset": offset, "size": size,
            "done": offset + len(data) >= size}


@router.get("/api/ops/backup/chunk")
async def read_dashboard_backup_chunk(archive: str, offset: int = 0, length: int = CHUNK_MAX,
                                      profile: Optional[str] = None):
    """One base64 slice of a backup this dashboard wrote."""
    from hermes_cli.web_routers.ops import _dashboard_backup_path
    if offset < 0 or not 0 < length <= CHUNK_MAX:
        raise HTTPException(status_code=400, detail="offset/length out of range")
    target = _dashboard_backup_path(archive, profile)
    return await asyncio.to_thread(_read_chunk, target, offset, length)


# One writer per upload: a late request for a chunk the client already retried must not interleave.
_UPLOAD_LOCKS: dict[str, threading.Lock] = {}
_UPLOAD_LOCKS_GUARD = threading.Lock()


def _upload_lock(path: Path) -> threading.Lock:
    with _UPLOAD_LOCKS_GUARD:
        return _UPLOAD_LOCKS.setdefault(path.name, threading.Lock())


def _append(path: Path, offset: int, data: bytes, create: bool) -> int:
    """Write ``data`` at ``offset``; return the size. Nothing is ever truncated: a retried chunk
    (offset below the size) must match the bytes already there, so a late duplicate cannot cut
    off the chunks written after it."""
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    with _upload_lock(path), path.open("r+b") as handle:
        received = handle.seek(0, os.SEEK_END)
        if offset > received:
            raise HTTPException(status_code=409, detail={"code": "offset_gap", "received": received,
                                                         "message": f"offset_gap: received {received}"})
        if offset + len(data) > TRANSFER_MAX_BYTES:
            raise HTTPException(status_code=413, detail="The backup is too large to transfer")
        handle.seek(offset)
        held = handle.read(len(data))
        if held != data[:len(held)]:
            raise HTTPException(status_code=409, detail="The chunk differs from the bytes already received")
        handle.write(data[len(held):])
        return max(received, handle.tell())


@router.post("/api/ops/transfer/chunk")
async def append_transfer_chunk(request: Request, profile: Optional[str] = None):
    """Append one base64 chunk to a transfer upload; ``upload_id`` empty at offset 0 starts one."""
    profile = destructive_profile(profile, "POST /api/ops/transfer/chunk")
    body = await _bounded_json(request)
    offset, upload_id = body.get("offset"), str(body.get("upload_id") or "")
    if not isinstance(offset, int) or offset < 0:
        raise HTTPException(status_code=400, detail="offset must be a non-negative integer")
    try:
        data = base64.b64decode(str(body.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="data is not base64")
    if len(data) > CHUNK_MAX:
        raise HTTPException(status_code=413, detail="Chunk is too large")
    create = not upload_id
    if create:
        if offset:
            raise HTTPException(status_code=400, detail="A new upload starts at offset 0")
        upload_id = secrets.token_hex(_UPLOAD_ID_LEN // 2)
    path = _upload_path(profile, upload_id)
    if not create and not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown upload")
    received = await asyncio.to_thread(_append, path, offset, data, create)
    if create:  # a canceled or failed transfer never reaches restore, which also prunes
        await asyncio.to_thread(_prune_uploads, path)
    return {"upload_id": upload_id, "received": received}


@router.post("/api/ops/transfer/restore")
async def restore_transfer(request: Request, profile: Optional[str] = None):
    """Restore a finished transfer upload with ``hermes import --keep-host --messaging-off``: the
    agent's bots arrive switched off, so the computer it came from keeps answering."""
    from hermes_cli.web_routers.ops import _spawn_import
    profile = destructive_profile(profile, "POST /api/ops/transfer/restore")
    body = await _bounded_json(request)
    path, size = _upload_path(profile, str(body.get("upload_id") or "")), body.get("size")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown upload")
    received = path.stat().st_size
    if not isinstance(size, int) or received != size:
        raise HTTPException(status_code=409, detail=f"The upload holds {received} bytes, not {size}")
    if not await asyncio.to_thread(zipfile.is_zipfile, path):
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The upload is not a valid zip file")
    started = _spawn_import(str(path), True, profile, keep_host=True, messaging_off=True)
    await asyncio.to_thread(_prune_uploads, path)
    return {**started, "archive": str(path), "keep_host": True}


def _prune_uploads(current: Path) -> None:
    """Drop earlier uploads (restored or abandoned) untouched for an hour; a live one is kept."""
    for path in current.parent.glob(f"{_UPLOAD_PREFIX}*.zip"):
        with contextlib.suppress(OSError):
            if path != current and time.time() - path.stat().st_mtime > _UPLOAD_STALE_SECONDS:
                path.unlink()
