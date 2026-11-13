"""``/api/move/*``: build a move bundle here, or receive one from another computer.

Source side (the user's own machine, called by Desktop's local backend):
``POST /api/move/bundle``          build (or dry-run inventory) a bundle of one profile
``GET  /api/move/bundle/{name}``   read the built bundle back in base64 chunks

Target side (a Hermes Cloud agent, called by Desktop over its OAuth connection):
``POST /api/move/uploads``                     begin / resume an upload by digest
``POST /api/move/uploads/{id}/chunk``          append one chunk
``POST /api/move/uploads/{id}/finish``         verify + preview
``POST /api/move/uploads/{id}/apply``          merge
``POST /api/move/uploads/{id}/undo``           restore the snapshot
``GET  /api/move/uploads`` / ``/{id}``         status, for resume after a dropped connection
``DELETE /api/move/uploads/{id}``              discard an unapplied upload

Every route is JSON. Chunks travel as base64 inside JSON so the OAuth cookie path (which carries
JSON only) and the bearer path behave the same.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from hermes_cli.web_routers._common import destructive_profile

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

_BUNDLE_NAME_RE = re.compile(r"^move-[a-f0-9]{16}\.zip$")
_READ_CHUNK_MAX = 4 * 1024 * 1024


class MoveBundleRequest(BaseModel):
    include_history: bool = True
    history_limit: int = Field(200, ge=0, le=500)
    dry_run: bool = False


class MoveBegin(BaseModel):
    size: int
    sha256: str


class MoveChunk(BaseModel):
    offset: int
    data: str


class MoveApply(BaseModel):
    include: Optional[list[str]] = None


def _home(profile: Optional[str]) -> tuple[str, Path]:
    from hermes_cli.web_server_cron import _cron_profile_home
    return _cron_profile_home(profile)


def _refusal(exc: Exception) -> HTTPException:
    from hermes_cli.move_receive import MoveError
    if isinstance(exc, MoveError):
        return HTTPException(status_code=exc.status, detail={"code": exc.code, "message": str(exc)})
    return HTTPException(status_code=500, detail={"code": "move_failed", "message": str(exc)[:300]})


async def _run(fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:  # health: allow BLE001 -- every failure answers as a structured refusal
        from hermes_cli.move_receive import MoveError
        if not isinstance(exc, MoveError):
            _log.exception("move route failed")
        raise _refusal(exc) from exc


def _outbox(home: Path) -> Path:
    from hermes_constants import mkdir_under_hermes_home
    path = home / "moves" / "outbox"
    mkdir_under_hermes_home(path)
    return path


# --------------------------------------------------------------------------- source side


@router.post("/api/move/bundle")
async def build_move_bundle(body: MoveBundleRequest, profile: Optional[str] = None):
    profile = destructive_profile(profile, "POST /api/move/bundle")
    name, home = _home(profile)

    def _build():
        from hermes_cli.move_bundle import build_bundle
        outbox = _outbox(home)
        for old in outbox.glob("move-*.zip"):
            old.unlink(missing_ok=True)
        out = outbox / f"move-{uuid.uuid4().hex[:16]}.zip"
        report = build_bundle(home, name, out, include_history=body.include_history,
                              history_limit=body.history_limit, dry_run=body.dry_run)
        data = report.as_dict()
        data["path"] = None
        data["name"] = None if body.dry_run else out.name
        return data

    return await _run(_build)


@router.get("/api/move/bundle/{bundle_name}")
async def read_move_bundle(bundle_name: str, offset: int = 0, length: int = _READ_CHUNK_MAX,
                           profile: Optional[str] = None):
    if not _BUNDLE_NAME_RE.match(bundle_name):
        raise HTTPException(status_code=404, detail="Bundle not found")
    _name, home = _home(profile)
    path = home / "moves" / "outbox" / bundle_name

    def _read():
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Bundle not found")
        size = path.stat().st_size
        start = max(0, offset)
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read(max(1, min(length, _READ_CHUNK_MAX)))
        return {"offset": start, "size": size, "data": base64.b64encode(data).decode("ascii"),
                "done": start + len(data) >= size}

    return await _run(_read)


@router.delete("/api/move/bundle/{bundle_name}")
async def delete_move_bundle(bundle_name: str, profile: Optional[str] = None):
    if not _BUNDLE_NAME_RE.match(bundle_name):
        raise HTTPException(status_code=404, detail="Bundle not found")
    _name, home = _home(profile)
    (home / "moves" / "outbox" / bundle_name).unlink(missing_ok=True)
    return {"ok": True}


# --------------------------------------------------------------------------- target side


@router.get("/api/move/uploads")
async def list_move_uploads(profile: Optional[str] = None):
    from hermes_cli import move_receive
    _name, home = _home(profile)
    return {"moves": await _run(move_receive.list_moves, home),
            "limits": {"bundle_bytes": move_receive.MAX_BUNDLE_BYTES,
                       "chunk_bytes": move_receive.MAX_CHUNK_BYTES}}


@router.get("/api/move/uploads/{move_id}")
async def get_move_upload(move_id: str, profile: Optional[str] = None):
    from hermes_cli import move_receive
    _name, home = _home(profile)
    return await _run(move_receive.get_move, home, move_id)


@router.post("/api/move/uploads")
async def begin_move_upload(body: MoveBegin, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "POST /api/move/uploads")
    _name, home = _home(profile)
    return await _run(move_receive.begin, home, size=body.size, sha256=body.sha256)


@router.post("/api/move/uploads/{move_id}/chunk")
async def move_upload_chunk(move_id: str, body: MoveChunk, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "POST /api/move/uploads/{id}/chunk")
    _name, home = _home(profile)
    return await _run(move_receive.chunk, home, move_id, offset=body.offset, data_b64=body.data)


@router.post("/api/move/uploads/{move_id}/finish")
async def finish_move_upload(move_id: str, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "POST /api/move/uploads/{id}/finish")
    _name, home = _home(profile)
    return await _run(move_receive.finish, home, move_id)


@router.post("/api/move/uploads/{move_id}/apply")
async def apply_move_upload(move_id: str, body: MoveApply, request: Request, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "POST /api/move/uploads/{id}/apply")
    _name, home = _home(profile)
    loop = asyncio.get_running_loop()

    def import_session(session: dict[str, Any]) -> dict[str, Any]:
        """One conversation through the same revision-fenced import the Sessions page uses."""
        from hermes_cli.web_server_sessions import _mutate_session_request
        future = asyncio.run_coroutine_threadsafe(_mutate_session_request(
            request, profile, session["id"], request_id=f"move-{move_id}-{session['id']}",
            expected_revision=0, operation="import", payload={"sessions": [session]}), loop)
        try:
            return future.result(timeout=120)
        except HTTPException as exc:
            if exc.status_code == 409 and "revision_conflict" in str(exc.detail):
                return {"imported": 0, "skipped": 1}
            raise RuntimeError(str(exc.detail)) from None

    include = set(body.include) if body.include else None
    return await _run(move_receive.apply, home, move_id, import_session=import_session, include=include)


@router.post("/api/move/uploads/{move_id}/undo")
async def undo_move_upload(move_id: str, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "POST /api/move/uploads/{id}/undo")
    _name, home = _home(profile)
    return await _run(move_receive.undo, home, move_id)


@router.delete("/api/move/uploads/{move_id}")
async def discard_move_upload(move_id: str, profile: Optional[str] = None):
    from hermes_cli import move_receive
    profile = destructive_profile(profile, "DELETE /api/move/uploads/{id}")
    _name, home = _home(profile)
    return await _run(move_receive.discard, home, move_id)
