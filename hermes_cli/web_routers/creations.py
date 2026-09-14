from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from hermes_cli.config import get_hermes_home, load_config
from hermes_cli.creations.contracts import CreationsConfig
from hermes_cli.creations.service import CreationsService
from hermes_cli.web_deps import late

router = APIRouter()


_cron_profile_home = late("_cron_profile_home", "hermes_cli.web_server_cron")


def _profile_home(profile: str | None) -> Path:
    if profile:
        _name, home = _cron_profile_home(profile)
        return Path(home)
    return Path(get_hermes_home())


def _service(profile: str | None = None) -> CreationsService:
    cfg = load_config().get("dashboard", {}).get("creations", {})
    return CreationsService(_profile_home(profile), CreationsConfig.from_mapping(cfg))


async def _call(fn, profile: str | None = None):
    service = _service(profile)
    try: return await fn(service)
    except KeyError as exc: raise HTTPException(404, str(exc))
    except ValueError as exc: raise HTTPException(400, str(exc))
    except RuntimeError as exc: raise HTTPException(502, str(exc))
    finally: await service.close()


@router.get("/api/creations/status")
async def creations_status(profile: str | None = None): return await _call(lambda s: s.status(), profile)

@router.get("/api/creations/workflows")
async def creations_workflows(profile: str | None = None): return await _call(lambda s: s.workflows(), profile)

@router.post("/api/creations/plans")
async def creations_plan(body: dict, profile: str | None = None): return await _call(lambda s: s.plan(body), profile)

@router.post("/api/creations/runs")
async def creations_submit(body: dict, profile: str | None = None): return await _call(lambda s: s.submit(body), profile)

@router.get("/api/creations/runs")
async def creations_runs(profile: str | None = None): return await _call(lambda s: s.runs(), profile)

@router.get("/api/creations/library")
async def creations_library(kind: str | None = None, cursor: str | None = None, profile: str | None = None):
    if kind not in (None, "image", "video"): raise HTTPException(400, "kind must be image or video")
    return await _call(lambda s: s.library(kind, cursor), profile)

@router.get("/api/creations/library/{item_id}/content")
async def creations_library_content(item_id: int, profile: str | None = None):
    item = await _call(lambda s: s.library_item(item_id), profile)
    if not item or item.get("storageState") != "local" or not item.get("path"):
        raise HTTPException(404, "library_not_found")
    path = Path(item["path"]).resolve()
    # The path comes from the server-side SQLite record, never from the browser.
    if not path.is_file(): raise HTTPException(404, "library_not_found")
    return FileResponse(path, media_type=item.get("mimeType"), filename=item.get("filename"), headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=0"})

@router.post("/api/creations/uploads")
async def creations_upload(request: Request, profile: str | None = None):
    content_type = request.headers.get("content-type", "")
    data = await request.body()
    return await _call(lambda s: s.upload(data, content_type), profile)

@router.post("/api/creations/library/{item_id}/reuse")
async def creations_reuse(item_id: int, profile: str | None = None): return await _call(lambda s: s.reuse(item_id), profile)

@router.post("/api/creations/library/{item_id}/copy-retry")
async def creations_copy_retry(item_id: int, profile: str | None = None): return await _call(lambda s: s.copy_retry(item_id), profile)

@router.get("/api/creations/runs/{run_id}")
async def creations_run(run_id: str, profile: str | None = None):
    result = await _call(lambda s: s.run(run_id), profile)
    if result is None: raise HTTPException(404, "run_not_found")
    return result

@router.post("/api/creations/runs/{run_id}/retry")
async def creations_retry(run_id: str, body: dict, profile: str | None = None): return await _call(lambda s: s.retry(run_id, body), profile)
