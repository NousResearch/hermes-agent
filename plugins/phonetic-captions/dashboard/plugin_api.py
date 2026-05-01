"""Phonetic Captions plugin — FastAPI backend for the dashboard editor.

Routes are mounted at /api/plugins/phonetic-captions/ by the framework.
Auth is bypassed for /api/plugins/* (dashboard binds localhost only).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from hermes_constants import get_hermes_home

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _caption_jobs_dir() -> Path:
    return get_hermes_home() / "caption-jobs"


def _load_caption_job(job_id: str) -> dict:
    """Load a caption job JSON by ID. Raises HTTPException if not found."""
    # Sanitize job_id to prevent path traversal
    safe_id = Path(job_id).name
    if safe_id != job_id or ".." in job_id or "/" in job_id:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job_file = _caption_jobs_dir() / f"{safe_id}.json"
    if not job_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return json.loads(job_file.read_text(encoding="utf-8"))


def _save_caption_job_data(job_id: str, data: dict) -> None:
    """Persist caption job data to disk."""
    safe_id = Path(job_id).name
    if safe_id != job_id or ".." in job_id or "/" in job_id:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    jobs_dir = _caption_jobs_dir()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{safe_id}.json"
    job_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SegmentsPayload(BaseModel):
    segments: list[dict[str, Any]]


class StylePayload(BaseModel):
    style: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/jobs")
async def list_jobs():
    """List all caption jobs (summary only)."""
    jobs_dir = _caption_jobs_dir()
    if not jobs_dir.exists():
        return []

    summaries = []
    for f in sorted(jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summaries.append({
                "id": data.get("id", f.stem),
                "created_at": data.get("created_at", ""),
                "video_filename": Path(data.get("video_path", "")).name,
                "segment_count": len(data.get("segments", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get full job state including segments and style."""
    return _load_caption_job(job_id)


@router.put("/jobs/{job_id}/segments")
async def save_segments(job_id: str, payload: SegmentsPayload):
    """Save edited segments back to the job."""
    data = _load_caption_job(job_id)
    data["segments"] = payload.segments
    _save_caption_job_data(job_id, data)
    return {"ok": True}


@router.put("/jobs/{job_id}/style")
async def save_style(job_id: str, payload: StylePayload):
    """Save style changes back to the job."""
    data = _load_caption_job(job_id)
    data["style"] = payload.style
    _save_caption_job_data(job_id, data)
    return {"ok": True}


@router.post("/jobs/{job_id}/burn")
async def reburn_job(job_id: str):
    """Re-burn captions into video with current segments + style."""
    from tools.video_caption import _build_ass_content, burn

    data = _load_caption_job(job_id)
    segments = data.get("segments", [])
    style = data.get("style", {})
    video_path = data.get("video_path", "")

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Source video not found: {video_path}")

    # Build ASS content and write to temp file
    ass_content = _build_ass_content(segments, style)
    cache_dir = get_hermes_home() / "cache" / "captions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ass_path = str(cache_dir / f"reburn_{job_id}.ass")
    Path(ass_path).write_text(ass_content, encoding="utf-8")

    # Burn in background thread (FFmpeg is CPU-bound)
    output_path = str(cache_dir / f"{Path(video_path).stem}_captioned_{job_id}.mp4")

    def _do_burn():
        return burn(video_path, ass_path, output_path)

    result_path = await asyncio.to_thread(_do_burn)

    # Update job with new output path
    data["output_path"] = result_path
    _save_caption_job_data(job_id, data)

    return {"ok": True, "output_path": result_path}


@router.get("/jobs/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    """Stream the output video for the <video> player."""
    data = _load_caption_job(job_id)
    output_path = data.get("output_path", "")

    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output video not found — run burn first")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=Path(output_path).name,
    )


@router.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    """Download the final captioned video."""
    data = _load_caption_job(job_id)
    output_path = data.get("output_path", "")

    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output video not found — run burn first")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"captioned_{job_id}.mp4",
        headers={"Content-Disposition": f'attachment; filename="captioned_{job_id}.mp4"'},
    )
