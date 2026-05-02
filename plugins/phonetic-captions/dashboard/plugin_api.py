"""Phonetic Captions plugin — FastAPI backend for the dashboard editor.

Routes are mounted at /api/plugins/phonetic-captions/ by the framework.
Auth is bypassed for /api/plugins/* (dashboard binds localhost only).
"""

import asyncio
import importlib.util
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from hermes_constants import get_hermes_home

_log = logging.getLogger(__name__)

# Load pipeline.py by absolute path — relative imports don't work here because
# the dashboard loader imports this file by path (not as a package module).
_PLUGIN_ROOT = Path(__file__).parent.parent


def _get_pipeline():
    """Return the pipeline module, importing it by file path if necessary."""
    mod_name = "phonetic_captions_pipeline"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _PLUGIN_ROOT / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

router = APIRouter()


@router.get("/health")
async def health_check():
    """Return dependency status for the prerequisites banner in the frontend."""
    import shutil

    ffmpeg_ok = shutil.which("ffmpeg") is not None

    faster_whisper_ok = False
    try:
        import faster_whisper  # type: ignore  # noqa: F401
        faster_whisper_ok = True
    except ImportError:
        pass

    nvidia_key = bool(os.environ.get("NVIDIA_API_KEY", ""))

    if nvidia_key:
        phonetics_source = "nvidia"
    elif faster_whisper_ok:
        # Hermes model path is always available when Hermes is running
        phonetics_source = "hermes"
    else:
        phonetics_source = "unavailable"

    hermes_model = ""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        hermes_model = cfg.get("model", {}).get("default", "")
    except Exception:
        pass

    return {
        "ffmpeg": ffmpeg_ok,
        "faster_whisper": faster_whisper_ok,
        "phonetics_source": phonetics_source,
        "hermes_model": hermes_model,
    }


# Allowed video extensions for uploads (server-side validation)
_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts", ".mts"}

# Default style values mirrored here for diff computation (avoids importing private symbol)
_STYLE_DEFAULTS: dict[str, Any] = {
    "font": "Arial",
    "font_size": 48,
    "primary_color": "&H00FFFFFF",
    "outline_color": "&H00000000",
    "outline_width": 3,
    "alignment": 2,
    "margin_edge": 80,
    "max_line_length": 42,
}

# Memory key prefix for style diff entries
_STYLE_MEMORY_PREFIX = "Caption style edit"


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


def _update_job_status(
    job_id: str,
    status: str,
    status_message: str = "",
    segments: Optional[list] = None,
) -> None:
    """Update status fields on a job in-place (called from background pipeline thread)."""
    safe_id = Path(job_id).name
    job_file = _caption_jobs_dir() / f"{safe_id}.json"
    if not job_file.exists():
        return
    try:
        data = json.loads(job_file.read_text(encoding="utf-8"))
        data["status"] = status
        data["status_message"] = status_message
        if segments is not None:
            data["segments"] = segments
        job_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        _log.exception("_update_job_status failed for %s", job_id)


def _run_pipeline(job_id: str, video_path: str) -> None:
    """Transcribe + generate phonetics for a newly uploaded video (blocking, run in thread)."""
    pipeline = _get_pipeline()
    generate_phonetics = pipeline.generate_phonetics
    transcribe = pipeline.transcribe

    try:
        _update_job_status(job_id, "transcribing", "Transcribing audio…")
        segments = transcribe(video_path)

        _update_job_status(job_id, "generating_phonetics", "Generating phonetics…", segments=segments)
        segments = generate_phonetics(segments)

        _update_job_status(job_id, "ready", "", segments=segments)
    except Exception as exc:
        _log.exception("Pipeline failed for job %s", job_id)
        _update_job_status(job_id, "error", str(exc))


def _call_agent(system_prompt: str, user_message: str) -> str:
    """Instantiate a minimal AIAgent and call it synchronously. Returns the response string."""
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    cfg = load_config()
    model = cfg.get("model", {}).get("default", "")
    runtime = resolve_runtime_provider(requested=None, target_model=model)

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="api_server",
        max_iterations=1,
        enabled_toolsets=[],  # no tools — forces direct text response, prevents empty final_response
    )
    result = agent.run_conversation(user_message, system_message=system_prompt)
    response = result.get("final_response") or ""
    if not response.strip():
        raise ValueError(
            "Agent returned an empty response — the model may have attempted a tool call "
            "or the provider returned no content. Check agent.log for details."
        )
    return response


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SegmentsPayload(BaseModel):
    segments: list[dict[str, Any]]


class StylePayload(BaseModel):
    style: dict[str, Any]


class NLEditPayload(BaseModel):
    instruction: str


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
            created_at = data.get("created_at") or datetime.utcfromtimestamp(
                f.stat().st_mtime
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            summaries.append({
                "id": data.get("id", f.stem),
                "created_at": created_at,
                "video_filename": Path(data.get("video_path", "")).name,
                "segment_count": len(data.get("segments", [])),
                "status": data.get("status", "ready"),
                "status_message": data.get("status_message", ""),
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
    try:
        data = _load_caption_job(job_id)
        data["segments"] = payload.model_dump()["segments"]
        _save_caption_job_data(job_id, data)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        _log.error("save_segments failed for job %s: %s\n%s", job_id, exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"save_segments error: {exc}") from exc


@router.put("/jobs/{job_id}/style")
async def save_style(job_id: str, payload: StylePayload):
    """Save style changes back to the job."""
    data = _load_caption_job(job_id)
    new_style = payload.model_dump()["style"]
    _log.info(
        "save_style job=%s alignment=%s margin_edge=%s",
        job_id, new_style.get("alignment"), new_style.get("margin_edge"),
    )
    data["style"] = new_style
    _save_caption_job_data(job_id, data)
    return {"ok": True, "saved": {"alignment": new_style.get("alignment"), "margin_edge": new_style.get("margin_edge")}}


@router.post("/jobs/{job_id}/burn")
async def reburn_job(job_id: str):
    """Re-burn captions into video with current segments + style."""
    pipeline = _get_pipeline()
    _build_ass_content = pipeline._build_ass_content
    burn = pipeline.burn

    data = _load_caption_job(job_id)
    segments = data.get("segments", [])
    style = data.get("style", {})
    video_path = data.get("video_path", "")

    _log.info(
        "reburn_job job=%s alignment=%s margin_edge=%s video=%s",
        job_id, style.get("alignment"), style.get("margin_edge"), video_path,
    )

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

    # Record style diff in Hermes memory for cross-session style learning
    try:
        from tools.memory_tool import MemoryStore

        diff = {k: v for k, v in style.items() if v != _STYLE_DEFAULTS.get(k)}
        if diff:
            store = MemoryStore()
            store.load_from_disk()
            store.add("memory", f"{_STYLE_MEMORY_PREFIX} (job {job_id}): {json.dumps(diff)}")
    except Exception:
        _log.debug("Style memory write failed (non-fatal)", exc_info=True)

    return {
        "ok": True,
        "output_path": result_path,
        "style_used": {
            "alignment": style.get("alignment"),
            "margin_edge": style.get("margin_edge"),
            "font_size": style.get("font_size"),
        },
    }


@router.get("/jobs/{job_id}/ass")
async def get_ass_content(job_id: str):
    """Return the last generated ASS subtitle file for this job (debug)."""
    cache_dir = get_hermes_home() / "cache" / "captions"
    ass_path = cache_dir / f"reburn_{job_id}.ass"
    if not ass_path.exists():
        raise HTTPException(status_code=404, detail="No ASS file found for this job — burn first")
    content = ass_path.read_text(encoding="utf-8")
    # Extract just the Style lines and first few Dialogue lines for readability
    lines = content.splitlines()
    style_lines = [l for l in lines if l.startswith(("Style:", "PlayRes", "ScaledBorder"))]
    dialogue_lines = [l for l in lines if l.startswith("Dialogue:")][:3]
    return {
        "ass_path": str(ass_path),
        "style_lines": style_lines,
        "first_dialogues": dialogue_lines,
        "full_content": content,
    }


@router.get("/jobs/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    """Stream the output video for the <video> player."""
    data = _load_caption_job(job_id)
    output_path = data.get("output_path", "")

    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output video not found — run burn first")

    resp = FileResponse(
        output_path,
        media_type="video/mp4",
        filename=Path(output_path).name,
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


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


# ---------------------------------------------------------------------------
# Upload — create a new job from a local video file
# ---------------------------------------------------------------------------


@router.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    segments: Optional[UploadFile] = File(None),
    run_pipeline: bool = Form(True),
):
    """Create a new caption job from an uploaded video file.

    If ``segments`` JSON is provided the pipeline is skipped and the job is
    immediately ready.  Otherwise, if ``run_pipeline`` is true, transcription
    + phonetics run in a background thread and job status is updated as they
    progress.
    """
    # Validate extension
    ext = Path(video.filename or "").suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_VIDEO_EXTS))}",
        )

    job_id = uuid.uuid4().hex[:12]
    upload_dir = get_hermes_home() / "cache" / "captions" / "uploads" / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save video to disk
    safe_filename = Path(video.filename).name if video.filename else f"video{ext}"
    video_path = upload_dir / safe_filename
    content = await video.read()
    video_path.write_bytes(content)

    # Parse segments if provided
    parsed_segments: list = []
    has_segments = False
    if segments is not None:
        try:
            raw = await segments.read()
            parsed_segments = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed_segments, list):
                raise ValueError("Segments file must be a JSON array")
            has_segments = True
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid segments JSON: {exc}") from exc

    initial_status = "ready" if has_segments else "pending"
    job: dict[str, Any] = {
        "id": job_id,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "video_path": str(video_path),
        "output_path": "",
        "style": dict(_STYLE_DEFAULTS),
        "segments": parsed_segments,
        "status": initial_status,
        "status_message": "",
    }
    _save_caption_job_data(job_id, job)

    if not has_segments and run_pipeline:
        # Fire pipeline in background thread — do not await
        asyncio.create_task(asyncio.to_thread(_run_pipeline, job_id, str(video_path)))

    return {"job_id": job_id, "status": initial_status}


# ---------------------------------------------------------------------------
# Status polling for async pipeline jobs
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Return current pipeline status for a job (for polling during upload)."""
    data = _load_caption_job(job_id)
    return {
        "status": data.get("status", "ready"),
        "status_message": data.get("status_message", ""),
        "segment_count": len(data.get("segments", [])),
    }


# ---------------------------------------------------------------------------
# NL edit — apply natural-language instructions to segments
# ---------------------------------------------------------------------------

_NL_SYSTEM_PROMPT = """You are a caption segment editor for bilingual EN/VI educational videos.
You receive a JSON array of caption segments and a natural-language instruction from the user.
You MUST respond with ONLY a valid JSON array of patch operations — no prose, no markdown fences.

Each segment in the input has two numbering fields:
  "num"  — 1-based display number (this is what the user sees in the UI as "#1", "#2", etc.)
  "id"   — 0-based internal identifier used in patches

When the user references a segment by number (e.g. "#2", "segment 2", "2nd segment"),
they are referring to the segment whose "num" equals that number.
Always use the corresponding "id" value as "segment_id" in your patches.
Example: if the user says "fix #3", find the segment where num=3 and use its id as segment_id.

Each patch is one of:
  {"op":"edit",    "segment_id":<int>, "field":"text"|"phonetic"|"lang"|"start"|"end", "old":<any>, "new":<any>}
  {"op":"merge",   "segment_ids":[<int>, ...]}
  {"op":"split",   "segment_id":<int>, "at_word_index":<int>}

Rules:
- Only include patches that actually change something.
- For "merge", list segment IDs (id values, not num values) in order; the merged text is the concatenation of their texts separated by a space.
- For "split", at_word_index is 0-based and must be within the words array of that segment.
- "lang" must be "en" or "vi".
- When changing lang to "en", also emit an edit patch clearing "phonetic" to "".
- "phonetic" values MUST use only regular English letters, hyphens, spaces, and square brackets.
  NEVER use IPA symbols (no ɓ ɔ ŋ ʔ ː ˨ ˩ ˧ ˥ or any Unicode phonetic/tone characters).
  Write how the Vietnamese sounds to an English speaker, e.g. "[hong biet]", "[zuh yee]", "[toh ee ten lah]".
- If the instruction cannot produce any meaningful patches (e.g. no relevant segments), return [].
- Return ONLY the JSON array.
"""


@router.post("/jobs/{job_id}/nl-edit")
async def nl_edit_segments(job_id: str, payload: NLEditPayload):
    """Apply a natural-language instruction to segments and return proposed patches."""
    data = _load_caption_job(job_id)
    segments = data.get("segments", [])

    # Build a compact segment representation for the prompt.
    # Prepend "num" (1-indexed display number) so the AI can reliably map
    # user references like "#2" or "segment 3" to the correct 0-based id.
    seg_compact = [
        {"num": i + 1, **{k: v for k, v in s.items() if k != "words"}}
        for i, s in enumerate(segments)
    ]
    user_message = (
        f"Segments:\n{json.dumps(seg_compact, ensure_ascii=False, indent=2)}\n\n"
        f"Instruction: {payload.instruction}"
    )

    try:
        raw = await asyncio.to_thread(_call_agent, _NL_SYSTEM_PROMPT, user_message)
        # Strip markdown fences if model wrapped response anyway
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.splitlines()[1:])
            if clean.endswith("```"):
                clean = clean[: clean.rfind("```")]
        patches = json.loads(clean)
        if not isinstance(patches, list):
            raise ValueError("Agent did not return a JSON array")
    except Exception as exc:
        _log.warning("nl-edit agent call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    return {"patches": patches}


# ---------------------------------------------------------------------------
# QA review — flag segment issues
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = """You are a quality reviewer for bilingual EN/VI caption segments.
Review the provided segments and return ONLY a JSON array of flags — no prose, no markdown fences.

Each segment has a "num" field (1-based display number) and an "id" field (0-based internal id).
Use the "id" value as "segment_id" in every flag you emit.

Each flag must have a "type" field — either "issue" or "praise":
  Issue:  {"segment_id":<int>, "type":"issue",  "issue":"<one-line description>", "suggestion":"<one-line fix>"}
  Praise: {"segment_id":<int>, "type":"praise", "issue":"<one-line note on what is correct>", "suggestion":""}

Check for issues:
1. Wrong language classification (e.g. Vietnamese text labelled "en" or vice versa)
2. Mangled Vietnamese diacritics (Whisper artifacts: missing tones, malformed characters)
3. Phonetic guide that does not phonetically match the Vietnamese text, or that uses IPA symbols
   (ɓ ɔ ŋ ʔ ː ˨ ˩ ˧ ˥ etc.) — guides must use only plain English letters/hyphens/brackets
4. Very short duration (<0.3 s) — likely a stray word that should be merged
5. Very long duration (>8 s) — likely should be split
6. Empty text field

Also add a praise flag for any segment that is clearly well-formed — correct language label,
clean diacritics, accurate phonetic guide, and good timing. Only praise segments that genuinely
stand out as correct; do not praise every segment.

Return [] if there is nothing noteworthy. Return ONLY the JSON array.
"""


@router.post("/jobs/{job_id}/qa")
async def qa_review(job_id: str):
    """Run an AI quality review over all segments and return a list of flags."""
    data = _load_caption_job(job_id)
    segments = data.get("segments", [])

    seg_compact = [
        {"num": i + 1, **{k: v for k, v in s.items() if k != "words"}}
        for i, s in enumerate(segments)
    ]
    user_message = f"Segments:\n{json.dumps(seg_compact, ensure_ascii=False, indent=2)}"

    try:
        raw = await asyncio.to_thread(_call_agent, _QA_SYSTEM_PROMPT, user_message)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.splitlines()[1:])
            if clean.endswith("```"):
                clean = clean[: clean.rfind("```")]
        flags = json.loads(clean)
        if not isinstance(flags, list):
            raise ValueError("Agent did not return a JSON array")
    except Exception as exc:
        _log.warning("qa-review agent call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    return {"flags": flags}


# ---------------------------------------------------------------------------
# Style suggestion — cross-session learned preset
# ---------------------------------------------------------------------------

_STYLE_SUGGESTION_SYSTEM_PROMPT = """You are a caption style advisor.
You receive a list of style diff entries from a user's past caption burns.
Each entry shows which style fields they changed away from the defaults.

Analyse the pattern and return ONLY a JSON object with two keys:
  "style": a complete CaptionStyle object (all 8 fields must be present)
  "explanation": a single short sentence (max 15 words) explaining the pattern

The CaptionStyle fields and their defaults:
  font (str, default "Arial")
  font_size (int, default 48)
  primary_color (str ASS hex &HAABBGGRR, default "&H00FFFFFF")
  outline_color (str ASS hex, default "&H00000000")
  outline_width (int, default 3)
  alignment (int ASS numpad, default 2)
  margin_edge (int px, default 80)
  max_line_length (int chars, default 42)

Return ONLY the JSON object.
"""

_STYLE_MIN_ENTRIES = 3


@router.get("/style/suggestion")
async def style_suggestion():
    """Return a learned style preset derived from past burns, if enough history exists."""
    try:
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        store.load_from_disk()
        result = store.read("memory")
        entries: list[str] = [
            e for e in (result.get("entries") or [])
            if isinstance(e, str) and e.startswith(_STYLE_MEMORY_PREFIX)
        ]
    except Exception as exc:
        _log.debug("Style suggestion memory read failed: %s", exc)
        return {"available": False}

    if len(entries) < _STYLE_MIN_ENTRIES:
        return {"available": False}

    diffs_text = "\n".join(entries)
    user_message = f"Style diff history:\n{diffs_text}"

    try:
        raw = await asyncio.to_thread(_call_agent, _STYLE_SUGGESTION_SYSTEM_PROMPT, user_message)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.splitlines()[1:])
            if clean.endswith("```"):
                clean = clean[: clean.rfind("```")]
        suggestion = json.loads(clean)
        if not isinstance(suggestion, dict) or "style" not in suggestion:
            raise ValueError("Unexpected agent response shape")
    except Exception as exc:
        _log.warning("style-suggestion agent call failed: %s", exc)
        return {"available": False}

    return {"available": True, "style": suggestion["style"], "explanation": suggestion.get("explanation", "")}


# ---------------------------------------------------------------------------
# Named preset library
# ---------------------------------------------------------------------------

def _presets_dir() -> Path:
    return get_hermes_home() / "caption-presets"


def _safe_preset_name(name: str) -> str:
    """Sanitize a preset name to a safe filename stem (alphanumeric + dash/underscore/space)."""
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid preset name")
    return safe


_STYLE_GENERATE_SYSTEM_PROMPT = """You are a caption style assistant for bilingual EN/VI short-form videos.
The user will describe a visual style in natural language.
Return ONLY a JSON object with these exact 8 fields (no other keys, no prose, no markdown fences):
  font           (str) — font family name, e.g. "Impact", "Arial", "Trebuchet MS"
  font_size      (int) — point size, typical range 36-72
  primary_color  (str) — ASS hex &HAABBGGRR e.g. "&H00FFFFFF" white, "&H0000FFFF" yellow
  outline_color  (str) — ASS hex e.g. "&H00000000" black, "&H000000FF" red
  outline_width  (int) — 0-5
  alignment      (int) — ASS numpad: 2 bottom-center (most common), 8 top-center
  margin_edge    (int) — pixels from the nearest edge (bottom for btm-*, top for top-*, left/right for side alignments), typical 60-120
  max_line_length (int) — characters before hard-wrap, typical 30-50

Return ONLY the JSON object."""


class PresetPayload(BaseModel):
    style: dict[str, Any]


class GenerateStylePayload(BaseModel):
    description: str


@router.get("/presets")
async def list_presets():
    """List all saved named style presets, newest first."""
    d = _presets_dir()
    if not d.exists():
        return []
    presets = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            presets.append({"name": f.stem, "style": data})
        except (json.JSONDecodeError, OSError):
            continue
    return presets


@router.put("/presets/{name}")
async def save_preset(name: str, payload: PresetPayload):
    """Create or overwrite a named style preset."""
    safe = _safe_preset_name(name)
    d = _presets_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{safe}.json").write_text(
        json.dumps(payload.style, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True, "name": safe}


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    """Delete a named style preset (no-op if it does not exist)."""
    safe = _safe_preset_name(name)
    f = _presets_dir() / f"{safe}.json"
    if f.exists():
        f.unlink()
    return {"ok": True}


@router.post("/presets/generate")
async def generate_style(payload: GenerateStylePayload):
    """Generate a CaptionStyle from a natural-language description (not saved automatically)."""
    if not payload.description.strip():
        raise HTTPException(status_code=400, detail="description is required")

    user_message = f"Style description: {payload.description}"
    try:
        raw = await asyncio.to_thread(_call_agent, _STYLE_GENERATE_SYSTEM_PROMPT, user_message)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.splitlines()[1:])
            if clean.endswith("```"):
                clean = clean[: clean.rfind("```")]
        style = json.loads(clean)
        if not isinstance(style, dict):
            raise ValueError("Agent did not return a JSON object")
    except Exception as exc:
        _log.warning("generate-style agent call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    return {"style": style}
