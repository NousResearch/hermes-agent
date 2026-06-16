"""Core SurfSense plugin implementation."""

from __future__ import annotations

import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

try:
    from hermes_cli.config import get_env_value, save_env_value
except Exception:  # pragma: no cover - import safety during early plugin load
    get_env_value = None  # type: ignore[assignment]
    save_env_value = None  # type: ignore[assignment]


SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(api[_-]?key|secret[_-]?key|token|password|passwd|authorization)\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)([?&](code|token|auth|key|secret)=)[^\s&#]+"),
    re.compile(r"\b[A-Za-z0-9_-]{24,}\b"),
)


STATUS_SCHEMA = {
    "name": "surfsense_status",
    "description": "Show self-hosted SurfSense readiness without revealing secrets.",
    "parameters": {"type": "object", "properties": {}},
}

LOGIN_SCHEMA = {
    "name": "surfsense_login",
    "description": "Log in to SurfSense and optionally save the bearer token to Hermes env storage.",
    "parameters": {
        "type": "object",
        "properties": {
            "username": {"type": "string", "description": "SurfSense account email or username."},
            "password": {"type": "string", "description": "SurfSense account password."},
            "save": {
                "type": "boolean",
                "description": "Save the returned token as SURFSENSE_ACCESS_TOKEN.",
            },
        },
        "required": ["username", "password"],
    },
}

SEARCHSPACES_SCHEMA = {
    "name": "surfsense_searchspaces",
    "description": "List SurfSense search spaces available to the authenticated user.",
    "parameters": {
        "type": "object",
        "properties": {
            "owned_only": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
    },
}

UPLOAD_SCHEMA = {
    "name": "surfsense_upload",
    "description": "Upload local files into a SurfSense search space for indexing.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_space_id": {"type": "integer", "minimum": 1},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Local file paths to upload.",
            },
            "use_vision_llm": {"type": "boolean"},
            "processing_mode": {
                "type": "string",
                "description": "SurfSense processing mode, usually basic.",
            },
        },
        "required": ["search_space_id", "paths"],
    },
}

SEARCH_SCHEMA = {
    "name": "surfsense_search",
    "description": "Search SurfSense documents with citations and metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "search_space_id": {"type": "integer", "minimum": 1},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["query", "search_space_id"],
    },
}

ASK_SCHEMA = {
    "name": "surfsense_ask",
    "description": "Ask a SurfSense chat question against a search space and return bounded SSE events.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "search_space_id": {"type": "integer", "minimum": 1},
            "thread_id": {"type": "integer", "minimum": 1},
            "title": {"type": "string"},
            "mentioned_document_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "disabled_tools": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["query", "search_space_id"],
    },
}

VIDEO_PLAN_SCHEMA = {
    "name": "surfsense_video_plan",
    "description": (
        "Create NotebookLM-style video planning artifacts from SurfSense/source text "
        "for Manim, HeyGen Video Agent, HyperFrames, irodoriTTS, AITuber OnAir, "
        "and MP4 audio muxing without calling external renderers."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Video topic or source-set title."},
            "source_text": {
                "type": "string",
                "description": "Bounded source summary, citations, or notes to turn into a video overview.",
            },
            "renderer": {
                "type": "string",
                "enum": ["all", "manim", "heygen", "hyperframes"],
                "description": "Renderer artifact set to write.",
            },
            "output_dir": {
                "type": "string",
                "description": "Optional output directory. Must not contain parent traversal.",
            },
            "duration_seconds": {"type": "integer", "minimum": 15, "maximum": 900},
            "language": {"type": "string", "description": "Narration language, for example ja or en."},
            "style": {
                "type": "string",
                "description": "Visual style hint, default swiss_pulse.",
            },
            "llm_wiki_text": {
                "type": "string",
                "description": "Optional LLM-wiki page text or notes to carry as derived knowledge.",
            },
            "codegraph_text": {
                "type": "string",
                "description": "Optional codegraph summary or symbol context to visualize.",
            },
            "sleep_text": {
                "type": "string",
                "description": "Optional sleep-consolidation digest to treat as synthesis hints.",
            },
            "memory_text": {
                "type": "string",
                "description": "Optional Hermes memory context. Treated as preference/prior context, not evidence.",
            },
            "evidence_policy": {
                "type": "string",
                "enum": ["strict", "balanced"],
                "description": "strict keeps only source_text as evidence; balanced allows labeled derived hints.",
            },
            "voice_pipeline": {
                "type": "string",
                "enum": ["none", "irodori_tts", "aituber_onair", "all"],
                "description": "Optional voice packet to write for irodoriTTS, AITuber OnAir, or both.",
            },
            "audio_text": {
                "type": "string",
                "description": "Optional narration text override for TTS; defaults to the generated script.",
            },
            "audio_output_path": {
                "type": "string",
                "description": "Destination audio path for the generated voice request. Defaults to voice.wav under output_dir.",
            },
            "video_input_path": {
                "type": "string",
                "description": "Silent rendered video path to mux with audio. Defaults to rendered_video.mp4 under output_dir.",
            },
            "output_mp4_path": {
                "type": "string",
                "description": "Final narrated MP4 path. Defaults to final_with_voice.mp4 under output_dir.",
            },
            "tts_voice": {
                "type": "string",
                "description": "Optional irodoriTTS/AITuber OnAir voice id.",
            },
            "tts_model": {
                "type": "string",
                "description": "Optional irodoriTTS model id.",
            },
            "tts_speed": {
                "type": "number",
                "description": "Optional speech speed multiplier.",
            },
            "tts_format": {
                "type": "string",
                "enum": ["wav", "mp3", "flac", "opus", "aac", "pcm"],
                "description": "Audio format for the voice request. MP4 muxing expects wav by default.",
            },
        },
        "required": ["topic", "source_text"],
    },
}

VIDEO_MUX_SCHEMA = {
    "name": "surfsense_video_mux",
    "description": "Mux an existing silent video and narration audio into an MP4 with ffmpeg.",
    "parameters": {
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Existing silent input video, usually a Manim/HyperFrames MP4.",
            },
            "audio_path": {
                "type": "string",
                "description": "Existing narration audio, usually voice.wav from irodoriTTS or AITuber OnAir.",
            },
            "output_path": {
                "type": "string",
                "description": "Output MP4 path.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "When true, return the ffmpeg argv without executing it.",
            },
        },
        "required": ["video_path", "audio_path", "output_path"],
    },
}


@dataclass
class Settings:
    base_url: str
    frontend_url: str
    access_token: str
    token_file: Path
    surfsense_root: Path
    timeout: int
    max_stream_chars: int


def check_available() -> bool:
    return True


def _env(name: str, default: str = "") -> str:
    if get_env_value is not None:
        try:
            value = get_env_value(name)
            if value is not None:
                return str(value)
        except Exception:
            pass
    return os.environ.get(name, default)


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = _env(name, str(default)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def settings() -> Settings:
    home = Path(get_hermes_home())
    token_file = Path(
        _env("SURFSENSE_TOKEN_FILE", str(home / "surfsense" / "token.json"))
    ).expanduser()
    root_default = str(Path.cwd() / "SurfSense")
    return Settings(
        base_url=_env("SURFSENSE_BASE_URL", "http://localhost:8929").strip().rstrip("/"),
        frontend_url=_env("SURFSENSE_FRONTEND_URL", "http://localhost:3929").strip().rstrip("/"),
        access_token=_env("SURFSENSE_ACCESS_TOKEN", "").strip(),
        token_file=token_file,
        surfsense_root=Path(_env("SURFSENSE_ROOT", root_default)).expanduser(),
        timeout=_int_env("SURFSENSE_TIMEOUT", 60, minimum=5, maximum=600),
        max_stream_chars=_int_env("SURFSENSE_MAX_STREAM_CHARS", 12000, minimum=1000, maximum=100000),
    )


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def redact_sensitive_text(text: str) -> str:
    cleaned = text or ""
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub(
            lambda m: (m.group(1) + "=[REDACTED]") if m.lastindex else "[REDACTED]",
            cleaned,
        )
    return cleaned


def _token_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(data, dict):
        return str(data.get("access_token") or data.get("token") or "").strip()
    return ""


def _effective_token(cfg: Settings) -> str:
    return cfg.access_token or _token_from_file(cfg.token_file)


def _headers(token: str | None = None, *, content_type: str | None = "application/json") -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _url(cfg: Settings, path: str) -> str:
    clean_path = "/" + path.lstrip("/")
    return f"{cfg.base_url}{clean_path}"


def _http_json(
    method: str,
    path: str,
    *,
    cfg: Settings,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    form: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict[str, Any] | list[Any]:
    if form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        headers = _headers(token, content_type="application/x-www-form-urlencoded")
    elif payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = _headers(token)
    else:
        body = None
        headers = _headers(token, content_type=None)
    request = urllib.request.Request(
        _url(cfg, path),
        data=body,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout or cfg.timeout) as response:  # nosec B310 - user-configured local SurfSense URL.
        raw = response.read().decode("utf-8", errors="replace")
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, (dict, list)):
        raise RuntimeError("SurfSense returned a non-object JSON response")
    return data


def _http_text(
    method: str,
    path: str,
    *,
    cfg: Settings,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> str:
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib.request.Request(
        _url(cfg, path),
        data=body,
        method=method,
        headers=_headers(token),
    )
    with urllib.request.urlopen(request, timeout=timeout or cfg.timeout) as response:  # nosec B310 - user-configured local SurfSense URL.
        chunks: list[str] = []
        total = 0
        while True:
            raw = response.read(4096)
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            chunks.append(text)
            total += len(text)
            if total >= cfg.max_stream_chars:
                chunks.append("\n[TRUNCATED]\n")
                break
    return "".join(chunks)


def _http_multipart(
    path: str,
    *,
    cfg: Settings,
    token: str,
    fields: dict[str, str],
    files: Iterable[Path],
) -> dict[str, Any] | list[Any]:
    boundary = f"----hermes-surfsense-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("ascii"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for file_path in files:
        name = file_path.name
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                (
                    'Content-Disposition: form-data; name="files"; '
                    f'filename="{name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime}\r\n\r\n".encode("ascii"),
                file_path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    body = b"".join(chunks)
    request = urllib.request.Request(
        _url(cfg, path),
        data=body,
        method="POST",
        headers=_headers(token, content_type=f"multipart/form-data; boundary={boundary}"),
    )
    with urllib.request.urlopen(request, timeout=cfg.timeout) as response:  # nosec B310 - user-configured local SurfSense URL.
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
        except Exception:
            detail = str(exc)
        return redact_sensitive_text(f"HTTP {exc.code}: {exc.reason or exc.msg}: {detail}")
    return redact_sensitive_text(str(exc) or exc.__class__.__name__)


def _safe_call(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"ok": False, "error": _safe_error(exc)}


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip().lower()).strip("-")
    return slug[:64] or "surfsense-video"


def _allowed_media_roots() -> list[Path]:
    return [
        Path.cwd().resolve(),
        Path(get_hermes_home()).resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def _is_under_allowed_root(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in _allowed_media_roots())


def _resolve_media_path(
    value: str | None,
    *,
    default: Path | None = None,
    base_dir: Path | None = None,
    label: str = "path",
) -> Path:
    raw_text = str(value or "").strip()
    if raw_text:
        raw = Path(raw_text).expanduser()
        if ".." in raw.parts:
            raise ValueError(f"{label} must not contain parent-directory traversal")
        if not raw.is_absolute() and base_dir is not None:
            raw = base_dir / raw
    elif default is not None:
        raw = default
    else:
        raise ValueError(f"{label} is required")
    resolved = raw.resolve()
    if not _is_under_allowed_root(resolved):
        raise ValueError(f"{label} must be under the workspace, Hermes home, or temp directory")
    return resolved


def _resolve_video_output_dir(topic: str, output_dir: str | None, cfg: Settings) -> Path:
    if output_dir:
        raw = Path(output_dir).expanduser()
        if ".." in raw.parts:
            raise ValueError("output_dir must not contain parent-directory traversal")
        resolved = raw.resolve()
        if not _is_under_allowed_root(resolved):
            raise ValueError("output_dir must be under the workspace, Hermes home, or temp directory")
        return resolved
    return (Path(get_hermes_home()) / "surfsense" / "videos" / _slug(topic)).resolve()


def _sentences(source_text: str, *, limit: int = 5) -> list[str]:
    cleaned = re.sub(r"\s+", " ", source_text or "").strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [part.strip() for part in parts if part.strip()]
    if len(sentences) == 1:
        chunks = re.split(r"[;\n]", cleaned)
        sentences = [chunk.strip() for chunk in chunks if chunk.strip()]
    return sentences[:limit]


def _build_video_plan(
    *,
    topic: str,
    source_text: str,
    duration_seconds: int,
    language: str,
    style: str,
    knowledge_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    points = _sentences(source_text, limit=5) or [source_text.strip()[:240] or topic]
    hook = f"{topic}: what the sources say"
    chapters = [
        {"title": "Context", "narration": points[0]},
        {"title": "Evidence", "narration": points[1] if len(points) > 1 else points[0]},
        {"title": "Tension", "narration": points[2] if len(points) > 2 else "Compare the claims before deciding what follows."},
        {"title": "Takeaway", "narration": points[3] if len(points) > 3 else "Close with the action the research now supports."},
    ]
    if len(points) > 4:
        chapters.append({"title": "Next action", "narration": points[4]})
    plan = {
        "topic": topic,
        "language": language,
        "duration_seconds": duration_seconds,
        "style": style,
        "hook": hook,
        "chapters": chapters,
        "source_excerpt": source_text[:4000],
        "production_notes": {
            "notebooklm_mode": "source-grounded overview",
            "external_renderers_called": False,
            "heygen_transport": "Pass heygen_prompt.txt to the HeyGen app or heygen video-agent CLI.",
            "hyperframes_validation": "Run npx hyperframes lint and inspect before render.",
        },
    }
    if knowledge_context:
        plan["knowledge_context"] = knowledge_context
        plan["integration_mode"] = "knowledge_cycle"
    else:
        plan["integration_mode"] = "source_only"
    return plan


def _context_layer(
    *,
    text: str,
    provenance: str,
    claim_role: str,
    max_chars: int = 1800,
) -> dict[str, Any]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    return {
        "available": bool(clean),
        "provenance": provenance,
        "claim_role": claim_role,
        "text_excerpt": clean[:max_chars],
    }


def _build_knowledge_context(
    *,
    source_text: str,
    llm_wiki_text: str = "",
    codegraph_text: str = "",
    sleep_text: str = "",
    memory_text: str = "",
    evidence_policy: str = "strict",
) -> dict[str, Any]:
    policy = (evidence_policy or "strict").strip().lower()
    if policy not in {"strict", "balanced"}:
        policy = "strict"
    return {
        "evidence_policy": policy,
        "layers": {
            "source_backed": _context_layer(
                text=source_text,
                provenance="SurfSense/source_text",
                claim_role="evidence",
            ),
            "llm_wiki": _context_layer(
                text=llm_wiki_text,
                provenance="LLM-wiki",
                claim_role="derived_knowledge",
            ),
            "codegraph": _context_layer(
                text=codegraph_text,
                provenance="codegraph",
                claim_role="code_structure",
            ),
            "sleep_digest": _context_layer(
                text=sleep_text,
                provenance="Hermes sleep consolidation",
                claim_role="consolidation_hint",
            ),
            "memory": _context_layer(
                text=memory_text,
                provenance="Hermes memory",
                claim_role="preference_or_prior_context",
            ),
        },
    }


def _has_integrated_context(context: dict[str, Any]) -> bool:
    layers = context.get("layers", {})
    return any(
        bool(layer.get("available"))
        for key, layer in layers.items()
        if key != "source_backed" and isinstance(layer, dict)
    )


def _llm_wiki_page(plan: dict[str, Any]) -> str:
    lines = [
        f"# {plan['topic']}",
        "",
        "## Source-Backed Summary",
        plan.get("source_excerpt", ""),
        "",
        "## Video Chapters",
    ]
    for chapter in plan["chapters"]:
        lines.append(f"- {chapter['title']}: {chapter['narration']}")
    context = plan.get("knowledge_context") or {}
    if context:
        lines.extend(["", "## Integration Layers"])
        for name, layer in context.get("layers", {}).items():
            if not isinstance(layer, dict) or not layer.get("available"):
                continue
            lines.append(f"- {name}: {layer['claim_role']} from {layer['provenance']}")
    lines.extend(
        [
            "",
            "## Evidence Policy",
            str(context.get("evidence_policy") or "strict"),
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _memory_sleep_packet(plan: dict[str, Any]) -> dict[str, Any]:
    context = plan.get("knowledge_context") or {}
    layers = context.get("layers", {})
    return {
        "topic": plan["topic"],
        "integration_mode": plan.get("integration_mode", "source_only"),
        "sleep_digest": layers.get("sleep_digest", {}),
        "memory": layers.get("memory", {}),
        "writeback_policy": {
            "memory_is_not_evidence": True,
            "sleep_digest_requires_source_review": True,
        },
    }


def _script_text(plan: dict[str, Any]) -> str:
    lines = [plan["hook"], ""]
    for chapter in plan["chapters"]:
        lines.append(f"{chapter['title']}: {chapter['narration']}")
    context = plan.get("knowledge_context") or {}
    if _has_integrated_context(context):
        lines.append("")
        lines.append("Integrated context: keep source evidence, LLM-wiki, codegraph, sleep digest, and memory visibly labeled.")
    lines.append("")
    lines.append("End with one clear research next step.")
    return "\n".join(lines)


def _coerce_voice_pipeline(value: str | None) -> str:
    pipeline = (value or "none").strip().lower().replace("-", "_")
    if pipeline not in {"none", "irodori_tts", "aituber_onair", "all"}:
        raise ValueError(f"unsupported voice_pipeline: {value}")
    return pipeline


def _coerce_tts_format(value: str | None) -> str:
    fmt = (value or "wav").strip().lower()
    if fmt not in {"wav", "mp3", "flac", "opus", "aac", "pcm"}:
        return "wav"
    return fmt


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _voice_script_text(plan: dict[str, Any], audio_text: str = "") -> str:
    override = (audio_text or "").strip()
    if override:
        return override
    return _script_text(plan)


def _ffmpeg_mux_argv(video_path: Path, audio_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]


def _write_voice_packet(
    *,
    plan: dict[str, Any],
    out_dir: Path,
    files: list[Path],
    voice_pipeline: str,
    audio_text: str = "",
    audio_output_path: str | None = None,
    video_input_path: str | None = None,
    output_mp4_path: str | None = None,
    tts_voice: str = "",
    tts_model: str = "",
    tts_speed: float | None = None,
    tts_format: str = "wav",
) -> dict[str, Any]:
    pipeline = _coerce_voice_pipeline(voice_pipeline)
    if pipeline == "none":
        return {}
    fmt = _coerce_tts_format(tts_format)
    audio_path = _resolve_media_path(
        audio_output_path,
        default=out_dir / f"voice.{fmt}",
        base_dir=out_dir,
        label="audio_output_path",
    )
    video_path = _resolve_media_path(
        video_input_path,
        default=out_dir / "rendered_video.mp4",
        base_dir=out_dir,
        label="video_input_path",
    )
    mp4_path = _resolve_media_path(
        output_mp4_path,
        default=out_dir / "final_with_voice.mp4",
        base_dir=out_dir,
        label="output_mp4_path",
    )
    if mp4_path.suffix.lower() != ".mp4":
        mp4_path = mp4_path.with_suffix(".mp4")
    script = _voice_script_text(plan, audio_text)
    voice_script_path = out_dir / "voice_script.txt"
    voice_script_path.write_text(script, encoding="utf-8")
    files.append(voice_script_path)

    mux_plan = {
        "video_input": str(video_path),
        "audio_input": str(audio_path),
        "output_mp4": str(mp4_path),
        "ffmpeg": {"argv": _ffmpeg_mux_argv(video_path, audio_path, mp4_path)},
        "dry_run_tool": {
            "tool": "surfsense_video_mux",
            "arguments": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "output_path": str(mp4_path),
                "dry_run": True,
            },
        },
    }
    mux_plan_path = out_dir / "mp4_mux_plan.json"
    mux_plan_path.write_text(json.dumps(mux_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(mux_plan_path)

    packet: dict[str, Any] = {
        "pipeline": pipeline,
        "voice_script_path": str(voice_script_path),
        "audio_output_path": str(audio_path),
        "video_input_path": str(video_path),
        "output_mp4_path": str(mp4_path),
        "mp4_mux_plan_path": str(mux_plan_path),
    }
    if pipeline in {"irodori_tts", "all"}:
        irodori_request = {
            "tool": "irodori_tts_synthesize",
            "arguments": {
                "text": script,
                "output_path": str(audio_path),
                "voice": tts_voice or None,
                "model": tts_model or None,
                "format": fmt,
                "speed": tts_speed,
            },
        }
        irodori_path = out_dir / "irodori_tts_request.json"
        irodori_path.write_text(json.dumps(irodori_request, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(irodori_path)
        packet["irodori_tts_request_path"] = str(irodori_path)
    if pipeline in {"aituber_onair", "all"}:
        aituber_cue = {
            "tool": "aituber_onair_speak",
            "arguments": {
                "text": script,
                "provider": "irodori",
                "output_path": str(audio_path),
                "format": fmt,
                "voice": tts_voice or None,
                "model": tts_model or None,
                "speed": tts_speed,
                "play": False,
            },
            "mp4_followup_tool": "surfsense_video_mux",
            "mp4_followup_arguments": {
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "output_path": str(mp4_path),
                "dry_run": False,
            },
        }
        aituber_path = out_dir / "aituber_onair_cue.json"
        aituber_path.write_text(json.dumps(aituber_cue, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(aituber_path)
        packet["aituber_onair_cue_path"] = str(aituber_path)
    return packet


def _manim_scene(plan: dict[str, Any]) -> str:
    chapters = plan["chapters"][:4]
    chapter_data = json.dumps(
        [(chapter["title"], chapter["narration"][:120]) for chapter in chapters],
        ensure_ascii=False,
        indent=8,
    )
    title = json.dumps(str(plan["topic"])[:80], ensure_ascii=False)
    context = plan.get("knowledge_context") or {}
    layer_labels = []
    for key, label in (
        ("source_backed", "SurfSense evidence"),
        ("llm_wiki", "LLM-wiki"),
        ("codegraph", "codegraph"),
        ("sleep_digest", "sleep digest"),
        ("memory", "memory"),
    ):
        layer = context.get("layers", {}).get(key, {})
        if key == "source_backed" or (isinstance(layer, dict) and layer.get("available")):
            layer_labels.append(label)
    layer_data = json.dumps(layer_labels, ensure_ascii=False, indent=8)
    return f'''"""Generated SurfSense NotebookLM-style Manim scene.

Render manually after review:
  manim -pql manim_scene.py SurfSenseNotebookOverview
"""

from manim import *


class SurfSenseNotebookOverview(Scene):
    def construct(self):
        title = Text({title}, font_size=42)
        subtitle = Text("SurfSense source-grounded overview", font_size=24, color=BLUE)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.25)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN))
        self.wait(0.8)
        self.play(header.animate.to_edge(UP))

        chapters = {chapter_data}
        rows = VGroup()
        for idx, (name, note) in enumerate(chapters, start=1):
            number = Text(str(idx), font_size=28, color=BLUE)
            heading = Text(name, font_size=26)
            body = Text(note, font_size=18, color=GREY_A).scale_to_fit_width(9.5)
            row = VGroup(number, VGroup(heading, body).arrange(DOWN, aligned_edge=LEFT, buff=0.12))
            row.arrange(RIGHT, aligned_edge=UP, buff=0.35)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(header, DOWN, buff=0.6)

        for row in rows:
            self.play(FadeIn(row, shift=RIGHT), run_time=0.55)
        self.wait(1.2)

        layers = {layer_data}
        if layers:
            layer_title = Text("Integration layers", font_size=24, color=YELLOW)
            layer_items = VGroup(*[Text(item, font_size=18) for item in layers])
            layer_panel = VGroup(layer_title, layer_items.arrange(DOWN, aligned_edge=LEFT, buff=0.16))
            layer_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_edge(RIGHT).shift(DOWN * 0.4)
            self.play(FadeIn(layer_panel, shift=LEFT), run_time=0.7)
            self.wait(0.8)
        self.play(Circumscribe(rows[-1], color=YELLOW), run_time=1.0)
        self.wait(0.8)
'''


def _heygen_prompt(plan: dict[str, Any]) -> str:
    script = _script_text(plan)
    context_note = ""
    context = plan.get("knowledge_context") or {}
    if _has_integrated_context(context):
        context_note = (
            "\nLabel claim types clearly: SurfSense/source_text is evidence; "
            "LLM-wiki is derived knowledge; codegraph is code structure; "
            "sleep digest is a consolidation hint; memory is preference or prior context.\n"
        )
    return f"""Create a {plan['duration_seconds']}-second presenter-led explainer in {plan['language']}.

Topic: {plan['topic']}

Narration concept:
{script}
{context_note}

This script is a concept and theme to convey, not a verbatim transcript. You have full creative freedom to expand, elaborate, add examples, and fill the duration naturally. Do not pad with silence or pauses.

Use motion graphics for source structure, evidence highlights, and chapter breaks. Use stock media only when it clarifies a real-world context. Keep one topic and make the final takeaway source-grounded.

STYLE - SWISS PULSE: Black/white with electric blue #0066FF. Grid-locked layouts, Helvetica-style bold type, animated counters, and clean grid wipe transitions.
"""


def _hyperframes_html(plan: dict[str, Any]) -> str:
    chapters = plan["chapters"][:4]
    scene_duration = max(4, int(plan["duration_seconds"] / max(1, len(chapters))))
    escaped_topic = html.escape(str(plan["topic"]))
    scene_markup: list[str] = []
    tweens: list[str] = []
    for idx, chapter in enumerate(chapters):
        start = idx * scene_duration
        scene_id = f"scene-{idx + 1}"
        scene_markup.append(
            f'''    <section id="{scene_id}" class="scene" data-start="{start}" data-duration="{scene_duration}" data-track-index="{idx + 1}">
      <div class="scene-content">
        <p class="kicker">SurfSense overview / {idx + 1:02d}</p>
        <h1>{html.escape(str(chapter["title"]))}</h1>
        <p>{html.escape(str(chapter["narration"][:260]))}</p>
      </div>
    </section>'''
        )
        offset = start + 0.2
        tweens.append(
            f'''      tl.from("#{scene_id} .kicker", {{ y: 24, opacity: 0, duration: 0.45, ease: "power3.out" }}, {offset});
      tl.from("#{scene_id} h1", {{ x: -48, opacity: 0, duration: 0.65, ease: "expo.out" }}, {offset + 0.18});
      tl.from("#{scene_id} p:last-child", {{ y: 34, opacity: 0, duration: 0.6, ease: "back.out(1.2)" }}, {offset + 0.38});'''
        )
    scenes = "\n".join(scene_markup)
    timeline = "\n".join(tweens)
    duration = scene_duration * max(1, len(chapters))
    return f'''<!doctype html>
<html lang="{html.escape(str(plan['language']))}">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escaped_topic}</title>
  <style>
    [data-composition-id="surfsense-overview"] {{
      width: 100%;
      height: 100%;
      background: #05070b;
      color: #f7f8fb;
      font-family: Inter, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    .scene {{
      position: absolute;
      inset: 0;
      display: grid;
      background:
        linear-gradient(90deg, rgba(0, 102, 255, 0.18) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255, 255, 255, 0.07) 1px, transparent 1px),
        #05070b;
      background-size: 96px 96px, 96px 96px, auto;
    }}
    .scene-content {{
      display: flex;
      flex-direction: column;
      justify-content: center;
      width: 100%;
      height: 100%;
      padding: 110px 150px;
      gap: 26px;
      box-sizing: border-box;
    }}
    .kicker {{
      color: #69a1ff;
      font-size: 24px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    h1 {{
      max-width: 1180px;
      margin: 0;
      color: #ffffff;
      font-size: 92px;
      line-height: 0.96;
      letter-spacing: 0;
    }}
    p {{
      max-width: 1080px;
      margin: 0;
      color: #d6deea;
      font-size: 36px;
      line-height: 1.25;
      letter-spacing: 0;
    }}
  </style>
</head>
<body>
  <main data-composition-id="surfsense-overview" data-width="1920" data-height="1080" data-duration="{duration}">
{scenes}
  </main>
  <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
{timeline}
    window.__timelines["surfsense-overview"] = tl;
  </script>
</body>
</html>
'''


def video_plan(
    *,
    topic: str,
    source_text: str,
    renderer: str = "all",
    output_dir: str | None = None,
    duration_seconds: int = 120,
    language: str = "ja",
    style: str = "swiss_pulse",
    llm_wiki_text: str = "",
    codegraph_text: str = "",
    sleep_text: str = "",
    memory_text: str = "",
    evidence_policy: str = "strict",
    voice_pipeline: str = "none",
    audio_text: str = "",
    audio_output_path: str | None = None,
    video_input_path: str | None = None,
    output_mp4_path: str | None = None,
    tts_voice: str = "",
    tts_model: str = "",
    tts_speed: float | None = None,
    tts_format: str = "wav",
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    clean_topic = (topic or "").strip()
    clean_source = (source_text or "").strip()
    if not clean_topic:
        return {"ok": False, "error": "topic is required"}
    if not clean_source:
        return {"ok": False, "error": "source_text is required"}
    renderer = (renderer or "all").strip().lower()
    if renderer not in {"all", "manim", "heygen", "hyperframes"}:
        return {"ok": False, "error": f"unsupported renderer: {renderer}"}
    try:
        voice_pipeline = _coerce_voice_pipeline(voice_pipeline)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        out_dir = _resolve_video_output_dir(clean_topic, output_dir, cfg)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    out_dir.mkdir(parents=True, exist_ok=True)
    bounded_duration = max(15, min(int(duration_seconds or 120), 900))
    knowledge_context = _build_knowledge_context(
        source_text=clean_source,
        llm_wiki_text=llm_wiki_text,
        codegraph_text=codegraph_text,
        sleep_text=sleep_text,
        memory_text=memory_text,
        evidence_policy=evidence_policy,
    )
    integrated = _has_integrated_context(knowledge_context)
    plan = _build_video_plan(
        topic=clean_topic,
        source_text=clean_source,
        duration_seconds=bounded_duration,
        language=(language or "ja").strip() or "ja",
        style=(style or "swiss_pulse").strip() or "swiss_pulse",
        knowledge_context=knowledge_context if integrated else None,
    )
    files: list[Path] = []
    plan_path = out_dir / "video_plan.json"
    script_path = out_dir / "script.txt"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    script_path.write_text(_script_text(plan), encoding="utf-8")
    files.extend([plan_path, script_path])
    if integrated:
        context_path = out_dir / "knowledge_context.json"
        wiki_path = out_dir / "llm_wiki_page.md"
        memory_sleep_path = out_dir / "memory_sleep_packet.json"
        context_path.write_text(json.dumps(knowledge_context, ensure_ascii=False, indent=2), encoding="utf-8")
        wiki_path.write_text(_llm_wiki_page(plan), encoding="utf-8")
        memory_sleep_path.write_text(
            json.dumps(_memory_sleep_packet(plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        files.extend([context_path, wiki_path, memory_sleep_path])
    if renderer in {"all", "manim"}:
        path = out_dir / "manim_scene.py"
        path.write_text(_manim_scene(plan), encoding="utf-8")
        files.append(path)
    if renderer in {"all", "heygen"}:
        path = out_dir / "heygen_prompt.txt"
        path.write_text(_heygen_prompt(plan), encoding="utf-8")
        files.append(path)
    if renderer in {"all", "hyperframes"}:
        path = out_dir / "hyperframes_index.html"
        path.write_text(_hyperframes_html(plan), encoding="utf-8")
        files.append(path)
    try:
        voice_packet = _write_voice_packet(
            plan=plan,
            out_dir=out_dir,
            files=files,
            voice_pipeline=voice_pipeline,
            audio_text=audio_text,
            audio_output_path=audio_output_path,
            video_input_path=video_input_path,
            output_mp4_path=output_mp4_path,
            tts_voice=tts_voice,
            tts_model=tts_model,
            tts_speed=tts_speed,
            tts_format=tts_format,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    next_steps = {
        "manim": "Review manim_scene.py, then render with manim -pql manim_scene.py SurfSenseNotebookOverview.",
        "heygen": "Pass heygen_prompt.txt to the HeyGen app or heygen video-agent create flow.",
        "hyperframes": "Rename hyperframes_index.html to index.html in a HyperFrames project, then run npx hyperframes lint and inspect.",
        "knowledge_cycle": "Review knowledge_context.json before promoting any memory or sleep-derived hint to source-backed narration.",
    }
    if voice_packet:
        plan["production_notes"]["voice_pipeline"] = voice_pipeline
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        next_steps["mp4_audio"] = (
            "Create voice.wav with irodoriTTS or AITuber OnAir, render the silent video, "
            "then run surfsense_video_mux or the ffmpeg argv in mp4_mux_plan.json."
        )
    return {
        "ok": True,
        "renderer": renderer,
        "voice_pipeline": voice_pipeline,
        "integration_mode": plan.get("integration_mode", "source_only"),
        "output_dir": str(out_dir),
        "files": [str(path) for path in files],
        "voice_packet": voice_packet,
        "next_steps": next_steps,
    }


def video_mux(
    *,
    video_path: str,
    audio_path: str,
    output_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    try:
        video = _resolve_media_path(video_path, label="video_path")
        audio = _resolve_media_path(audio_path, label="audio_path")
        output = _resolve_media_path(output_path, label="output_path")
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    argv = _ffmpeg_mux_argv(video, audio, output)
    missing_inputs = [str(path) for path in (video, audio) if not path.is_file()]
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "missing_inputs": missing_inputs,
            "ffmpeg": {"argv": argv},
            "output_path": str(output),
        }
    if missing_inputs:
        return {
            "ok": False,
            "error": "video_path and audio_path must exist before muxing",
            "missing_inputs": missing_inputs,
            "ffmpeg": {"argv": argv},
        }
    if not shutil.which("ffmpeg"):
        return {
            "ok": False,
            "error": "ffmpeg was not found on PATH",
            "ffmpeg": {"argv": argv},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    return {
        "ok": completed.returncode == 0 and output.is_file(),
        "dry_run": False,
        "returncode": completed.returncode,
        "output_path": str(output),
        "ffmpeg": {"argv": argv},
        "stdout": redact_sensitive_text((completed.stdout or "")[-4000:]),
        "stderr": redact_sensitive_text((completed.stderr or "")[-4000:]),
    }


def status(*, cfg: Settings | None = None) -> dict[str, Any]:
    cfg = cfg or settings()
    compose_file = cfg.surfsense_root / "docker" / "docker-compose.yml"
    token = _effective_token(cfg)
    health: dict[str, Any] = {"checked": False}
    try:
        health_data = _http_json("GET", "/health", cfg=cfg, timeout=min(cfg.timeout, 10))
        health = {"checked": True, "ok": True, "response": health_data}
    except Exception as exc:
        health = {"checked": True, "ok": False, "error": _safe_error(exc)}
    return {
        "ok": True,
        "base_url": cfg.base_url,
        "frontend_url": cfg.frontend_url,
        "access_token_set": bool(token),
        "access_token": "[REDACTED]" if token else "",
        "token_file": str(cfg.token_file),
        "token_file_exists": cfg.token_file.exists(),
        "surfsense_root": str(cfg.surfsense_root),
        "docker_compose_file": str(compose_file),
        "docker_compose_file_exists": compose_file.exists(),
        "health": health,
    }


def login(
    *,
    username: str,
    password: str,
    save: bool = True,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    data = _http_json(
        "POST",
        "/auth/jwt/login",
        cfg=cfg,
        form={"username": username, "password": password, "grant_type": "password"},
    )
    if not isinstance(data, dict):
        return {"ok": False, "error": "SurfSense login returned an unexpected response"}
    token = str(data.get("access_token") or "").strip()
    if not token:
        return {"ok": False, "error": "SurfSense login did not return access_token"}
    saved: list[str] = []
    if save:
        if save_env_value is not None:
            save_env_value("SURFSENSE_ACCESS_TOKEN", token)
            saved.append("SURFSENSE_ACCESS_TOKEN")
        cfg.token_file.parent.mkdir(parents=True, exist_ok=True)
        cfg.token_file.write_text(json.dumps({"access_token": token}), encoding="utf-8")
        saved.append(str(cfg.token_file))
    return {
        "ok": True,
        "access_token": "[REDACTED]",
        "token_type": data.get("token_type", "bearer"),
        "saved": saved,
    }


def list_searchspaces(
    *,
    owned_only: bool = False,
    limit: int | None = None,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    params = {"owned_only": str(bool(owned_only)).lower()}
    if limit is not None:
        params["limit"] = str(max(1, min(int(limit), 200)))
    query = urllib.parse.urlencode(params)
    data = _http_json(
        "GET",
        f"/api/v1/searchspaces?{query}",
        cfg=cfg,
        token=_effective_token(cfg),
    )
    return {"ok": True, "searchspaces": data}


def upload_files(
    *,
    paths: list[str],
    search_space_id: int,
    use_vision_llm: bool = False,
    processing_mode: str = "basic",
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    token = _effective_token(cfg)
    if not token:
        return {"ok": False, "error": "SURFSENSE_ACCESS_TOKEN is not configured"}
    file_paths = [Path(path).expanduser() for path in paths]
    missing = [str(path) for path in file_paths if not path.is_file()]
    if missing:
        return {"ok": False, "missing": missing, "error": "One or more files do not exist"}
    data = _http_multipart(
        "/api/v1/documents/fileupload",
        cfg=cfg,
        token=token,
        fields={
            "search_space_id": str(search_space_id),
            "use_vision_llm": str(bool(use_vision_llm)).lower(),
            "processing_mode": processing_mode or "basic",
        },
        files=file_paths,
    )
    return {"ok": True, "upload": data}


def search_documents(
    *,
    query: str,
    search_space_id: int,
    page_size: int = 10,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    params = urllib.parse.urlencode(
        {
            "search_space_id": int(search_space_id),
            "page_size": max(1, min(int(page_size or 10), 100)),
            "q": query,
        }
    )
    data = _http_json(
        "GET",
        f"/api/v1/documents/search?{params}",
        cfg=cfg,
        token=_effective_token(cfg),
    )
    return {"ok": True, "results": data}


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        data_text = "\n".join(data_lines)
        if data_text == "[DONE]":
            events.append({"type": "done"})
            continue
        try:
            value = json.loads(data_text)
        except json.JSONDecodeError:
            value = {"type": "raw", "text": data_text}
        if isinstance(value, dict):
            events.append(value)
        else:
            events.append({"type": "data", "value": value})
    return events


def ask(
    *,
    query: str,
    search_space_id: int,
    thread_id: int | None = None,
    title: str | None = None,
    mentioned_document_ids: list[int] | None = None,
    disabled_tools: list[str] | None = None,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    token = _effective_token(cfg)
    if not token:
        return {"ok": False, "error": "SURFSENSE_ACCESS_TOKEN is not configured"}
    effective_thread_id = thread_id
    created_thread: dict[str, Any] | None = None
    if not effective_thread_id:
        data = _http_json(
            "POST",
            "/api/v1/threads",
            cfg=cfg,
            token=token,
            payload={
                "title": title or "Hermes SurfSense chat",
                "search_space_id": search_space_id,
                "visibility": "PRIVATE",
            },
        )
        if not isinstance(data, dict) or not data.get("id"):
            return {"ok": False, "error": "SurfSense did not return a thread id"}
        created_thread = data
        effective_thread_id = int(data["id"])
    payload = {
        "chat_id": int(effective_thread_id),
        "user_query": query,
        "search_space_id": int(search_space_id),
        "messages": [],
        "mentioned_document_ids": mentioned_document_ids or None,
        "disabled_tools": disabled_tools or None,
        "filesystem_mode": "cloud",
        "client_platform": "web",
    }
    raw = _http_text("POST", "/api/v1/new_chat", cfg=cfg, token=token, payload=payload, timeout=cfg.timeout)
    return {
        "ok": True,
        "thread_id": effective_thread_id,
        "created_thread": created_thread,
        "events": _parse_sse(raw),
        "raw_chars": len(raw),
        "raw_truncated": "[TRUNCATED]" in raw,
    }


def docker_compose(
    *,
    action: str = "ps",
    cfg: Settings | None = None,
) -> dict[str, Any]:
    cfg = cfg or settings()
    compose_file = cfg.surfsense_root / "docker" / "docker-compose.yml"
    if not compose_file.exists():
        return {"ok": False, "error": f"docker compose file not found: {compose_file}"}
    allowed = {"ps", "up", "pull", "logs"}
    if action not in allowed:
        return {"ok": False, "error": f"unsupported docker action: {action}"}
    cmd = ["docker", "compose", "-f", str(compose_file)]
    if action == "up":
        cmd.extend(["up", "-d"])
    elif action == "logs":
        cmd.extend(["logs", "--tail", "80"])
    else:
        cmd.append(action)
    result = subprocess.run(
        cmd,
        cwd=str(compose_file.parent),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": redact_sensitive_text((result.stdout or "")[-8000:]),
        "stderr": redact_sensitive_text((result.stderr or "")[-4000:]),
    }


def handle_status(args: dict[str, Any] | None = None, **_: Any) -> str:
    return _json(_safe_call(status))


def handle_login(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            login,
            username=str(args.get("username") or ""),
            password=str(args.get("password") or ""),
            save=bool(args.get("save", True)),
        )
    )


def handle_searchspaces(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            list_searchspaces,
            owned_only=bool(args.get("owned_only", False)),
            limit=int(args.get("limit") or 0) or None,
        )
    )


def handle_upload(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            upload_files,
            paths=[str(path) for path in args.get("paths") or []],
            search_space_id=int(args.get("search_space_id") or 0),
            use_vision_llm=bool(args.get("use_vision_llm", False)),
            processing_mode=str(args.get("processing_mode") or "basic"),
        )
    )


def handle_search(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            search_documents,
            query=str(args.get("query") or ""),
            search_space_id=int(args.get("search_space_id") or 0),
            page_size=int(args.get("page_size") or 10),
        )
    )


def handle_ask(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    thread = args.get("thread_id")
    return _json(
        _safe_call(
            ask,
            query=str(args.get("query") or ""),
            search_space_id=int(args.get("search_space_id") or 0),
            thread_id=int(thread) if thread else None,
            title=str(args.get("title") or "") or None,
            mentioned_document_ids=args.get("mentioned_document_ids") or None,
            disabled_tools=args.get("disabled_tools") or None,
        )
    )


def handle_video_plan(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            video_plan,
            topic=str(args.get("topic") or ""),
            source_text=str(args.get("source_text") or ""),
            renderer=str(args.get("renderer") or "all"),
            output_dir=str(args.get("output_dir") or "") or None,
            duration_seconds=int(args.get("duration_seconds") or 120),
            language=str(args.get("language") or "ja"),
            style=str(args.get("style") or "swiss_pulse"),
            llm_wiki_text=str(args.get("llm_wiki_text") or ""),
            codegraph_text=str(args.get("codegraph_text") or ""),
            sleep_text=str(args.get("sleep_text") or ""),
            memory_text=str(args.get("memory_text") or ""),
            evidence_policy=str(args.get("evidence_policy") or "strict"),
            voice_pipeline=str(args.get("voice_pipeline") or "none"),
            audio_text=str(args.get("audio_text") or ""),
            audio_output_path=str(args.get("audio_output_path") or "") or None,
            video_input_path=str(args.get("video_input_path") or "") or None,
            output_mp4_path=str(args.get("output_mp4_path") or "") or None,
            tts_voice=str(args.get("tts_voice") or ""),
            tts_model=str(args.get("tts_model") or ""),
            tts_speed=_optional_float(args.get("tts_speed")),
            tts_format=str(args.get("tts_format") or "wav"),
        )
    )


def handle_video_mux(args: dict[str, Any] | None = None, **_: Any) -> str:
    args = args or {}
    return _json(
        _safe_call(
            video_mux,
            video_path=str(args.get("video_path") or ""),
            audio_path=str(args.get("audio_path") or ""),
            output_path=str(args.get("output_path") or ""),
            dry_run=bool(args.get("dry_run", False)),
        )
    )


HELP = """surfsense commands:
  /surfsense status
  /surfsense spaces
  /surfsense search <search_space_id> <query>
  /surfsense ask <search_space_id> <query>
  /surfsense video-plan <topic> :: <source text>
  /surfsense video-mux <video.mp4> <voice.wav> <output.mp4>
"""


def handle_slash(raw_args: str) -> str:
    argv = (raw_args or "").strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return HELP
    command = argv[0].lower()
    if command == "status":
        return _json(_safe_call(status))
    if command in {"spaces", "searchspaces"}:
        return _json(_safe_call(list_searchspaces))
    if command == "search" and len(argv) >= 3:
        return _json(
            _safe_call(
                search_documents,
                search_space_id=int(argv[1]),
                query=" ".join(argv[2:]),
            )
        )
    if command == "ask" and len(argv) >= 3:
        return _json(
            _safe_call(
                ask,
                search_space_id=int(argv[1]),
                query=" ".join(argv[2:]),
            )
        )
    if command == "video-plan" and "::" in argv:
        divider = argv.index("::")
        return _json(
            _safe_call(
                video_plan,
                topic=" ".join(argv[1:divider]),
                source_text=" ".join(argv[divider + 1 :]),
            )
        )
    if command == "video-mux" and len(argv) >= 4:
        return _json(
            _safe_call(
                video_mux,
                video_path=argv[1],
                audio_path=argv[2],
                output_path=argv[3],
            )
        )
    return f"Unknown or incomplete surfsense command: {command}\n\n{HELP}"
