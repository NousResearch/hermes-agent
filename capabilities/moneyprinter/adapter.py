"""Hermes Desktop adapter for the vendored MoneyPrinterTurbo capability.

This module keeps the Video Studio integration outside Hermes Agent Core. It is
used by the Desktop API server as a thin service/proxy layer around the upstream
MoneyPrinterTurbo FastAPI app.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from functools import lru_cache
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse

try:
    from aiohttp import ClientSession, ClientTimeout, web
except Exception:  # pragma: no cover - import guard for optional server deps
    ClientSession = None  # type: ignore[assignment]
    ClientTimeout = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
MONEYPRINTER_ROOT = REPO_ROOT / "external" / "MoneyPrinterTurbo"
UPSTREAM_COMMIT = "63113a3"
DEFAULT_BASE_URL = os.getenv("HERMES_MONEYPRINTER_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT_SECONDS = 30
MINIMAX_PROXY_TIMEOUT_SECONDS = {
    "/api/v1/minimax/lyrics": 90,
    "/api/v1/minimax/music": 210,
    "/api/v1/minimax/tts": 180,
    "/api/v1/minimax/voices/clone": 420,
}
CONFIG_PATH = MONEYPRINTER_ROOT / "config.toml"
CONFIG_EXAMPLE_PATH = MONEYPRINTER_ROOT / "config.example.toml"
TASKS_DIR = MONEYPRINTER_ROOT / "storage" / "tasks"
LOCAL_MATERIALS_DIR = MONEYPRINTER_ROOT / "storage" / "local_videos"
SONGS_DIR = MONEYPRINTER_ROOT / "resource" / "songs"
FONTS_DIR = MONEYPRINTER_ROOT / "resource" / "fonts"
CUSTOM_AUDIO_DIR = MONEYPRINTER_ROOT / "storage" / "custom_audio"
MINIMAX_VOICES_DIR = MONEYPRINTER_ROOT / "storage" / "minimax" / "voices"
MAX_MINIMAX_AUDIO_BYTES = 20 * 1024 * 1024
MINIMAX_AUDIO_EXTS = {".m4a", ".mp3", ".wav"}
SIDECAR_TOKEN_HEADER = "X-Hermes-MoneyPrinter-Token"
_SIDECAR_TOKEN = secrets.token_urlsafe(32)
SUPPORTED_LOCAL_MATERIAL_EXTS = {".avi", ".flv", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png"}
SUPPORTED_AUDIO_EXTS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
SUPPORTED_BGM_EXTS = {".mp3"}
SUPPORTED_FONT_EXTS = {".ttc", ".ttf"}
PUBLIC_VIDEO_OUTPUT_RE = re.compile(r"^(?:combined|final)-\d+\.mp4$", re.IGNORECASE)
MEDIA_PROXY_HEADERS = ("content-type", "content-length", "content-disposition", "accept-ranges", "content-range")

_process: Optional[asyncio.subprocess.Process] = None


def _media_proxy_request_headers(request: Any = None) -> dict[str, str]:
    headers = {SIDECAR_TOKEN_HEADER: _SIDECAR_TOKEN}
    request_headers = getattr(request, "headers", None)
    range_header = str(request_headers.get("Range") or "").strip() if request_headers is not None else ""
    if range_header:
        headers["Range"] = range_header
    return headers


def _envelope(data: Any = None, *, ok: bool = True, error: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {"ok": ok, "data": data, "error": error}


def _error(message: str, code: str = "MONEYPRINTER_ERROR", *, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _envelope(None, ok=False, error={"code": code, "message": message, "details": details or {}})


def _json(data: dict[str, Any], status: int = 200) -> Any:
    return web.json_response(data, status=status)


def _is_installed() -> bool:
    return (MONEYPRINTER_ROOT / "main.py").exists() and (MONEYPRINTER_ROOT / "app").is_dir()


def _service_running() -> bool:
    return _process is not None and _process.returncode is None


def _moneyprinter_python() -> str:
    for relative_path in (("bin", "python"), ("Scripts", "python.exe")):
        venv_python = MONEYPRINTER_ROOT / ".venv" / Path(*relative_path)
        if venv_python.is_file():
            return str(venv_python)
    return sys.executable


@lru_cache(maxsize=4)
def _moneyprinter_runtime_status() -> dict[str, Any]:
    python = _moneyprinter_python()
    required = ("fastapi", "uvicorn", "moviepy", "imageio_ffmpeg")
    probe = (
        "import importlib.util,json,os\n"
        f"required={required!r}\n"
        "missing=[name for name in required if importlib.util.find_spec(name) is None]\n"
        "ffmpeg=''\n"
        "if 'imageio_ffmpeg' not in missing:\n"
        " import imageio_ffmpeg\n"
        " try: ffmpeg=imageio_ffmpeg.get_ffmpeg_exe() or ''\n"
        " except Exception: ffmpeg=''\n"
        "if not ffmpeg or not os.path.isfile(ffmpeg):\n"
        " missing.append('ffmpeg')\n"
        "print(json.dumps({'missing':missing,'ffmpeg':ffmpeg}))\n"
        "raise SystemExit(0 if not missing else 3)\n"
    )
    try:
        completed = subprocess.run(
            [python, "-c", probe],
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        payload = json.loads((completed.stdout or "").strip() or "{}")
        missing = payload.get("missing") if isinstance(payload.get("missing"), list) else []
        ffmpeg_path = str(payload.get("ffmpeg") or "")
        ready = completed.returncode == 0 and not missing and bool(ffmpeg_path)
    except Exception:
        missing = ["python-runtime"]
        ffmpeg_path = ""
        ready = False
    return {
        "ffmpegPath": ffmpeg_path,
        "missingDependencies": [str(item) for item in missing],
        "runtimePython": python,
        "runtimeReady": ready,
    }


def _build_service_env() -> dict[str, str]:
    """Build a clean env for the MoneyPrinter sidecar.

    Hermes' own venv/site-packages (often a different Python minor) must not
    leak into MoneyPrinterTurbo; mixing them breaks native wheels such as
    pydantic_core. Prefer a small whitelist over ``os.environ.copy()``.
    """
    keep_keys = (
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "HERMES_MONEYPRINTER_URL",
    )
    env: dict[str, str] = {}
    for key in keep_keys:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    venv_bin = MONEYPRINTER_ROOT / ".venv" / "bin"
    base_path = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    # Drop other virtualenv bins that would shadow MoneyPrinter's python.
    path_parts = [part for part in base_path.split(os.pathsep) if part and "site-packages" not in part]
    if venv_bin.is_dir():
        path_parts = [str(venv_bin), *[p for p in path_parts if p != str(venv_bin)]]
    env["PATH"] = os.pathsep.join(path_parts) if path_parts else "/usr/bin:/bin"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["MONEYPRINTER_HERMES_TOKEN"] = _SIDECAR_TOKEN
    return env


def _managed_sidecar_command() -> tuple[str, ...]:
    parsed = urlparse(DEFAULT_BASE_URL)
    port = parsed.port or 8080
    return (
        _moneyprinter_python(),
        "-m",
        "uvicorn",
        "app.asgi:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    )


def _health_payload(*, service_running: bool, message: str = "") -> dict[str, Any]:
    storage = MONEYPRINTER_ROOT / "storage"
    return {
        "apiBaseUrl": DEFAULT_BASE_URL,
        "config": _config_summary(),
        "installed": _is_installed(),
        "message": message,
        "serviceRunning": service_running,
        "storageWritable": storage.exists() and os.access(storage, os.W_OK),
        "upstreamCommit": UPSTREAM_COMMIT,
        **_moneyprinter_runtime_status(),
    }


def _managed_identity_valid(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    data = payload.get("data")
    return (
        isinstance(data, dict)
        and data.get("service") == "moneyprinterturbo"
        and data.get("managed") is True
        and data.get("protocol_version") == 1
    )


async def _probe_managed_sidecar() -> bool:
    if ClientSession is None or ClientTimeout is None:
        return False
    try:
        async with ClientSession() as session:
            async with session.get(
                f"{DEFAULT_BASE_URL}/api/v1/hermes/health",
                headers={SIDECAR_TOKEN_HEADER: _SIDECAR_TOKEN},
                timeout=ClientTimeout(total=3),
            ) as response:
                if response.status != 200:
                    return False
                return _managed_identity_valid(await response.json())
    except Exception:
        return False


async def _sidecar_port_in_use() -> bool:
    parsed = urlparse(DEFAULT_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=0.75)
        del reader
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


def _read_config_text() -> str:
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text(encoding="utf-8")
    if CONFIG_EXAMPLE_PATH.exists():
        return CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8")
    return "[app]\n"


def _insert_config_line(text: str, section: str, line: str) -> str:
    section_match = re.search(rf"^\[{re.escape(section)}\]\s*$", text, flags=re.MULTILINE)
    if section_match is None:
        prefix = text.rstrip()
        separator = "\n\n" if prefix else ""
        return f"{prefix}{separator}[{section}]\n{line}\n"

    remaining = text[section_match.end() :]
    next_section = re.search(r"^\s*\[", remaining, flags=re.MULTILINE)
    insert_at = section_match.end() + (next_section.start() if next_section else len(remaining))
    before = text[:insert_at].rstrip()
    after = text[insert_at:]
    separator = "\n" if not after.startswith("\n") else ""
    return f"{before}\n{line}{separator}{after}"


def _replace_config_line(text: str, section: str, key: str, line: str, value_pattern: str) -> str:
    section_match = re.search(rf"^\[{re.escape(section)}\]\s*$", text, flags=re.MULTILINE)
    if section_match is None:
        return _insert_config_line(text, section, line)

    content_start = section_match.end()
    remaining = text[content_start:]
    next_section = re.search(r"^\s*\[", remaining, flags=re.MULTILINE)
    content_end = content_start + (next_section.start() if next_section else len(remaining))
    section_text = text[content_start:content_end]
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*{value_pattern}$", re.MULTILINE)
    if pattern.search(section_text):
        replaced = pattern.sub(line, section_text, count=1)
        return f"{text[:content_start]}{replaced}{text[content_end:]}"
    return _insert_config_line(text, section, line)


def _replace_scalar(text: str, key: str, value: str, *, section: str = "app") -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("config key contains unsupported characters")
    if re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError(f"{key} contains unsupported control characters")
    line = f"{key} = {json.dumps(value, ensure_ascii=False)}"
    return _replace_config_line(text, section, key, line, r".*")


def _replace_list(text: str, key: str, values: list[str], *, section: str = "app") -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("config key contains unsupported characters")
    if any(re.search(r"[\x00-\x1f\x7f]", value) for value in values):
        raise ValueError(f"{key} contains unsupported control characters")
    rendered = ", ".join(json.dumps(value, ensure_ascii=False) for value in values if value)
    line = f"{key} = [{rendered}]"
    return _replace_config_line(text, section, key, line, r"\[.*?\]")


def _split_config_list_input(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = [str(item) for item in value]
    else:
        raw_values = re.split(r"[\n,，]", str(value or ""))
    return [item.strip() for item in raw_values if item.strip()]


def _has_config_value(text: str, key: str) -> bool:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return False
    value = match.group(1).strip()
    return value not in {'""', "''", "[]"}


def _read_config_scalar(text: str, key: str, default: str = "") -> str:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return default
    value = match.group(1).strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    return value.strip() or default


def _config_summary() -> dict[str, Any]:
    text = _read_config_text()
    provider = _read_config_scalar(text, "llm_provider", "openai")
    api_key_configured = _has_config_value(text, f"{provider}_api_key")
    try:
        parsed_config = tomllib.loads(text)
    except Exception:
        parsed_config = {}
    minimax_config = parsed_config.get("minimax") if isinstance(parsed_config.get("minimax"), dict) else {}
    minimax_api_key = str(minimax_config.get("api_key") or _read_config_scalar(text, "minimax_api_key")).strip()
    return {
        "apiKeyConfigured": api_key_configured,
        "baseUrl": _read_config_scalar(text, f"{provider}_base_url"),
        "configExists": CONFIG_PATH.exists(),
        "llmProvider": provider,
        "materialProviders": {
            "coverr": _has_config_value(text, "coverr_api_keys"),
            "pexels": _has_config_value(text, "pexels_api_keys"),
            "pixabay": _has_config_value(text, "pixabay_api_keys"),
        },
        "minimax": {
            "apiKeyConfigured": bool(minimax_api_key),
            "baseUrl": str(
                minimax_config.get("base_url")
                or _read_config_scalar(text, "minimax_base_url", "https://api.minimaxi.com")
            ),
            "musicModel": str(minimax_config.get("music_model") or "music-2.6-free"),
            "t2aModel": str(minimax_config.get("t2a_model") or "speech-2.8-hd"),
            "voiceCloneModel": str(minimax_config.get("voice_clone_model") or "speech-2.8-hd"),
        },
        "modelConfigured": api_key_configured,
        "modelName": _read_config_scalar(text, f"{provider}_model_name"),
    }


def _write_config_text(text: str) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{CONFIG_PATH.name}.",
        suffix=".tmp",
        dir=CONFIG_PATH.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(text)
            fp.flush()
            os.fsync(fp.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, CONFIG_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


async def health() -> Any:
    if ClientSession is None:
        return _json(_error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE"), status=503)

    if not _is_installed():
        return _json(_envelope(_health_payload(service_running=False, message="MoneyPrinterTurbo is not vendored.")))

    if await _probe_managed_sidecar():
        return _json(
            _envelope(
                _health_payload(
                    service_running=True,
                    message="Authenticated MoneyPrinter sidecar is reachable.",
                )
            )
        )

    return _json(
        _envelope(
            _health_payload(
                service_running=False,
                message="Authenticated MoneyPrinter sidecar is not reachable.",
            )
        )
    )


async def get_config() -> Any:
    return _json(_envelope(_config_summary()))


async def save_config(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)

    if not isinstance(body, dict):
        return _json(_error("Config body must be an object", "MONEYPRINTER_INVALID_CONFIG"), status=400)

    text = _read_config_text()
    provider = str(body.get("llmProvider") or "openai").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", provider):
        return _json(_error("Invalid LLM provider name", "MONEYPRINTER_INVALID_CONFIG"), status=400)

    try:
        text = _replace_scalar(text, "llm_provider", provider)

        for body_key, config_key in {
            "apiKey": f"{provider}_api_key",
            "baseUrl": f"{provider}_base_url",
            "modelName": f"{provider}_model_name",
        }.items():
            value = str(body.get(body_key) or "").strip()
            if value:
                text = _replace_scalar(text, config_key, value)

        for body_key, config_key in {
            "minimaxApiKey": "api_key",
            "minimaxBaseUrl": "base_url",
            "minimaxMusicModel": "music_model",
            "minimaxT2aModel": "t2a_model",
            "minimaxVoiceCloneModel": "voice_clone_model",
        }.items():
            value = str(body.get(body_key) or "").strip()
            if value:
                text = _replace_scalar(text, config_key, value, section="minimax")

        for body_key, config_key in (
            ("pexelsApiKey", "pexels_api_keys"),
            ("pixabayApiKey", "pixabay_api_keys"),
            ("coverrApiKey", "coverr_api_keys"),
        ):
            values = _split_config_list_input(body.get(body_key))
            if values:
                text = _replace_list(text, config_key, values)
    except ValueError as exc:
        return _json(_error(str(exc), "MONEYPRINTER_INVALID_CONFIG"), status=400)

    _write_config_text(text)
    return _json(_envelope(_config_summary()))


async def start_service() -> Any:
    global _process

    if not _is_installed():
        return _json(_error("external/MoneyPrinterTurbo is missing.", "MONEYPRINTER_NOT_INSTALLED"), status=404)

    runtime = _moneyprinter_runtime_status()
    if not runtime["runtimeReady"]:
        missing = ", ".join(runtime["missingDependencies"]) or "unknown dependencies"
        return _json(
            _error(
                f"MoneyPrinter runtime is incomplete: {missing}",
                "MONEYPRINTER_RUNTIME_NOT_READY",
                details=runtime,
            ),
            status=503,
        )

    if await _probe_managed_sidecar():
        return _json(
            _envelope(
                _health_payload(
                    service_running=True,
                    message="MoneyPrinter service is already running.",
                )
            )
        )

    if await _sidecar_port_in_use():
        return _json(
            _error(
                f"Port {urlparse(DEFAULT_BASE_URL).port or 8080} is occupied by a service that is not this managed MoneyPrinter sidecar.",
                "MONEYPRINTER_PORT_CONFLICT",
            ),
            status=409,
        )

    try:
        _process = await asyncio.create_subprocess_exec(
            *_managed_sidecar_command(),
            cwd=str(MONEYPRINTER_ROOT),
            env=_build_service_env(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as exc:
        return _json(_error(str(exc), "MONEYPRINTER_START_FAILED"), status=500)

    for _ in range(100):
        if _process.returncode is not None:
            return _json(
                _error(
                    f"MoneyPrinter process exited during startup with code {_process.returncode}.",
                    "MONEYPRINTER_START_FAILED",
                ),
                status=503,
            )
        if await _probe_managed_sidecar():
            return _json(
                _envelope(
                    _health_payload(
                        service_running=True,
                        message="MoneyPrinter service is ready.",
                    )
                )
            )
        await asyncio.sleep(0.1)

    return _json(
        _error(
            "MoneyPrinter process started but did not pass its authenticated health check.",
            "MONEYPRINTER_START_TIMEOUT",
        ),
        status=504,
    )


def _normalize_task_state(value: Any) -> str:
    state = str(value or "unknown").lower()
    return {
        "-1": "failed",
        "1": "complete",
        "4": "processing",
        "completed": "complete",
        "success": "complete",
    }.get(state, state)


def _as_task(raw: dict[str, Any]) -> dict[str, Any]:
    task_id = str(raw.get("task_id") or raw.get("id") or "")
    state = _normalize_task_state(raw.get("state") or raw.get("status") or "unknown")
    progress = raw.get("progress")
    if progress is None:
        progress = 100 if state in {"complete", "completed", "success"} else 0
    try:
        progress_num = float(progress)
    except Exception:
        progress_num = 0

    videos = raw.get("videos") or raw.get("combined_videos") or []
    normalized_videos = []
    for index, item in enumerate(videos if isinstance(videos, list) else []):
        if not item:
            continue
        value = str(item)
        media_path = _normalize_media_path(value)
        name = media_path.rstrip("/").split("/")[-1] or f"video-{index + 1}.mp4"
        normalized_videos.append({
            "downloadUrl": f"/api/capabilities/moneyprinter/download/{media_path}",
            "file": media_path,
            "name": name,
            "streamUrl": f"/api/capabilities/moneyprinter/stream/{media_path}",
        })

    return {
        "audioFile": raw.get("audio_file") or raw.get("audioFile"),
        "error": raw.get("error"),
        "id": task_id,
        "progress": max(0, min(100, progress_num)),
        "script": raw.get("script") or raw.get("video_script"),
        "state": state,
        "subject": raw.get("video_subject") or (raw.get("params") or {}).get("video_subject") if isinstance(raw.get("params"), dict) else raw.get("subject"),
        "subtitlePath": raw.get("subtitle_path") or raw.get("subtitlePath"),
        "terms": raw.get("terms"),
        "videos": normalized_videos,
    }


def _task_from_disk(task_dir: Path) -> Optional[dict[str, Any]]:
    if not task_dir.is_dir():
        return None

    script_path = task_dir / "script.json"
    script_data: dict[str, Any] = {}
    if script_path.exists():
        try:
            loaded = json.loads(script_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                script_data = loaded
        except Exception:
            script_data = {}

    params = script_data.get("params") if isinstance(script_data.get("params"), dict) else {}
    videos = [
        f"{task_dir.name}/{path.name}"
        for pattern in ("final-*.mp4", "combined-*.mp4")
        for path in sorted(task_dir.glob(pattern))
        if path.is_file() and PUBLIC_VIDEO_OUTPUT_RE.match(path.name)
    ]

    state = "complete" if videos else "unknown"
    raw = {
        "task_id": task_dir.name,
        "params": params,
        "progress": 100 if videos else 0,
        "script": script_data.get("script"),
        "state": state,
        "terms": script_data.get("search_terms"),
        "videos": videos,
    }
    task = _as_task(raw)
    try:
        task["updatedAt"] = task_dir.stat().st_mtime
    except OSError:
        task["updatedAt"] = 0
    return task


def _recover_disk_tasks() -> list[dict[str, Any]]:
    if not TASKS_DIR.exists():
        return []

    tasks = []
    for task_dir in TASKS_DIR.iterdir():
        task = _task_from_disk(task_dir)
        if task:
            tasks.append(task)
    tasks.sort(key=lambda item: item.get("updatedAt", 0), reverse=True)
    return tasks


def _merge_tasks(upstream_tasks: list[dict[str, Any]], disk_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in upstream_tasks + disk_tasks:
        task_id = str(task.get("id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        merged.append(task)
    return merged


def _sanitize_local_material_filename(filename: str) -> str:
    raw = str(filename or "").replace("\\", "/")
    name = Path(raw).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not name:
        raise ValueError("material filename is required")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_LOCAL_MATERIAL_EXTS:
        allowed = ", ".join(sorted(SUPPORTED_LOCAL_MATERIAL_EXTS))
        raise ValueError(f"unsupported material type {suffix or '<none>'}; allowed: {allowed}")
    return name


def _sanitize_asset_filename(filename: str, allowed_exts: set[str], label: str) -> str:
    raw = str(filename or "").replace("\\", "/")
    name = Path(raw).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    if not name:
        raise ValueError(f"{label} filename is required")
    suffix = Path(name).suffix.lower()
    if suffix not in allowed_exts:
        allowed = ", ".join(sorted(allowed_exts))
        raise ValueError(f"unsupported {label} type {suffix or '<none>'}; allowed: {allowed}")
    return name


def _sanitize_bgm_filename(filename: str) -> str:
    return _sanitize_asset_filename(filename, SUPPORTED_BGM_EXTS, "BGM")


def _sanitize_custom_audio_filename(filename: str) -> str:
    return _sanitize_asset_filename(filename, SUPPORTED_AUDIO_EXTS, "custom audio")


def _local_material_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    stat = path.stat()
    return {
        "file": path.name,
        "kind": "image" if suffix in {".jpg", ".jpeg", ".png"} else "video",
        "name": path.name,
        "size": stat.st_size,
        "updatedAt": stat.st_mtime,
    }


def _audio_asset_payload(path: Path, *, file_value: Optional[str] = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "file": file_value or path.name,
        "kind": "audio",
        "name": path.name,
        "size": stat.st_size,
        "updatedAt": stat.st_mtime,
    }


def _font_asset_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "file": path.name,
        "name": path.name,
        "size": stat.st_size,
        "updatedAt": stat.st_mtime,
    }


def _custom_audio_relative_path(path: Path) -> str:
    return path.resolve(strict=False).relative_to(MONEYPRINTER_ROOT.resolve(strict=False)).as_posix()


def _normalize_minimax_audio_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    source = Path(raw).expanduser()
    if source.is_absolute():
        try:
            raw = source.resolve(strict=False).relative_to(
                MONEYPRINTER_ROOT.resolve(strict=False)
            ).as_posix()
        except ValueError:
            return ""
    else:
        raw = raw.lstrip("/")
    parts = Path(raw).parts
    allowed_prefixes = (
        ("storage", "custom_audio"),
        ("storage", "minimax", "tts"),
        ("storage", "minimax", "voices"),
    )
    if ".." in parts or not any(parts[: len(prefix)] == prefix for prefix in allowed_prefixes):
        return ""
    if Path(raw).suffix.lower() not in {".m4a", ".mp3", ".wav"}:
        return ""
    return Path(raw).as_posix()


def _minimax_audio_descriptor(value: Any) -> dict[str, Any]:
    media_path = _normalize_minimax_audio_path(value)
    if not media_path:
        return {}
    candidate = MONEYPRINTER_ROOT / media_path
    return {
        "downloadUrl": f"/api/capabilities/moneyprinter/download/{media_path}",
        "file": media_path,
        "kind": "audio",
        "name": Path(media_path).name,
        "size": candidate.stat().st_size if candidate.is_file() else 0,
        "streamUrl": f"/api/capabilities/moneyprinter/stream/{media_path}",
    }


def _decode_upload_content(value: str) -> bytes:
    content = str(value or "")
    if "," in content and content.lstrip().lower().startswith("data:"):
        content = content.split(",", 1)[1]
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("contentBase64 is not valid base64") from exc


def _materialize_minimax_audio_asset(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")

    source_path = str(value.get("sourcePath") or value.get("source_path") or "").strip()
    requested_name = str(value.get("filename") or value.get("name") or "").strip()
    if source_path:
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"{label} sourcePath is not a file")
        if source.suffix.lower() not in MINIMAX_AUDIO_EXTS:
            allowed = ", ".join(sorted(MINIMAX_AUDIO_EXTS))
            raise ValueError(f"unsupported {label} type {source.suffix.lower() or '<none>'}; allowed: {allowed}")
        if source.stat().st_size > MAX_MINIMAX_AUDIO_BYTES:
            raise ValueError(f"{label} exceeds the MiniMax 20 MB limit")
        filename = _sanitize_asset_filename(requested_name or source.name, MINIMAX_AUDIO_EXTS, label)
        raw = source.read_bytes()
    else:
        filename = _sanitize_asset_filename(requested_name, MINIMAX_AUDIO_EXTS, label)
        content_base64 = value.get("contentBase64") or value.get("content_base64")
        if not content_base64:
            raise ValueError(f"{label} contentBase64 is required")
        raw = _decode_upload_content(str(content_base64))
        if len(raw) > MAX_MINIMAX_AUDIO_BYTES:
            raise ValueError(f"{label} exceeds the MiniMax 20 MB limit")

    return {
        "contentBase64": base64.b64encode(raw).decode("ascii"),
        "filename": filename,
    }


def _normalize_local_video_materials(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("video_materials must be a list when video_source is local")

    materials: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            filename = item
            duration = 0
        elif isinstance(item, dict):
            filename = str(item.get("url") or item.get("file") or item.get("name") or "")
            duration = item.get("duration") or 0
        else:
            raise ValueError("video_materials entries must be strings or objects")

        safe_name = _sanitize_local_material_filename(filename)
        if not (LOCAL_MATERIALS_DIR / safe_name).is_file():
            raise ValueError(f"local material not found: {safe_name}")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = 0
        materials.append({"provider": "local", "url": safe_name, "duration": duration_value})
    return materials


def _normalize_bgm_file(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    filename = _sanitize_bgm_filename(raw)
    if not (SONGS_DIR / filename).is_file():
        raise ValueError(f"BGM file not found: {filename}")
    return filename


def _normalize_custom_audio_file(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    filename = _sanitize_custom_audio_filename(raw)
    target = CUSTOM_AUDIO_DIR / filename
    if not target.is_file():
        raise ValueError(f"custom audio file not found: {filename}")
    return _custom_audio_relative_path(target)


def _unwrap_upstream(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def _local_minimax_voices() -> list[str]:
    if not MINIMAX_VOICES_DIR.is_dir():
        return []

    voices = []
    for metadata_path in sorted(MINIMAX_VOICES_DIR.glob("*/metadata.json"), key=lambda p: p.parent.name.lower()):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        voice_id = str(metadata.get("voice_id") or metadata_path.parent.name).strip()
        if not voice_id:
            continue
        display_name = str(metadata.get("display_name") or metadata.get("voice_name") or voice_id).strip()
        voices.append(f"minimax:{voice_id}:{display_name}")
    return voices


def list_local_materials_data() -> tuple[int, dict[str, Any]]:
    LOCAL_MATERIALS_DIR.mkdir(parents=True, exist_ok=True)
    materials = [
        _local_material_payload(path)
        for path in sorted(LOCAL_MATERIALS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if path.is_file() and path.suffix.lower() in SUPPORTED_LOCAL_MATERIAL_EXTS
    ]
    return 200, _envelope({"materials": materials})


def upload_local_material_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        filename = _sanitize_local_material_filename(
            str(body.get("filename") or body.get("name") or body.get("sourcePath") or body.get("source_path") or "")
        )
        target = LOCAL_MATERIALS_DIR / filename
        LOCAL_MATERIALS_DIR.mkdir(parents=True, exist_ok=True)

        source_path = str(body.get("sourcePath") or body.get("source_path") or "").strip()
        content_base64 = body.get("contentBase64") or body.get("content_base64")
        if source_path:
            source = Path(source_path).expanduser().resolve(strict=True)
            if not source.is_file():
                raise ValueError("sourcePath is not a file")
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        elif content_base64:
            target.write_bytes(_decode_upload_content(str(content_base64)))
        else:
            raise ValueError("sourcePath or contentBase64 is required")
    except (OSError, ValueError) as exc:
        return 400, _error(str(exc), "MONEYPRINTER_LOCAL_MATERIAL_INVALID")

    return 200, _envelope({"material": _local_material_payload(target)})


def _write_uploaded_asset(body: dict[str, Any], target: Path) -> None:
    source_path = str(body.get("sourcePath") or body.get("source_path") or "").strip()
    content_base64 = body.get("contentBase64") or body.get("content_base64")
    if source_path:
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("sourcePath is not a file")
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    elif content_base64:
        target.write_bytes(_decode_upload_content(str(content_base64)))
    else:
        raise ValueError("sourcePath or contentBase64 is required")


def list_assets_data() -> tuple[int, dict[str, Any]]:
    bgms = []
    if SONGS_DIR.is_dir():
        bgms = [
            _audio_asset_payload(path)
            for path in sorted(SONGS_DIR.iterdir(), key=lambda p: p.name.lower())
            if path.is_file() and path.suffix.lower() in SUPPORTED_BGM_EXTS
        ]

    CUSTOM_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    custom_audio = [
        _audio_asset_payload(path, file_value=_custom_audio_relative_path(path))
        for path in sorted(CUSTOM_AUDIO_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]

    fonts = []
    if FONTS_DIR.is_dir():
        fonts = [
            _font_asset_payload(path)
            for path in sorted(FONTS_DIR.iterdir(), key=lambda p: p.name.lower())
            if path.is_file() and path.suffix.lower() in SUPPORTED_FONT_EXTS
        ]

    voices = [
        "no-voice",
        "zh-CN-XiaoxiaoNeural-Female",
        "zh-CN-XiaoyiNeural-Female",
        "zh-CN-YunjianNeural-Male",
        "zh-CN-YunxiNeural-Male",
        "zh-CN-YunxiaNeural-Male",
        "zh-CN-YunyangNeural-Male",
        "en-US-JennyNeural-Female",
        "en-US-GuyNeural-Male",
        "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male",
        "siliconflow:FunAudioLLM/CosyVoice2-0.5B:anna-Female",
        "gemini:Zephyr-Female",
        "gemini:Puck-Male",
        "mimo:mimo_default-Female",
        "chatterbox:default-Female",
    ]
    voices.extend(_local_minimax_voices())

    return 200, _envelope({"bgms": bgms, "customAudio": custom_audio, "fonts": fonts, "voices": voices})


async def list_minimax_voices_data() -> tuple[int, dict[str, Any]]:
    status, upstream = await _proxy_json("GET", "/api/v1/minimax/voices?voice_type=all")
    if status >= 400:
        local = [
            {
                "category": "local_preview",
                "id": value.split(":", 2)[1],
                "name": value.split(":", 2)[-1],
                "providerConfirmed": False,
            }
            for value in _local_minimax_voices()
        ]
        if local:
            return 200, _envelope({"voices": local})
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_MINIMAX_VOICES_FAILED")
    data = _unwrap_upstream(upstream)
    voices = data.get("voices", []) if isinstance(data, dict) else []
    return 200, _envelope({"voices": voices})


def upload_bgm_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        filename = _sanitize_bgm_filename(
            str(body.get("filename") or body.get("name") or body.get("sourcePath") or body.get("source_path") or "")
        )
        target = SONGS_DIR / filename
        SONGS_DIR.mkdir(parents=True, exist_ok=True)
        _write_uploaded_asset(body, target)
    except (OSError, ValueError) as exc:
        return 400, _error(str(exc), "MONEYPRINTER_BGM_INVALID")

    return 200, _envelope({"bgm": _audio_asset_payload(target)})


def upload_custom_audio_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        filename = _sanitize_custom_audio_filename(
            str(body.get("filename") or body.get("name") or body.get("sourcePath") or body.get("source_path") or "")
        )
        target = CUSTOM_AUDIO_DIR / filename
        CUSTOM_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        _write_uploaded_asset(body, target)
    except (OSError, ValueError) as exc:
        return 400, _error(str(exc), "MONEYPRINTER_CUSTOM_AUDIO_INVALID")

    return 200, _envelope({"audio": _audio_asset_payload(target, file_value=_custom_audio_relative_path(target))})


async def _proxy_json(
    method: str,
    upstream_path: str,
    body: Optional[dict[str, Any]] = None,
    *,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> tuple[int, Any]:
    if ClientSession is None:
        return 503, _error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE")
    try:
        async with ClientSession() as session:
            async with session.request(
                method,
                f"{DEFAULT_BASE_URL}{upstream_path}",
                headers={SIDECAR_TOKEN_HEADER: _SIDECAR_TOKEN},
                json=body,
                timeout=ClientTimeout(total=timeout_seconds),
            ) as response:
                try:
                    payload = await response.json()
                except Exception:
                    payload = {"message": await response.text()}
                return response.status, payload
    except Exception as exc:
        return 503, _error(str(exc), "MONEYPRINTER_UPSTREAM_UNREACHABLE")


def _default_create_video_body(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize create-video payload with safe MoneyPrinter defaults."""
    payload = dict(body or {})
    defaults = {
        "video_subject": "",
        "video_script": "",
        "video_language": "zh-CN",
        "video_aspect": "9:16",
        "video_concat_mode": "random",
        "video_transition_mode": None,
        "video_count": 1,
        "video_clip_duration": 5,
        "video_source": "pexels",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        "voice_rate": 1.0,
        "voice_volume": 1.0,
        "subtitle_enabled": True,
        "subtitle_position": "bottom",
        "bgm_type": "random",
        "bgm_file": "",
        "bgm_volume": 0.2,
        "custom_audio_file": "",
        "font_name": "STHeitiMedium.ttc",
        "font_size": 60,
        "text_fore_color": "#FFFFFF",
        "text_background_color": True,
        "rounded_subtitle_background": False,
        "custom_position": 70.0,
        "stroke_color": "#000000",
        "stroke_width": 1.5,
        "match_materials_to_script": False,
        "paragraph_number": 1,
        "video_script_prompt": "",
        "custom_system_prompt": "",
    }
    for key, value in defaults.items():
        if key not in payload or payload[key] in (None, ""):
            if key == "video_subject":
                continue
            payload[key] = value
    if not str(payload.get("video_subject") or "").strip():
        raise ValueError("video_subject is required")
    payload["video_subject"] = str(payload["video_subject"]).strip()
    if payload.get("video_source") == "local":
        materials = _normalize_local_video_materials(payload.get("video_materials"))
        if not materials:
            raise ValueError("at least one local material is required when video_source is local")
        payload["video_materials"] = materials
    if payload.get("bgm_file"):
        payload["bgm_file"] = _normalize_bgm_file(payload.get("bgm_file"))
    if payload.get("custom_audio_file"):
        payload["custom_audio_file"] = _normalize_custom_audio_file(payload.get("custom_audio_file"))
    return payload


async def create_video_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        payload = _default_create_video_body(body)
    except ValueError as exc:
        return 400, _error(str(exc), "MONEYPRINTER_INVALID_PARAMS")

    status, upstream = await _proxy_json("POST", "/api/v1/videos", payload)
    if status >= 400:
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_CREATE_FAILED")

    data = _unwrap_upstream(upstream)
    task = _as_task(data if isinstance(data, dict) else {})
    return 200, _envelope({"task": task})


def _default_audio_subtitle_body(body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body or {})
    defaults = {
        "video_language": "zh-CN",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        "voice_rate": 1.0,
        "voice_volume": 1.0,
        "custom_audio_file": "",
        "bgm_type": "random",
        "bgm_file": "",
        "bgm_volume": 0.2,
        "video_source": "local",
        "subtitle_position": "bottom",
        "font_name": "STHeitiMedium.ttc",
        "font_size": 60,
        "text_fore_color": "#FFFFFF",
        "text_background_color": True,
        "rounded_subtitle_background": False,
        "custom_position": 70.0,
        "stroke_color": "#000000",
        "stroke_width": 1.5,
        "subtitle_enabled": True,
    }
    for key, value in defaults.items():
        if key not in payload or payload[key] in (None, ""):
            payload[key] = value
    if not str(payload.get("video_script") or "").strip():
        raise ValueError("video_script is required")
    payload["video_script"] = str(payload["video_script"]).strip()
    if payload.get("bgm_file"):
        payload["bgm_file"] = _normalize_bgm_file(payload.get("bgm_file"))
    if payload.get("custom_audio_file"):
        payload["custom_audio_file"] = _normalize_custom_audio_file(payload.get("custom_audio_file"))
    return payload


def _subtitle_enabled_for_upstream(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return "true"
        if normalized in {"0", "false", "no", "off"}:
            return "false"
        return value
    return "true" if bool(value) else "false"


async def create_audio_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        payload = _default_audio_subtitle_body(body)
    except ValueError as exc:
        return 400, _error(str(exc), "MONEYPRINTER_INVALID_PARAMS")

    status, upstream = await _proxy_json("POST", "/api/v1/audio", payload)
    if status >= 400:
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_AUDIO_FAILED")

    data = _unwrap_upstream(upstream)
    task = _as_task(data if isinstance(data, dict) else {})
    return 200, _envelope({"task": task})


async def create_subtitle_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        payload = _default_audio_subtitle_body(body)
    except ValueError as exc:
        return 400, _error(str(exc), "MONEYPRINTER_INVALID_PARAMS")

    # MoneyPrinterTurbo accepts booleans for full /videos generation, but the
    # staged /subtitle request schema is typed as Optional[str]. Keep /videos
    # unchanged and stringify only at this endpoint boundary.
    payload["subtitle_enabled"] = _subtitle_enabled_for_upstream(payload.get("subtitle_enabled"))

    status, upstream = await _proxy_json("POST", "/api/v1/subtitle", payload)
    if status >= 400:
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_SUBTITLE_FAILED")

    data = _unwrap_upstream(upstream)
    task = _as_task(data if isinstance(data, dict) else {})
    return 200, _envelope({"task": task})


async def list_tasks_data() -> tuple[int, dict[str, Any]]:
    status, payload = await _proxy_json("GET", "/api/v1/tasks")
    if status >= 400:
        disk_tasks = _recover_disk_tasks()
        if disk_tasks:
            return 200, _envelope({"tasks": disk_tasks})
        if isinstance(payload, dict) and payload.get("ok") is False:
            return status, payload
        return status, _error(str(payload), "MONEYPRINTER_LIST_TASKS_FAILED")
    data = _unwrap_upstream(payload)
    raw_tasks = data.get("tasks", []) if isinstance(data, dict) else []
    upstream_tasks = [_as_task(task) for task in raw_tasks if isinstance(task, dict)]
    return 200, _envelope({"tasks": _merge_tasks(upstream_tasks, _recover_disk_tasks())})


async def get_task_data(task_id: str) -> tuple[int, dict[str, Any]]:
    status, payload = await _proxy_json("GET", f"/api/v1/tasks/{task_id}")
    if status >= 400:
        disk_task = _task_from_disk(TASKS_DIR / task_id)
        if disk_task:
            return 200, _envelope(disk_task)
        if isinstance(payload, dict) and payload.get("ok") is False:
            return status, payload
        return status, _error(str(payload), "MONEYPRINTER_GET_TASK_FAILED")
    data = _unwrap_upstream(payload)
    return 200, _envelope(_as_task(data if isinstance(data, dict) else {}))


async def delete_task_data(task_id: str) -> tuple[int, dict[str, Any]]:
    status, payload = await _proxy_json("DELETE", f"/api/v1/tasks/{task_id}")
    if status >= 400:
        task_dir = TASKS_DIR / task_id
        if task_dir.is_dir():
            shutil.rmtree(task_dir)
            return 200, _envelope({"taskId": task_id})
        if isinstance(payload, dict) and payload.get("ok") is False:
            return status, payload
        return status, _error(str(payload), "MONEYPRINTER_DELETE_TASK_FAILED")
    return 200, _envelope({"taskId": task_id})


async def generate_script_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    payload = {
        "video_subject": str((body or {}).get("video_subject") or "").strip() or "春天的花海",
        "video_language": str((body or {}).get("video_language") or "zh-CN"),
        "paragraph_number": int((body or {}).get("paragraph_number") or 1),
        "video_script_prompt": str((body or {}).get("video_script_prompt") or ""),
        "custom_system_prompt": str((body or {}).get("custom_system_prompt") or ""),
    }
    status, upstream = await _proxy_json("POST", "/api/v1/scripts", payload)
    if status >= 400:
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_SCRIPT_FAILED")
    data = _unwrap_upstream(upstream)
    if isinstance(data, dict):
        return 200, _envelope(data)
    return 200, _envelope({"video_script": data})


async def generate_terms_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    payload = {
        "video_subject": str((body or {}).get("video_subject") or "").strip() or "春天的花海",
        "video_script": str((body or {}).get("video_script") or ""),
        "amount": int((body or {}).get("amount") or 5),
        "match_materials_to_script": bool((body or {}).get("match_materials_to_script") or False),
    }
    status, upstream = await _proxy_json("POST", "/api/v1/terms", payload)
    if status >= 400:
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), "MONEYPRINTER_TERMS_FAILED")
    data = _unwrap_upstream(upstream)
    if isinstance(data, dict):
        return 200, _envelope(data)
    return 200, _envelope({"video_terms": data})


async def clone_minimax_voice_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    try:
        payload = dict(body or {})
        payload["clone_audio"] = _materialize_minimax_audio_asset(payload.get("clone_audio"), "clone audio")
        if payload.get("prompt_audio"):
            payload["prompt_audio"] = _materialize_minimax_audio_asset(payload["prompt_audio"], "prompt audio")
    except (OSError, ValueError) as exc:
        return 400, _error(str(exc), "MONEYPRINTER_MINIMAX_AUDIO_INVALID")

    return await _proxy_minimax_data(
        "/api/v1/minimax/voices/clone", payload, "MONEYPRINTER_MINIMAX_CLONE_FAILED"
    )


async def _proxy_minimax_data(
    path: str, body: dict[str, Any], error_code: str
) -> tuple[int, dict[str, Any]]:
    status, upstream = await _proxy_json(
        "POST",
        path,
        body,
        timeout_seconds=MINIMAX_PROXY_TIMEOUT_SECONDS[path],
    )
    if status >= 400:
        message = str(upstream.get("message") if isinstance(upstream, dict) else upstream)
        if "voice id duplicate" in message.lower():
            return 409, _error(
                "该 ID 已存在；请在已有音色中使用它，或为克隆生成新的 ID。",
                "MONEYPRINTER_MINIMAX_VOICE_ID_DUPLICATE",
            )
        if isinstance(upstream, dict) and upstream.get("ok") is False:
            return status, upstream
        return status, _error(str(upstream), error_code)
    data = _unwrap_upstream(upstream)
    if isinstance(data, dict):
        audio = data.get("audio")
        if isinstance(audio, dict):
            descriptor = _minimax_audio_descriptor(audio.get("file"))
            if descriptor:
                data["audio"] = {**audio, **descriptor}
        trial_descriptor = _minimax_audio_descriptor(data.get("trialAudioFile"))
        if trial_descriptor:
            data["trialAudio"] = trial_descriptor
    return 200, _envelope(data if isinstance(data, dict) else {"result": data})


async def generate_minimax_tts_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return await _proxy_minimax_data(
        "/api/v1/minimax/tts", dict(body or {}), "MONEYPRINTER_MINIMAX_TTS_FAILED"
    )


async def generate_minimax_lyrics_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return await _proxy_minimax_data(
        "/api/v1/minimax/lyrics", dict(body or {}), "MONEYPRINTER_MINIMAX_LYRICS_FAILED"
    )


async def generate_minimax_music_data(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return await _proxy_minimax_data(
        "/api/v1/minimax/music", dict(body or {}), "MONEYPRINTER_MINIMAX_MUSIC_FAILED"
    )


async def list_outputs_data() -> tuple[int, dict[str, Any]]:
    status, payload = await list_tasks_data()
    if status >= 400 or not payload.get("ok"):
        return status, payload
    tasks = (payload.get("data") or {}).get("tasks") or []
    outputs = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for video in task.get("videos") or []:
            if not isinstance(video, dict):
                continue
            outputs.append(
                {
                    "taskId": task.get("id"),
                    "state": task.get("state"),
                    "subject": task.get("subject"),
                    "name": video.get("name"),
                    "file": video.get("file"),
                    "streamUrl": video.get("streamUrl"),
                    "downloadUrl": video.get("downloadUrl"),
                }
            )
    return 200, _envelope({"outputs": outputs, "total": len(outputs)})


def _normalize_media_path(file_path: str) -> str:
    """Normalize upstream/Desktop media paths to task-dir-relative paths.

    MoneyPrinterTurbo's task status currently returns output paths like
    ``/tasks/<task_id>/final-1.mp4`` while its own stream/download endpoints
    resolve paths relative to ``storage/tasks`` and therefore expect
    ``<task_id>/final-1.mp4``. Accept both forms so Desktop preview/download
    links keep working across upstream variants.
    """
    raw_path = str(file_path or "").strip()
    if "://" in raw_path:
        raw_path = urlparse(raw_path).path

    try:
        raw_file = Path(raw_path).expanduser()
        if raw_file.is_absolute():
            return raw_file.resolve(strict=False).relative_to(TASKS_DIR.resolve(strict=False)).as_posix()
    except Exception:
        pass

    safe_path = raw_path.lstrip("/")

    for prefix in (
        "api/capabilities/moneyprinter/stream/",
        "api/capabilities/moneyprinter/download/",
        "api/v1/stream/",
        "api/v1/download/",
    ):
        if safe_path.startswith(prefix):
            safe_path = safe_path[len(prefix) :]
            break

    if safe_path.startswith("tasks/"):
        safe_path = safe_path[len("tasks/") :]
    return safe_path


async def create_video(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await create_video_data(body)
    return _json(payload, status=status)


async def create_audio(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await create_audio_data(body)
    return _json(payload, status=status)


async def create_subtitle(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await create_subtitle_data(body)
    return _json(payload, status=status)


async def list_local_materials() -> Any:
    status, payload = list_local_materials_data()
    return _json(payload, status=status)


async def list_assets() -> Any:
    status, payload = list_assets_data()
    return _json(payload, status=status)


async def upload_local_material(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = upload_local_material_data(body)
    return _json(payload, status=status)


async def upload_bgm(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = upload_bgm_data(body)
    return _json(payload, status=status)


async def upload_custom_audio(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = upload_custom_audio_data(body)
    return _json(payload, status=status)


async def list_tasks() -> Any:
    status, payload = await list_tasks_data()
    return _json(payload, status=status)


async def get_task(task_id: str) -> Any:
    status, payload = await get_task_data(task_id)
    return _json(payload, status=status)


async def delete_task(task_id: str) -> Any:
    status, payload = await delete_task_data(task_id)
    return _json(payload, status=status)


async def generate_script(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await generate_script_data(body)
    return _json(payload, status=status)


async def generate_terms(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await generate_terms_data(body)
    return _json(payload, status=status)


async def clone_minimax_voice(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await clone_minimax_voice_data(body)
    return _json(payload, status=status)


async def list_minimax_voices() -> Any:
    status, payload = await list_minimax_voices_data()
    return _json(payload, status=status)


async def generate_minimax_tts(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await generate_minimax_tts_data(body)
    return _json(payload, status=status)


async def generate_minimax_lyrics(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await generate_minimax_lyrics_data(body)
    return _json(payload, status=status)


async def generate_minimax_music(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)
    if not isinstance(body, dict):
        return _json(_error("Body must be an object", "MONEYPRINTER_INVALID_JSON"), status=400)
    status, payload = await generate_minimax_music_data(body)
    return _json(payload, status=status)


def _local_file_response(kind: str, candidate: Path, content_type: str, request: Any = None) -> Any:
    if web is None:
        return None
    if not candidate.is_file():
        return None

    payload = candidate.read_bytes()
    total = len(payload)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Type": content_type,
    }
    if kind == "download":
        headers["Content-Disposition"] = f'attachment; filename="{candidate.name}"'

    request_headers = getattr(request, "headers", None)
    range_header = str(request_headers.get("Range") or "").strip() if request_headers is not None else ""
    if not range_header:
        return web.Response(body=payload, headers=headers, status=200)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
    if match is None or (not match.group(1) and not match.group(2)) or total == 0:
        return web.Response(headers={**headers, "Content-Range": f"bytes */{total}"}, status=416)

    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else total - 1
    else:
        suffix_length = int(match.group(2))
        start = max(total - suffix_length, 0)
        end = total - 1
    if start >= total or start > end:
        return web.Response(headers={**headers, "Content-Range": f"bytes */{total}"}, status=416)

    end = min(end, total - 1)
    partial = payload[start : end + 1]
    headers.update(
        {
            "Content-Length": str(len(partial)),
            "Content-Range": f"bytes {start}-{end}/{total}",
        }
    )
    return web.Response(body=partial, headers=headers, status=206)


def _local_media_response(kind: str, safe_path: str, request: Any = None) -> Any:
    parts = Path(safe_path).parts
    if len(parts) == 2 and PUBLIC_VIDEO_OUTPUT_RE.fullmatch(parts[1]):
        root = TASKS_DIR.resolve(strict=False)
        candidate = (root / Path(*parts)).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return _local_file_response(kind, candidate, "video/mp4", request)

    audio_roots = {
        ("storage", "custom_audio"): MONEYPRINTER_ROOT / "storage" / "custom_audio",
        ("storage", "minimax", "tts"): MONEYPRINTER_ROOT / "storage" / "minimax" / "tts",
        ("storage", "minimax", "voices"): MONEYPRINTER_ROOT / "storage" / "minimax" / "voices",
    }
    for prefix, root_value in audio_roots.items():
        if parts[: len(prefix)] != prefix or len(parts) <= len(prefix):
            continue
        root = root_value.resolve(strict=False)
        candidate = (root / Path(*parts[len(prefix) :])).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        content_type = {
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
        }.get(candidate.suffix.lower())
        if content_type is None:
            return None
        return _local_file_response(kind, candidate, content_type, request)
    return None


async def proxy_media(kind: str, file_path: str, request: Any = None) -> Any:
    if kind not in {"download", "stream"}:
        return _json(_error("Unsupported media proxy kind", "MONEYPRINTER_BAD_MEDIA_KIND"), status=400)

    safe_path = _normalize_media_path(file_path)
    if not safe_path or ".." in Path(safe_path).parts:
        return _json(_error("Invalid media path", "MONEYPRINTER_INVALID_MEDIA_PATH"), status=400)

    local_response = _local_media_response(kind, safe_path, request)
    if local_response is not None:
        return local_response

    if ClientSession is None:
        return _json(_error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE"), status=503)

    upstream_path = quote(safe_path, safe="/")
    try:
        async with ClientSession() as session:
            async with session.get(
                f"{DEFAULT_BASE_URL}/api/v1/{kind}/{upstream_path}",
                headers=_media_proxy_request_headers(request),
                timeout=None,
            ) as response:
                body = await response.read()
                headers = {}
                for name in MEDIA_PROXY_HEADERS:
                    value = response.headers.get(name)
                    if value:
                        headers[name] = value
                return web.Response(body=body, headers=headers, status=response.status)
    except Exception as exc:
        return _json(_error(str(exc), "MONEYPRINTER_MEDIA_PROXY_FAILED"), status=503)
