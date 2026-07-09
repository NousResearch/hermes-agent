"""Hermes Desktop adapter for the vendored MoneyPrinterTurbo capability.

This module keeps the Video Studio integration outside Hermes Agent Core. It is
used by the Desktop API server as a thin service/proxy layer around the upstream
MoneyPrinterTurbo FastAPI app.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import sys
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
CONFIG_PATH = MONEYPRINTER_ROOT / "config.toml"
CONFIG_EXAMPLE_PATH = MONEYPRINTER_ROOT / "config.example.toml"
TASKS_DIR = MONEYPRINTER_ROOT / "storage" / "tasks"
LOCAL_MATERIALS_DIR = MONEYPRINTER_ROOT / "storage" / "local_videos"
SUPPORTED_LOCAL_MATERIAL_EXTS = {".avi", ".flv", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png"}
PUBLIC_VIDEO_OUTPUT_RE = re.compile(r"^(?:combined|final)-\d+\.mp4$", re.IGNORECASE)
MEDIA_PROXY_HEADERS = ("content-type", "content-length", "content-disposition", "accept-ranges", "content-range")

_process: Optional[asyncio.subprocess.Process] = None


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
    venv_python = MONEYPRINTER_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


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
    return env


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
    }


def _read_config_text() -> str:
    if CONFIG_PATH.exists():
        return CONFIG_PATH.read_text(encoding="utf-8")
    if CONFIG_EXAMPLE_PATH.exists():
        return CONFIG_EXAMPLE_PATH.read_text(encoding="utf-8")
    return "[app]\n"


def _replace_scalar(text: str, key: str, value: str) -> str:
    escaped = value.replace('"', '\\"')
    line = f'{key} = "{escaped}"'
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(line, text, count=1)
    return f"{text.rstrip()}\n{line}\n"


def _replace_list(text: str, key: str, values: list[str]) -> str:
    rendered = ", ".join(f'"{value.replace(chr(34), chr(92) + chr(34))}"' for value in values if value)
    line = f"{key} = [{rendered}]"
    pattern = re.compile(rf"^{re.escape(key)}\s*=\s*\[.*?\]", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(line, text, count=1)
    return f"{text.rstrip()}\n{line}\n"


def _has_config_value(text: str, key: str) -> bool:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return False
    value = match.group(1).strip()
    return value not in {'""', "''", "[]"}


def _config_summary() -> dict[str, Any]:
    text = _read_config_text()
    provider = "openai"
    provider_match = re.search(r'^llm_provider\s*=\s*["\']([^"\']*)["\']', text, flags=re.MULTILINE)
    if provider_match:
        provider = provider_match.group(1) or provider
    return {
        "configExists": CONFIG_PATH.exists(),
        "llmProvider": provider,
        "materialProviders": {
            "coverr": _has_config_value(text, "coverr_api_keys"),
            "pexels": _has_config_value(text, "pexels_api_keys"),
            "pixabay": _has_config_value(text, "pixabay_api_keys"),
        },
        "modelConfigured": any(
            _has_config_value(text, key)
            for key in ("openai_api_key", "gemini_api_key", "qwen_api_key", "deepseek_api_key", "grok_api_key")
        ),
    }


async def health() -> Any:
    if ClientSession is None:
        return _json(_error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE"), status=503)

    if not _is_installed():
        return _json(_envelope(_health_payload(service_running=False, message="MoneyPrinterTurbo is not vendored.")))

    try:
        async with ClientSession() as session:
            async with session.get(f"{DEFAULT_BASE_URL}/docs", timeout=ClientTimeout(total=3)) as response:
                if response.status < 500:
                    return _json(_envelope(_health_payload(service_running=True, message="MoneyPrinter API is reachable.")))
    except Exception:
        pass

    return _json(_envelope(_health_payload(service_running=_service_running(), message="MoneyPrinter API is not reachable yet.")))


async def get_config() -> Any:
    return _json(_envelope(_config_summary()))


async def save_config(request: Any) -> Any:
    try:
        body = await request.json()
    except Exception:
        return _json(_error("Invalid JSON body", "MONEYPRINTER_INVALID_JSON"), status=400)

    if not isinstance(body, dict):
        return _json(_error("Config body must be an object", "MONEYPRINTER_INVALID_CONFIG"), status=400)

    if not CONFIG_PATH.exists() and CONFIG_EXAMPLE_PATH.exists():
        shutil.copyfile(CONFIG_EXAMPLE_PATH, CONFIG_PATH)

    text = _read_config_text()
    provider = str(body.get("llmProvider") or "openai").strip()
    if provider:
        text = _replace_scalar(text, "llm_provider", provider)

    for body_key, config_key in {
        "apiKey": f"{provider}_api_key",
        "baseUrl": f"{provider}_base_url",
        "modelName": f"{provider}_model_name",
    }.items():
        value = str(body.get(body_key) or "").strip()
        if value:
            text = _replace_scalar(text, config_key, value)

    for body_key, config_key in (
        ("pexelsApiKey", "pexels_api_keys"),
        ("pixabayApiKey", "pixabay_api_keys"),
        ("coverrApiKey", "coverr_api_keys"),
    ):
        value = str(body.get(body_key) or "").strip()
        if value:
            text = _replace_list(text, config_key, [value])

    CONFIG_PATH.write_text(text, encoding="utf-8")
    return _json(_envelope(_config_summary()))


async def start_service() -> Any:
    global _process

    if not _is_installed():
        return _json(_error("external/MoneyPrinterTurbo is missing.", "MONEYPRINTER_NOT_INSTALLED"), status=404)

    if _service_running():
        return _json(_envelope(_health_payload(service_running=True, message="MoneyPrinter service is already running.")))

    # If an external process already owns the API port, treat as running.
    if ClientSession is not None and ClientTimeout is not None:
        try:
            async with ClientSession() as session:
                async with session.get(f"{DEFAULT_BASE_URL}/docs", timeout=ClientTimeout(total=2)) as response:
                    if response.status < 500:
                        return _json(
                            _envelope(
                                _health_payload(
                                    service_running=True,
                                    message="MoneyPrinter API is already reachable (external process).",
                                )
                            )
                        )
        except Exception:
            pass

    try:
        _process = await asyncio.create_subprocess_exec(
            _moneyprinter_python(),
            "main.py",
            cwd=str(MONEYPRINTER_ROOT),
            env=_build_service_env(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except Exception as exc:
        return _json(_error(str(exc), "MONEYPRINTER_START_FAILED"), status=500)

    return _json(_envelope(_health_payload(service_running=True, message="MoneyPrinter service start requested.")))


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


def _decode_upload_content(value: str) -> bytes:
    content = str(value or "")
    if "," in content and content.lstrip().lower().startswith("data:"):
        content = content.split(",", 1)[1]
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("contentBase64 is not valid base64") from exc


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


def _unwrap_upstream(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


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


async def _proxy_json(method: str, upstream_path: str, body: Optional[dict[str, Any]] = None) -> tuple[int, Any]:
    if ClientSession is None:
        return 503, _error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE")
    try:
        async with ClientSession() as session:
            async with session.request(
                method,
                f"{DEFAULT_BASE_URL}{upstream_path}",
                json=body,
                timeout=ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
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
        "video_count": 1,
        "video_clip_duration": 5,
        "video_source": "pexels",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        "voice_rate": 1.0,
        "voice_volume": 1.0,
        "subtitle_enabled": True,
        "subtitle_position": "bottom",
        "bgm_type": "random",
        "bgm_volume": 0.2,
        "font_name": "STHeitiMedium.ttc",
        "font_size": 60,
        "text_fore_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 1.5,
        "match_materials_to_script": False,
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


async def list_local_materials() -> Any:
    status, payload = list_local_materials_data()
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


async def proxy_media(kind: str, file_path: str) -> Any:
    if ClientSession is None:
        return _json(_error("aiohttp is unavailable", "MONEYPRINTER_AIOHTTP_UNAVAILABLE"), status=503)
    if kind not in {"download", "stream"}:
        return _json(_error("Unsupported media proxy kind", "MONEYPRINTER_BAD_MEDIA_KIND"), status=400)

    safe_path = _normalize_media_path(file_path)
    if not safe_path or ".." in Path(safe_path).parts:
        return _json(_error("Invalid media path", "MONEYPRINTER_INVALID_MEDIA_PATH"), status=400)

    upstream_path = quote(safe_path, safe="/")
    try:
        async with ClientSession() as session:
            async with session.get(f"{DEFAULT_BASE_URL}/api/v1/{kind}/{upstream_path}", timeout=None) as response:
                body = await response.read()
                headers = {}
                for name in MEDIA_PROXY_HEADERS:
                    value = response.headers.get(name)
                    if value:
                        headers[name] = value
                return web.Response(body=body, headers=headers, status=response.status)
    except Exception as exc:
        return _json(_error(str(exc), "MONEYPRINTER_MEDIA_PROXY_FAILED"), status=503)
