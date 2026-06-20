from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent.tts_provider import TTSProvider


PLUGIN_DIR = Path(__file__).resolve().parent
HERMES_ROOT = PLUGIN_DIR.parents[1]
DEFAULT_IRODORI_REPO_DIR = HERMES_ROOT.parent / "irodori-tts-server"
DEFAULT_BASE_URL = "http://127.0.0.1:8088"
DEFAULT_MODEL = "irodori-tts"
DEFAULT_VOICE = "none"
DEFAULT_SPEED = 1.0
DEFAULT_TIMEOUT = 900
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus", "aac", "pcm"}


@dataclass(frozen=True)
class IrodoriSettings:
    repo_dir: Path
    start_script: Path
    invoke_script: Path
    base_url: str
    model: str
    voice: str
    speed: float
    timeout: int


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def settings() -> IrodoriSettings:
    repo_dir = _env_path("IRODORI_TTS_REPO_DIR", DEFAULT_IRODORI_REPO_DIR)
    start_script = _env_path(
        "IRODORI_TTS_START_SCRIPT",
        HERMES_ROOT / "scripts" / "windows" / "start-irodori-tts.ps1",
    )
    invoke_script = _env_path(
        "IRODORI_TTS_INVOKE_SCRIPT",
        HERMES_ROOT / "scripts" / "windows" / "invoke-irodori-tts.ps1",
    )
    base_url = os.environ.get("IRODORI_TTS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("IRODORI_TTS_MODEL", DEFAULT_MODEL)
    voice = os.environ.get("IRODORI_TTS_VOICE", DEFAULT_VOICE)
    speed = _float_env("IRODORI_TTS_SPEED", DEFAULT_SPEED)
    timeout = _int_env("IRODORI_TTS_TIMEOUT", DEFAULT_TIMEOUT)
    return IrodoriSettings(
        repo_dir=repo_dir,
        start_script=start_script,
        invoke_script=invoke_script,
        base_url=base_url,
        model=model,
        voice=voice,
        speed=speed,
        timeout=timeout,
    )


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def powershell_path() -> str | None:
    return shutil.which("powershell") or shutil.which("pwsh")


def server_health(base_url: str | None = None, timeout: float = 3.0) -> dict[str, Any]:
    endpoint = f"{(base_url or settings().base_url).rstrip('/')}/health"
    request = Request(endpoint, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            return {
                "ok": 200 <= response.status < 300,
                "status_code": response.status,
                "endpoint": endpoint,
                "payload": payload,
            }
    except HTTPError as exc:
        return {
            "ok": False,
            "status_code": exc.code,
            "endpoint": endpoint,
            "error": str(exc),
        }
    except (OSError, URLError) as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "error": str(exc),
        }


def status_payload() -> dict[str, Any]:
    cfg = settings()
    ps = powershell_path()
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    health = server_health(cfg.base_url)
    checks = {
        "repo_dir": cfg.repo_dir.exists(),
        "start_script": cfg.start_script.exists(),
        "invoke_script": cfg.invoke_script.exists(),
        "powershell": bool(ps),
        "curl": bool(curl_path),
    }
    return {
        "ok": all(checks.values()),
        "provider": "irodori",
        "available": all(checks.values()),
        "server": health,
        "checks": checks,
        "paths": {
            "repo_dir": str(cfg.repo_dir),
            "start_script": str(cfg.start_script),
            "invoke_script": str(cfg.invoke_script),
            "powershell": ps,
            "curl": curl_path,
        },
        "defaults": {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "voice": cfg.voice,
            "speed": cfg.speed,
            "timeout": cfg.timeout,
        },
    }


def list_local_voices() -> list[dict[str, str]]:
    cfg = settings()
    voices = [{"id": DEFAULT_VOICE, "name": "Default Irodori voice"}]
    voices_dir = cfg.repo_dir / "voices"
    if voices_dir.exists():
        for path in sorted(voices_dir.glob("*.toml")):
            voice_id = path.stem
            if voice_id == DEFAULT_VOICE:
                continue
            voices.append({"id": voice_id, "name": voice_id})
    return voices


def normalize_format(output_format: str | None, output_path: str | Path | None = None) -> str:
    candidate = (output_format or "").strip().lower()
    if not candidate and output_path:
        suffix = Path(output_path).suffix.lower().lstrip(".")
        candidate = suffix
    if candidate in SUPPORTED_FORMATS:
        return candidate
    return "wav"


def default_output_path(output_format: str = "wav") -> Path:
    cache_dir = Path(os.environ.get("HERMES_AUDIO_CACHE_DIR", Path.home() / "AppData" / "Local" / "hermes" / "audio_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return cache_dir / f"irodori_plugin_{stamp}.{output_format}"


def resolved_output_path(output_path: str | Path | None, output_format: str) -> Path:
    path = Path(output_path).expanduser() if output_path else default_output_path(output_format)
    if path.suffix.lower().lstrip(".") != output_format:
        path = path.with_suffix(f".{output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def synthesize_text(
    text: str,
    output_path: str | Path | None = None,
    voice: str | None = None,
    model: str | None = None,
    output_format: str | None = None,
    speed: float | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    cfg = settings()
    fmt = normalize_format(output_format, output_path)
    destination = resolved_output_path(output_path, fmt)
    voice_id = voice or cfg.voice
    model_id = model or cfg.model
    speed_value = cfg.speed if speed is None else float(speed)
    ps = powershell_path()
    if not ps:
        raise RuntimeError("PowerShell was not found on PATH")
    if not cfg.invoke_script.exists():
        raise FileNotFoundError(f"Irodori invoke script not found: {cfg.invoke_script}")
    if not cfg.repo_dir.exists():
        raise FileNotFoundError(f"Irodori repository not found: {cfg.repo_dir}")

    with tempfile.TemporaryDirectory(prefix="hermes-irodori-") as temp_dir:
        input_path = Path(temp_dir) / "input.txt"
        input_path.write_text(text, encoding="utf-8")
        command = [
            ps,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(cfg.invoke_script),
            "-InputPath",
            str(input_path),
            "-OutputPath",
            str(destination),
            "-Format",
            fmt,
            "-Voice",
            voice_id,
            "-Model",
            model_id,
            "-Speed",
            str(speed_value),
            "-BaseUrl",
            cfg.base_url,
        ]
        completed = subprocess.run(
            command,
            cwd=str(cfg.repo_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=cfg.timeout,
        )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Irodori TTS script failed: {detail}")
    if not destination.exists() or destination.stat().st_size <= 0:
        raise RuntimeError(f"Irodori TTS script did not create audio: {destination}")

    return {
        "ok": True,
        "provider": "irodori",
        "file_path": str(destination),
        "format": fmt,
        "voice": voice_id,
        "model": model_id,
        "speed": speed_value,
        "media_tag": f"MEDIA:{destination}",
    }


class IrodoriScriptTTSProvider(TTSProvider):
    name = "irodori"
    display_name = "Irodori TTS"
    voice_compatible = True

    def is_available(self) -> bool:
        payload = status_payload()
        return bool(payload["available"])

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Irodori TTS",
            "badge": "local · free",
            "tag": "Local Japanese TTS through the Irodori script harness",
            "env_vars": [],
        }

    def list_voices(self) -> list[dict[str, str]]:
        return list_local_voices()

    def default_voice(self) -> str | None:
        return settings().voice

    def list_models(self) -> list[dict[str, Any]]:
        cfg = settings()
        return [
            {
                "id": cfg.model,
                "name": "Irodori TTS",
                "max_text_length": 4096,
            }
        ]

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str | None = None,
        **_: Any,
    ) -> str:
        result = synthesize_text(
            text=text,
            output_path=output_path,
            voice=voice,
            model=model,
            speed=speed,
            output_format=format,
        )
        return str(result["file_path"])
