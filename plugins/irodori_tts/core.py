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
HAKUA_VOICE_ID = "hakua"
DEFAULT_SPEED = 1.0
DEFAULT_TIMEOUT = 900
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus", "aac", "pcm"}
VOICE_FILE_EXTENSIONS = (".toml", ".ogg", ".wav", ".mp3", ".flac", ".opus", ".aac")
HAKUA_REFERENCE_NAMES = ("hakua.ogg", "Hakua.ogg", "hakua.wav", "Hakua.wav")


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



def _load_tts_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    tts = config.get("tts", {})
    return tts if isinstance(tts, dict) else {}


def _irodori_config(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    tts = tts_config if isinstance(tts_config, dict) else _load_tts_section()
    section = tts.get("irodori", {})
    return section if isinstance(section, dict) else {}


def _path_value(value: Any, default: Path) -> Path:
    if value is None or value == "":
        return default
    return Path(str(value)).expanduser()


def hakua_reference_path(repo_dir: Path) -> Path | None:
    voices_dir = repo_dir / "voices"
    for name in HAKUA_REFERENCE_NAMES:
        candidate = voices_dir / name
        if candidate.is_file():
            return candidate
    return None


def resolve_default_voice(repo_dir: Path, configured_voice: str | None) -> str:
    voice = (configured_voice or "").strip()
    if voice and voice.lower() != DEFAULT_VOICE:
        return voice
    if hakua_reference_path(repo_dir):
        return HAKUA_VOICE_ID
    return DEFAULT_VOICE


def settings(tts_config: dict[str, Any] | None = None) -> IrodoriSettings:
    cfg = _irodori_config(tts_config)
    repo_dir = _path_value(
        cfg.get("repo_dir") or os.environ.get("IRODORI_TTS_REPO_DIR"),
        DEFAULT_IRODORI_REPO_DIR,
    )
    start_script = _path_value(
        cfg.get("start_script") or os.environ.get("IRODORI_TTS_START_SCRIPT"),
        HERMES_ROOT / "scripts" / "windows" / "start-irodori-tts.ps1",
    )
    invoke_script = _path_value(
        cfg.get("invoke_script") or os.environ.get("IRODORI_TTS_INVOKE_SCRIPT"),
        HERMES_ROOT / "scripts" / "windows" / "invoke-irodori-tts.ps1",
    )
    base_url = str(
        cfg.get("base_url")
        or cfg.get("url")
        or os.environ.get("IRODORI_TTS_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    model = str(
        cfg.get("model") or os.environ.get("IRODORI_TTS_MODEL") or DEFAULT_MODEL
    )
    configured_voice = cfg.get("voice") or os.environ.get("IRODORI_TTS_VOICE")
    voice = resolve_default_voice(
        repo_dir,
        str(configured_voice) if configured_voice else None,
    )
    speed = _float_value(
        cfg.get("speed", os.environ.get("IRODORI_TTS_SPEED")),
        DEFAULT_SPEED,
    )
    timeout = _int_value(
        cfg.get("timeout", os.environ.get("IRODORI_TTS_TIMEOUT")),
        DEFAULT_TIMEOUT,
    )
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


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    return _float_value(os.environ.get(name), default)


def _int_env(name: str, default: int) -> int:
    return _int_value(os.environ.get(name), default)


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


def status_payload(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = settings(tts_config)
    ps = powershell_path()
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    health = server_health(cfg.base_url)
    hakua_ref = hakua_reference_path(cfg.repo_dir)
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
        "hakua_ref": {
            "path": str(hakua_ref) if hakua_ref else None,
            "present": hakua_ref is not None,
            "default_voice": cfg.voice,
        },
        "defaults": {
            "base_url": cfg.base_url,
            "model": cfg.model,
            "voice": cfg.voice,
            "speed": cfg.speed,
            "timeout": cfg.timeout,
        },
    }


def list_local_voices(tts_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    cfg = settings(tts_config)
    voices = [{"id": DEFAULT_VOICE, "name": "Default Irodori voice", "display": "Default Irodori voice"}]
    seen = {DEFAULT_VOICE}
    voices_dir = cfg.repo_dir / "voices"
    if not voices_dir.exists():
        return voices

    for ext in VOICE_FILE_EXTENSIONS:
        for path in sorted(voices_dir.glob(f"*{ext}")):
            voice_id = path.stem
            key = voice_id.lower()
            if key in seen:
                continue
            seen.add(key)
            display = voice_id
            if key == HAKUA_VOICE_ID:
                display = "Hakua (reference clone voice)"
            voices.append({"id": voice_id, "name": voice_id, "display": display})
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
