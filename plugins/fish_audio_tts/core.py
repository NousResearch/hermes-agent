from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent.tts_provider import TTSProvider


DEFAULT_BASE_URL = "https://api.fish.audio"
DEFAULT_MODEL = "s2.1-pro-free"
DEFAULT_TIMEOUT = 120.0
DEFAULT_SPEED = 1.0
DEFAULT_FORMAT = "mp3"
DEFAULT_REFERENCE_ID = ""
SUPPORTED_FORMATS = frozenset({"mp3", "wav", "opus", "pcm"})
MAX_TEXT_LENGTH = 15000


@dataclass(frozen=True)
class FishAudioSettings:
    api_key: str
    base_url: str
    model: str
    reference_id: str
    timeout: float
    speed: float
    output_format: str
    sample_rate: int | None
    mp3_bitrate: int


def _load_tts_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        return {}
    tts = config.get("tts", {})
    return tts if isinstance(tts, dict) else {}


def _fish_config(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    tts = tts_config if isinstance(tts_config, dict) else _load_tts_section()
    section = tts.get("fishaudio", tts.get("fish_audio", {}))
    return section if isinstance(section, dict) else {}


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


def _api_key() -> str:
    try:
        from hermes_cli.config import get_env_value
        return str(get_env_value("FISH_AUDIO_API_KEY") or get_env_value("FISH_API_KEY") or "").strip()
    except Exception:
        return str(os.environ.get("FISH_AUDIO_API_KEY") or os.environ.get("FISH_API_KEY") or "").strip()


def settings(tts_config: dict[str, Any] | None = None) -> FishAudioSettings:
    cfg = _fish_config(tts_config)
    output_format = str(cfg.get("format") or cfg.get("output_format") or DEFAULT_FORMAT).lower().strip()
    if output_format not in SUPPORTED_FORMATS:
        output_format = DEFAULT_FORMAT
    speed = max(0.5, min(2.0, _float_value(cfg.get("speed"), DEFAULT_SPEED)))
    bitrate = _int_value(cfg.get("mp3_bitrate"), 128)
    if bitrate not in {64, 128, 192}:
        bitrate = 128
    sample_rate_raw = cfg.get("sample_rate")
    sample_rate = _int_value(sample_rate_raw, 0) if sample_rate_raw is not None else 0
    if sample_rate <= 0:
        sample_rate = None
    return FishAudioSettings(
        api_key=_api_key(),
        base_url=str(cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/"),
        model=str(cfg.get("model") or DEFAULT_MODEL),
        reference_id=str(cfg.get("reference_id") or cfg.get("voice") or DEFAULT_REFERENCE_ID).strip(),
        timeout=max(1.0, _float_value(cfg.get("timeout"), DEFAULT_TIMEOUT)),
        speed=speed,
        output_format=output_format,
        sample_rate=sample_rate,
        mp3_bitrate=bitrate,
    )


def status_payload(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = settings(tts_config)
    return {
        "ok": bool(cfg.api_key),
        "provider": "fishaudio",
        "available": bool(cfg.api_key),
        "credentials": {"api_key_present": bool(cfg.api_key)},
        "endpoint": f"{cfg.base_url}/v1/tts",
        "defaults": {
            "model": cfg.model,
            "reference_id_configured": bool(cfg.reference_id),
            "format": cfg.output_format,
            "speed": cfg.speed,
            "timeout": cfg.timeout,
        },
    }


def list_voices(tts_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = settings(tts_config)
    if cfg.reference_id:
        return [{"id": cfg.reference_id, "display": f"Fish Audio voice {cfg.reference_id}", "language": "multi"}]
    return []


def list_models(_: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        {"id": "s2.1-pro-free", "display": "S2.1 Pro Free", "languages": ["multi"], "max_text_length": MAX_TEXT_LENGTH},
        {"id": "s2.1-pro", "display": "S2.1 Pro", "languages": ["multi"], "max_text_length": MAX_TEXT_LENGTH},
        {"id": "s2-pro", "display": "S2 Pro", "languages": ["multi"], "max_text_length": MAX_TEXT_LENGTH},
        {"id": "s1", "display": "S1", "languages": ["multi"], "max_text_length": MAX_TEXT_LENGTH},
    ]


def _resolved_output_path(output_path: str | Path | None, output_format: str) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
    else:
        cache_dir = Path(os.environ.get("HERMES_AUDIO_CACHE_DIR", Path.home() / ".hermes" / "cache" / "audio"))
        path = cache_dir / f"fishaudio_{time.strftime('%Y%m%d-%H%M%S')}.{output_format}"
    if path.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
        path = path.with_suffix(f".{output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        body = exc.read(512).decode("utf-8", errors="replace")
        return f"Fish Audio HTTP {exc.code}: {body[:400]}"
    if isinstance(exc, URLError):
        return f"Fish Audio connection failed: {exc.reason}"
    return f"Fish Audio request failed: {exc}"


def synthesize_text(
    text: str,
    output_path: str | Path | None = None,
    voice: str | None = None,
    model: str | None = None,
    output_format: str | None = None,
    speed: float | None = None,
    tts_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"text exceeds Fish Audio limit of {MAX_TEXT_LENGTH} characters")
    cfg = settings(tts_config)
    if not cfg.api_key:
        raise RuntimeError("Fish Audio API key is missing; set FISH_AUDIO_API_KEY in Hermes .env")
    fmt = str(output_format or cfg.output_format).lower().strip()
    if fmt not in SUPPORTED_FORMATS:
        fmt = DEFAULT_FORMAT
    destination = _resolved_output_path(output_path, fmt)
    speed_value = cfg.speed if speed is None else max(0.5, min(2.0, float(speed)))
    payload: dict[str, Any] = {
        "text": text,
        "format": fmt,
        "prosody": {"speed": speed_value, "volume": 0, "normalize_loudness": True},
        "normalize": True,
        "mp3_bitrate": cfg.mp3_bitrate,
        "latency": "normal",
        "chunk_length": 300,
        "min_chunk_length": 50,
        "condition_on_previous_chunks": True,
    }
    reference_id = str(voice or cfg.reference_id or "").strip()
    if reference_id:
        payload["reference_id"] = reference_id
    if cfg.sample_rate is not None:
        payload["sample_rate"] = cfg.sample_rate
    request = Request(
        f"{cfg.base_url}/v1/tts",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "model": str(model or cfg.model),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=cfg.timeout) as response:
            audio = response.read()
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError(_error_message(exc)) from exc
    if not audio:
        raise RuntimeError("Fish Audio returned empty audio")
    destination.write_bytes(audio)
    return {
        "ok": True,
        "provider": "fishaudio",
        "file_path": str(destination),
        "format": fmt,
        "voice": reference_id or None,
        "model": str(model or cfg.model),
        "speed": speed_value,
        "media_tag": f"MEDIA:{destination}",
    }


class FishAudioTTSProvider(TTSProvider):
    name = "fishaudio"
    display_name = "Fish Audio"
    voice_compatible = True

    def is_available(self) -> bool:
        return bool(status_payload()["available"])

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "Fish Audio",
            "badge": "cloud",
            "tag": "Fish Audio REST TTS (S2.1 Pro Free supported)",
            "env_vars": [
                {
                    "key": "FISH_AUDIO_API_KEY",
                    "prompt": "Fish Audio API key",
                    "url": "https://fish.audio/ja/app/developers/",
                }
            ],
        }

    def list_voices(self) -> list[dict[str, Any]]:
        return list_voices()

    def default_voice(self) -> str | None:
        return settings().reference_id or None

    def list_models(self) -> list[dict[str, Any]]:
        return list_models()

    def default_model(self) -> str | None:
        return settings().model

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
            output_format=format,
            speed=speed,
        )
        return str(result["file_path"])
