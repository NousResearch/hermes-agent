from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agent.tts_provider import TTSProvider


DEFAULT_BASE_URL = "http://127.0.0.1:50021"
DEFAULT_SPEAKER = 8
DEFAULT_TIMEOUT = 30.0
DEFAULT_SPEED = 1.0
DEFAULT_MODEL = "voicevox-engine"


@dataclass(frozen=True)
class VoicevoxSettings:
    base_url: str
    speaker: int
    timeout: float
    speed: float


def _load_tts_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    tts = config.get("tts", {})
    return tts if isinstance(tts, dict) else {}


def _voicevox_config(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    tts = tts_config if isinstance(tts_config, dict) else _load_tts_section()
    section = tts.get("voicevox", {})
    return section if isinstance(section, dict) else {}


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def settings(tts_config: dict[str, Any] | None = None) -> VoicevoxSettings:
    cfg = _voicevox_config(tts_config)
    base_url = (
        cfg.get("base_url")
        or cfg.get("url")
        or os.environ.get("VOICEVOX_URL")
        or DEFAULT_BASE_URL
    )
    speaker = _int_value(
        cfg.get("speaker", os.environ.get("VOICEVOX_SPEAKER")),
        DEFAULT_SPEAKER,
    )
    timeout = _float_value(
        cfg.get("timeout", os.environ.get("VOICEVOX_TIMEOUT")),
        DEFAULT_TIMEOUT,
    )
    speed = _float_value(
        cfg.get("speed", os.environ.get("VOICEVOX_SPEED")),
        DEFAULT_SPEED,
    )
    return VoicevoxSettings(
        base_url=str(base_url).rstrip("/"),
        speaker=speaker,
        timeout=timeout,
        speed=speed,
    )


def status_payload(tts_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = settings(tts_config)
    endpoint = f"{cfg.base_url}/version"
    try:
        response = requests.get(endpoint, timeout=min(cfg.timeout, 5.0))
        response.raise_for_status()
        version = response.text.strip().strip('"')
        reachable = True
        error = ""
    except Exception as exc:
        version = ""
        reachable = False
        error = str(exc)
    return {
        "ok": reachable,
        "provider": "voicevox",
        "available": reachable,
        "server": {
            "endpoint": endpoint,
            "reachable": reachable,
            "version": version,
            "error": error,
        },
        "defaults": {
            "base_url": cfg.base_url,
            "speaker": cfg.speaker,
            "speed": cfg.speed,
            "timeout": cfg.timeout,
        },
    }


def list_speakers(tts_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = settings(tts_config)
    try:
        response = requests.get(f"{cfg.base_url}/speakers", timeout=min(cfg.timeout, 10.0))
        response.raise_for_status()
        speakers = response.json()
    except Exception:
        return [
            {
                "id": str(cfg.speaker),
                "display": f"VOICEVOX speaker {cfg.speaker}",
                "language": "ja",
            }
        ]

    voices: list[dict[str, Any]] = []
    for speaker in speakers if isinstance(speakers, list) else []:
        speaker_name = str(speaker.get("name") or "VOICEVOX")
        for style in speaker.get("styles", []) if isinstance(speaker, dict) else []:
            style_id = style.get("id") if isinstance(style, dict) else None
            if style_id is None:
                continue
            style_name = str(style.get("name") or style_id)
            voices.append(
                {
                    "id": str(style_id),
                    "display": f"{speaker_name} - {style_name}",
                    "language": "ja",
                }
            )
    return voices or [
        {
            "id": str(cfg.speaker),
            "display": f"VOICEVOX speaker {cfg.speaker}",
            "language": "ja",
        }
    ]


def _speaker_id(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    return _int_value(value, default)


def _resolved_output_path(output_path: str | Path | None) -> Path:
    if output_path:
        path = Path(output_path).expanduser()
    else:
        cache_dir = Path(
            os.environ.get(
                "HERMES_AUDIO_CACHE_DIR",
                Path.home() / "AppData" / "Local" / "hermes" / "audio_cache",
            )
        )
        path = cache_dir / f"voicevox_plugin_{time.strftime('%Y%m%d-%H%M%S')}.wav"
    if path.suffix.lower() != ".wav":
        path = path.with_suffix(".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def synthesize_text(
    text: str,
    output_path: str | Path | None = None,
    voice: str | int | None = None,
    speed: float | None = None,
    tts_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    cfg = settings(tts_config)
    speaker = _speaker_id(voice, cfg.speaker)
    speed_value = cfg.speed if speed is None else float(speed)
    destination = _resolved_output_path(output_path)

    query_response = requests.post(
        f"{cfg.base_url}/audio_query",
        params={"speaker": speaker, "text": text},
        timeout=cfg.timeout,
    )
    query_response.raise_for_status()
    query = query_response.json()
    if isinstance(query, dict) and speed_value > 0:
        query["speedScale"] = speed_value

    synthesis_response = requests.post(
        f"{cfg.base_url}/synthesis",
        params={"speaker": speaker},
        json=query,
        timeout=cfg.timeout,
    )
    synthesis_response.raise_for_status()
    destination.write_bytes(synthesis_response.content)
    if destination.stat().st_size <= 0:
        raise RuntimeError(f"VOICEVOX created empty audio: {destination}")

    return {
        "ok": True,
        "provider": "voicevox",
        "file_path": str(destination),
        "format": "wav",
        "voice": str(speaker),
        "model": DEFAULT_MODEL,
        "speed": speed_value,
        "media_tag": f"MEDIA:{destination}",
    }


class VoicevoxTTSProvider(TTSProvider):
    name = "voicevox"
    display_name = "VOICEVOX"
    voice_compatible = True

    def is_available(self) -> bool:
        return bool(status_payload()["available"])

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "VOICEVOX",
            "badge": "local · free",
            "tag": "Japanese local TTS via VOICEVOX Engine",
            "env_vars": [],
        }

    def list_voices(self) -> list[dict[str, Any]]:
        return list_speakers()

    def default_voice(self) -> str | None:
        return str(settings().speaker)

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "VOICEVOX Engine",
                "languages": ["ja"],
                "max_text_length": 5000,
            }
        ]

    def default_model(self) -> str | None:
        return DEFAULT_MODEL

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
            speed=speed,
        )
        return str(result["file_path"])
