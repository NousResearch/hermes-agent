"""Inworld bundled backend plugin.

Provides Text-to-Speech and Speech-to-Text providers selected with
``tts.provider: inworld`` and ``stt.provider: inworld``.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.inworld.ai"
DEFAULT_TTS_MODEL = "inworld-tts-2"
DEFAULT_TTS_VOICE = "Dennis"
DEFAULT_TTS_FORMAT = "ogg"
DEFAULT_TTS_SAMPLE_RATE = 48000
DEFAULT_TTS_DELIVERY_MODE = "BALANCED"
DEFAULT_STT_MODEL = "inworld/inworld-stt-1"
DEFAULT_STT_ENCODING = "AUTO_DETECT"
DEFAULT_STT_SAMPLE_RATE = 16000
DEFAULT_STT_CHANNELS = 1

_TTS_MODELS = [
    {
        "id": "inworld-tts-2",
        "display": "Inworld TTS-2",
        "languages": ["100+"],
        "max_text_length": 10000,
    },
    {
        "id": "inworld-tts-1.5-max",
        "display": "Inworld TTS 1.5 Max",
        "languages": [
            "en", "zh", "ja", "ko", "ru", "it", "es", "pt", "fr", "de",
            "pl", "nl", "hi", "he", "ar",
        ],
        "max_text_length": 10000,
    },
    {
        "id": "inworld-tts-1.5-mini",
        "display": "Inworld TTS 1.5 Mini",
        "languages": [
            "en", "zh", "ja", "ko", "ru", "it", "es", "pt", "fr", "de",
            "pl", "nl", "hi", "he", "ar",
        ],
        "max_text_length": 10000,
    },
]

_STT_MODELS = [
    {"id": "inworld/inworld-stt-1", "display": "Inworld STT 1", "languages": ["30+"]},
    {"id": "groq/whisper-large-v3", "display": "Groq Whisper Large v3", "languages": ["100+"]},
    {
        "id": "assemblyai/u3-rt-pro",
        "display": "AssemblyAI Universal-3 Pro Realtime",
        "languages": ["en", "es", "fr", "de", "it", "pt"],
    },
    {
        "id": "soniox/stt-rt-v4",
        "display": "Soniox STT RT v4",
        "languages": ["multilingual"],
    },
]


def _get_env_value(name: str) -> Optional[str]:
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
    except Exception:
        import os

        value = os.environ.get(name)
    return str(value).strip() if value else None


def _load_config_section(section: str) -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = cfg.get(section) if isinstance(cfg, dict) else None
        return value if isinstance(value, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load %s config: %s", section, exc)
        return {}


def _provider_cfg(section: str) -> Dict[str, Any]:
    root = _load_config_section(section)
    value = root.get("inworld")
    return value if isinstance(value, dict) else {}


def _base_url(section: str) -> str:
    cfg = _provider_cfg(section)
    return str(cfg.get("base_url") or DEFAULT_BASE_URL).strip().rstrip("/")


def _api_key() -> Optional[str]:
    return _get_env_value("INWORLD_API_KEY")


def _headers() -> Dict[str, str]:
    key = _api_key()
    if not key:
        raise ValueError(
            "INWORLD_API_KEY is not set. Create one in Inworld Portal and save it in ~/.hermes/.env."
        )
    return {
        "Authorization": f"Basic {key}",
        "Content-Type": "application/json",
    }


def _require_requests():
    import requests

    return requests


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _audio_encoding_for_format(fmt: str) -> str:
    key = (fmt or DEFAULT_TTS_FORMAT).lower().strip().lstrip(".")
    return {
        "mp3": "MP3",
        "ogg": "OGG_OPUS",
        "opus": "OGG_OPUS",
        "wav": "WAV",
        "flac": "WAV",
        "pcm": "PCM",
    }.get(key, "OGG_OPUS")


def _suffix_for_encoding(encoding: str) -> str:
    return {
        "MP3": ".mp3",
        "OGG_OPUS": ".ogg",
        "WAV": ".wav",
        "PCM": ".pcm",
        "LINEAR16": ".wav",
    }.get(encoding, ".ogg")


def _stt_encoding_for_path(file_path: str, configured: str) -> str:
    cfg = str(configured or DEFAULT_STT_ENCODING).strip().upper()
    if cfg and cfg != "AUTO":
        return cfg
    suffix = Path(file_path).suffix.lower()
    return {
        ".mp3": "MP3",
        ".ogg": "OGG_OPUS",
        ".opus": "OGG_OPUS",
        ".flac": "FLAC",
        ".wav": "LINEAR16",
        ".pcm": "LINEAR16",
    }.get(suffix, "AUTO_DETECT")


def _error_message(response: Any) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            msg = data.get("message") or data.get("error")
            if isinstance(msg, str) and msg:
                return msg
    except Exception:
        pass
    text = getattr(response, "text", "") or ""
    return text[:500] or f"HTTP {getattr(response, 'status_code', '?')}"


class InworldTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "inworld"

    @property
    def display_name(self) -> str:
        return "Inworld Realtime TTS"

    @property
    def voice_compatible(self) -> bool:
        return True

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Inworld Realtime TTS",
            "badge": "paid",
            "tag": "Realtime TTS-2, voice cloning, native Opus",
            "env_vars": [
                {
                    "key": "INWORLD_API_KEY",
                    "prompt": "Inworld API key",
                    "url": "https://platform.inworld.ai/api-keys",
                }
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_TTS_MODELS)

    def list_voices(self) -> List[Dict[str, Any]]:
        if not _api_key():
            return []
        requests = _require_requests()
        url = f"{_base_url('tts')}/voices/v1/voices"
        voices: List[Dict[str, Any]] = []
        params: Dict[str, Any] = {"pageSize": 2000, "orderBy": "display_name asc"}
        for _ in range(10):
            response = requests.get(
                url,
                headers={"Authorization": f"Basic {_api_key()}"},
                params=params,
                timeout=30,
            )
            if response.status_code >= 400:
                logger.debug("Inworld list voices failed: %s", _error_message(response))
                return voices
            data = response.json()
            for voice in data.get("voices", []) if isinstance(data, dict) else []:
                vid = voice.get("voiceId") or voice.get("voice_id")
                if not vid:
                    continue
                voices.append(
                    {
                        "id": vid,
                        "display": voice.get("displayName") or voice.get("display_name") or vid,
                        "language": voice.get("langCode") or voice.get("lang_code"),
                        "gender": voice.get("gender"),
                    }
                )
            token = data.get("nextPageToken") if isinstance(data, dict) else ""
            if not token:
                break
            params["pageToken"] = token
        return voices

    def default_model(self) -> Optional[str]:
        return DEFAULT_TTS_MODEL

    def default_voice(self) -> Optional[str]:
        return DEFAULT_TTS_VOICE

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_TTS_FORMAT,
        **extra: Any,
    ) -> str:
        cfg = _provider_cfg("tts")
        model_id = model or cfg.get("model") or DEFAULT_TTS_MODEL
        voice_id = voice or cfg.get("voice") or cfg.get("voice_id") or DEFAULT_TTS_VOICE
        requested_suffix = Path(output_path).suffix.lower().lstrip(".")
        path_format = requested_suffix if requested_suffix and requested_suffix != "mp3" else ""
        arg_format = str(format or "").lower().strip().lstrip(".")
        call_format = arg_format if arg_format and arg_format != "mp3" else ""
        fmt = str(
            cfg.get("output_format")
            or cfg.get("format")
            or path_format
            or call_format
            or DEFAULT_TTS_FORMAT
        )
        encoding = _audio_encoding_for_format(fmt)
        sample_rate = _coerce_int(
            cfg.get("sample_rate", cfg.get("sample_rate_hertz")),
            DEFAULT_TTS_SAMPLE_RATE,
        )
        speaking_rate = _coerce_float(speed if speed is not None else cfg.get("speed"), 1.0)
        delivery_mode = str(cfg.get("delivery_mode") or DEFAULT_TTS_DELIVERY_MODE).strip().upper()
        language = str(cfg.get("language") or "").strip()
        apply_text_normalization = cfg.get("apply_text_normalization")
        timestamp_type = str(cfg.get("timestamp_type") or "").strip().upper()

        payload: Dict[str, Any] = {
            "text": text,
            "voiceId": str(voice_id),
            "modelId": str(model_id),
            "audioConfig": {
                "audioEncoding": encoding,
                "sampleRateHertz": sample_rate,
                "speakingRate": max(0.5, min(1.5, speaking_rate)),
            },
            "deliveryMode": delivery_mode
            if delivery_mode in {"STABLE", "BALANCED", "CREATIVE"}
            else DEFAULT_TTS_DELIVERY_MODE,
        }
        if language:
            payload["language"] = language
        if isinstance(apply_text_normalization, str) and apply_text_normalization.strip():
            payload["applyTextNormalization"] = apply_text_normalization.strip().upper()
        if timestamp_type in {"WORD", "CHARACTER"}:
            payload["timestampType"] = timestamp_type

        out = Path(output_path).expanduser()
        suffix = _suffix_for_encoding(encoding)
        if out.suffix.lower() != suffix:
            out = out.with_suffix(suffix)
        out.parent.mkdir(parents=True, exist_ok=True)

        requests = _require_requests()
        response = requests.post(
            f"{_base_url('tts')}/tts/v1/voice",
            headers=_headers(),
            json=payload,
            timeout=_coerce_float(cfg.get("timeout"), 60.0),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Inworld TTS failed: {_error_message(response)}")
        data = response.json()
        audio_b64 = data.get("audioContent") if isinstance(data, dict) else None
        if not isinstance(audio_b64, str) or not audio_b64:
            raise RuntimeError("Inworld TTS returned no audioContent")
        out.write_bytes(base64.b64decode(audio_b64))
        return str(out)


class InworldTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "inworld"

    @property
    def display_name(self) -> str:
        return "Inworld Realtime STT"

    def is_available(self) -> bool:
        return bool(_api_key())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Inworld Realtime STT",
            "badge": "paid",
            "tag": "Unified STT with Inworld, Groq, AssemblyAI, Soniox models",
            "env_vars": [
                {
                    "key": "INWORLD_API_KEY",
                    "prompt": "Inworld API key",
                    "url": "https://platform.inworld.ai/api-keys",
                }
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_STT_MODELS)

    def default_model(self) -> Optional[str]:
        return DEFAULT_STT_MODEL

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        try:
            cfg = _provider_cfg("stt")
            model_id = model or cfg.get("model") or DEFAULT_STT_MODEL
            lang = language or cfg.get("language") or ""
            encoding = _stt_encoding_for_path(
                file_path,
                str(cfg.get("audio_encoding") or DEFAULT_STT_ENCODING),
            )
            enable_voice_profile = _bool_cfg(cfg.get("voice_profile"), False)

            audio = Path(file_path).expanduser()
            audio_b64 = base64.b64encode(audio.read_bytes()).decode("ascii")
            transcribe_config: Dict[str, Any] = {
                "modelId": str(model_id),
                "audioEncoding": encoding,
                "numberOfChannels": _coerce_int(cfg.get("number_of_channels"), DEFAULT_STT_CHANNELS),
            }
            if lang:
                transcribe_config["language"] = str(lang)
            if encoding == "LINEAR16":
                transcribe_config["sampleRateHertz"] = _coerce_int(
                    cfg.get("sample_rate", cfg.get("sample_rate_hertz")),
                    DEFAULT_STT_SAMPLE_RATE,
                )
            if enable_voice_profile:
                transcribe_config["voiceProfileConfig"] = {
                    "enableVoiceProfile": True,
                    "topN": _coerce_int(cfg.get("voice_profile_top_n"), 10),
                }

            requests = _require_requests()
            response = requests.post(
                f"{_base_url('stt')}/stt/v1/transcribe",
                headers=_headers(),
                json={
                    "transcribeConfig": transcribe_config,
                    "audioData": {"content": audio_b64},
                },
                timeout=_coerce_float(cfg.get("timeout"), 90.0),
            )
            if response.status_code >= 400:
                return {
                    "success": False,
                    "transcript": "",
                    "provider": self.name,
                    "error": f"Inworld STT failed: {_error_message(response)}",
                }
            data = response.json()
            transcription = data.get("transcription") if isinstance(data, dict) else {}
            transcript = transcription.get("transcript") if isinstance(transcription, dict) else ""
            result: Dict[str, Any] = {
                "success": True,
                "transcript": transcript or "",
                "provider": self.name,
            }
            if isinstance(data, dict):
                if data.get("voiceProfile") is not None:
                    result["voice_profile"] = data.get("voiceProfile")
                if data.get("usage") is not None:
                    result["usage"] = data.get("usage")
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Inworld STT transcription failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "transcript": "",
                "provider": self.name,
                "error": f"Inworld STT transcription failed: {exc}",
            }


def register(ctx) -> None:
    ctx.register_tts_provider(InworldTTSProvider())
    ctx.register_transcription_provider(InworldTranscriptionProvider())
