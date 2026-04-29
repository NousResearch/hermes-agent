"""Volcengine (Doubao) text-to-speech provider.

Implements the TtsProvider ABC.  The WebSocket client (``client.py``) is
imported lazily inside ``synthesize()`` so machines without the optional
``voice-volcengine`` extra installed can still discover and introspect the
plugin without ImportError.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tts_provider import TtsProvider

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
SETUP_URL = "https://console.volcengine.com/speech/service"


def _get_env(key: str, default: Any = None) -> Any:
    """Read a credential/config value via the Hermes profile-aware helper."""
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(key)
        return val if val else default
    except ImportError:
        return os.getenv(key, default)


def _env_or(config: Dict[str, Any], key: str, env_key: str, default: Any) -> Any:
    """Resolution order: config dict > env var > default.

    The env fallback lets a user tweak voice/emotion/speed by editing .env
    and having the next synthesize() call read the new value — no restart.
    """
    val = config.get(key)
    if val is not None and val != "":
        return val
    env_val = _get_env(env_key)
    if env_val is not None and env_val != "":
        return env_val
    return default


def _run(coro: Any) -> None:
    """Run a coroutine from sync provider code, even if a loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: asyncio.run(coro)).result()


class VolcengineTtsProvider(TtsProvider):
    """Volcengine bidirectional streaming TTS backend (seed-tts-2.0)."""

    @property
    def name(self) -> str:
        return "volcengine"

    @property
    def display_name(self) -> str:
        return "Volcengine (Doubao)"

    def is_available(self) -> bool:
        """Credential check only — no heavy imports."""
        return bool(_get_env("VOLCENGINE_APP_ID") and _get_env("VOLCENGINE_ACCESS_TOKEN"))

    def max_text_length(self) -> int:
        return 1024

    def list_voices(self) -> List[Dict[str, Any]]:
        return [{"id": DEFAULT_SPEAKER, "display": "Doubao default female voice"}]

    def default_voice(self) -> Optional[str]:
        return DEFAULT_SPEAKER

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Volcengine (Doubao)",
            "badge": "paid",
            "tag": "火山引擎 bidirectional streaming TTS (seed-tts-2.0)",
            "env_vars": [
                {"key": "VOLCENGINE_APP_ID", "prompt": "Volcengine App ID", "url": SETUP_URL},
                {"key": "VOLCENGINE_ACCESS_TOKEN", "prompt": "Volcengine Access Token", "url": SETUP_URL},
            ],
        }

    def synthesize(self, text: str, output_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech and write it to *output_path*.

        Parameter resolution order (for every runtime knob):
            config dict (config.yaml/tts.volcengine.*)
            > env var (VOLCENGINE_TTS_*)
            > built-in default

        Returns:
            Result dict per TtsProvider.synthesize() contract.
        """
        # Lazy-import the WebSocket client only when actually synthesizing
        try:
            from .client import VolcengineAuthError, VolcengineParamError, VolcengineVoiceError, tts_to_file
        except ImportError as exc:
            return {
                "success": False,
                "error": (
                    "voice-volcengine plugin requires the 'websockets' package. "
                    "Install with: uv pip install 'hermes-agent[voice-volcengine]'"
                ),
                "error_type": "dependency",
            }

        app_id = _get_env("VOLCENGINE_APP_ID")
        access_token = _get_env("VOLCENGINE_ACCESS_TOKEN")
        if not app_id or not access_token:
            return {
                "success": False,
                "error": (
                    "VOLCENGINE_APP_ID or VOLCENGINE_ACCESS_TOKEN not set. "
                    f"Get credentials at {SETUP_URL}"
                ),
                "error_type": "config",
            }

        config = config or {}

        speaker = str(_env_or(config, "speaker", "VOLCENGINE_TTS_SPEAKER", DEFAULT_SPEAKER))
        audio_format = str(_env_or(config, "audio_format", "VOLCENGINE_TTS_AUDIO_FORMAT", "mp3"))
        sample_rate = int(_env_or(config, "sample_rate", "VOLCENGINE_TTS_SAMPLE_RATE", 24000))
        speed_ratio = float(_env_or(config, "speed_ratio", "VOLCENGINE_TTS_SPEED_RATIO", 1.0))
        resource_id = str(_env_or(config, "resource_id", "VOLCENGINE_TTS_RESOURCE_ID", DEFAULT_RESOURCE_ID))
        timeout = float(_env_or(config, "timeout", "VOLCENGINE_TTS_TIMEOUT", 30.0))

        emotion = _env_or(config, "emotion", "VOLCENGINE_TTS_EMOTION", None)
        if emotion is not None and str(emotion).strip().lower() in ("", "none", "null"):
            emotion = None
        emotion_scale_raw = _env_or(config, "emotion_scale", "VOLCENGINE_TTS_EMOTION_SCALE", None)
        try:
            emotion_scale = int(emotion_scale_raw) if emotion_scale_raw is not None else None
        except (TypeError, ValueError):
            emotion_scale = None

        try:
            _run(tts_to_file(
                text,
                Path(output_path),
                speaker=speaker,
                audio_format=audio_format,
                sample_rate=sample_rate,
                speed_ratio=speed_ratio,
                emotion=str(emotion) if emotion else None,
                emotion_scale=emotion_scale,
                app_id=app_id,
                access_token=access_token,
                resource_id=resource_id,
                timeout=timeout,
            ))
        except (VolcengineAuthError, VolcengineParamError, ValueError) as exc:
            return {
                "success": False,
                "error": f"Volcengine TTS configuration error: {exc}",
                "error_type": "config",
            }
        except VolcengineVoiceError as exc:
            logger.error("Volcengine TTS failed: %s", exc)
            return {
                "success": False,
                "error": f"Volcengine TTS failed: {exc}",
                "error_type": "runtime",
            }
        except Exception as exc:
            logger.exception("Volcengine TTS failed")
            return {
                "success": False,
                "error": f"Volcengine TTS failed: {type(exc).__name__}: {exc}",
                "error_type": "runtime",
            }

        # Determine output metadata based on format
        native_opus = audio_format == "ogg_opus"
        voice_compatible = native_opus  # ogg_opus is ready for Telegram voice bubbles

        return {
            "success": True,
            "file_path": str(output_path),
            "format": "ogg" if native_opus else audio_format,
            "native_opus": native_opus,
            "voice_compatible": voice_compatible,
        }
