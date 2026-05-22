"""Gateway voice mode management — extracted from gateway/run.py.

Handles per-chat persistent voice mode state (on/off/tts), adapter sync,
and voice mode persistence across gateway restarts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def voice_key(platform: Any, chat_id: str) -> str:
    """Return a storage key for per-chat voice mode."""
    return f"{platform}:{chat_id}"


def load_voice_modes(persist_path: Path) -> Dict[str, str]:
    """Load persistent voice modes from disk."""
    try:
        if persist_path.exists():
            return json.loads(persist_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Failed to load voice modes: %s", e)
    return {}


def save_voice_modes(modes: Dict[str, str], persist_path: Path) -> None:
    """Save persistent voice modes to disk."""
    try:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        persist_path.write_text(
            json.dumps(modes, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("Failed to save voice modes: %s", e)


def set_adapter_auto_tts_disabled(
    adapter: Any,
    chat_id: str,
    disabled: bool,
) -> None:
    """Set auto-TTS disabled state for a chat on the adapter."""
    try:
        if hasattr(adapter, "set_auto_tts_disabled"):
            adapter.set_auto_tts_disabled(chat_id, disabled)
    except Exception as e:
        logger.debug("Failed to set auto-tts disabled: %s", e)


def set_adapter_auto_tts_enabled(
    adapter: Any,
    chat_id: str,
    enabled: bool,
) -> None:
    """Set auto-TTS enabled state for a chat on the adapter."""
    try:
        if hasattr(adapter, "set_auto_tts_enabled"):
            adapter.set_auto_tts_enabled(chat_id, enabled)
    except Exception as e:
        logger.debug("Failed to set auto-tts enabled: %s", e)


def sync_voice_mode_state_to_adapter(
    adapter: Any,
    chat_id: str,
    voice_modes: Dict[str, str],
    default_mode: str = "off",
) -> None:
    """Sync current voice mode state for a chat to the adapter."""
    key = voice_key(getattr(adapter, "platform_name", ""), chat_id)
    mode = voice_modes.get(key, default_mode)
    if mode == "off":
        set_adapter_auto_tts_disabled(adapter, chat_id, True)
    else:
        set_adapter_auto_tts_enabled(adapter, chat_id, True)
