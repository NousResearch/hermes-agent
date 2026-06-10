"""Routing helpers for inbound user-attached audio.

Mirrors ``agent/image_routing.py`` for audio: decides whether to attach raw
audio as OpenAI-style ``input_audio`` content parts on the user turn, or
fall back to STT transcription (text).

Two modes:

  native  — attach audio as ``input_audio`` content parts.  The model
            (e.g. mimo-v2.5) processes the raw audio directly.

  stt     — run STT transcription and prepend the text.  The model never
            hears the audio; it only sees the transcript.

The decision is made once per message turn by :func:`decide_audio_input_mode`.
It reads ``agent.audio_input_mode`` from config.yaml (``auto`` | ``native``
| ``stt``, default ``auto``) and the active model's capability metadata.

In ``auto`` mode:
  - If the model reports ``supports_audio_input=True`` via config override
    (models.dev doesn't track audio for most models yet), we attach natively.
  - Otherwise, we fall back to STT.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"auto", "native", "stt"})

# Audio extensions that can be passed to the model as input_audio.
_AUDIO_EXTS = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".webm")

# Max audio file size for base64 embedding (50 MB — Xiaomi API limit).
_MAX_AUDIO_BYTES = 50 * 1024 * 1024

# MIME type mapping for audio extensions.
_AUDIO_MIME_MAP = {
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    return val if val in _VALID_MODES else "auto"


def _coerce_bool(raw: Any) -> Optional[bool]:
    """Strict YAML/JSON boolean coercion (same logic as image_routing)."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw) if raw in (0, 1) else None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"true", "yes", "on", "1"}:
            return True
        if s in {"false", "no", "off", "0"}:
            return False
    return None


def _supports_audio_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
) -> Optional[bool]:
    """Resolve user-declared audio capability from config.yaml.

    Resolution order, first hit wins:
      1. ``model.supports_audio`` (top-level shortcut)
      2. ``providers.<provider>.models.<model>.supports_audio``
    Returns None when no override is set.
    """
    if not isinstance(cfg, dict):
        return None

    # 1. Top-level shortcut
    model_cfg_raw = cfg.get("model")
    model_cfg: Dict[str, Any] = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    top = _coerce_bool(model_cfg.get("supports_audio"))
    if top is not None:
        return top

    # 2. Per-provider, per-model
    config_provider = str(model_cfg.get("provider") or "").strip()
    providers_raw = cfg.get("providers")
    providers_cfg: Dict[str, Any] = providers_raw if isinstance(providers_raw, dict) else {}
    for p in dict.fromkeys(filter(None, (provider, config_provider))):
        entry_raw = providers_cfg.get(p)
        entry: Dict[str, Any] = entry_raw if isinstance(entry_raw, dict) else {}
        models_raw = entry.get("models")
        models_cfg: Dict[str, Any] = models_raw if isinstance(models_raw, dict) else {}
        per_model_raw = models_cfg.get(model)
        per_model: Dict[str, Any] = per_model_raw if isinstance(per_model_raw, dict) else {}
        coerced = _coerce_bool(per_model.get("supports_audio"))
        if coerced is not None:
            return coerced
    return None


def _lookup_supports_audio(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return True if the model supports native audio input.

    Checks config override first, then falls back to models.dev
    (``ModelInfo.supports_audio_input``), then returns False.
    """
    override = _supports_audio_override(cfg, provider, model)
    if override is not None:
        return override

    if not provider or not model:
        return False

    try:
        from agent.models_dev import get_model_info
        info = get_model_info(provider, model)
        if info is not None and info.supports_audio_input():
            return True
    except Exception as exc:
        logger.debug("audio_routing: models.dev lookup failed for %s:%s — %s", provider, model, exc)

    return False


def decide_audio_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> str:
    """Return ``"native"`` or ``"stt"`` for the given turn.

    Args:
      provider: active inference provider ID (e.g. ``"xiaomi"``).
      model:    active model slug (e.g. ``"mimo-v2.5"``).
      cfg:      loaded config.yaml dict, or None. When None, behaves as auto.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("audio_input_mode"))

    if mode_cfg == "native":
        return "native"
    if mode_cfg == "stt":
        return "stt"

    # auto
    if _lookup_supports_audio(provider, model, cfg):
        return "native"
    return "stt"


def build_audio_content_parts(
    user_text: str,
    audio_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list for a user turn with audio.

    Shape:
      [{"type": "text", "text": "..."},
       {"type": "input_audio", "input_audio": {"data": "data:audio/ogg;base64,..."}},
       ...]

    Each audio file is read from disk, base64-encoded, and embedded as a
    ``data:`` URL.  Files exceeding ``_MAX_AUDIO_BYTES`` are skipped.

    Returns (content_parts, skipped).  Skipped entries are paths that
    couldn't be read or were too large.
    """
    skipped: List[str] = []
    audio_parts: List[Dict[str, Any]] = []

    for raw_path in audio_paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            skipped.append(str(raw_path))
            continue

        file_size = p.stat().st_size
        if file_size > _MAX_AUDIO_BYTES:
            logger.warning(
                "audio_routing: skipping %s (%s bytes exceeds %s limit)",
                raw_path, file_size, _MAX_AUDIO_BYTES,
            )
            skipped.append(str(raw_path))
            continue

        try:
            raw = p.read_bytes()
        except Exception as exc:
            logger.warning("audio_routing: failed to read %s — %s", raw_path, exc)
            skipped.append(str(raw_path))
            continue

        # Determine MIME type from extension.
        ext = p.suffix.lower()
        mime = _AUDIO_MIME_MAP.get(ext, "audio/ogg")

        b64 = base64.b64encode(raw).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        audio_parts.append({
            "type": "input_audio",
            "input_audio": {
                "data": data_url,
            },
        })

    text = (user_text or "").strip()

    if audio_parts:
        base_text = text or "Please listen to this audio and respond."
        hint_lines = [f"[Audio attached at: {p}]" for p in audio_paths if p not in skipped]
        combined_text = f"{base_text}\n\n" + "\n".join(hint_lines) if hint_lines else base_text
        parts: List[Dict[str, Any]] = [{"type": "text", "text": combined_text}]
        parts.extend(audio_parts)
        return parts, skipped

    # No audio successfully attached — return empty list (caller falls back to text).
    return [], skipped
