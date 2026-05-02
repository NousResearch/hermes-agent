"""Routing helpers for inbound user-attached audio.

Two modes:

  native  — pass ``input_audio`` content parts through to the model unchanged.
            Only works with models that accept audio natively (e.g. OpenAI
            ``gpt-audio``, Gemini 2.5 Flash with audio modality).

  text    — run ``transcribe_audio`` on each audio attachment up-front and
            prepend the transcript to the user's text. The model never hears
            the raw audio; it only sees the transcribed text. This is the
            right choice for the vast majority of models (non-audio-native).

The decision is made once per message turn by :func:`decide_audio_input_mode`.
It reads ``agent.audio_input_mode`` from config.yaml (``auto`` | ``native``
| ``text``, default ``auto``) and the active model's capability metadata.

In ``auto`` mode:
  - If the active model reports ``supports_audio_input=True`` in its
    models.dev metadata, attach natively.
  - Otherwise, fall back to transcription (text path).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"auto", "native", "text"})


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    if val in _VALID_MODES:
        return val
    return "auto"


def _lookup_supports_audio(provider: str, model: str) -> Optional[bool]:
    """Return True/False if we can resolve caps, None if unknown."""
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "audio_routing: caps lookup failed for %s:%s — %s",
            provider, model, exc,
        )
        return None
    if caps is None:
        return None
    return bool(caps.supports_audio_input())


def decide_audio_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> str:
    """Return ``"native"`` or ``"text"`` for the given turn.

    Args:
        provider: active inference provider ID (e.g. ``"openai"``, ``"openrouter"``).
        model:    active model slug as it would be sent to the provider.
        cfg:      loaded config.yaml dict, or None. When None, behaves as auto.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("audio_input_mode"))

    if mode_cfg == "native":
        return "native"
    if mode_cfg == "text":
        return "text"

    # auto
    supports = _lookup_supports_audio(provider, model)
    if supports is True:
        return "native"
    return "text"


__all__ = [
    "decide_audio_input_mode",
]
