"""Prepare assistant responses for text-to-speech.

By default this preserves Hermes' historical behavior: strip markdown and cap
input at 4000 chars. When ``voice.tts_summary_enabled`` is true, it asks the
auxiliary LLM router for a concise spoken summary while the full text response
still goes to the chat.
"""

from __future__ import annotations

import logging
from typing import Any

from tools.tts_tool import _strip_markdown_for_tts

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 700
_DEFAULT_MAX_TOKENS = 180
_FALLBACK_INPUT_CHARS = 4000
_SUMMARY_INPUT_CHARS = 8000


def _fallback_tts_text(original: str) -> str:
    """Previous behavior: markdown-stripped original capped before TTS."""
    return _strip_markdown_for_tts((original or "")[:_FALLBACK_INPUT_CHARS])


def _bounded_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _settings() -> tuple[bool, int, int]:
    """Return (enabled, max_chars, max_tokens) from config with safe defaults."""
    try:
        from hermes_cli.config import load_config

        voice_cfg = (load_config() or {}).get("voice", {}) or {}
    except Exception as exc:
        logger.debug("Could not load TTS summary settings: %s", exc)
        voice_cfg = {}

    enabled = bool(voice_cfg.get("tts_summary_enabled", False))
    max_chars = _bounded_int(
        voice_cfg.get("tts_summary_max_chars", _DEFAULT_MAX_CHARS),
        _DEFAULT_MAX_CHARS,
        minimum=120,
        maximum=4000,
    )
    max_tokens = _bounded_int(
        voice_cfg.get("tts_summary_max_tokens", _DEFAULT_MAX_TOKENS),
        _DEFAULT_MAX_TOKENS,
        minimum=32,
        maximum=1000,
    )
    return enabled, max_chars, max_tokens


def _extract_content(response: Any) -> str:
    """Extract text from common LLM response shapes."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        for key in ("content", "text"):
            value = response.get(key)
            if isinstance(value, str):
                return value.strip()
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        if isinstance(message, str):
            return message.strip()

    try:
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
    except Exception:
        pass

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()
    return ""


def _clamp_for_speech(text: str, max_chars: int) -> str:
    """Clamp text to max_chars, preferring natural-ish punctuation boundaries."""
    cleaned = " ".join((text or "").split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned

    candidate = cleaned[:max_chars].rstrip()
    boundary_floor = max(80, int(max_chars * 0.55))
    for marker in (". ", "! ", "? ", "; ", ": ", ", "):
        idx = candidate.rfind(marker)
        if idx >= boundary_floor:
            return candidate[: idx + 1].rstrip()
    return candidate.rstrip(" ,;:-") + "…"


def prepare_tts_text(original: str) -> str:
    """Return TTS-ready text.

    Disabled/default mode preserves the old behavior exactly. Summary mode is
    best-effort and always falls back to the old behavior on errors or empty
    output, so text delivery is never blocked by summarization.
    """
    if not original or not original.strip():
        return ""

    fallback = _fallback_tts_text(original)
    enabled, max_chars, max_tokens = _settings()
    if not enabled:
        return fallback

    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task="tts_summary",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Convert assistant replies into concise spoken Telegram voice notes. "
                        "Preserve the key answer, decisions, blockers, and next actions. "
                        "Omit code blocks, raw logs, long paths, URLs, tables, markdown, "
                        "and implementation detail unless critical. Use natural speech. "
                        "No intro, no sign-off."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Summarize this assistant response for speech in at most {max_chars} characters.\n\n"
                        f"Assistant response:\n{original[:_SUMMARY_INPUT_CHARS]}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        summary = _strip_markdown_for_tts(_extract_content(response))
        if summary and summary.strip():
            return _clamp_for_speech(summary, max_chars)
        logger.warning("TTS summary was empty; using fallback text")
    except Exception as exc:
        logger.warning("TTS summary failed; using fallback text: %s", exc)

    return fallback
