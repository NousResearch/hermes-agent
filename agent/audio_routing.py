"""Routing helpers for inbound user-attached audio and voice notes.

Two modes:

  native  — attach audio directly as multimodal parts (OpenAI-style ``input_audio``
            content parts, or Gemini inlineData) on the user turn without an
            intermediate speech-to-text (STT) transcription step.

  stt     — run the configured STT provider up-front and prepend the transcription
            to the user's text. This is the fallback / classic flow for models
            that do not accept native audio input.

The decision is made once per message turn by :func:`decide_audio_input_mode`.
It reads ``agent.audio_input_mode`` from config.yaml (``auto`` | ``native``
| ``stt``, default ``auto``) and the active model's capability metadata.

In ``auto`` mode:
  - If the active model or provider reports native audio support (via
    config override, models.dev metadata, or built-in model capability rules),
    we attach natively.
  - Otherwise (non-audio model), routes to the STT pipeline.
  ``agent.audio_input_mode: native`` remains an explicit override.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"auto", "native", "stt"})

_AUDIO_EXTS = (
    ".oga", ".ogg", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".webm", ".opus", ".wma",
)
_AUDIO_EXT_PATTERN = "|".join(e.lstrip(".") for e in _AUDIO_EXTS)

_LOCAL_AUDIO_PATH_RE = re.compile(
    r"(?<![/:\\w.])(?:~/|/)(?:[\w.\-]+/)*[\w.\-]+\.(?:" + _AUDIO_EXT_PATTERN + r")\b",
    re.IGNORECASE,
)

_AUDIO_URL_RE = re.compile(
    r"https?://[^\s<>\"']+?\.(?:" + _AUDIO_EXT_PATTERN + r")(?:\?[^\s<>\"']*)?",
    re.IGNORECASE,
)


def _coerce_mode(raw: Any) -> str:
    """Coerce arbitrary raw mode input to 'auto' | 'native' | 'stt'."""
    if not isinstance(raw, str):
        return "auto"
    s = raw.strip().lower()
    if s in ("native", "stt", "text", "auto"):
        return "stt" if s == "text" else s
    return "auto"


def extract_audio_refs(text: str) -> Tuple[List[str], List[str]]:
    """Scan free-form text for audio references the model should hear.

    Returns ``(local_paths, urls)``.
    """
    if not isinstance(text, str) or not text:
        return [], []

    code_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"```[^\n]*\n.*?```", text, re.DOTALL):
        code_spans.append((m.start(), m.end()))
    for m in re.finditer(r"`[^`\n]+`", text):
        code_spans.append((m.start(), m.end()))

    def _in_code(pos: int) -> bool:
        return any(s <= pos < e for s, e in code_spans)

    local_paths: list[str] = []
    seen_paths: set[str] = set()
    for match in _LOCAL_AUDIO_PATH_RE.finditer(text):
        if _in_code(match.start()):
            continue
        raw = match.group(0)
        expanded = os.path.expanduser(raw)
        try:
            if not os.path.isfile(expanded):
                continue
        except OSError:
            continue
        if expanded in seen_paths:
            continue
        seen_paths.add(expanded)
        local_paths.append(expanded)

    urls: list[str] = []
    seen_urls: set[str] = set()
    for match in _AUDIO_URL_RE.finditer(text):
        if _in_code(match.start()):
            continue
        url = match.group(0).rstrip(".,;:!?)]>")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        urls.append(url)

    return local_paths, urls


_TRUE_TOKENS = frozenset({"true", "yes", "on", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "off", "0"})


def _coerce_capability_bool(raw: Any) -> Optional[bool]:
    """Return True/False for recognised boolean values, None otherwise."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        if raw in (0, 1):
            return bool(raw)
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
    return None


def _supports_audio_override(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Resolve user-declared audio capability from config.yaml."""
    if not isinstance(cfg, dict):
        return None

    # 1. Top-level shortcut under model
    model_cfg_raw = cfg.get("model")
    model_cfg: Dict[str, Any] = model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    for key in ("supports_audio", "supports_audio_input", "audio"):
        if key in model_cfg:
            coerced = _coerce_capability_bool(model_cfg.get(key))
            if coerced is not None:
                return coerced

    # 2. Per-provider, per-model
    config_provider = str(model_cfg.get("provider") or "").strip()
    provider_candidates: List[str] = []
    for candidate in (requested_provider, provider, config_provider):
        if not candidate:
            continue
        provider_candidates.append(candidate)
        if candidate.startswith("custom:"):
            stripped_candidate = candidate[len("custom:"):]
            if stripped_candidate:
                provider_candidates.append(stripped_candidate)
    providers_raw = cfg.get("providers")
    providers_cfg: Dict[str, Any] = providers_raw if isinstance(providers_raw, dict) else {}
    for p in dict.fromkeys(provider_candidates):
        entry_raw = providers_cfg.get(p)
        entry: Dict[str, Any] = entry_raw if isinstance(entry_raw, dict) else {}
        models_raw = entry.get("models")
        models_cfg: Dict[str, Any] = models_raw if isinstance(models_raw, dict) else {}
        per_model_raw = models_cfg.get(model)
        per_model: Dict[str, Any] = per_model_raw if isinstance(per_model_raw, dict) else {}
        for key in ("supports_audio", "supports_audio_input", "audio"):
            if key in per_model:
                coerced = _coerce_capability_bool(per_model.get(key))
                if coerced is not None:
                    return coerced

    # 2b. Legacy list-style custom_providers
    custom_providers = cfg.get("custom_providers")
    if isinstance(custom_providers, list):
        for candidate in dict.fromkeys(provider_candidates):
            candidate_name = candidate.strip().lower()
            for entry_raw in custom_providers:
                if not isinstance(entry_raw, dict):
                    continue
                entry_name = str(entry_raw.get("name") or "").strip().lower()
                if entry_name != candidate_name:
                    continue
                models_raw = entry_raw.get("models")
                models_cfg = models_raw if isinstance(models_raw, dict) else {}
                per_model_raw = models_cfg.get(model)
                per_model = per_model_raw if isinstance(per_model_raw, dict) else {}
                for key in ("supports_audio", "supports_audio_input", "audio"):
                    if key in per_model:
                        coerced = _coerce_capability_bool(per_model.get(key))
                        if coerced is not None:
                            return coerced

    return None


def _is_known_audio_model_or_provider(provider: str, model: str) -> bool:
    """Heuristic check for known native-audio capable models and providers."""
    prov = (provider or "").strip().lower()
    mod = (model or "").strip().lower()

    # Gemini family (Flash / Pro natively accept audio)
    if prov in ("gemini", "google"):
        return True
    if "gemini" in mod:
        return True

    # OpenAI GPT-4o audio preview / realtime models
    if "gpt-4o-audio" in mod or "gpt-4o-mini-audio" in mod or "gpt-4o-realtime" in mod:
        return True

    return False


def _lookup_supports_audio(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    requested_provider: str = "",
) -> Optional[bool]:
    """Check if the active model/provider supports native audio input."""
    override = _supports_audio_override(
        cfg,
        provider,
        model,
        requested_provider=requested_provider,
    )
    if override is not None:
        return override
    if not provider and not model:
        return None

    # Try models.dev capabilities
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model, allow_network=True)
        if caps is not None and caps.supports_audio_input():
            return True
    except Exception as exc:
        logger.debug("audio_routing: models.dev lookup failed for %s:%s — %s", provider, model, exc)

    # Built-in heuristics for Gemini / GPT-4o Audio
    if _is_known_audio_model_or_provider(provider, model):
        return True

    return None


def decide_audio_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
    *,
    requested_provider: str = "",
) -> str:
    """Return "native" or "stt" for the given turn.

    Args:
      provider: active inference provider ID (e.g. "gemini", "openai").
      model:    active model slug as it would be sent to the provider.
      cfg:      loaded config.yaml dict, or None. When None, behaves as auto.
      requested_provider: provider identity before runtime canonicalization.
    """
    mode_cfg = "auto"
    if isinstance(cfg, dict):
        agent_cfg = cfg.get("agent") or {}
        if isinstance(agent_cfg, dict):
            mode_cfg = _coerce_mode(agent_cfg.get("audio_input_mode"))
        if mode_cfg == "auto" and "audio_input_mode" in cfg:
            mode_cfg = _coerce_mode(cfg.get("audio_input_mode"))
        if mode_cfg == "auto":
            gw_cfg = cfg.get("gateway") or {}
            if isinstance(gw_cfg, dict) and "audio_input_mode" in gw_cfg:
                mode_cfg = _coerce_mode(gw_cfg.get("audio_input_mode"))

    if mode_cfg == "native":
        return "native"
    if mode_cfg in ("stt", "text"):
        return "stt"

    if requested_provider:
        supports = _lookup_supports_audio(
            provider,
            model,
            cfg,
            requested_provider=requested_provider,
        )
    else:
        supports = _lookup_supports_audio(provider, model, cfg)

    if supports is True:
        return "native"
    return "stt"


def _sniff_audio_mime_from_bytes(raw: bytes) -> Optional[str]:
    """Detect audio MIME from magic bytes. Returns None if unrecognised."""
    if not raw:
        return None
    # WAV: RIFF....WAVE
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    # Ogg / Opus / OGA: OggS
    if raw.startswith(b"OggS"):
        return "audio/ogg"
    # MP3: ID3 or frame sync
    if raw.startswith(b"ID3"):
        return "audio/mp3"
    if len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0:
        return "audio/mp3"
    # FLAC: fLaC
    if raw.startswith(b"fLaC"):
        return "audio/flac"
    # M4A / MP4: bytes 4..8 == 'ftyp'
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        return "audio/m4a"
    # WebM: \x1a\x45\xdf\xa3
    if raw.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"
    return None


def _guess_audio_mime(path: Path, raw: Optional[bytes] = None) -> str:
    """Return audio MIME type for path."""
    if raw is not None:
        sniffed = _sniff_audio_mime_from_bytes(raw)
        if sniffed:
            return sniffed
    suffix = path.suffix.lower()
    suffix_map = {
        ".oga": "audio/ogg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
        ".wma": "audio/x-ms-wma",
    }
    if suffix in suffix_map:
        return suffix_map[suffix]
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("audio/"):
        if mime in ("audio/x-wav", "audio/wave"):
            return "audio/wav"
        if mime in ("audio/mpeg", "audio/mpeg3"):
            return "audio/mp3"
        return mime
    return "audio/ogg"


def _mime_to_audio_format(mime: str, suffix: str = "") -> str:
    """Normalize MIME or suffix to audio format identifier for input_audio."""
    mime_clean = (mime or "").strip().lower()
    if mime_clean in ("audio/wav", "audio/x-wav", "audio/wave"):
        return "wav"
    if mime_clean in ("audio/mp3", "audio/mpeg", "audio/mpeg3"):
        return "mp3"
    if mime_clean in ("audio/ogg", "audio/oga", "audio/opus", "application/ogg"):
        return "ogg"
    if mime_clean in ("audio/m4a", "audio/mp4", "audio/x-m4a"):
        return "m4a"
    if mime_clean in ("audio/aac", "audio/x-aac"):
        return "aac"
    if mime_clean in ("audio/flac", "audio/x-flac"):
        return "flac"
    if mime_clean in ("audio/webm",):
        return "webm"
    if suffix:
        s = suffix.lower().lstrip(".")
        if s in ("wav", "mp3", "ogg", "oga", "m4a", "aac", "flac", "webm", "opus"):
            return "ogg" if s in ("oga", "opus") else s
    return "ogg"


def _file_to_audio_part(path: Path) -> Optional[Dict[str, Any]]:
    """Encode local audio file as an input_audio multimodal part."""
    try:
        from agent.file_safety import raise_if_read_blocked

        raise_if_read_blocked(str(path))
    except ValueError as exc:
        logger.warning("audio_routing: blocked local audio attachment %s -- %s", path, exc)
        return None
    except Exception:
        pass

    try:
        raw = path.read_bytes()
    except Exception as exc:
        logger.warning("audio_routing: failed to read %s — %s", path, exc)
        return None

    if not raw:
        logger.warning("audio_routing: audio file is empty: %s", path)
        return None

    mime = _guess_audio_mime(path, raw=raw)
    fmt = _mime_to_audio_format(mime, path.suffix)
    b64 = base64.b64encode(raw).decode("ascii")

    return {
        "type": "input_audio",
        "input_audio": {
            "data": b64,
            "format": fmt,
        },
    }


def build_native_audio_content_parts(
    user_text: str,
    audio_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build multimodal content parts for a user turn with audio attachments.

    Returns ``(content_parts, skipped)``.
    """
    skipped: List[str] = []
    audio_parts: List[Dict[str, Any]] = []
    attached_paths: List[str] = []

    for raw_path in audio_paths:
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            skipped.append(str(raw_path))
            continue
        part = _file_to_audio_part(p)
        if not part:
            skipped.append(str(raw_path))
            continue
        audio_parts.append(part)
        attached_paths.append(str(raw_path))

    text = (user_text or "").strip()

    if attached_paths:
        base_text = text or "Listen to the attached audio and respond."
        hint_lines: List[str] = []
        hint_lines.extend(f"[Audio attached at: {p}]" for p in attached_paths)
        combined_text = f"{base_text}\n\n" + "\n".join(hint_lines)
        parts: List[Dict[str, Any]] = [{"type": "text", "text": combined_text}]
        parts.extend(audio_parts)
        return parts, skipped

    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    return parts, skipped


__all__ = [
    "decide_audio_input_mode",
    "build_native_audio_content_parts",
    "extract_audio_refs",
]
