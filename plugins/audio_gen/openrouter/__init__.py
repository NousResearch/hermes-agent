"""OpenRouter audio generation backend.

Surface: text-to-music through OpenRouter's audio-output chat models. One
``OPENROUTER_API_KEY`` routes to Google's Lyria 3 (music, 48kHz stereo,
optional vocals/lyrics) — the same key the agent already uses for its main
model, so no extra setup is needed when Hermes runs on OpenRouter.

Why chat/completions: OpenRouter has **no** dedicated ``/audio/generate``
endpoint. Music models are ordinary chat-completions models that declare
``audio`` in their output modalities. Verified live against OpenRouter's
API, the call requires BOTH ``modalities: ["text", "audio"]`` (audio-only
is rejected) and ``stream: true`` (non-streaming returns HTTP 400). The
base64 audio arrives across streamed ``delta.audio.data`` chunks, which we
concatenate.

Scope: this backend targets the **Lyria** music family. OpenAI's
``gpt-audio`` models also advertise audio output but additionally require
``audio.voice`` and behave like speech synthesis — that belongs to the
``text_to_speech`` tool (OpenRouter TTS built-in), not audio generation, so
they are excluded from this catalog.

Flow:
  1. ``POST {base}/chat/completions`` (streamed) with the prompt as the user
     message, ``modalities: ["text", "audio"]`` and ``audio: {format}``.
  2. Concatenate the base64 from each ``delta.audio.data`` chunk, decode,
     and write it to the audio-gen cache; return the absolute path. The
     gateway delivers the file.

Model discovery: the audio-output catalog is fetched live from
``GET {base}/models`` filtered on ``architecture.output_modalities``
containing ``"audio"`` AND the Lyria family, cached for the process, with a
small static fallback when the network call fails.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from agent.audio_gen_provider import (
    COMMON_AUDIO_FORMATS,
    DEFAULT_AUDIO_FORMAT,
    AudioGenProvider,
    error_response,
    save_b64_audio,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/lyria-3-pro-preview"
DEFAULT_TIMEOUT_SECONDS = 300
MODELS_CACHE_TTL_SECONDS = 3600

# OpenRouter audio output is delivered as wav or mp3 (the ChatAudioOutput
# format field). We default to mp3 and only forward formats we know the
# chat-audio path accepts.
_SUPPORTED_FORMATS = ("mp3", "wav")

# Audio-generation models this backend serves. OpenRouter exposes more
# audio-OUTPUT models (openai/gpt-audio*) but those are speech models that
# require ``audio.voice`` — they belong to text_to_speech, not audio gen.
# We keep this backend scoped to the Lyria music family.
_AUDIO_GEN_MODEL_PREFIXES = ("google/lyria",)

# Static fallback catalog — used only when the live /models call fails
# (offline, transient error). The live fetch is the source of truth.
# NOT asserted by any test (catalog data changes upstream).
_FALLBACK_MODELS: List[Dict[str, Any]] = [
    {
        "id": "google/lyria-3-pro-preview",
        "display": "Lyria 3 Pro",
        "strengths": "Music, 48kHz stereo, optional vocals + lyrics",
        "kinds": ["music"],
        "supports_lyrics": True,
    },
    {
        "id": "google/lyria-3-clip-preview",
        "display": "Lyria 3 Clip",
        "strengths": "Short music clips with vocals",
        "kinds": ["music"],
        "supports_lyrics": True,
    },
]

# Process-wide cache for the live audio-model catalog: (timestamp, models).
_models_cache: tuple[float, List[Dict[str, Any]]] | None = None


# ---------------------------------------------------------------------------
# Credential + HTTP helpers
# ---------------------------------------------------------------------------


def _resolve_credentials() -> tuple[str, str]:
    """Return ``(api_key, base_url)`` from the shared OpenRouter resolver."""
    try:
        from tools.tool_backend_helpers import resolve_openrouter_credentials

        creds = resolve_openrouter_credentials()
        return creds["api_key"], creds["base_url"]
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenRouter credential resolver failed: %s", exc)
        return "", "https://openrouter.ai/api/v1"


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
        "X-Title": "Hermes Agent",
    }


def _fetch_models() -> List[Dict[str, Any]]:
    """Fetch the live audio-output model catalog, cached for the process.

    Filters ``GET /models`` on ``architecture.output_modalities`` containing
    ``"audio"`` AND a known audio-generation (Lyria) model prefix — the
    gpt-audio speech models are deliberately excluded (they belong to TTS).
    On any failure returns the static fallback so the picker / capabilities
    never crash on a network blip.
    """
    global _models_cache
    now = time.time()
    if _models_cache is not None and (now - _models_cache[0]) < MODELS_CACHE_TTL_SECONDS:
        return _models_cache[1]

    api_key, base_url = _resolve_credentials()
    if not api_key:
        return _FALLBACK_MODELS
    try:
        resp = httpx.get(
            f"{base_url}/models",
            headers=_headers(api_key),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        out: List[Dict[str, Any]] = []
        for entry in data if isinstance(data, list) else []:
            arch = entry.get("architecture") or {}
            out_modalities = arch.get("output_modalities") or []
            if "audio" not in out_modalities:
                continue
            mid = entry.get("id")
            if not mid:
                continue
            if not any(mid.lower().startswith(p) for p in _AUDIO_GEN_MODEL_PREFIXES):
                continue
            out.append({
                "id": mid,
                "display": entry.get("name", mid),
                "strengths": (entry.get("description") or "")[:120],
                "kinds": ["music"],
                "supports_lyrics": True,  # Lyria family does vocals/lyrics
            })
        if out:
            _models_cache = (now, out)
            return out
    except Exception as exc:  # noqa: BLE001
        logger.debug("OpenRouter audio model fetch failed: %s", exc)
    return _FALLBACK_MODELS


def _model_entry(model_id: str) -> Optional[Dict[str, Any]]:
    """Return the catalog entry for *model_id*, or None."""
    for entry in _fetch_models():
        if entry.get("id") == model_id:
            return entry
    return None


def _clamp_format(audio_format: Optional[str]) -> str:
    fmt = (audio_format or DEFAULT_AUDIO_FORMAT).strip().lower()
    return fmt if fmt in _SUPPORTED_FORMATS else "mp3"


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenRouterAudioGenProvider(AudioGenProvider):
    """OpenRouter audio backend (music / audio from a text prompt)."""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        api_key, _ = _resolve_credentials()
        return bool(api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_fetch_models())

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter Audio",
            "badge": "paid",
            "tag": (
                "One OpenRouter key for Google Lyria 3 music generation "
                "(text-to-music with optional lyrics); uses OPENROUTER_API_KEY"
            ),
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "kinds": ["music"],
            "formats": list(_SUPPORTED_FORMATS),
            "max_duration": 60,
            "min_duration": 1,
            "supports_lyrics": True,
            "supports_negative_prompt": False,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
        negative_prompt: Optional[str] = None,
        lyrics: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key, base_url = _resolve_credentials()
        if not api_key:
            return error_response(
                error=(
                    "No OpenRouter credentials found. Set OPENROUTER_API_KEY "
                    "(https://openrouter.ai/keys) or run `hermes setup`."
                ),
                error_type="auth_required",
                provider="openrouter", prompt=prompt,
            )

        prompt = (prompt or "").strip()
        if not prompt:
            return error_response(
                error="prompt is required for OpenRouter audio generation",
                error_type="missing_prompt",
                provider="openrouter", prompt=prompt,
            )

        resolved_model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        # Reject models this backend doesn't serve. The catalog is scoped to
        # the Lyria music family; openai/gpt-audio* require audio.voice and
        # belong to text_to_speech, not audio generation. Honour the tool
        # schema's promise that unknown models are rejected.
        if not resolved_model.lower().startswith(_AUDIO_GEN_MODEL_PREFIXES):
            return error_response(
                error=(
                    f"Model '{resolved_model}' is not an audio-generation model "
                    f"for this backend. Use a Lyria model (e.g. {DEFAULT_MODEL}); "
                    f"gpt-audio models are speech models — use text_to_speech."
                ),
                error_type="unsupported_model",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )
        fmt = _clamp_format(audio_format)

        # Clamp duration to the advertised range (matches video_gen's
        # provider-side clamping; the tool layer does only soft validation).
        clamped_duration: Optional[int] = None
        if duration is not None:
            try:
                clamped_duration = max(1, min(int(duration), 60))
            except (TypeError, ValueError):
                clamped_duration = None

        # Build the user message. Lyrics ride along in the prompt text so
        # Lyria's song mode picks them up; instrumental models ignore them.
        user_content = prompt
        if lyrics:
            user_content = f"{prompt}\n\nLyrics:\n{lyrics.strip()}"
        if clamped_duration:
            user_content = f"{user_content}\n\nTarget duration: ~{clamped_duration} seconds."

        # OpenRouter audio-output models (Lyria 3, etc.) require BOTH:
        #   * modalities == ["text", "audio"]  (NOT ["audio"] — rejected)
        #   * stream: true                     (non-streaming returns 400)
        # The audio is delivered as base64 across streamed ``delta.audio.data``
        # chunks that we concatenate. Verified live against
        # google/lyria-3-clip-preview (see PR notes).
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "modalities": ["text", "audio"],
            "audio": {"format": fmt},
            "stream": True,
            "messages": [{"role": "user", "content": user_content}],
        }
        if seed is not None:
            payload["seed"] = seed

        audio_parts: List[str] = []
        transcript_parts: List[str] = []
        usage: Optional[Dict[str, Any]] = None
        saw_done = False
        stream_error: Optional[str] = None
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=_headers(api_key),
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            ) as resp:
                if not 200 <= resp.status_code < 300:
                    detail = ""
                    try:
                        detail = resp.read().decode("utf-8", "replace")[:500]
                    except Exception:
                        pass
                    return error_response(
                        error=f"OpenRouter audio request failed ({resp.status_code}): {detail}",
                        error_type="api_error",
                        provider="openrouter", model=resolved_model, prompt=prompt,
                    )
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].lstrip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(data)
                    except (ValueError, TypeError):
                        continue
                    # Streaming APIs can emit an error object inside a 200.
                    if isinstance(chunk.get("error"), dict):
                        stream_error = str(chunk["error"].get("message") or chunk["error"])
                        break
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta") if choices else None
                    aud = delta.get("audio") if isinstance(delta, dict) else None
                    if isinstance(aud, dict):
                        if aud.get("data"):
                            audio_parts.append(aud["data"])
                        if aud.get("transcript"):
                            transcript_parts.append(aud["transcript"])
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
        except httpx.TimeoutException as exc:
            return error_response(
                error=f"OpenRouter audio request timed out: {exc}",
                error_type="timeout",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )
        except httpx.HTTPError as exc:
            return error_response(
                error=f"OpenRouter audio network error: {exc}",
                error_type="network_error",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter audio gen unexpected failure: %s", exc, exc_info=True)
            return error_response(
                error=f"OpenRouter audio generation failed: {exc}",
                error_type="api_error",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )

        if stream_error:
            return error_response(
                error=f"OpenRouter audio stream error: {stream_error}",
                error_type="api_error",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )

        b64 = "".join(audio_parts)
        if not b64:
            return error_response(
                error=(
                    "OpenRouter audio response did not include audio data. "
                    "The selected model may not support music/audio output — "
                    "pick a Lyria model via `hermes tools` → Audio Generation."
                ),
                error_type="empty_response",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )

        # Guard against a stream that delivered partial audio then dropped
        # before signalling completion — saving that as success would yield a
        # truncated/corrupt file.
        if not saw_done:
            return error_response(
                error=(
                    "OpenRouter audio stream ended before completion "
                    "([DONE] not received); the audio may be truncated."
                ),
                error_type="incomplete_stream",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )

        try:
            path = save_b64_audio(b64, prefix="openrouter", extension=fmt)
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not save OpenRouter audio: {exc}",
                error_type="io_error",
                provider="openrouter", model=resolved_model, prompt=prompt,
            )

        extra: Dict[str, Any] = {}
        if usage:
            extra["usage"] = usage
        transcript = "".join(transcript_parts)
        if transcript:
            extra["transcript"] = transcript

        return success_response(
            audio=str(path),
            model=resolved_model,
            prompt=prompt,
            duration=clamped_duration or 0,
            audio_format=fmt,
            provider="openrouter",
            extra=extra or None,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenRouterAudioGenProvider`` into the registry."""
    ctx.register_audio_gen_provider(OpenRouterAudioGenProvider())
