"""Gemini (Google AI Studio) image generation via ``generateContent``.

Native ``https://generativelanguage.googleapis.com/v1beta`` backend using
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` directly — no FAL/OpenRouter proxy.
Selection: ``model`` kwarg → ``GEMINI_IMAGE_MODEL`` → ``image_gen.gemini.model``
→ ``image_gen.model`` → :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from agent.secret_scope import get_secret
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config,
    post_json, prompt_required_error, resolve_static_model,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_REQUEST_TIMEOUT = 300.0
_MAX_REFERENCE_IMAGES = 3  # Gemini image models accept up to 3 input images per prompt.

_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-2.5-flash-image": {
        "display": "Nano Banana (Gemini 2.5 Flash Image)",
        "speed": "~5-15s",
        "strengths": "Fast and cheap; good for iteration and edits",
    },
    "gemini-3-pro-image-preview": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "~30-60s",
        "strengths": "Highest fidelity; best prompt adherence",
    },
}
DEFAULT_MODEL = "gemini-2.5-flash-image"

# Semantic ratio → Gemini ``imageConfig.aspectRatio``.
_ASPECT_RATIOS = {"square": "1:1", "landscape": "16:9", "portrait": "9:16"}


def _resolve_api_key() -> str:
    """``GOOGLE_API_KEY`` first, ``GEMINI_API_KEY`` as alias (mirrors the text provider)."""
    return (get_secret("GOOGLE_API_KEY", "") or "").strip() or (get_secret("GEMINI_API_KEY", "") or "").strip()


def _resolve_model(explicit: Optional[str] = None) -> str:
    """Model id per the standard precedence chain (unknown ids fall through)."""
    model_id, _meta = resolve_static_model(
        _MODELS, DEFAULT_MODEL, env_var="GEMINI_IMAGE_MODEL", config_key="gemini", explicit=explicit,
    )
    return model_id


def _inline_ref(ref: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(inlineData part, None)`` for a data-URI/local-file ref; ``(None, reason)`` otherwise."""
    ref = (ref or "").strip()
    if ref.startswith("data:"):
        try:
            header, b64 = ref.split(",", 1)
            mime = header.split(";")[0].split(":", 1)[1] or "image/png"
        except (IndexError, ValueError):
            return None, f"Skipping malformed data-URI reference ({ref[:40]}…)"
        return {"inlineData": {"mimeType": mime, "data": b64}}, None
    if ref.startswith(("http://", "https://")):
        return None, (
            f"Remote reference images are not supported by this backend ({ref[:60]}…); "
            "pass a local file path or data URI instead."
        )
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before inlining

    raise_if_read_blocked(ref)
    try:
        raw = Path(ref).read_bytes()
    except OSError as exc:
        return None, f"Skipping unreadable reference image {ref}: {exc}"
    mime = mimetypes.guess_type(Path(ref).name)[0] or "image/png"
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode("ascii")}}, None


def _extract_b64(body: Any) -> Tuple[Optional[str], Optional[str]]:
    """``(b64, None)`` for the first inline image part, else ``(None, reason)``."""
    candidates = body.get("candidates") if isinstance(body, dict) else None
    parts: List[Any] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            content = candidate.get("content") if isinstance(candidate, dict) else None
            content_parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(content_parts, list):
                parts.extend(content_parts)
    for part in parts:
        inline = part.get("inlineData") if isinstance(part, dict) else None
        data = inline.get("data") if isinstance(inline, dict) else None
        if isinstance(data, str) and data:
            return data, None
    if isinstance(body, dict):
        feedback = body.get("promptFeedback") or {}
        reason = feedback.get("blockReason") if isinstance(feedback, dict) else None
        if reason:
            return None, f"Gemini blocked the request (blockReason: {reason})"
    return None, "Gemini returned no image data"


class GeminiImageGenProvider(StaticImageGenProvider):
    """Native Google AI Studio ``generateContent`` image backend."""

    provider_id = "gemini"
    label = "Google AI Studio"
    models = _MODELS
    default_model_id = DEFAULT_MODEL
    setup = dict(
        name="Google AI Studio", badge="paid",
        tag="Nano Banana / Nano Banana Pro direct from Google — text-to-image & image editing",
        key="GOOGLE_API_KEY", prompt="Google AI Studio API key",
        url="https://aistudio.google.com/apikey")

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": _MAX_REFERENCE_IMAGES}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id = _resolve_model(kwargs.get("model") if isinstance(kwargs.get("model"), str) else None)
        fail = error_factory("gemini", aspect, model=model_id, prompt=prompt)
        if not prompt:
            return prompt_required_error("gemini", aspect)
        api_key = _resolve_api_key()
        if not api_key:
            return fail(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Run `hermes tools` → Image "
                "Generation → Google AI Studio to configure, or get a key at "
                "https://aistudio.google.com/apikey.",
                "auth_required")
        parts: List[Dict[str, Any]] = []
        for ref in collect_source_images(image_url, reference_image_urls, limit=_MAX_REFERENCE_IMAGES):
            inline, skip = _inline_ref(ref)
            if inline is not None:
                parts.append(inline)
            else:
                logger.debug("Gemini reference dropped: %s", skip)
                if ref.startswith(("http://", "https://")):
                    return fail(str(skip), "modality_unsupported")
        parts.append({"text": prompt})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": _ASPECT_RATIOS.get(aspect, "16:9")},
            },
        }
        url = f"{_BASE_URL}/models/{model_id}:generateContent?key={api_key}"
        body, failure = post_json(
            url, headers={"Content-Type": "application/json"}, payload=payload,
            timeout=_REQUEST_TIMEOUT, label="Gemini",
        )
        if failure is not None:
            return fail(failure.error, failure.error_type)
        b64, reason = _extract_b64(body)
        if not b64:
            return fail(str(reason), "empty_response")
        from agent.image_gen_provider import save_b64_image

        try:
            image_ref = str(save_b64_image(b64, prefix=f"gemini_{model_id.replace(':', '_')}"))
        except Exception as exc:  # noqa: BLE001
            return fail(f"Could not save image to cache: {exc}", "io_error")
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="gemini")


def register(ctx) -> None:
    """Plugin entry point — wire ``GeminiImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(GeminiImageGenProvider())
