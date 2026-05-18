"""OpenRouter image generation backend.

Uses OpenRouter's Chat Completions image-output surface:
``POST /api/v1/chat/completions`` with ``modalities`` and optional
``image_config``. Responses contain generated images as base64 data URLs in
``choices[0].message.images[*].image_url.url``. The provider saves data URLs
under ``$HERMES_HOME/cache/images/`` and returns the absolute file path.

Selection precedence:
1. ``model=`` passed by the tool wrapper (usually ``image_gen.model``)
2. ``OPENROUTER_IMAGE_MODEL`` env var
3. ``image_gen.openrouter.model`` in config.yaml
4. ``image_gen.model`` in config.yaml
5. ``DEFAULT_MODEL``

Any configured model string is accepted so users can use newly-added
OpenRouter image-output models without waiting for Hermes catalog updates.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-pro-image-preview"
DEFAULT_IMAGE_SIZE = "1K"

_MODELS: Dict[str, Dict[str, Any]] = {
    "google/gemini-3-pro-image-preview": {
        "display": "Gemini 3 Pro Image Preview",
        "speed": "varies",
        "strengths": "High prompt adherence, reasoning, text rendering",
        "price": "see OpenRouter",
        "modalities": ["image", "text"],
    },
    "google/gemini-3.1-flash-image-preview": {
        "display": "Gemini 3.1 Flash Image Preview",
        "speed": "fast",
        "strengths": "Fast image generation; extended aspect ratios / 0.5K",
        "price": "see OpenRouter",
        "modalities": ["image", "text"],
    },
    "google/gemini-2.5-flash-image": {
        "display": "Gemini 2.5 Flash Image",
        "speed": "fast",
        "strengths": "Fast text+image output",
        "price": "see OpenRouter",
        "modalities": ["image", "text"],
    },
    "black-forest-labs/flux.2-pro": {
        "display": "FLUX.2 Pro",
        "speed": "varies",
        "strengths": "Photorealistic image-only generation",
        "price": "see OpenRouter",
        "modalities": ["image"],
    },
    "black-forest-labs/flux.2-flex": {
        "display": "FLUX.2 Flex",
        "speed": "varies",
        "strengths": "Flexible image-only generation",
        "price": "see OpenRouter",
        "modalities": ["image"],
    },
    "sourceful/riverflow-v2-standard-preview": {
        "display": "Sourceful Riverflow v2 Standard Preview",
        "speed": "varies",
        "strengths": "Design/image generation",
        "price": "see OpenRouter",
        "modalities": ["image"],
    },
}

_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def _load_image_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Optional[str]] = [explicit, os.environ.get("OPENROUTER_IMAGE_MODEL")]
    cfg = _load_image_gen_section()
    or_cfg = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
    if isinstance(or_cfg, dict):
        candidates.append(or_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            model_id = candidate.strip()
            return model_id, _MODELS.get(model_id, {
                "display": model_id,
                "speed": "unknown",
                "strengths": "Custom OpenRouter image-output model",
                "price": "see OpenRouter",
                "modalities": ["image", "text"],
            })
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def _resolve_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL


def _modalities_for_model(meta: Dict[str, Any]) -> List[str]:
    modalities = meta.get("modalities")
    if isinstance(modalities, list) and "image" in modalities:
        # Gemini-like models want text+image; image-only models can use image.
        return [str(m) for m in modalities]
    return ["image", "text"]


def _extract_data_url_b64(data_url: str) -> Tuple[str, str]:
    if not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError("not a base64 data URL")
    header, b64 = data_url.split(",", 1)
    mime = header[5:].split(";", 1)[0].lower()
    ext = "png"
    if mime.endswith("jpeg") or mime.endswith("jpg"):
        ext = "jpg"
    elif mime.endswith("webp"):
        ext = "webp"
    elif mime.endswith("gif"):
        ext = "gif"
    return b64, ext


class OpenRouterImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "paid",
            "tag": "OpenRouter image-output models via Chat Completions (Gemini, FLUX, Sourceful, etc.)",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/settings/keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id, meta = _resolve_model(kwargs.get("model"))

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openrouter",
                model=model_id,
                aspect_ratio=aspect,
            )

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "OPENROUTER_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenRouter to configure, or set OPENROUTER_API_KEY."
                ),
                error_type="auth_required",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        cfg = _load_image_gen_section()
        or_cfg = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
        image_size = str(
            kwargs.get("image_size")
            or os.environ.get("OPENROUTER_IMAGE_SIZE")
            or (or_cfg.get("image_size", "") if isinstance(or_cfg, dict) else "")
            or DEFAULT_IMAGE_SIZE
        ).strip()

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": _modalities_for_model(meta),
            "stream": False,
            "image_config": {
                "aspect_ratio": _ASPECT_RATIOS.get(aspect, "16:9"),
                "image_size": image_size,
            },
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-Title": "Hermes Agent",
        }

        try:
            with httpx.Client(timeout=180) as client:
                response = client.post(
                    f"{_resolve_base_url()}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            return error_response(
                error=f"OpenRouter image generation failed ({exc.response.status_code}): {detail}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("OpenRouter image generation failed", exc_info=True)
            return error_response(
                error=f"OpenRouter image generation failed: {exc}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        choices = body.get("choices") if isinstance(body, dict) else None
        message = (choices or [{}])[0].get("message", {}) if choices else {}
        images = message.get("images") if isinstance(message, dict) else None
        if not images:
            return error_response(
                error="OpenRouter response did not include message.images",
                error_type="empty_response",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = images[0] if isinstance(images, list) and images else {}
        image_url = ((first.get("image_url") or {}).get("url") if isinstance(first, dict) else None) or ""
        if not image_url:
            return error_response(
                error="OpenRouter image entry did not include image_url.url",
                error_type="empty_response",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_ref = image_url
        if image_url.startswith("data:"):
            try:
                b64, ext = _extract_data_url_b64(image_url)
                # Validate base64 early so save_b64_image raises only IO/decode issues.
                base64.b64decode(b64)
                image_ref = str(save_b64_image(b64, prefix="openrouter", extension=ext))
            except Exception as exc:
                return error_response(
                    error=f"Could not save OpenRouter data URL image: {exc}",
                    error_type="io_error",
                    provider="openrouter",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

        extra: Dict[str, Any] = {
            "image_config": payload["image_config"],
            "modalities": payload["modalities"],
        }
        if message.get("content"):
            extra["content"] = message["content"]
        if body.get("usage"):
            extra["usage"] = body["usage"]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openrouter",
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(OpenRouterImageGenProvider())
