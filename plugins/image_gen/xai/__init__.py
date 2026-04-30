"""xAI image generation backend.

Exposes xAI's ``grok-imagine-image`` model as an
:class:`ImageGenProvider` implementation.

Features:
- Text-to-image generation
- Multiple aspect ratios (1:1, 16:9, 9:16, etc.)
- Multiple resolutions (1K, 2K)
- Base64 output saved to cache

Selection precedence (first hit wins):
1. ``XAI_IMAGE_MODEL`` env var
2. ``image_gen.xai.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from tools.xai_http import hermes_xai_user_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

API_MODEL = "grok-imagine-image"

_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-imagine-image": {
        "display": "Grok Imagine Image",
        "speed": "~5-10s",
        "strengths": "Fast, high-quality",
    },
}

DEFAULT_MODEL = "grok-imagine-image"

# xAI aspect ratios (more options than FAL/OpenAI)
_XAI_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
    "3:2": "3:2",
    "2:3": "2:3",
}

# xAI resolutions
_XAI_RESOLUTIONS = {
    "1k": "1024",
    "2k": "2048",
}

DEFAULT_RESOLUTION = "1k"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_config() -> Dict[str, Any]:
    """Read ``image_gen.xai`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        xai_section = section.get("xai") if isinstance(section, dict) else None
        return xai_section if isinstance(xai_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.xai config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("XAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_xai_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate and candidate in _MODELS:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_resolution() -> str:
    """Get configured resolution."""
    cfg = _load_xai_config()
    res = cfg.get("resolution") if isinstance(cfg.get("resolution"), str) else None
    if res and res in _XAI_RESOLUTIONS:
        return res
    return DEFAULT_RESOLUTION


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIImageGenProvider(ImageGenProvider):
    """xAI ``grok-imagine-image`` backend."""

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI (Grok)"

    def is_available(self) -> bool:
        return bool(os.getenv("XAI_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
            }
            for model_id, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "xAI (Grok)",
            "badge": "paid",
            "tag": "Native xAI image generation via grok-imagine-image",
            "env_vars": [
                {
                    "key": "XAI_API_KEY",
                    "prompt": "xAI API key",
                    "url": "https://console.x.ai/",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image using xAI's grok-imagine-image."""
        api_key = os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            return error_response(
                error="XAI_API_KEY not set. Get one at https://console.x.ai/",
                error_type="missing_api_key",
                provider="xai",
                aspect_ratio=aspect_ratio,
            )

        model_id, meta = _resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)
        xai_ar = _XAI_ASPECT_RATIOS.get(aspect, "1:1")
        resolution = _resolve_resolution()
        xai_res = _XAI_RESOLUTIONS.get(resolution, "1024")

        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "aspect_ratio": xai_ar,
            "resolution": xai_res,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        }

        base_url = (os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1").strip().rstrip("/")

        try:
            response = requests.post(
                f"{base_url}/images/generations",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            try:
                err_msg = response.json().get("error", {}).get("message", response.text[:300])
            except Exception:
                err_msg = response.text[:300] if response is not None else str(exc)
            logger.error("xAI image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"xAI image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="xAI image generation timed out (120s)",
                error_type="timeout",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"xAI connection error: {exc}",
                error_type="connection_error",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:
            return error_response(
                error=f"xAI returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Parse response — xAI returns data[0].b64_json or data[0].url
        data = result.get("data", [])
        if not data:
            return error_response(
                error="xAI returned no image data",
                error_type="empty_response",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = first.get("b64_json")
        url = first.get("url")

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"xai_{model_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="xai",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            image_ref = url
        else:
            return error_response(
                error="xAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "resolution": xai_res,
        }

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="xai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(XAIImageGenProvider())
