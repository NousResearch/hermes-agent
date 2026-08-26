"""Meta Model API image generation backend.

Exposes Meta's ``muse-image`` model(s) as an :class:`ImageGenProvider`.
The Meta Model API (https://api.meta.ai/v1) is OpenAI-compatible, so we reuse
the OpenAI Python SDK pointed at Meta's base URL and authenticate with
``META_MODEL_API_KEY``.

Output is base64 JSON (WebP) -> saved under ``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):
  1. ``META_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
  2. ``image_gen.meta-ai.model`` in ``config.yaml``
  3. ``image_gen.model`` in ``config.yaml`` (when it's one of our IDs)
  4. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
# Auth env vars, in priority order. Mirrors the bundled ``meta-ai`` chat
# provider (plugins/model-providers/meta-ai): MODEL_API_KEY is Meta's
# documented var; the rest are accepted aliases.
API_KEY_ENVS = ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY")
# Primary key shown in setup prompts / error messages.
API_KEY_ENV = "META_MODEL_API_KEY"
# Optional base-url override (same var the chat provider honors).
BASE_URL_ENV = "META_BASE_URL"


def _resolve_api_key() -> Optional[str]:
    """First non-empty auth env var, checked in priority order."""
    for env in API_KEY_ENVS:
        val = get_secret(env)
        if val:
            return val
    return None


def _resolve_base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or "").strip() or DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
# Catalog shown in `hermes tools` and matched against `image_gen.model`.
# The model id is sent verbatim to the Meta Model API (`/v1/images/generations`).
_MODELS: Dict[str, Dict[str, Any]] = {
    "muse-image-1.0": {
        "display": "Muse Image 1.0",
        "speed": "~10s",
        "strengths": "Meta Model API image generation",
        "price": "$0.01/image",
    },
}
DEFAULT_MODEL = "muse-image-1.0"

# aspect_ratio -> OpenAI-style size string
_SIZES: Dict[str, str] = {
    "square": "1024x1024",
    "landscape": "1536x1024",
    "portrait": "1024x1536",
}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Return (model_id, metadata) using the documented precedence chain."""
    env_model = os.environ.get("META_IMAGE_MODEL")
    if env_model and env_model in _MODELS:
        return env_model, _MODELS[env_model]

    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        ig = cfg.get("image_gen") or {}
        scoped = (ig.get("meta-ai") or {}).get("model")
        if scoped and scoped in _MODELS:
            return scoped, _MODELS[scoped]
        top = ig.get("model")
        if top and top in _MODELS:
            return top, _MODELS[top]
    except Exception:
        logger.debug("Could not read image_gen model from config", exc_info=True)

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


class MetaImageGenProvider(ImageGenProvider):
    """Meta Model API ``images.generate`` backend (muse-image)."""

    @property
    def name(self) -> str:
        return "meta-ai"

    @property
    def display_name(self) -> str:
        return "Meta Model API"

    def is_available(self) -> bool:
        if not _resolve_api_key():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": m["display"],
                "speed": m["speed"],
                "strengths": m["strengths"],
                "price": m["price"],
            }
            for mid, m in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Meta Model API",
            "badge": "internal",
            "tag": "Muse Image via Meta Model API (api.meta.ai)",
            "env_vars": [
                {
                    "key": API_KEY_ENV,
                    "prompt": "Meta Model API key (LLM|... token)",
                    "url": "https://api.meta.ai",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Text-to-image only for now. Bump this once image-to-image is verified
        # against the Meta endpoint.
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="meta-ai",
                aspect_ratio=aspect,
            )

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    f"{API_KEY_ENV} not set. Run `hermes tools` -> Image "
                    "Generation -> Meta Model API to configure."
                ),
                error_type="auth_required",
                provider="meta-ai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="meta-ai",
                aspect_ratio=aspect,
            )

        model_id, _meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        client = openai.OpenAI(api_key=api_key, base_url=_resolve_base_url())

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }

        try:
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("Meta image generation failed", exc_info=True)
            return error_response(
                error=f"Meta image generation failed: {exc}",
                error_type="api_error",
                provider="meta-ai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            first = response.data[0]
        except (AttributeError, IndexError, TypeError):
            return error_response(
                error="Meta response contained no image data",
                error_type="empty_response",
                provider="meta-ai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)

        try:
            if b64:
                path = save_b64_image(b64, prefix="meta", extension="webp")
                image_ref = str(path)
            elif url:
                path = save_url_image(url, prefix="meta")
                image_ref = str(path)
            else:
                return error_response(
                    error="Meta response contained neither b64_json nor URL",
                    error_type="empty_response",
                    provider="meta-ai",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        except Exception as exc:
            return error_response(
                error=f"Failed to save Meta image: {exc}",
                error_type="io_error",
                provider="meta-ai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        revised_prompt = getattr(first, "revised_prompt", None)
        extra: Dict[str, Any] = {"size": size}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="meta-ai",
            modality="text",
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point -- wire ``MetaImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MetaImageGenProvider())
