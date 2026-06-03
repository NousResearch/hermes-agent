"""OpenRouter image generation backend.

Supports all image-capable models on OpenRouter (GPT Image, FLUX, etc.)
via the chat completions API with ``modalities: ["image"]``.

The provider auto-selects a sensible default model (``openai/gpt-5.4-image-2``)
but users can override via ``image_gen.openrouter.model`` in ``config.yaml`` or
the ``OPENROUTER_IMAGE_MODEL`` environment variable.

Configuration via environment variables:

- ``OPENROUTER_API_KEY`` (required)
- ``OPENROUTER_IMAGE_MODEL`` (optional, e.g. ``openai/gpt-5.4-image-2``)
- ``OPENROUTER_IMAGE_BASE_URL`` (optional, defaults to ``https://openrouter.ai/api/v1``)

OpenRouter image models can be browsed at:
https://openrouter.ai/models?output_modalities=image
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-5.4-image-2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Map Hermes aspect_ratio → OpenRouter image_config.aspect_ratio
_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
    "wide": "16:9",
    "tall": "9:16",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> Optional[str]:
    """Return the OpenRouter API key from environment."""
    return os.environ.get("OPENROUTER_API_KEY")


def _get_base_url() -> str:
    """Return the OpenRouter API base URL (overridable via env)."""
    return os.environ.get("OPENROUTER_IMAGE_BASE_URL", DEFAULT_BASE_URL)


def _resolve_model() -> str:
    """Resolve the model ID from env or config, falling back to default."""
    env_override = os.environ.get("OPENROUTER_IMAGE_MODEL")
    if env_override:
        return env_override

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen", {})
        if isinstance(section, dict):
            # image_gen.openrouter.model takes priority
            or_cfg = section.get("openrouter", {})
            if isinstance(or_cfg, dict):
                model = or_cfg.get("model")
                if isinstance(model, str) and model.strip():
                    return model.strip()
            # Then image_gen.model (when it looks like an OpenRouter model ID)
            top_model = section.get("model")
            if isinstance(top_model, str) and "/" in top_model:
                return top_model.strip()
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)

    return DEFAULT_MODEL


def _save_image_to_cache(image_bytes: bytes, model: str) -> str:
    """Save raw image bytes to ``$HERMES_HOME/cache/images/`` and return the path."""
    from hermes_constants import get_hermes_home

    cache_dir = get_hermes_home() / "cache" / "images"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    safe_model = model.replace("/", "_").replace(":", "_")
    out_path = cache_dir / f"openrouter_{safe_model}_{ts}_{short}.png"
    out_path.write_bytes(image_bytes)
    return str(out_path)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenRouterImageGenProvider(ImageGenProvider):
    """OpenRouter image generation via chat completions + modalities."""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        if not _get_api_key():
            return False
        try:
            import httpx  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return a short list of popular image models on OpenRouter."""
        popular = [
            {
                "id": "openai/gpt-5.4-image-2",
                "display": "GPT 5.4 Image 2",
                "speed": "~10s",
                "strengths": "Latest OpenAI image model, high quality",
                "price": "pay-per-use",
            },
            {
                "id": "openai/gpt-5-image",
                "display": "GPT 5 Image",
                "speed": "~8s",
                "strengths": "Fast, good quality",
                "price": "pay-per-use",
            },
            {
                "id": "openai/gpt-5-image-mini",
                "display": "GPT 5 Image Mini",
                "speed": "~5s",
                "strengths": "Ultra-fast, economical",
                "price": "pay-per-use",
            },
            {
                "id": "black-forest-labs/flux-schnell",
                "display": "FLUX Schnell",
                "speed": "~2s",
                "strengths": "Ultra-fast, artistic styles",
                "price": "~$0.003 / img",
            },
            {
                "id": "black-forest-labs/flux-dev",
                "display": "FLUX Dev",
                "speed": "~10s",
                "strengths": "High quality, open weights",
                "price": "~$0.02 / img",
            },
            {
                "id": "google/gemini-3.1-flash-image-preview",
                "display": "Gemini Flash Image",
                "speed": "~5s",
                "strengths": "Fast, extended aspect ratios",
                "price": "~$0.01 / img",
            },
            {
                "id": "x-ai/grok-imagine-image-quality",
                "display": "Grok Imagine",
                "speed": "~10s",
                "strengths": "xAI's image model",
                "price": "~$0.01 / img",
            },
        ]
        return popular

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "pay-per-use",
            "tag": "GPT Image, FLUX, Imagen via chat completions",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/settings/keys",
                },
                {
                    "key": "OPENROUTER_IMAGE_MODEL",
                    "prompt": "Default image model (e.g. openai/gpt-5.4-image-2)",
                    "default": DEFAULT_MODEL,
                    "optional": True,
                },
            ],
        }

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = _resolve_model()

        api_key = _get_api_key()
        if not api_key:
            return error_response(
                error=(
                    "OPENROUTER_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenRouter to configure, or add it to ~/.hermes/.env."
                ),
                error_type="auth_required",
                provider=self.name,
                aspect_ratio=aspect,
                model=model,
            )

        # Build OpenRouter request
        base_url = _get_base_url()
        or_aspect = _ASPECT_RATIOS.get(aspect, "1:1")

        # Determine modalities: image models output images
        modalities = ["image", "text"]

        # Build image_config
        image_config: Dict[str, Any] = {"aspect_ratio": or_aspect}

        # Some models support image_size
        if "gemini" in model.lower():
            image_config["image_size"] = "1K"

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
            "X-Title": "Hermes Agent",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "modalities": modalities,
            "image_config": image_config,
            "stream": False,
        }

        # Call OpenRouter
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("OpenRouter image generation HTTP error: %s", exc)
            return error_response(
                error=f"OpenRouter request failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.warning("OpenRouter image generation error: %s", exc)
            return error_response(
                error=f"OpenRouter request failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Parse response
        try:
            data = resp.json()
        except Exception as exc:
            return error_response(
                error=f"Invalid JSON from OpenRouter: {exc}",
                error_type="parse_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        choices = data.get("choices", [])
        if not choices:
            return error_response(
                error="OpenRouter returned no choices in response",
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        message = choices[0].get("message", {})
        images = message.get("images", [])

        if not images:
            # Some models may return text only; surface the text as context
            text_content = message.get("content", "")
            return error_response(
                error=(
                    f"Model returned no image. Response text: "
                    f"{text_content[:200] if text_content else 'empty'}"
                ),
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first_image = images[0]
        image_url = first_image.get("image_url", {}).get("url")

        if not image_url:
            return error_response(
                error="OpenRouter image missing image_url",
                error_type="empty_response",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # If the URL is a data: URL (base64), save to disk and return path
        if image_url.startswith("data:image"):
            try:
                header, b64_data = image_url.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                image_ref = _save_image_to_cache(image_bytes, model)
            except Exception as exc:
                logger.warning("Could not save base64 image from OpenRouter: %s", exc)
                image_ref = image_url  # fall back to raw data URL
        else:
            image_ref = image_url

        return success_response(
            image=image_ref,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra={"modalities": modalities, "image_config": image_config},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenRouterImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenRouterImageGenProvider())
