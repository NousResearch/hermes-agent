"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model as an :class:`ImageGenProvider`
implementation. We intentionally only support this one model — the older
``gpt-image-1.5`` / ``gpt-image-1`` / ``dall-e-*`` models are slower, lower
quality, or have quirky parameter constraints (dall-e-2 squares only, etc.)
with nothing to gain.

Outputs are base64 JSON → saved under ``$HERMES_HOME/cache/images/``.

Config overrides live at ``image_gen.openai.*`` in ``config.yaml``. Today
that's just ``quality`` (``low`` / ``medium`` / ``high`` / ``auto``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_ID = "gpt-image-2"

MODEL_META: Dict[str, Any] = {
    "display": "GPT Image 2",
    "speed": "~15-50s",
    "strengths": "Highest quality, strong prompt adherence, excellent text rendering",
    "price": "varies",
    "sizes": {
        "landscape": "1536x1024",
        "square": "1024x1024",
        "portrait": "1024x1536",
    },
}


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 only."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": MODEL_ID,
                "display": MODEL_META["display"],
                "speed": MODEL_META["speed"],
                "strengths": MODEL_META["strengths"],
                "price": MODEL_META["price"],
            }
        ]

    def default_model(self) -> Optional[str]:
        return MODEL_ID

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        if not os.environ.get("OPENAI_API_KEY"):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        size = MODEL_META["sizes"].get(aspect, MODEL_META["sizes"]["square"])

        payload: Dict[str, Any] = {
            "model": MODEL_ID,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }

        # gpt-image-2 unconditionally returns b64_json and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        cfg = _load_openai_config()
        openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
        if isinstance(openai_cfg, dict):
            quality = openai_cfg.get("quality")
            if isinstance(quality, str) and quality and quality != "auto":
                payload["quality"] = quality

        try:
            client = openai.OpenAI()
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=MODEL_ID,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=MODEL_ID,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix="openai_gpt-image-2")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=MODEL_ID,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64, but fall back gracefully
            # if OpenAI ever changes the response shape.
            image_ref = url
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=MODEL_ID,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=MODEL_ID,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
