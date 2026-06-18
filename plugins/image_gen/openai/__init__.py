"""OpenAI image generation backend.

Exposes OpenAI image models (gpt-image-1, gpt-image-1.5, gpt-image-2) as an
:class:`ImageGenProvider` implementation. Models that support the ``quality``
parameter (gpt-image-2, gpt-image-1.5) expose three quality tiers:

    <model>-low     ~15s   fastest, good for iteration
    <model>-medium  ~40s   default — balanced
    <model>-high    ~2min  slowest, highest fidelity

Models without quality support (gpt-image-1, gpt-image-1-mini) use a flat
tier mapped directly to the model name, sending no ``quality`` parameter.

Configuration precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (any model name or tier ID)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml``
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# ``_MODELS`` maps tier IDs (e.g. "gpt-image-2-medium") to their metadata.
# Each entry has an ``api_model`` field that determines what's sent to OpenAI.
# Models that support ``quality`` include it; others omit it.

_MODELS: Dict[str, Dict[str, Any]] = {
    # gpt-image-2 tiers (supports quality parameter)
    "gpt-image-2-low": {
        "api_model": "gpt-image-2",
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "api_model": "gpt-image-2",
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "api_model": "gpt-image-2",
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
    # gpt-image-1.5 tiers (supports quality parameter)
    "gpt-image-1.5-low": {
        "api_model": "gpt-image-1.5",
        "display": "GPT Image 1.5 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-1.5-medium": {
        "api_model": "gpt-image-1.5",
        "display": "GPT Image 1.5 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced",
        "quality": "medium",
    },
    "gpt-image-1.5-high": {
        "api_model": "gpt-image-1.5",
        "display": "GPT Image 1.5 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity",
        "quality": "high",
    },
    # gpt-image-1 (no quality parameter — flat tier)
    "gpt-image-1": {
        "api_model": "gpt-image-1",
        "display": "GPT Image 1",
        "speed": "~40s",
        "strengths": "Balanced quality, no quality tiers",
    },
    # gpt-image-1-mini (no quality parameter — flat tier)
    "gpt-image-1-mini": {
        "api_model": "gpt-image-1-mini",
        "display": "GPT Image 1 Mini",
        "speed": "~10s",
        "strengths": "Fastest, lowest cost",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
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


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model/tier to use and return ``(tier_id, meta)``.

    Accepts both tier IDs (e.g. ``gpt-image-2-medium``) and bare model names
    (e.g. ``gpt-image-1``). Bare names that aren't in the catalog are passed
    through as-is with default metadata (no quality parameter).
    """
    # 1. Env override
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override:
        if env_override in _MODELS:
            return env_override, _MODELS[env_override]
        # Pass through unknown model names with default metadata
        return env_override, {"api_model": env_override, "display": env_override,
                              "speed": "unknown", "strengths": "Custom model"}

    # 2-3. Config file: image_gen.openai.model, then image_gen.model
    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top.strip():
            candidate = top.strip()

    if candidate is not None:
        # Exact tier ID match
        if candidate in _MODELS:
            return candidate, _MODELS[candidate]
        # Pass through bare model names or custom values
        return candidate, {"api_model": candidate, "display": candidate,
                           "speed": "unknown", "strengths": "Custom model"}

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — supports gpt-image-1/1.5/2."""

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
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-1 / gpt-image-1.5 / gpt-image-2 with quality tiers",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
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

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        # Determine the actual API model name to send to OpenAI.
        api_model = meta.get("api_model", tier_id)

        # gpt-image-2/1.5 return b64_json unconditionally and REJECT
        # ``response_format`` as an unknown parameter. Don't send it.
        # gpt-image-1/1-mini also return b64_json but do NOT support
        # the ``quality`` parameter.
        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
        # Only include quality for models that support it
        if "quality" in meta:
            payload["quality"] = meta["quality"]

        try:
            client = openai.OpenAI()
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64 today, but OpenAI's API
            # has previously returned URLs.  Cache the bytes locally so the
            # gateway never tries to fetch an ephemeral / signed URL after
            # it expires — same rationale as the xAI provider (#26942).
            try:
                saved_path = save_url_image(url, prefix=f"openai_{tier_id}")
            except Exception as exc:
                logger.warning(
                    "OpenAI image URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "api_model": api_model}
        if "quality" in meta:
            extra["quality"] = meta["quality"]
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
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