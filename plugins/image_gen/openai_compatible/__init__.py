"""OpenAI-compatible image generation backend.

Routes Hermes' native ``image_generate`` tool through an OpenAI-compatible
``/images/generations`` endpoint, such as a self-hosted gateway or Bifrost.
This intentionally lives beside the direct ``openai`` provider instead of
changing it: users who want first-party OpenAI can keep ``provider: openai``,
while custom gateways can set ``provider: openai-compatible``.

Configuration precedence:

1. ``OPENAI_COMPATIBLE_IMAGE_*`` environment variables
2. ``image_gen.openai_compatible`` or ``image_gen.openai-compatible``
3. Top-level ``image_gen`` keys
4. Current ``model`` config when it is a custom provider

Example::

    image_gen:
      provider: openai-compatible
      model: gpt-image-2-medium
      openai_compatible:
        base_url: https://bifrost.example.com/v1
        api_key: sk-bf-...
        api_model: gpt-image-2
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
    success_response,
)

logger = logging.getLogger(__name__)

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced - default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_image_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _load_provider_config() -> Dict[str, Any]:
    cfg = _load_image_config()
    for key in ("openai_compatible", "openai-compatible"):
        section = cfg.get(key)
        if isinstance(section, dict):
            return section
    return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    env_override = os.environ.get("OPENAI_COMPATIBLE_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_config()
    provider_cfg = _load_provider_config()

    candidate: Optional[str] = None
    value = provider_cfg.get("model")
    if isinstance(value, str) and value in _MODELS:
        candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_model() -> str:
    provider_cfg = _load_provider_config()
    value = (
        os.environ.get("OPENAI_COMPATIBLE_IMAGE_API_MODEL")
        or provider_cfg.get("api_model")
        or provider_cfg.get("model_id")
        or _load_image_config().get("api_model")
        or API_MODEL
    )
    return str(value).strip() or API_MODEL


def _resolve_client_config() -> Tuple[Optional[str], Optional[str]]:
    cfg = _load_image_config()
    provider_cfg = _load_provider_config()

    api_key = (
        os.environ.get("OPENAI_COMPATIBLE_IMAGE_API_KEY")
        or provider_cfg.get("api_key")
        or cfg.get("api_key")
    )
    base_url = (
        os.environ.get("OPENAI_COMPATIBLE_IMAGE_BASE_URL")
        or provider_cfg.get("base_url")
        or cfg.get("base_url")
    )

    if not api_key or not base_url:
        try:
            from hermes_cli.config import load_config

            full_cfg = load_config()
            model_cfg = full_cfg.get("model") if isinstance(full_cfg, dict) else None
            if isinstance(model_cfg, dict) and str(model_cfg.get("provider") or "").startswith("custom:"):
                api_key = api_key or model_cfg.get("api_key")
                base_url = base_url or model_cfg.get("base_url")
        except Exception as exc:
            logger.debug("Could not inherit custom model provider config: %s", exc)

    return (
        str(api_key).strip() if api_key else None,
        str(base_url).strip() if base_url else None,
    )


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    """OpenAI-compatible ``images.generate`` backend."""

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def display_name(self) -> str:
        return "OpenAI-Compatible"

    def is_available(self) -> bool:
        api_key, base_url = _resolve_client_config()
        if not api_key or not base_url:
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
                "price": "gateway-defined",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI-Compatible",
            "badge": "custom",
            "tag": "Custom OpenAI-compatible image endpoint such as Bifrost",
            "env_vars": [
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
                    "prompt": "OpenAI-compatible image base URL",
                },
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_API_KEY",
                    "prompt": "OpenAI-compatible image API key",
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
                provider=self.name,
                aspect_ratio=aspect,
            )

        api_key, base_url = _resolve_client_config()
        if not api_key or not base_url:
            return error_response(
                error=(
                    "OpenAI-compatible image provider requires api_key and "
                    "base_url via image_gen.openai_compatible, top-level "
                    "image_gen keys, OPENAI_COMPATIBLE_IMAGE_* env vars, or "
                    "an active custom model provider."
                ),
                error_type="auth_required",
                provider=self.name,
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider=self.name,
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])
        api_model = _resolve_api_model()

        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": meta["quality"],
        }

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI-compatible image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI-compatible image generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI-compatible endpoint returned no image data",
                error_type="empty_response",
                provider=self.name,
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
                saved_path = save_b64_image(b64, prefix=f"openai_compatible_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider=self.name,
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            image_ref = url
        else:
            return error_response(
                error="OpenAI-compatible response contained neither b64_json nor URL",
                error_type="empty_response",
                provider=self.name,
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "api_model": api_model,
            "size": size,
            "quality": meta["quality"],
        }
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())
