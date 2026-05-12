"""Clawbay OpenAI-compatible image generation backend.

Routes Hermes `image_generate` calls to the user's Clawbay/OpenAI-compatible
Images API using the same provider credentials configured under
`providers.clawbay` in config.yaml.
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

DEFAULT_MODEL = "gpt-image-1.5"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-1.5": {
        "display": "GPT Image 1.5 (Clawbay)",
        "speed": "provider-dependent",
        "strengths": "OpenAI-compatible GPT image generation",
        "quality": "medium",
        "api_model": "gpt-image-1.5",
    },
    "gpt-image-2": {
        "display": "GPT Image 2 (Clawbay)",
        "speed": "provider-dependent",
        "strengths": "OpenAI-compatible GPT image generation",
        "quality": "medium",
        "api_model": "gpt-image-2",
    },
}

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("Could not load config: %s", exc)
        return {}


def _provider_config() -> Dict[str, Any]:
    cfg = _load_config()
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    provider = providers.get("clawbay") if isinstance(providers, dict) else None
    return provider if isinstance(provider, dict) else {}


def _image_config() -> Dict[str, Any]:
    cfg = _load_config()
    section = cfg.get("image_gen") if isinstance(cfg.get("image_gen"), dict) else {}
    clawbay = section.get("clawbay") if isinstance(section, dict) else {}
    return clawbay if isinstance(clawbay, dict) else {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    env_override = os.environ.get("CLAWBAY_IMAGE_MODEL")
    if env_override:
        if env_override in _MODELS:
            return env_override, _MODELS[env_override]
        return env_override, {
            "display": env_override,
            "speed": "provider-dependent",
            "strengths": "custom Clawbay image model",
            "quality": "medium",
            "api_model": env_override,
        }

    cfg = _image_config()
    candidate = cfg.get("model")
    if isinstance(candidate, str) and candidate.strip():
        candidate = candidate.strip()
        if candidate in _MODELS:
            return candidate, _MODELS[candidate]
        return candidate, {
            "display": candidate,
            "speed": "provider-dependent",
            "strengths": "custom Clawbay image model",
            "quality": "medium",
            "api_model": candidate,
        }

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


class ClawbayImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "clawbay"

    @property
    def display_name(self) -> str:
        return "Clawbay"

    def is_available(self) -> bool:
        pcfg = _provider_config()
        return bool(pcfg.get("api_key") and pcfg.get("base_url"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "provider billing",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Clawbay",
            "badge": "custom",
            "tag": "OpenAI-compatible GPT image generation via providers.clawbay",
            "env_vars": [],
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
                provider="clawbay",
                aspect_ratio=aspect,
            )

        pcfg = _provider_config()
        api_key = pcfg.get("api_key") or os.environ.get("CLAWBAY_API_KEY")
        base_url = pcfg.get("base_url") or os.environ.get("CLAWBAY_BASE_URL") or "https://api.theclawbay.com/v1"
        if not api_key:
            return error_response(
                error="Clawbay API key not found in providers.clawbay.api_key or CLAWBAY_API_KEY",
                error_type="auth_required",
                provider="clawbay",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="clawbay",
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        api_model = meta.get("api_model", model_id)
        size = _SIZES.get(aspect, _SIZES["square"])
        quality = str(_image_config().get("quality") or meta.get("quality") or "medium")

        # Use OpenAI Images-compatible parameters. Keep payload minimal; some
        # providers reject newer optional fields.
        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
        if quality:
            payload["quality"] = quality

        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("Clawbay image generation failed", exc_info=True)
            return error_response(
                error=f"Clawbay image generation failed: {exc}",
                error_type="api_error",
                provider="clawbay",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="Clawbay returned no image data",
                error_type="empty_response",
                provider="clawbay",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)
        if b64:
            try:
                image_ref = str(save_b64_image(b64, prefix=f"clawbay_{model_id}"))
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="clawbay",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif url:
            image_ref = url
        else:
            return error_response(
                error="Clawbay response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="clawbay",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "quality": quality, "base_url": base_url}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt
        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="clawbay",
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ClawbayImageGenProvider())
