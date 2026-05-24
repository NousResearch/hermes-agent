"""OpenAI-compatible image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The same provider class is also used
for named ``custom:<name>`` providers that expose the OpenAI-compatible
``/images/generations`` and ``/images/edits`` endpoints. It does not use the
Responses API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

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
        "strengths": "Balanced — default",
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _clean_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _resolve_effective_model(requested: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    """Return ``(reported_model, api_model, meta)`` for a tier or raw model id."""
    requested = (requested or "").strip()
    if requested:
        if requested in _MODELS:
            return requested, API_MODEL, _MODELS[requested]
        return requested, requested, {}
    tier_id, meta = _resolve_model()
    return tier_id, API_MODEL, meta


def _open_file_input(value: Any, *, field: str) -> Tuple[Any, List[Any]]:
    """Open local image file path(s) for OpenAI's multipart images.edit API."""
    handles: List[Any] = []

    def _open_one(item: Any) -> Any:
        if hasattr(item, "read"):
            return item
        path_text = str(item or "").strip()
        if not path_text:
            raise ValueError(f"{field} must be a non-empty local file path")
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise ValueError(f"{field} must be a local file path; not found: {path_text}")
        handle = path.open("rb")
        handles.append(handle)
        return handle

    if isinstance(value, (list, tuple)):
        opened = [_open_one(item) for item in value if item]
        if not opened:
            raise ValueError(f"{field} must include at least one local file path")
        return opened, handles
    return _open_one(value), handles


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` / ``images.edit`` backend."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_name: str = "openai",
        default_model: Optional[str] = None,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._base_url = (base_url or "").strip().rstrip("/") or None
        self._provider_name = (provider_name or "openai").strip() or "openai"
        self._default_model = (default_model or "").strip() or None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def display_name(self) -> str:
        if self._provider_name == "openai":
            return "OpenAI"
        return self._provider_name

    def is_available(self) -> bool:
        if not (self._api_key or os.environ.get("OPENAI_API_KEY")):
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
            "tag": "gpt-image-2 at low/medium/high quality tiers",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
        }

    def _client_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return kwargs

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

        if not (self._api_key or os.environ.get("OPENAI_API_KEY")):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
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

        tier_id, api_model, meta = _resolve_effective_model(
            _clean_str(kwargs.get("model")) or self._default_model
        )
        size = _clean_str(kwargs.get("size")) or _SIZES.get(aspect, _SIZES["square"])
        quality = _clean_str(kwargs.get("quality")) or meta.get("quality")
        output_format = _clean_str(kwargs.get("output_format"))
        n = kwargs.get("n", kwargs.get("num_images", 1))
        operation = "edit" if kwargs.get("image") is not None else "generate"

        common_payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }
        if quality:
            common_payload["quality"] = quality
        for key in ("background", "output_compression", "output_format", "user"):
            if kwargs.get(key) is not None:
                common_payload[key] = kwargs[key]

        handles: List[Any] = []
        try:
            client = openai.OpenAI(**self._client_kwargs())
            if operation == "edit":
                image_value, image_handles = _open_file_input(kwargs.get("image"), field="image")
                handles.extend(image_handles)
                payload = dict(common_payload)
                payload["image"] = image_value
                if kwargs.get("mask") is not None:
                    mask_value, mask_handles = _open_file_input(kwargs.get("mask"), field="mask")
                    handles.extend(mask_handles)
                    payload["mask"] = mask_value
                if kwargs.get("input_fidelity") is not None:
                    payload["input_fidelity"] = kwargs["input_fidelity"]
                response = client.images.edit(**payload)
            else:
                payload = dict(common_payload)
                for key in ("moderation", "style"):
                    if kwargs.get(key) is not None:
                        payload[key] = kwargs[key]
                response = client.images.generate(**payload)
        except ValueError as exc:
            return error_response(
                error=str(exc),
                error_type="invalid_argument",
                provider=self.name,
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
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
                extension = output_format if output_format in {"png", "jpeg", "webp"} else "png"
                saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}", extension=extension)
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
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider=self.name,
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size, "operation": operation}
        if quality:
            extra["quality"] = quality
        if output_format:
            extra["output_format"] = output_format
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
