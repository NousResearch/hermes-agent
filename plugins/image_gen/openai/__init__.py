"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2``) with a
different ``quality`` parameter. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import get_env_value

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


def _resolve_timeout(cfg: Optional[Dict[str, Any]] = None) -> float:
    """Resolve OpenAI image timeout in seconds."""
    env_value = get_env_value("OPENAI_IMAGE_TIMEOUT") or os.environ.get("OPENAI_IMAGE_TIMEOUT")
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            logger.warning("Invalid OPENAI_IMAGE_TIMEOUT=%r; ignoring", env_value)

    cfg = cfg if isinstance(cfg, dict) else _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    for section in (openai_cfg, cfg):
        if isinstance(section, dict):
            value = section.get("timeout")
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                logger.warning("Invalid image_gen timeout value=%r; ignoring", value)

    return 180.0


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


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 at low/medium/high."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not get_env_value("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
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

    def supports_image_edit(self) -> bool:
        return True

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

        if not get_env_value("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
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
        cfg = _load_openai_config()
        timeout_seconds = _resolve_timeout(cfg)
        size = _SIZES.get(aspect, _SIZES["square"])

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": meta["quality"],
        }

        try:
            api_key = get_env_value("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = get_env_value("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            client_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout_seconds,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)
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

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
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

    def edit_image(
        self,
        image_path: str,
        instruction: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        instruction = (instruction or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not instruction:
            return error_response(
                error="Instruction is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        source = Path(image_path).expanduser()
        if not source.is_absolute() or not source.exists() or not source.is_file():
            return error_response(
                error=f"Source image does not exist or is not a file: {source}",
                error_type="invalid_argument",
                provider="openai",
                prompt=instruction,
                aspect_ratio=aspect,
            )

        if not get_env_value("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                prompt=instruction,
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                prompt=instruction,
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        cfg = _load_openai_config()
        timeout_seconds = _resolve_timeout(cfg)
        size = _SIZES.get(aspect, _SIZES["square"])

        try:
            api_key = get_env_value("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = get_env_value("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            client_kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "timeout": timeout_seconds,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)

            with source.open("rb") as image_file:
                response = client.images.edit(
                    model=API_MODEL,
                    image=image_file,
                    prompt=instruction,
                    size=size,
                    quality=meta["quality"],
                )
        except Exception as exc:
            logger.debug("OpenAI image edit failed", exc_info=True)
            return error_response(
                error=f"OpenAI image edit failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=instruction,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=instruction,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_edit_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save edited image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=instruction,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            image_ref = url
        else:
            return error_response(
                error="OpenAI edit response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=instruction,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "size": size,
            "quality": meta["quality"],
            "source_image": str(source),
            "instruction": instruction,
        }
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=instruction,
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
