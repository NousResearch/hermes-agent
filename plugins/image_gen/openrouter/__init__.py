"""
OpenRouter image generation backend.

Exposes popular image models available on OpenRouter (Nano Banana / Gemini image
models, FLUX.2 variants, etc.) via the unified ``/v1/chat/completions`` endpoint
using the ``modalities`` and ``image_config`` parameters.

Features:
- Text-to-image generation
- Image-to-image / editing (via reference images in message content)
- Aspect ratio control via ``image_config.aspect_ratio``
- Model catalog focused on the main "Nano Banana" and FLUX families
- Automatic handling of OPENROUTER_API_KEY (or OPENROUTER_API_KEY in config)
- Proper use of success_response / error_response helpers
- Support for image_url (primary edit source) and reference_image_urls

Selection precedence (first hit wins):
1. ``OPENROUTER_IMAGE_MODEL`` env var
2. ``image_gen.openrouter.model`` in config.yaml
3. :data:`DEFAULT_MODEL`

OpenRouter docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

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

# ---------------------------------------------------------------------------
# Model catalog (main Nano Banana / FLUX models)
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "google/gemini-2.5-flash-image-preview": {
        "display": "Nano Banana (Gemini 2.5 Flash Image)",
        "speed": "~5-15s",
        "strengths": "Fast, strong prompt adherence, excellent editing",
        "supports_edit": True,
    },
    "google/gemini-3-pro-image-preview": {
        "display": "Nano Banana Pro (Gemini 3 Pro Image)",
        "speed": "~15-40s",
        "strengths": "Highest quality, advanced reasoning + editing",
        "supports_edit": True,
    },
    "black-forest-labs/flux.2-pro": {
        "display": "FLUX.2 Pro",
        "speed": "~10-30s",
        "strengths": "State-of-the-art photorealism and prompt following",
        "supports_edit": False,  # primarily text-to-image; editing via other routes
    },
    "black-forest-labs/flux.2-dev": {
        "display": "FLUX.2 Dev",
        "speed": "~8-20s",
        "strengths": "High quality open weights variant",
        "supports_edit": False,
    },
}

DEFAULT_MODEL = "google/gemini-2.5-flash-image-preview"

# Map Hermes aspect ratios to OpenRouter image_config values (most models accept 16:9, 1:1, 9:16 etc.)
_ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_openrouter_config() -> Dict[str, Any]:
    """Read ``image_gen.openrouter`` section from config.yaml (returns {} on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        or_section = section.get("openrouter") if isinstance(section, dict) else None
        return or_section if isinstance(or_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.openrouter config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return (model_id, meta)."""
    env_override = os.environ.get("OPENROUTER_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openrouter_config()
    cfg_model = cfg.get("model")
    if isinstance(cfg_model, str) and cfg_model in _MODELS:
        return cfg_model, _MODELS[cfg_model]

    # Also allow top-level image_gen.model if it matches one of ours
    try:
        from hermes_cli.config import load_config

        cfg_all = load_config()
        ig = cfg_all.get("image_gen", {}) if isinstance(cfg_all, dict) else {}
        top_model = ig.get("model")
        if isinstance(top_model, str) and top_model in _MODELS:
            return top_model, _MODELS[top_model]
    except Exception:
        pass

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _get_api_key() -> Optional[str]:
    """Return OPENROUTER_API_KEY from env (preferred) or config."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()

    cfg = _load_openrouter_config()
    key = cfg.get("api_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


class OpenRouterImageGenProvider(ImageGenProvider):
    """OpenRouter-backed image generation provider."""

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        return bool(_get_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for mid, meta in _MODELS.items():
            entry = {
                "id": mid,
                "display": meta.get("display", mid),
                "speed": meta.get("speed"),
                "strengths": meta.get("strengths"),
            }
            if "price" in meta:
                entry["price"] = meta["price"]
            out.append(entry)
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "paid",
            "tag": "Access to Nano Banana, FLUX.2, and 100+ image models via one key",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # Most listed models support both text and image (editing)
        return {"modalities": ["text", "image"], "max_reference_images": 4}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate or edit an image using OpenRouter /chat/completions + modalities."""
        api_key = _get_api_key()
        if not api_key:
            return error_response(
                error="No OPENROUTER_API_KEY found. Set the environment variable or configure it via `hermes tools`.",
                error_type="missing_api_key",
                provider=self.name,
                aspect_ratio=aspect_ratio,
            )

        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        or_aspect = _ASPECT_RATIO_MAP.get(aspect, "16:9")

        model_id, meta = _resolve_model()
        # Allow per-call override
        if "model" in kwargs and isinstance(kwargs["model"], str) and kwargs["model"] in _MODELS:
            model_id = kwargs["model"]
            meta = _MODELS[model_id]

        # Build sources for image-to-image
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        refs = normalize_reference_images(reference_image_urls)
        if refs:
            sources.extend(refs)

        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        if not prompt and not is_edit:
            return error_response(
                error="Prompt is required for text-to-image generation",
                error_type="invalid_input",
                provider=self.name,
                aspect_ratio=aspect,
            )

        # Build chat completion payload
        content: List[Dict[str, Any]] = []
        if sources:
            for src in sources[:4]:  # respect max_reference_images
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": src},
                    }
                )
        if prompt:
            content.append({"type": "text", "text": prompt})
        if not content:
            content = [{"type": "text", "text": "Generate an image"}]

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": or_aspect,
            },
        }

        # Some models support additional image_config options; keep minimal for broad compatibility
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
            "X-Title": "Hermes Agent",
        }

        try:
            resp = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as exc:
            try:
                err_detail = exc.response.json() if exc.response is not None else {}
                err_msg = err_detail.get("error", {}).get("message", str(exc))
            except Exception:
                err_msg = str(exc)
            return error_response(
                error=f"OpenRouter API error: {err_msg}",
                error_type="http_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            return error_response(
                error=f"Request failed: {exc}",
                error_type=type(exc).__name__,
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Parse response — OpenRouter returns images under choices[0].message.images
        try:
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            images = message.get("images", [])
            if not images:
                # Some models may return content with image urls in a different shape
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            images = [{"image_url": item.get("image_url", {})}]
                            break

            if not images:
                return error_response(
                    error="No image returned by OpenRouter",
                    error_type="no_image",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            first = images[0]
            img_url = None
            img_b64 = None

            if isinstance(first, dict):
                img_info = first.get("image_url", first)
                if isinstance(img_info, dict):
                    img_url = img_info.get("url")
                    # Some responses embed base64 data
                    if not img_url and img_info.get("data"):
                        img_b64 = img_info["data"]
                elif isinstance(img_info, str):
                    img_url = img_info

            if img_b64:
                # data may be data:image/...;base64,.... strip prefix if present
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                path = save_b64_image(img_b64, prefix="openrouter", extension="png")
                image_out = str(path)
            elif img_url:
                # Prefer to materialize locally for reliability (URLs can expire)
                try:
                    path = save_url_image(img_url, prefix="openrouter")
                    image_out = str(path)
                except Exception:
                    image_out = img_url  # fall back to remote URL
            else:
                return error_response(
                    error="Could not extract image URL or base64 from response",
                    error_type="parse_error",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            return success_response(
                image=image_out,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self.name,
                modality=modality,
            )

        except Exception as exc:
            return error_response(
                error=f"Failed to parse OpenRouter response: {exc}",
                error_type="parse_error",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )


def register(ctx) -> None:
    """Plugin entry point — register the OpenRouter provider."""
    ctx.register_image_gen_provider(OpenRouterImageGenProvider())
