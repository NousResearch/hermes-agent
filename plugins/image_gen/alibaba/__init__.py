"""Alibaba Cloud DashScope image generation backend.

Exposes Alibaba's Wan 2.7 image models (``wan2.7-image-pro``, ``wan2.7-image``)
as an :class:`ImageGenProvider` implementation via the DashScope
``image-generation/generation`` API.

Features:
- Text-to-image generation (up to 4K on wan2.7-image-pro)
- Image editing with up to 9 reference images (URL or base64)
- Thinking mode for enhanced quality (text-to-image only)
- Configurable output size, seed, watermark, and color palette
- Ephemeral OSS URLs materialised to local cache

Selection precedence (first hit wins):
1. ``DASHSCOPE_IMAGE_MODEL`` env var
2. ``image_gen.alibaba.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`

API reference:
https://www.alibabacloud.com/help/en/model-studio/wan-image-generation-and-editing-api-reference
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "wan2.7-image-pro": {
        "display": "Wan 2.7 Image Pro",
        "speed": "~5-20s",
        "strengths": "4K text-to-image, thinking mode, image editing (up to 9 refs)",
        "price": "token-plan",
    },
    "wan2.7-image": {
        "display": "Wan 2.7 Image",
        "speed": "~5-15s",
        "strengths": "Fast text-to-image and image editing (up to 9 refs)",
        "price": "token-plan",
    },
}

DEFAULT_MODEL = "wan2.7-image-pro"

# Aspect ratio → DashScope size string mapping.
# The API accepts preset strings ("1K", "2K", "4K") or explicit "WxH".
# We use explicit dimensions for deterministic aspect ratios.
_SIZES = {
    "landscape": "1280*720",
    "square": "1024*1024",
    "portrait": "720*1280",
}

# Maximum reference images the DashScope API accepts per request.
_MAX_REFERENCE_IMAGES = 9

# Default request timeout in seconds. Wan 2.7 with thinking_mode can take
# 20-30s; 4K generation can take longer.
_REQUEST_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_alibaba_image_config() -> Dict[str, Any]:
    """Read ``image_gen.alibaba`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        alibaba_section = section.get("alibaba") if isinstance(section, dict) else None
        return alibaba_section if isinstance(alibaba_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.alibaba config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("DASHSCOPE_IMAGE_MODEL", "").strip()
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_alibaba_image_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate and candidate in _MODELS:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_base_url() -> str:
    """Resolve the DashScope API base URL.

    Precedence:
    1. ``image_gen.alibaba.base_url`` in config.yaml
    2. ``DASHSCOPE_BASE_URL`` env var (shared with the LLM provider)
    3. Default international endpoint

    The image-generation endpoint is derived from the base URL by replacing
    the ``/compatible-mode/v1`` suffix (used by the LLM chat API) with the
    native DashScope path, or appending to the MaaS root.
    """
    cfg = _load_alibaba_image_config()
    configured = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else None
    if configured and configured.strip():
        return configured.strip().rstrip("/")

    env_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    return "https://dashscope-intl.aliyuncs.com/api/v1"


def _image_generation_endpoint(base_url: str) -> str:
    """Derive the image-generation endpoint from the base URL.

    Handles three base URL shapes:
    - ``https://dashscope-intl.aliyuncs.com/api/v1`` (standard)
    - ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1`` (LLM compat)
    - ``https://{workspace}.maas.aliyuncs.com/compatible-mode/v1`` (token plan)
    - ``https://{workspace}.maas.aliyuncs.com/api/v1`` (token plan native)
    """
    url = base_url.rstrip("/")
    # Strip the compatible-mode/v1 suffix if present
    if url.endswith("/compatible-mode/v1"):
        url = url[: -len("/compatible-mode/v1")]
    # Strip trailing /api/v1 if present (we'll re-add the full path)
    if url.endswith("/api/v1"):
        url = url[: -len("/api/v1")]
    return f"{url}/api/v1/services/aigc/image-generation/generation"


def _resolve_api_key() -> str:
    """Read the DashScope API key from the environment."""
    return os.environ.get("DASHSCOPE_API_KEY", "").strip()


def _resolve_size(aspect: str) -> str:
    """Map aspect ratio to a DashScope size string.

    wan2.7-image-pro supports 4K for text-to-image only. Image editing
    and image set generation are capped at 2K. We use explicit pixel
    dimensions for deterministic aspect ratios. Users can override via
    ``image_gen.alibaba.size`` in config.yaml (e.g. "2K", "4K", "1280*720").
    """
    cfg = _load_alibaba_image_config()
    configured_size = cfg.get("size") if isinstance(cfg.get("size"), str) else None
    if configured_size and configured_size.strip():
        return configured_size.strip()

    return _SIZES.get(aspect, _SIZES["square"])


def _image_to_content_item(source: str) -> Dict[str, str]:
    """Convert an image source (URL or local path) to a DashScope content item.

    DashScope accepts:
    - Public HTTPS URLs
    - Base64 data URIs: ``data:{MIME};base64,{data}``

    Local file paths are read and encoded into a data URI.
    """
    source = source.strip()
    lower = source.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return {"image": source}

    # Local file path → base64 data URI.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(source)
    path = Path(source).expanduser()
    with path.open("rb") as fh:
        raw = fh.read()
    ext = (path.suffix.lstrip(".") or "png").lower()
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    return {"image": f"data:image/{ext};base64,{b64}"}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AlibabaImageGenProvider(ImageGenProvider):
    """Alibaba Cloud DashScope Wan 2.7 image generation backend."""

    @property
    def name(self) -> str:
        return "alibaba"

    @property
    def display_name(self) -> str:
        return "Alibaba (Wan 2.7)"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
                "price": meta.get("price", ""),
            }
            for model_id, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Alibaba Wan 2.7 (image)",
            "badge": "paid",
            "tag": (
                "wan2.7-image-pro / wan2.7-image — text-to-image (4K) and "
                "image editing; uses DASHSCOPE_API_KEY"
            ),
            "env_vars": [
                {
                    "key": "DASHSCOPE_API_KEY",
                    "prompt": "DashScope API key",
                    "url": "https://www.alibabacloud.com/help/en/model-studio/get-api-key",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "max_reference_images": _MAX_REFERENCE_IMAGES,
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image (text-to-image) or edit source images (image-to-image).

        Routing: when ``image_url`` or ``reference_image_urls`` are provided,
        the images are included in the content array for image editing;
        otherwise text-to-image.
        """
        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "DASHSCOPE_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Alibaba to configure, or set the env var."
                ),
                error_type="missing_api_key",
                provider="alibaba",
                aspect_ratio=aspect_ratio,
            )

        model_id, meta = _resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)

        # Collect source images (image_url first, then references).
        source_images: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            source_images.append(image_url.strip())
        refs = normalize_reference_images(reference_image_urls)
        if refs:
            source_images.extend(refs)

        if len(source_images) > _MAX_REFERENCE_IMAGES:
            return error_response(
                error=(
                    f"DashScope supports at most {_MAX_REFERENCE_IMAGES} "
                    f"reference images per request, got {len(source_images)}"
                ),
                error_type="too_many_references",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        is_edit = bool(source_images)
        modality = "image" if is_edit else "text"
        size = _resolve_size(aspect)

        # Build the content array.
        content: List[Dict[str, str]] = []
        for source in source_images:
            try:
                content.append(_image_to_content_item(source))
            except Exception as exc:
                return error_response(
                    error=f"Could not load source image '{source}': {exc}",
                    error_type="io_error",
                    provider="alibaba",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        content.append({"text": (prompt or "").strip()})

        # Build parameters.
        parameters: Dict[str, Any] = {
            "size": size,
            "n": 1,
            "watermark": False,
        }

        # thinking_mode is only effective for text-to-image without image input.
        if not is_edit and model_id == "wan2.7-image-pro":
            cfg = _load_alibaba_image_config()
            thinking = cfg.get("thinking_mode")
            parameters["thinking_mode"] = bool(thinking) if thinking is not None else True

        # Optional seed from config.
        cfg = _load_alibaba_image_config()
        seed = cfg.get("seed")
        if isinstance(seed, int) and 0 <= seed <= 2147483647:
            parameters["seed"] = seed

        payload: Dict[str, Any] = {
            "model": model_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            },
            "parameters": parameters,
        }

        base_url = _resolve_base_url()
        endpoint = _image_generation_endpoint(base_url)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                body = resp.json() if resp is not None else {}
                err_msg = body.get("message", "") or body.get("error", {}).get("message", "")
                if not err_msg and resp is not None:
                    err_msg = resp.text[:300]
            except Exception:
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("DashScope image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"DashScope image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"DashScope image generation timed out ({_REQUEST_TIMEOUT}s)",
                error_type="timeout",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"DashScope connection error: {exc}",
                error_type="connection_error",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:
            return error_response(
                error=f"DashScope returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Parse response — DashScope returns output.choices[].message.content[]
        # with items of type "image" containing an "image" URL.
        output = result.get("output", {})
        choices = output.get("choices", [])
        if not choices:
            return error_response(
                error="DashScope returned no image data (empty choices)",
                error_type="empty_response",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        message = choices[0].get("message", {})
        content_items = message.get("content", [])
        image_url_result: Optional[str] = None
        for item in content_items:
            if isinstance(item, dict) and item.get("type") == "image":
                image_url_result = item.get("image")
                break
            # Some responses use {"image": "url"} without "type"
            if isinstance(item, dict) and "image" in item:
                image_url_result = item["image"]
                break

        if not image_url_result:
            return error_response(
                error="DashScope response contained no image URL",
                error_type="empty_response",
                provider="alibaba",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # DashScope returns ephemeral OSS URLs that expire. Materialise
        # locally so downstream consumers (Telegram send_photo, browser
        # fetch) have a stable file path.
        try:
            saved_path = save_url_image(
                image_url_result,
                prefix=f"dashscope_{model_id.replace('.', '_')}",
            )
        except Exception as exc:
            logger.warning(
                "DashScope image URL could not be cached (%s); returning bare URL.",
                exc,
            )
            image_ref = image_url_result
        else:
            image_ref = str(saved_path)

        extra: Dict[str, Any] = {"size": size}
        debug_info = output.get("debug_info")
        if isinstance(debug_info, list) and debug_info:
            first_debug = debug_info[0]
            if isinstance(first_debug, dict):
                for key in ("actual_seed", "output_W", "output_H", "inference_cost"):
                    if key in first_debug:
                        extra[key] = first_debug[key]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="alibaba",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(AlibabaImageGenProvider())
