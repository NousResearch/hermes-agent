"""xAI image generation backend.

Exposes xAI's ``grok-imagine-image`` model as an
:class:`ImageGenProvider` implementation.

Features:
- Text-to-image generation
- Multiple aspect ratios (1:1, 16:9, 9:16, etc.)
- Multiple resolutions (1K, 2K)
- Base64 output saved to cache

Selection precedence (first hit wins):
1. ``XAI_IMAGE_MODEL`` env var
2. ``image_gen.xai.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

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
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-imagine-image": {
        "display": "Grok Imagine Image",
        "speed": "~5-10s",
        "strengths": "Fast, high-quality",
    },
    "grok-imagine-image-quality": {
        "display": "Grok Imagine Image (Quality)",
        "speed": "~10-20s",
        "strengths": "Higher fidelity / detail; slower than the standard model.",
    },
}

DEFAULT_MODEL = "grok-imagine-image"

# xAI aspect ratios (more options than FAL/OpenAI)
_XAI_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
    "4:3": "4:3",
    "3:4": "3:4",
    "3:2": "3:2",
    "2:3": "2:3",
}

# xAI resolutions
_XAI_RESOLUTIONS = {"1k", "2k"}

DEFAULT_RESOLUTION = "1k"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_config() -> Dict[str, Any]:
    """Read ``image_gen.xai`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        xai_section = section.get("xai") if isinstance(section, dict) else None
        return xai_section if isinstance(xai_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.xai config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("XAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_xai_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate and candidate in _MODELS:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_resolution() -> str:
    """Get configured resolution."""
    cfg = _load_xai_config()
    res = cfg.get("resolution") if isinstance(cfg.get("resolution"), str) else None
    if res and res in _XAI_RESOLUTIONS:
        return res
    return DEFAULT_RESOLUTION


def _xai_image_field(source: str) -> Dict[str, str]:
    """Build the xAI ``image`` field for an edit request.

    xAI's ``/v1/images/edits`` accepts ``{"url": <ref>, "type": "image_url"}``
    where ``<ref>`` is a public URL or a base64 data URI. Public URLs and
    existing data URIs pass through unchanged; local file paths are read and
    encoded into a ``data:`` URI.
    """
    source = source.strip()
    lower = source.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return {"url": source, "type": "image_url"}
    # Local file path → base64 data URI.
    import base64
    import os as _os

    with open(source, "rb") as fh:
        raw = fh.read()
    ext = (_os.path.splitext(source)[1].lstrip(".") or "png").lower()
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(raw).decode("utf-8")
    return {"url": f"data:image/{ext};base64,{b64}", "type": "image_url"}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIImageGenProvider(ImageGenProvider):
    """xAI ``grok-imagine-image`` backend."""

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI (Grok)"

    def is_available(self) -> bool:
        creds = resolve_xai_http_credentials()
        return bool(creds.get("api_key"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
            }
            for model_id, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution is delegated to the shared ``xai_grok`` post_setup
        # hook (``hermes_cli/tools_config.py``); identical to the TTS / video
        # gen entries so users see the same OAuth-or-API-key choice for every
        # xAI service.
        return {
            "name": "xAI Grok Imagine (image)",
            "badge": "paid",
            "tag": "grok-imagine-image — text-to-image & image editing; uses xAI Grok OAuth or XAI_API_KEY",
            "env_vars": [],
            "post_setup": "xai_grok",
        }

    def capabilities(self) -> Dict[str, Any]:
        # xAI's /v1/images/edits supports image editing via grok-imagine-image
        # -quality. Single primary source image (multi-image editing exists as
        # a separate capability but we keep the primary edit surface here).
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image (text-to-image) or edit a source image (image-to-image).

        Routing: when ``image_url`` is provided, POST to ``/v1/images/edits``
        with the source image; otherwise POST to ``/v1/images/generations``.
        Per xAI docs, editing uses the ``grok-imagine-image-quality`` model and
        a JSON body (the OpenAI SDK's multipart ``images.edit()`` is NOT
        supported by xAI).
        """
        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        provider_name = str(creds.get("provider") or "xai").strip() or "xai"
        if not api_key:
            return error_response(
                error="No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.",
                error_type="missing_api_key",
                provider=provider_name,
                aspect_ratio=aspect_ratio,
            )

        model_id, meta = _resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)
        xai_ar = _XAI_ASPECT_RATIOS.get(aspect, "1:1")
        resolution = _resolve_resolution()
        xai_res = resolution if resolution in _XAI_RESOLUTIONS else DEFAULT_RESOLUTION

        # Pick the primary source image: explicit image_url wins, else the
        # first reference image.
        source_image = None
        if isinstance(image_url, str) and image_url.strip():
            source_image = image_url.strip()
        else:
            refs = normalize_reference_images(reference_image_urls)
            if refs:
                source_image = refs[0]
        is_edit = bool(source_image)
        modality = "image" if is_edit else "text"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        }

        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")

        if is_edit:
            # Editing requires the quality model per xAI docs. The source
            # image may be a public URL or a base64 data URI; local file paths
            # are converted to a data URI here.
            edit_model = "grok-imagine-image-quality"
            try:
                image_field = _xai_image_field(source_image)
            except Exception as exc:
                return error_response(
                    error=f"Could not load source image for editing: {exc}",
                    error_type="io_error",
                    provider=provider_name,
                    model=edit_model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            payload: Dict[str, Any] = {
                "model": edit_model,
                "prompt": prompt,
                "image": image_field,
            }
            endpoint_url = f"{base_url}/images/edits"
            model_id = edit_model
        else:
            payload = {
                "model": model_id,
                "prompt": prompt,
                "aspect_ratio": xai_ar,
                "resolution": xai_res,
            }
            endpoint_url = f"{base_url}/images/generations"

        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            response = exc.response
            status = response.status_code if response is not None else 0
            try:
                err_msg = response.json().get("error", {}).get("message", response.text[:300])
            except Exception:
                err_msg = response.text[:300] if response is not None else str(exc)
            logger.error("xAI image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"xAI image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="xAI image generation timed out (120s)",
                error_type="timeout",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"xAI connection error: {exc}",
                error_type="connection_error",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:
            return error_response(
                error=f"xAI returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Parse response — xAI returns data[0].b64_json or data[0].url
        data = result.get("data", [])
        if not data:
            return error_response(
                error="xAI returned no image data",
                error_type="empty_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = first.get("b64_json")
        url = first.get("url")

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"xai_{model_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="xai",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # xAI's grok-imagine-image returns ephemeral ``imgen.x.ai/xai-tmp-*``
            # URLs that 404 within minutes — by the time Telegram's
            # ``send_photo`` or any downstream consumer fetches them, the
            # asset is gone (#26942).  Materialise the bytes locally at
            # tool-completion time so the gateway has a stable file path to
            # upload, mirroring the b64 branch above and the audio_cache
            # pattern used by text_to_speech.
            try:
                saved_path = save_url_image(url, prefix=f"xai_{model_id}")
            except Exception as exc:
                logger.warning(
                    "xAI image URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                image_ref = url
            else:
                image_ref = str(saved_path)
        else:
            return error_response(
                error="xAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="xai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {}
        if not is_edit:
            extra["resolution"] = xai_res

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="xai",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(XAIImageGenProvider())
