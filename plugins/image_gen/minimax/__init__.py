"""MiniMax H3-Image backend: ``/v1/images/generations`` and ``/v1/images/edits``.

Selection: ``MINIMAX_IMAGE_MODEL`` → ``image_gen.minimax.model`` → :data:`DEFAULT_MODEL`.
Endpoint: ``image_gen.minimax.base_url`` → ``MINIMAX_BASE_URL`` → :data:`DEFAULT_BASE_URL`;
key: env named by ``image_gen.minimax.key_env`` → ``MINIMAX_API_KEY``.
Resolution is native ``aspect_ratio`` + ``area`` (1K/2K/4K, default 4K) rather than the
OpenAI-compatible 1024-level ``size`` trio. Requests are not retried (H3 is not idempotent).
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Optional

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from agent.secret_scope import get_secret, get_secret_str
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config,
    materialize_image, post_json, prompt_required_error, resolve_static_model)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MiniMax-H3-Image"
DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_AREA = "4K"
# H3 4K generation is ~17-19s and edits can run longer; the API requires a timeout above 600s.
_REQUEST_TIMEOUT = (30, 610)
_AREAS = ("1K", "2K", "4K")
_ASPECT_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_MAX_SOURCE_IMAGES = 16

MODELS = {
    DEFAULT_MODEL: {
        "display": "MiniMax H3 Image",
        "speed": "~10-20s",
        "strengths": "Text-to-image and image editing up to 4K",
        "api_model": DEFAULT_MODEL,
    },
}


def _resolve_model() -> tuple[str, dict[str, Any]]:
    return resolve_static_model(
        MODELS, DEFAULT_MODEL, env_var="MINIMAX_IMAGE_MODEL", config_key="minimax")


def _resolve_endpoint() -> tuple[str, str]:
    """``(base_url, api_key)`` — scoped config, then ``MINIMAX_BASE_URL`` / ``MINIMAX_API_KEY``."""
    cfg = load_image_gen_config("minimax")
    base_url = (
        str(cfg.get("base_url") or "").strip().rstrip("/")
        or get_secret_str("MINIMAX_BASE_URL").strip().rstrip("/")
        or DEFAULT_BASE_URL
    )
    key_env = str(cfg.get("key_env") or "").strip()
    api_key = (get_secret(key_env) if key_env else None) or get_secret("MINIMAX_API_KEY") or ""
    return base_url, api_key


def _resolve_area() -> str:
    env = (os.environ.get("MINIMAX_IMAGE_AREA") or "").strip()
    if env in _AREAS:
        return env
    area = str(load_image_gen_config("minimax").get("area") or "").strip()
    return area if area in _AREAS else DEFAULT_AREA


def _image_url_field(ref: str) -> str:
    """Pass through URL/data-URI references; encode a local path as a data URI."""
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:")):
        return ref
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    data = Path(ref).expanduser().read_bytes()
    mime = mimetypes.guess_type(ref)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _payload(model: str, prompt: str, aspect: str, area: str) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": _ASPECT_RATIOS.get(aspect, "1:1"),
        "area": area,
    }


class MiniMaxImageGenProvider(StaticImageGenProvider):
    """MiniMax H3-Image text-to-image and image-editing backend."""

    provider_id = "minimax"
    label = "MiniMax"
    models = MODELS
    default_model_id = DEFAULT_MODEL
    price = "varies"
    setup = dict(
        name="MiniMax", badge="paid",
        tag="MiniMax-H3-Image — text-to-image & image editing up to 4K",
        key="MINIMAX_API_KEY", prompt="MiniMax API key",
        url="https://platform.minimax.io")

    def is_available(self) -> bool:
        return bool(_resolve_endpoint()[1])

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": _MAX_SOURCE_IMAGES}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("minimax", aspect)
        base_url, api_key = _resolve_endpoint()
        if not api_key:
            return error_factory("minimax", aspect)(
                "MINIMAX_API_KEY not set (or the variable named by image_gen.minimax.key_env is empty). "
                "Run `hermes tools` → Image Generation → MiniMax to configure, or set MINIMAX_API_KEY. "
                "For the CN endpoint, set image_gen.minimax.base_url to https://api.minimaxi.com/v1 "
                "and image_gen.minimax.key_env to MINIMAX_CN_API_KEY.",
                "auth_required")

        model_id, _meta = _resolve_model()
        area = _resolve_area()
        sources = collect_source_images(image_url, reference_image_urls, limit=_MAX_SOURCE_IMAGES)
        is_edit = bool(sources)
        fail = error_factory("minimax", aspect, model=model_id, prompt=prompt)
        payload = _payload(model_id, prompt, aspect, area)
        if is_edit:
            try:
                payload["images"] = [{"image_url": _image_url_field(ref)} for ref in sources]
            except Exception as exc:
                return fail(f"Could not load source image for editing: {exc}", "io_error")
            endpoint = f"{base_url}/images/edits"
        else:
            endpoint = f"{base_url}/images/generations"

        result, failure = post_json(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload, timeout=_REQUEST_TIMEOUT, label="MiniMax")
        if failure:
            return fail(failure.error, failure.error_type)

        base_resp = result.get("base_resp") if isinstance(result, dict) else None
        if isinstance(base_resp, dict) and base_resp.get("status_code") not in (0, None):
            msg = base_resp.get("status_msg") or base_resp.get("status_code")
            return fail(f"MiniMax image {'editing' if is_edit else 'generation'} failed: {msg}", "api_error")

        data = (result or {}).get("data") or []
        if not data:
            return fail("MiniMax returned no image data", "empty_response")
        first = data[0]
        image_ref, err = materialize_image(
            first.get("b64_json") if isinstance(first, dict) else None,
            first.get("url") if isinstance(first, dict) else None,
            prefix=f"minimax_{model_id}", label="MiniMax", provider="minimax",
            model=model_id, prompt=prompt, aspect=aspect, log=logger)
        if err:
            return err
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="minimax",
            modality="image" if is_edit else "text", extra={"area": area})


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
