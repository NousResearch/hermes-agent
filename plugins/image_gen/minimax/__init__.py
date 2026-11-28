"""MiniMax H3-Image backend: ``/v1/images/generations`` and ``/v1/images/edits``."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from agent.secret_scope import get_secret
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory, load_image_gen_config,
    materialize_image, post_json, prompt_required_error, resolve_static_model)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MiniMax-H3-Image"
DEFAULT_BASE_URL = "https://api.minimax.io/v1"
_REQUEST_TIMEOUT = (30, 610)  # H3 requires a read timeout above 600s
_ASPECT_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}

MODELS = {
    DEFAULT_MODEL: {
        "display": "MiniMax H3 Image",
        "speed": "~10-20s",
        "strengths": "Text-to-image and image editing up to 4K",
        "api_model": DEFAULT_MODEL,
    },
}


def _endpoint() -> tuple[str, str]:
    cfg = load_image_gen_config("minimax")
    base_url = str(cfg.get("base_url") or "").strip().rstrip("/") or DEFAULT_BASE_URL
    return base_url, get_secret("MINIMAX_API_KEY") or ""


def _area() -> str:
    area = str(load_image_gen_config("minimax").get("area") or "").strip()
    return area if area in ("1K", "2K", "4K") else "4K"


def _image_url_field(ref: str) -> str:
    ref = ref.strip()
    if ref.lower().startswith(("http://", "https://", "data:")):
        return ref
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    data = Path(ref).expanduser().read_bytes()
    mime = mimetypes.guess_type(ref)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class MiniMaxImageGenProvider(StaticImageGenProvider):
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
        return bool(_endpoint()[1])

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error("minimax", aspect)
        base_url, api_key = _endpoint()
        if not api_key:
            return error_factory("minimax", aspect)(
                "MINIMAX_API_KEY not set. Run `hermes tools` → Image Generation → MiniMax "
                "to configure, or set MINIMAX_API_KEY.",
                "auth_required")

        model_id, _meta = resolve_static_model(
            MODELS, DEFAULT_MODEL, env_var="MINIMAX_IMAGE_MODEL", config_key="minimax")
        area = _area()
        sources = collect_source_images(image_url, reference_image_urls, limit=16)
        is_edit = bool(sources)
        fail = error_factory("minimax", aspect, model=model_id, prompt=prompt)
        payload = {
            "model": model_id, "prompt": prompt, "n": 1,
            "aspect_ratio": _ASPECT_RATIOS.get(aspect, "1:1"), "area": area,
        }
        if is_edit:
            try:
                payload["images"] = [{"image_url": _image_url_field(ref)} for ref in sources]
            except Exception as exc:
                logger.debug("Could not load MiniMax source image for editing", exc_info=True)
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
            return fail(
                f"MiniMax image {'editing' if is_edit else 'generation'} failed: {msg}",
                "api_error")

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
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect,
            provider="minimax", modality="image" if is_edit else "text", extra={"area": area})


def register(ctx) -> None:
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
