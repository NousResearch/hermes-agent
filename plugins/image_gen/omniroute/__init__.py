"""Omnirequest (local proxy) image generation provider.

Routes through ``http://127.0.0.1:20128`` using the ``image_gen.omniroute``
config block (model, preferred_models).

Selection: ``image_gen.model`` / ``image_gen.omniroute.model`` → first
``preferred_models`` entry → ``minimax/image-01`` (default).

Supports text-to-image, image editing (``image_url``), and reference-image
grounding (``reference_image_urls``, up to 4).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response,
    normalize_reference_images, resolve_aspect_ratio, save_b64_image,
    success_response)
from plugins.image_gen._common import (
    StaticImageGenProvider, collect_source_images, error_factory,
    load_image_gen_config, materialize_image, post_json)

logger = logging.getLogger(__name__)

# ── model catalog ────────────────────────────────────────────────────────────

_MODELS: Dict[str, Dict[str, Any]] = {
    "minimax/MiniMax-M3": {
        "display": "MiniMax M3",
        "speed": "~5-15s",
        "strengths": "Fallback via Omnirequest; reliable quality",
        "price": "≈$0.04–0.08 / image (bulk)",
    },
    "xao/grok-4.6": {
        "display": "xAI Grok 4.6 (vision)",
        "speed": "~5-15s",
        "strengths": "Primary via Omnirequest; image+text; high quality",
        "price": "≈$0.03–0.08 / image (through Omnirequest)",
    },
    "xao/grok-4.6-high": {
        "display": "xAI Grok 4.6 (high quality)",
        "speed": "~10-30s",
        "strengths": "Higher quality variant; better detail",
        "price": "≈$0.05–0.10 / image",
    },
    "xao/grok-4.6-xhigh": {
        "display": "xAI Grok 4.6 (maximum quality)",
        "speed": "~15-45s",
        "strengths": "Maximum quality; best for complex scenes",
        "price": "≈$0.08–0.15 / image",
    },
    "aihorde/Animagine XL": {
        "display": "AI Horde Animagine XL",
        "speed": "~15-45s",
        "strengths": "Anime/art style; community GPUs",
        "price": "Free (queue-dependent)",
    },
}

DEFAULT_MODEL = "xao/grok-4.6"
DEFAULT_PREFERRED_MODELS: List[str] = [
    "xao/grok-4.6",
    "minimax/MiniMax-M3",
    "aihorde/Animagine XL",
]
ENV_KEY = "OMNIROUTE_MAC_VK_HERMES"
REQUEST_TIMEOUT = 120.0


def _load_section() -> Dict[str, Any]:
    top = load_image_gen_config()
    if not isinstance(top, dict):
        return {}
    val = top.get("omniroute")
    return val if isinstance(val, dict) else {}


def _api_key() -> str:
    for env in (ENV_KEY, "OMNIROUTE_MAC_API_KEY"):
        val = (os.environ.get(env) or "").strip()
        if val:
            return val
    try:
        from agent.secret_scope import get_secret
        for env in (ENV_KEY, "OMNIROUTE_MAC_API_KEY"):
            val = (get_secret(env) or "").strip()
            if val:
                return val
    except Exception:
        pass
    return ""


def _preferred_models(explicit: Optional[str] = None) -> List[str]:
    top = _load_section()
    nested = top.get("omniroute") if isinstance(top.get("omniroute"), dict) else {}
    primary = (
        (explicit or "").strip()
        or str(nested.get("model") or "").strip()
        or str(top.get("model") or "").strip()
        or DEFAULT_MODEL
    )
    preferred = nested.get("preferred_models")
    if not isinstance(preferred, list) or not preferred:
        preferred = list(DEFAULT_PREFERRED_MODELS)
    chain: List[str] = []
    for mid in [primary, *[str(x).strip() for x in preferred if str(x).strip()]]:
        if mid and mid not in chain:
            chain.append(mid)
    return chain


# ── provider class ───────────────────────────────────────────────────────────

class OmnirequestImageGenProvider(StaticImageGenProvider):
    """Omnirequest-local proxy: routes through ``http://127.0.0.1:20128``."""

    provider_id = "omniroute"
    label = "Omnirequest (local proxy)"
    models = _MODELS
    default_model_id = DEFAULT_MODEL

    def is_available(self) -> bool:
        return _api_key() != ""

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "max_reference_images": 4,
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Text-to-image or edit via the local Omnirequest proxy."""
        api_key = _api_key()
        if not api_key:
            return error_response(
                f"API key (``{ENV_KEY}``) not set. Set ``{ENV_KEY}`` environment variable.",
                "missing_api_key", model or DEFAULT_MODEL, aspect_ratio)

        resolved_model = model or _preferred_models()[0]
        aspect = resolve_aspect_ratio(aspect_ratio).lower().strip()

        # Size map (OpenAI-compatible)
        sizes = {"landscape": "1536x1024", "square": "1024x1024", "portrait": "1024x1536"}
        size = sizes.get(aspect, sizes["square"])

        # Collect source images (for editing / reference-image grounding)
        source_images = collect_source_images(image_url, reference_image_urls)
        is_edit = bool(source_images)

        base_url = "http://127.0.0.1:20128"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if is_edit:
            # Edit workflow: build source image fields for the proxy API
            payload: Dict[str, Any] = {"model": resolved_model, "prompt": prompt}
            if len(source_images) == 1:
                payload["image"] = materialize_image(source_images[0])
            else:
                payload["images"] = [materialize_image(s) for s in source_images]
            endpoint = f"{base_url}/v1/images/edits"
        else:
            # Text-to-image
            payload = {
                "model": resolved_model,
                "prompt": prompt,
                "size": size,
            }
            endpoint = f"{base_url}/v1/images/generations"

        fail = error_factory("omniroute", aspect, model=resolved_model, prompt=prompt)

        result, failure = post_json(
            endpoint, headers=headers, payload=payload,
            timeout=REQUEST_TIMEOUT, label="Omnirequest")

        if failure:
            if failure.kind == "http":
                logger.error("Omnirequest image gen failed (%d): %s",
                             failure.status, failure.message)
            return fail(failure.error, failure.error_type)

        # Parse the result — b64_json or url
        data = result
        choices = data.get("data", data.get("choices", []))
        if not choices:
            return fail("No images in response", "no_images", resolved_model, aspect)

        item = choices[0]
        b64 = item.get("b64_json") or item.get("b64")
        url = item.get("url") or item.get("image_url")

        if b64:
            img_path = save_b64_image(b64, prefix="omniroute", extension="png")
            return success_response(
                image=str(img_path), model=resolved_model, prompt=prompt,
                aspect_ratio=aspect, provider=self.provider_id, modality="image")

        if url:
            return success_response(
                image=url, model=resolved_model, prompt=prompt,
                aspect_ratio=aspect, provider=self.provider_id, modality="image")

        return fail("No image in response (expected b64_json or url)",
                    "invalid_response", resolved_model, aspect)


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(OmnirequestImageGenProvider())

