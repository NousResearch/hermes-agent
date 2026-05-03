"""Poe.com image generation backend using fastapi_poe SDK.

Uses Poe's native bot API (``fp.get_bot_response``) to call any image
generation bot by name. This replaces the OpenAI-compatible ``images/generations``
endpoint which was limited to a hardcoded catalog and did not support bots
like ``gpt-image-1`` or arbitrary future bots.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog (informational — any bot name is accepted)
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "DALL-E-3": {"display": "DALL-E 3", "speed": "~15-30s", "strengths": "Excellent prompt adherence, detailed images"},
    "FLUX-pro": {"display": "FLUX Pro", "speed": "~10-20s", "strengths": "Studio photorealism, high quality"},
    "FLUX-pro-1.1": {"display": "FLUX Pro 1.1", "speed": "~10-20s", "strengths": "Studio photorealism"},
    "FLUX-schnell": {"display": "FLUX Schnell", "speed": "~2-5s", "strengths": "Very fast, good quality"},
    "FLUX-dev": {"display": "FLUX Dev", "speed": "~5-10s", "strengths": "Open-source quality, customizable"},
    "Imagen-3": {"display": "Imagen 3 (Google)", "speed": "~15-30s", "strengths": "Photorealistic, great for products"},
    "Imagen-4": {"display": "Imagen 4 (Google)", "speed": "~15-30s", "strengths": "Photorealistic, great for products"},
    "GPT-Image-1": {"display": "GPT Image 1", "speed": "~10-20s", "strengths": "Text rendering, editing"},
    "GPT-Image-1.5": {"display": "GPT Image 1.5", "speed": "~10-20s", "strengths": "Text rendering, editing"},
    "StableDiffusionXL": {"display": "Stable Diffusion XL", "speed": "~5-10s", "strengths": "Versatile, open weights"},
    "gpt-image-1": {"display": "gpt-image-1", "speed": "~10-20s", "strengths": "Chat-based image generation"},
}

DEFAULT_MODEL = "GPT-Image-1"


# ---------------------------------------------------------------------------
# Config helpers (same pattern as before)
# ---------------------------------------------------------------------------

def _load_poe_config() -> Dict[str, Any]:
    """Read ``image_gen.poe`` from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        poe_section = section.get("poe") if isinstance(section, dict) else None
        return poe_section if isinstance(poe_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.poe config: %s", exc)
        return {}


def _resolve_model(override: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which bot to use and return ``(bot_name, meta)``.

    Per-call ``override`` (from the tool's ``model`` arg) takes highest
    precedence so the agent can request a specific bot on each call.
    """
    if override and isinstance(override, str):
        bot_name = override.strip()
        if bot_name:
            return bot_name, _MODELS.get(bot_name, {"speed": "?", "strengths": "Custom bot"})

    env_override = os.environ.get("POE_IMAGE_MODEL")
    if env_override:
        return env_override, _MODELS.get(env_override, {"speed": "?", "strengths": "Custom bot"})

    cfg = _load_poe_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate:
        return candidate, _MODELS.get(candidate, {"speed": "?", "strengths": "Custom bot"})

    # Fallback: check top-level image_gen.model
    try:
        from hermes_cli.config import load_config
        top_cfg = load_config()
        top_model = top_cfg.get("image_gen", {}).get("model")
        if isinstance(top_model, str):
            return top_model, _MODELS.get(top_model, {"speed": "?", "strengths": "Custom bot"})
    except Exception:
        pass

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# fastapi_poe helpers
# ---------------------------------------------------------------------------

async def _call_poe_bot(api_key: str, bot_name: str, prompt: str):
    """
    Stream a response from a Poe bot.

    Returns ``(final_text, attachments)`` where attachments is a list of
    objects or dicts that have at least a ``url`` field.
    """
    import fastapi_poe as fp

    message = fp.ProtocolMessage(role="user", content=prompt)
    final_text = ""
    attachments: List[Any] = []
    seen_urls = set()

    async for partial in fp.get_bot_response(
        messages=[message],
        bot_name=bot_name,
        api_key=api_key,
    ):
        text = getattr(partial, "text", "") or ""
        if getattr(partial, "is_replace_response", False):
            final_text = text
        else:
            final_text += text

        att = getattr(partial, "attachment", None)
        if att:
            url = getattr(att, "url", None) or (att.get("url") if isinstance(att, dict) else None)
            if url and url not in seen_urls:
                seen_urls.add(url)
                attachments.append(att)

        atts = getattr(partial, "attachments", None)
        if atts:
            for a in atts:
                url = getattr(a, "url", None) or (a.get("url") if isinstance(a, dict) else None)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    attachments.append(a)

    return final_text, attachments


def _extract_image_urls(text: str) -> List[str]:
    """Find inline image URLs in markdown or raw URLs."""
    md_urls = re.findall(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", text)
    if md_urls:
        return md_urls
    return re.findall(
        r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)(?:\?\S*)?",
        text,
        re.IGNORECASE,
    )


def _download_image(url: str, prefix: str = "poe") -> Optional[str]:
    """Download an image to the Hermes cache dir and return the local path."""
    import datetime
    import uuid

    from agent.image_gen_provider import _images_cache_dir

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

        ext = "png"
        content_type = resp.headers.get("Content-Type", "").lower()
        if "jpeg" in content_type or "jpg" in content_type:
            ext = "jpg"
        elif "webp" in content_type:
            ext = "webp"
        elif "gif" in content_type:
            ext = "gif"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:8]
        cache_dir = _images_cache_dir()
        path = cache_dir / f"{prefix}_{ts}_{short}.{ext}"
        with open(path, "wb") as f:
            for chunk in resp.iter_content(64 * 1024):
                f.write(chunk)
        return str(path)
    except Exception as exc:
        logger.warning("Failed to download image from %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class PoeImageGenProvider(ImageGenProvider):
    """Poe.com image generation backend via fastapi_poe."""

    @property
    def name(self) -> str:
        return "poe"

    @property
    def display_name(self) -> str:
        return "Poe.com"

    def is_available(self) -> bool:
        if not os.environ.get("POE_API_KEY"):
            return False
        try:
            import fastapi_poe  # noqa: F401
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
                "price": "Poe points",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Poe.com",
            "badge": "paid",
            "tag": "Image generation via Poe bot API (any bot name supported, e.g. FLUX-pro, gpt-image-1, Imagen-3)",
            "env_vars": [
                {
                    "key": "POE_API_KEY",
                    "prompt": "Poe API key",
                    "url": "https://poe.com/api/keys",
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
                provider="poe",
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("POE_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "POE_API_KEY not set. Get your key at https://poe.com/api/keys "
                    "and add it to your ~/.hermes/.env file."
                ),
                error_type="auth_required",
                provider="poe",
                aspect_ratio=aspect,
            )

        try:
            import fastapi_poe  # noqa: F401
        except ImportError:
            return error_response(
                error="fastapi_poe Python package not installed (pip install fastapi_poe)",
                error_type="missing_dependency",
                provider="poe",
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model(kwargs.get("model"))

        # Many bots understand aspect-ratio hints embedded in the prompt.
        if aspect != "square":
            aspect_hint = {
                "landscape": "widescreen landscape",
                "portrait": "tall portrait",
            }.get(aspect, aspect)
            full_prompt = f"{aspect_hint} image: {prompt}"
        else:
            full_prompt = prompt

        try:
            # Gateway runs inside an asyncio event loop, so asyncio.run()
            # would crash.  Run the coroutine in a background thread where
            # we can safely start a fresh event loop.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, _call_poe_bot(api_key, model_id, full_prompt)
                )
                final_text, attachments = future.result(timeout=300)
        except Exception as exc:
            logger.debug("Poe bot call failed", exc_info=True)
            return error_response(
                error=f"Poe image generation failed: {exc}",
                error_type="api_error",
                provider="poe",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Prefer attachments (the proper channel for image bots like gpt-image-1)
        urls = []
        for att in attachments:
            url = getattr(att, "url", None)
            if not url and isinstance(att, dict):
                url = att.get("url")
            if url:
                urls.append(url)

        # Fall back to inline URLs in markdown or raw text
        if not urls:
            urls = _extract_image_urls(final_text)

        if not urls:
            return error_response(
                error=f"Poe returned no image data. Bot response was: {final_text[:500]}",
                error_type="empty_response",
                provider="poe",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_url = urls[0]
        local_path = _download_image(
            image_url,
            prefix=f"poe_{model_id.lower().replace('-', '_')}",
        )
        image_ref = local_path if local_path else image_url

        extra: Dict[str, Any] = {}
        if len(urls) > 1:
            extra["alternate_urls"] = urls[1:]
        if final_text and final_text.strip():
            extra["bot_response_text"] = final_text.strip()

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="poe",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Wire ``PoeImageGenProvider`` into the plugin registry."""
    ctx.register_image_gen_provider(PoeImageGenProvider())
