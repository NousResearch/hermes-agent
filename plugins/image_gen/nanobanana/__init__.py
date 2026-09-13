"""NanoBanana 2 image generation backend.

Calls our self-hosted Gemini 3.1 Flash Image proxy on Cloud Run, which
speaks the OpenAI /v1/images/generations API. No external API key needed
(the proxy mints its own Google token downstream).

Config:
  image_gen:
    provider: nanobanana
    model: gemini-3.1-flash-image
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
import uuid
from typing import Any, Dict, Optional

import httpx

from agent.image_gen_provider import ImageGenProvider
from agent.image_gen_registry import register_provider

logger = logging.getLogger(__name__)

PROXY_BASE = os.environ.get(
    "NANOBANANA_API_BASE",
    "https://gemini-vlm-gcp.aac.adskeng.net/v1",
)
API_KEY = os.environ.get("NANOBANANA_API_KEY", "EMPTY")
DEFAULT_MODEL = os.environ.get("NANOBANANA_MODEL", "gemini-3.1-flash-image")
IMAGE_CACHE_DIR = os.path.expanduser(
    os.environ.get("IMAGE_CACHE_DIR", "~/.hermes/cache/images")
)

# OpenAI size -> Gemini aspect ratio
_SIZE_MAP = {
    "1024x1024": "1:1",
    "1024x1792": "9:16",
    "1792x1024": "16:9",
    "768x768": "1:1",
}


class NanoBananaProvider(ImageGenProvider):
    """Image generation via our Gemini 3.1 Flash Image proxy."""

    @property
    def name(self) -> str:
        return "nanobanana"

    @property
    def display_name(self) -> str:
        return "NanoBanana 2 (Gemini 3.1 Flash Image)"

    @property
    def supported_sizes(self) -> list[str]:
        return list(_SIZE_MAP.keys())

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image. Returns dict with 'images' list of file paths."""
        model = model or DEFAULT_MODEL
        aspect = _SIZE_MAP.get(size, "1:1")

        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        logger.info("nanobanana: generating image (model=%s, size=%s)", model, size)

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{PROXY_BASE}/images/generations",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        data = resp.json()
        images = data.get("data", [])

        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        saved = []
        for i, img in enumerate(images):
            b64 = img.get("b64_json")
            if not b64:
                # If url-based, download instead
                url = img.get("url")
                if url:
                    with httpx.Client(timeout=60) as client:
                        img_resp = client.get(url)
                        img_resp.raise_for_status()
                        raw = img_resp.content
                else:
                    continue
            else:
                raw = base64.b64decode(b64)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:6]
            fname = f"nb2_{ts}_{uid}.png"
            fpath = os.path.join(IMAGE_CACHE_DIR, fname)
            with open(fpath, "wb") as f:
                f.write(raw)
            saved.append(fpath)
            logger.info("nanobanana: saved %s (%d bytes)", fpath, len(raw))

        return {
            "images": saved,
            "model": model,
            "count": len(saved),
        }


register_provider(NanoBananaProvider())
