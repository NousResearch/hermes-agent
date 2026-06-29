"""Cloud Temple image generation provider.

Uses the OpenAI-compatible images endpoint at api.ai.cloud-temple.com.
Model: x/z-image-turbo:latest (fast text-to-image).

API:
  POST https://api.ai.cloud-temple.com/v1/images/generations
  Body: { model, prompt, size, response_format: "b64_json" }

The provider is auto-discovered by the image gen registry. Activate with:
  image_gen.provider: cloud-temple
in config.yaml, or via `hermes tools`.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)


def _cloud_temple_api_key() -> str:
    try:
        from hermes_cli.config import get_env_value

        return (get_env_value("CLOUD_TEMPLE_API_KEY") or "").strip()
    except Exception:
        return (os.environ.get("CLOUD_TEMPLE_API_KEY") or "").strip()

# Cloud Temple base URL (production)
BASE_URL = "https://api.ai.cloud-temple.com/v1"

DEFAULT_IMAGE_MODEL = "x/z-image-turbo:latest"

_SIZE_MAP = {
    "square": "1024x1024",
    "landscape": "1536x1024",
    "portrait": "1024x1536",
}


def _load_config() -> Dict[str, Any]:
    """Read ``image_gen`` section from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Helper: save base64 image to local cache
# ---------------------------------------------------------------------------


def _save_b64_image(b64_data: str, prefix: str = "cloud_temple") -> Path:
    """Decode base64 image and save to HERMES_HOME/cache/images/."""
    from hermes_constants import get_hermes_home

    images_dir = get_hermes_home() / "cache" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    import time

    ts = int(time.time() * 1000)
    path = images_dir / f"{prefix}_{ts}.png"

    raw = base64.b64decode(b64_data)
    path.write_bytes(raw)
    return path


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CloudTempleImageGenProvider(ImageGenProvider):
    """Cloud Temple image generation — z-image-turbo via OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return "cloud-temple"

    @property
    def display_name(self) -> str:
        return "Cloud Temple"

    def is_available(self) -> bool:
        return bool(_cloud_temple_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_IMAGE_MODEL,
                "display": "z-image-turbo",
                "speed": "~5s",
                "strengths": "Fast text-to-image generation",
                "price": "free",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_IMAGE_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Cloud Temple",
            "badge": "free",
            "tag": "z-image-turbo fast text-to-image via OpenAI-compatible API",
            "env_vars": [
                {
                    "key": "CLOUD_TEMPLE_API_KEY",
                    "prompt": "Cloud Temple API key",
                    "url": "https://api.ai.cloud-temple.com/",
                }
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text"]}

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
                provider="cloud-temple",
                aspect_ratio=aspect,
            )

        api_key = _cloud_temple_api_key()
        if not api_key:
            return error_response(
                error=(
                    "CLOUD_TEMPLE_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → Cloud Temple to configure, or `hermes setup`."
                ),
                error_type="auth_required",
                provider="cloud-temple",
                aspect_ratio=aspect,
            )

        size = _SIZE_MAP.get(aspect, _SIZE_MAP["square"])

        url = f"{BASE_URL}/images/generations"
        payload = {
            "model": DEFAULT_IMAGE_MODEL,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
        except Exception as exc:
            return error_response(
                error=f"Network error contacting Cloud Temple: {exc}",
                error_type="network_error",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if resp.status_code != 200:
            detail = resp.text[:300]
            return error_response(
                error=f"Cloud Temple image API returned HTTP {resp.status_code}: {detail}",
                error_type="api_error",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            data = resp.json()
        except Exception:
            return error_response(
                error="Cloud Temple returned non-JSON response",
                error_type="parse_error",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Extract b64_json from response (OpenAI-compatible format)
        data_list = data.get("data", [])
        if not data_list:
            return error_response(
                error="Cloud Temple returned no image data",
                error_type="empty_response",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data_list[0]
        b64 = first.get("b64_json") if isinstance(first, dict) else None

        if not b64:
            url_ref = first.get("url") if isinstance(first, dict) else None
            if url_ref:
                return success_response(
                    image=url_ref,
                    model=DEFAULT_IMAGE_MODEL,
                    prompt=prompt,
                    aspect_ratio=aspect,
                    provider="cloud-temple",
                    modality="text",
                    extra={"size": size},
                )
            return error_response(
                error="Cloud Temple response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Save the image locally
        try:
            saved_path = _save_b64_image(b64)
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="cloud-temple",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="cloud-temple",
            modality="text",
            extra={"size": size},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire CloudTempleImageGenProvider into the registry."""
    ctx.register_image_gen_provider(CloudTempleImageGenProvider())