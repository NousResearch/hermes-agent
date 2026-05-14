"""Cloudflare Workers AI image generation backend.

Exposes Cloudflare's hosted text-to-image models as an
:class:`ImageGenProvider` implementation.

Supported model families (verified May 2026):
- FLUX.2 [klein] 9B / 4B and FLUX.2 [dev] (Black Forest Labs, partner)
- FLUX.1 [schnell] (Black Forest Labs)
- Leonardo Phoenix 1.0 / Lucid Origin (partner)
- Stable Diffusion XL Lightning (ByteDance, beta)

Selection precedence (first hit wins):

1. ``CLOUDFLARE_IMAGE_MODEL`` env var
2. ``image_gen.cloudflare.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`
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
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4"

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# Verified against https://developers.cloudflare.com/workers-ai/models/ (May
# 2026). When adding entries, double-check the model id at
# /workers-ai/models/<slug>/ — Cloudflare reserves the right to rename or
# deprecate IDs, and a bad default surfaces as a hard 404 on every call.

_MODELS: Dict[str, Dict[str, Any]] = {
    "@cf/black-forest-labs/flux-1-schnell": {
        "display": "FLUX.1 [schnell]",
        "speed": "~3-5s",
        "strengths": "Fast, free-tier friendly, strong prompt adherence",
    },
    "@cf/black-forest-labs/flux-2-klein-9b": {
        "display": "FLUX.2 [klein] 9B",
        "speed": "~5-8s",
        "strengths": "Distilled 9B image-gen + edit, fixed 4-step inference",
    },
    "@cf/black-forest-labs/flux-2-klein-4b": {
        "display": "FLUX.2 [klein] 4B",
        "speed": "~3-6s",
        "strengths": "Smaller, cheaper FLUX.2 — same unified gen+edit pipeline",
    },
    "@cf/black-forest-labs/flux-2-dev": {
        "display": "FLUX.2 [dev]",
        "speed": "~15-30s",
        "strengths": "Highest fidelity FLUX.2; multi-reference images",
    },
    "@cf/leonardo/phoenix-1.0": {
        "display": "Leonardo Phoenix 1.0",
        "speed": "~10-15s",
        "strengths": "Exceptional prompt adherence, coherent text rendering",
    },
    "@cf/leonardo/lucid-origin": {
        "display": "Leonardo Lucid Origin",
        "speed": "~10-15s",
        "strengths": "Adaptable, accurate in-image text and graphic design",
    },
    "@cf/bytedance/stable-diffusion-xl-lightning": {
        "display": "SDXL Lightning",
        "speed": "~2-4s",
        "strengths": "Lightning-fast 1024px generation in few steps",
    },
}

DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Default pixel resolutions per aspect bucket. Workers AI accepts width/height
# as ints — values must be multiples of 8 for most models.
_ASPECT_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "landscape": (1280, 720),
    "square": (1024, 1024),
    "portrait": (720, 1280),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config() -> Dict[str, Any]:
    """Read ``image_gen.cloudflare`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        cf_section = section.get("cloudflare") if isinstance(section, dict) else None
        return cf_section if isinstance(cf_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.cloudflare config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("CLOUDFLARE_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate and candidate in _MODELS:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CloudflareImageGenProvider(ImageGenProvider):
    """Cloudflare Workers AI image generation backend."""

    @property
    def name(self) -> str:
        return "cloudflare"

    @property
    def display_name(self) -> str:
        return "Cloudflare Workers AI"

    def is_available(self) -> bool:
        return bool(
            os.getenv("CLOUDFLARE_API_TOKEN")
            and os.getenv("CLOUDFLARE_ACCOUNT_ID")
        )

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

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Cloudflare Workers AI",
            "badge": "free-tier-available",
            "tag": "Hosted image generation via Workers AI (FLUX.2, Leonardo, SDXL)",
            "env_vars": [
                {
                    "key": "CLOUDFLARE_API_TOKEN",
                    "prompt": "Cloudflare API token (Workers AI Read+Run)",
                    "url": "https://dash.cloudflare.com/profile/api-tokens",
                },
                {
                    "key": "CLOUDFLARE_ACCOUNT_ID",
                    "prompt": "Cloudflare account ID",
                    "url": "https://dash.cloudflare.com",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        if not token or not account_id:
            return error_response(
                error=(
                    "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID required. "
                    "Create a token at https://dash.cloudflare.com/profile/api-tokens "
                    "with the Workers AI Read+Run permissions."
                ),
                error_type="missing_api_key",
                provider="cloudflare",
                aspect_ratio=aspect_ratio,
            )

        model_id, _meta = _resolve_model()
        aspect = resolve_aspect_ratio(aspect_ratio)
        width, height = _ASPECT_RESOLUTIONS.get(aspect, _ASPECT_RESOLUTIONS["landscape"])

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        # FLUX schnell defaults to 4 steps; allow override but stay fast.
        if "schnell" in model_id:
            payload.setdefault("num_steps", 4)

        url = f"{CF_API_BASE}/accounts/{account_id}/ai/run/{model_id}"
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                err_msg = resp.json() if resp is not None else str(exc)
            except Exception:
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("Cloudflare image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"Cloudflare image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="cloudflare",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="Cloudflare image generation timed out (180s)",
                error_type="timeout",
                provider="cloudflare",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"Cloudflare connection error: {exc}",
                error_type="connection_error",
                provider="cloudflare",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64 = _extract_image_b64(response)
        if not b64:
            return error_response(
                error="Cloudflare returned no image data",
                error_type="empty_response",
                provider="cloudflare",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(
                b64,
                prefix=f"cf_{model_id.replace('/', '_').replace('@', '')}",
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="cloudflare",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="cloudflare",
            extra={"resolution": f"{width}x{height}"},
        )


def _extract_image_b64(response: requests.Response) -> Optional[str]:
    """Return base64-encoded image bytes from a Workers AI image response.

    FLUX returns ``{"result": {"image": "<base64>"}}`` while some image models
    (and the ``response_format=raw`` path) return the PNG bytes directly with
    a non-JSON content type. Handle both transparently.
    """
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            data = response.json()
        except Exception:
            return None
        result = data.get("result") if isinstance(data, dict) else None
        if isinstance(result, dict):
            img = result.get("image")
            if isinstance(img, str) and img:
                return img
        return None
    if response.content:
        return base64.b64encode(response.content).decode("ascii")
    return None


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(CloudflareImageGenProvider())
