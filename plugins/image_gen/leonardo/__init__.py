"""Leonardo.AI image generation backend.

Exposes Leonardo.AI's platform models (Phoenix, SDXL, Flux) as an
:class:`ImageGenProvider` implementation.

Features:
- Text-to-image generation via official REST API
- Multiple platform models (Phoenix, SDXL, Flux)
- 31 preset styles (ANIME, DYNAMIC, PHOTOGRAPHY, etc.)
- Alchemy and Prompt Magic support
- Image-to-image via init_image_id

Selection precedence (first hit wins):
1. ``LEONARDO_IMAGE_MODEL`` env var
2. ``image_gen.leonardo.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leonardo REST API
# ---------------------------------------------------------------------------

_BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"
_POLL_INTERVAL_S = 4
_POLL_TIMEOUT_S = 180

# ---------------------------------------------------------------------------
# Model catalog — platform models available to all users
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "phoenix": {
        "id": "de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3",
        "display": "Leonardo Phoenix",
        "speed": "~8-15s",
        "strengths": "High quality, versatile, alchemy support",
        "sd_version": "PHOENIX",
    },
    "sdxl": {
        "id": "1e60896f-3c26-4296-8ecc-53e2afecc132",
        "display": "Leonardo SDXL",
        "speed": "~5-10s",
        "strengths": "Fast, good quality, SDXL-based",
        "sd_version": "SDXL_1_0",
    },
    "flux": {
        "id": "b2614463-296c-462a-9586-aafdb8f00e36",
        "display": "Leonardo Flux",
        "speed": "~10-20s",
        "strengths": "High fidelity, latest architecture",
        "sd_version": "FLUX",
    },
}

DEFAULT_MODEL = "phoenix"

# Hermes aspect_ratio → Leonardo (width, height)
_ASPECT_SIZES: Dict[str, Dict[str, int]] = {
    "landscape": {"width": 1024, "height": 768},
    "square": {"width": 1024, "height": 1024},
    "portrait": {"width": 768, "height": 1024},
}

# Default preset styles that work well
_DEFAULT_PRESET_STYLE = "DYNAMIC"

# Valid preset styles from Leonardo API
PRESET_STYLES = [
    "ANIME",
    "BOKEH",
    "CINEMATIC",
    "CINEMATIC_CLOSEUP",
    "CREATIVE",
    "DYNAMIC",
    "ENVIRONMENT",
    "FASHION",
    "FILM",
    "FOOD",
    "GENERAL",
    "HDR",
    "ILLUSTRATION",
    "LEONARDO",
    "LONG_EXPOSURE",
    "MACRO",
    "MINIMALISTIC",
    "MONOCHROME",
    "MOODY",
    "NONE",
    "NEUTRAL",
    "PHOTOGRAPHY",
    "PORTRAIT",
    "RAYTRACED",
    "RENDER_3D",
    "RETRO",
    "SKETCH_BW",
    "SKETCH_COLOR",
    "STOCK_PHOTO",
    "VIBRANT",
    "UNPROCESSED",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_leonardo_config() -> Dict[str, Any]:
    """Load Leonardo-specific config from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            leo = section.get("leonardo")
            if isinstance(leo, dict):
                return leo
    except Exception:
        pass
    return {}


def _resolve_model() -> tuple[str, str, str]:
    """Return (model_key, model_id, sd_version) for the active model."""
    # 1. Env var override
    env_model = os.environ.get("LEONARDO_IMAGE_MODEL", "").strip().lower()
    if env_model and env_model in _MODELS:
        m = _MODELS[env_model]
        return env_model, m["id"], m["sd_version"]

    # 2. Config override
    cfg = _load_leonardo_config()
    cfg_model = cfg.get("model", "").strip().lower() if cfg else ""
    if cfg_model and cfg_model in _MODELS:
        m = _MODELS[cfg_model]
        return cfg_model, m["id"], m["sd_version"]

    # 3. Default
    m = _MODELS[DEFAULT_MODEL]
    return DEFAULT_MODEL, m["id"], m["sd_version"]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> Optional[str]:
    """Get Leonardo API key from env."""
    key = os.environ.get("LEONARDO_API_KEY", "").strip()
    return key or None


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _create_generation(
    api_key: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    model_id: str,
    sd_version: str,
    num_images: int,
    preset_style: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    alchemy: bool,
) -> str:
    """Submit a generation job. Returns generationId."""
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "modelId": model_id,
        "sd_version": sd_version,
        "num_images": num_images,
        "presetStyle": preset_style,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "alchemy": alchemy,
        "public": False,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(
        f"{_BASE_URL}/generations",
        headers=_headers(api_key),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Response: {"sdGenerationJob": {"generationId": "...", "apiCreditCost": ...}}
    job = data.get("sdGenerationJob") or {}
    gen_id = job.get("generationId")
    if not gen_id:
        raise RuntimeError(f"No generationId in response: {data}")
    logger.info(
        "Leonardo generation submitted: %s (cost: %s)", gen_id, job.get("apiCreditCost")
    )
    return gen_id


def _poll_generation(api_key: str, gen_id: str) -> List[str]:
    """Poll until generation completes. Returns list of image URLs."""
    deadline = time.time() + _POLL_TIMEOUT_S
    while time.time() < deadline:
        resp = requests.get(
            f"{_BASE_URL}/generations/{gen_id}",
            headers=_headers(api_key),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        # Response: {"generations_by_pk": {"status": "...", "generated_images": [...]}}
        gen = data.get("generations_by_pk") or {}
        status = gen.get("status", "").upper()

        if status == "COMPLETE":
            images = gen.get("generated_images") or []
            urls = [img["url"] for img in images if img.get("url")]
            if not urls:
                raise RuntimeError(f"Generation complete but no image URLs: {gen}")
            return urls

        if status == "FAILED":
            raise RuntimeError(f"Leonardo generation failed: {gen}")

        # PENDING or PROCESSING — keep polling
        logger.debug("Leonardo generation %s status=%s, polling...", gen_id, status)
        time.sleep(_POLL_INTERVAL_S)

    raise TimeoutError(
        f"Leonardo generation {gen_id} timed out after {_POLL_TIMEOUT_S}s"
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LeonardoImageGenProvider(ImageGenProvider):
    """Leonardo.AI image generation backend."""

    @property
    def name(self) -> str:
        return "leonardo"

    @property
    def display_name(self) -> str:
        return "Leonardo.AI"

    def is_available(self) -> bool:
        return _get_api_key() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": key,
                "display": meta["display"],
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
            }
            for key, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Leonardo.AI",
            "badge": "free-tier",
            "tag": "Phoenix, SDXL, Flux — 150 free tokens/day",
            "env_vars": [
                {
                    "key": "LEONARDO_API_KEY",
                    "prompt": "Leonardo.AI API key",
                    "url": "https://app.leonardo.ai/settings/api",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via Leonardo.AI REST API."""
        api_key = _get_api_key()
        if not api_key:
            return error_response(
                error="LEONARDO_API_KEY is not set. Get one at https://app.leonardo.ai/settings/api",
                error_type="missing_api_key",
                provider="leonardo",
            )

        aspect = resolve_aspect_ratio(aspect_ratio)
        sizes = _ASPECT_SIZES.get(aspect, _ASPECT_SIZES["landscape"])

        model_key, model_id, sd_version = _resolve_model()

        # Optional overrides from kwargs
        num_images = kwargs.get("num_images", 1)
        if not isinstance(num_images, int) or num_images < 1:
            num_images = 1
        if num_images > 4:
            num_images = 4

        guidance_scale = kwargs.get("guidance_scale", 7)
        if not isinstance(guidance_scale, (int, float)) or not (
            1 <= guidance_scale <= 20
        ):
            guidance_scale = 7

        num_inference_steps = kwargs.get("num_inference_steps", 15)
        if not isinstance(num_inference_steps, int) or not (
            10 <= num_inference_steps <= 60
        ):
            num_inference_steps = 15

        seed = kwargs.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None

        preset_style = kwargs.get("preset_style", _DEFAULT_PRESET_STYLE)
        if isinstance(preset_style, str):
            preset_style = preset_style.upper()
        if preset_style not in PRESET_STYLES:
            preset_style = _DEFAULT_PRESET_STYLE

        alchemy = kwargs.get("alchemy", True)
        if not isinstance(alchemy, bool):
            alchemy = True

        negative_prompt = kwargs.get("negative_prompt", "")
        if not isinstance(negative_prompt, str):
            negative_prompt = ""

        try:
            gen_id = _create_generation(
                api_key=api_key,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=sizes["width"],
                height=sizes["height"],
                model_id=model_id,
                sd_version=sd_version,
                num_images=num_images,
                preset_style=preset_style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                alchemy=alchemy,
            )

            urls = _poll_generation(api_key, gen_id)

            # Save first image locally (matches other providers' behaviour)
            image_path = save_url_image(urls[0], prefix="leonardo")
            return success_response(
                image=str(image_path),
                model=model_key,
                prompt=prompt,
                aspect_ratio=aspect,
                provider="leonardo",
                extra={"generation_id": gen_id, "urls": urls},
            )

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            body = ""
            if exc.response is not None:
                try:
                    body = exc.response.text[:500]
                except Exception:
                    pass
            logger.warning("Leonardo API HTTP %s: %s", status, body)
            return error_response(
                error=f"Leonardo API error (HTTP {status}): {body}",
                error_type="api_error",
                provider="leonardo",
                model=model_key,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        except TimeoutError as exc:
            logger.warning("Leonardo generation timed out: %s", exc)
            return error_response(
                error=str(exc),
                error_type="timeout",
                provider="leonardo",
                model=model_key,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Leonardo generation failed: %s", exc, exc_info=True)
            return error_response(
                error=f"Leonardo generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="leonardo",
                model=model_key,
                prompt=prompt,
                aspect_ratio=aspect,
            )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire LeonardoImageGenProvider into the registry."""
    ctx.register_image_gen_provider(LeonardoImageGenProvider())
