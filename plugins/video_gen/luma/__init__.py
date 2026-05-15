"""Luma Dream Machine video generation backend.

User-facing surface: pick **ray-2** (latest) or **ray-flash-2** (faster).
The plugin posts to Luma's ``/generations/video`` endpoint and polls
``/generations/{id}`` until the video is ready.

Models:
  ray-2         — Latest Luma video model, highest quality
  ray-flash-2   — Fast mode, lower quality but quicker results

Selection precedence:
    1. ``model=`` arg from the tool call
    2. ``LUMAAI_VIDEO_MODEL`` env var
    3. ``video_gen.luma.model`` in ``config.yaml``
    4. ``video_gen.model`` in ``config.yaml`` (when it's one of our IDs)
    5. ``DEFAULT_MODEL`` (ray-2)

Authentication via ``LUMAAI_API_KEY`` (Bearer token). Output is an HTTPS URL
from Luma's CDN; the gateway downloads and delivers it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

LUMAAI_MODELS: Dict[str, Dict[str, Any]] = {
    "ray-2": {
        "display": "Ray 2",
        "model_id": "ray-2",
        "speed": "~1-3 min",
        "price": "$0.10/s",
        "strengths": "Latest model, highest quality, cinematic",
        "modalities": ["text", "image"],
    },
    "ray-flash-2": {
        "display": "Ray Flash 2",
        "model_id": "ray-flash-2",
        "speed": "~30-60s",
        "price": "$0.05/s",
        "strengths": "Faster generation, good quality",
        "modalities": ["text", "image"],
    },
}

DEFAULT_MODEL = "ray-2"

# Luma supports standard aspect ratios
VALID_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21"]

# Luma supports duration as string ("5s" or "9s")
VALID_DURATIONS = ["5s", "9s"]

# Luma supports resolution
VALID_RESOLUTIONS = ["540p", "720p", "1080p", "4k"]


class LumaProvider(VideoGenProvider):
    """Luma Dream Machine (Ray 2) backend."""

    @property
    def name(self) -> str:
        return "luma"

    @property
    def display_name(self) -> str:
        return "Luma Dream Machine"

    def is_available(self) -> bool:
        return bool(os.environ.get("LUMAAI_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for mid, meta in LUMAAI_MODELS.items():
            out.append({
                "id": mid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
                "modalities": meta["modalities"],
            })
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": VALID_ASPECT_RATIOS,
            "resolutions": VALID_RESOLUTIONS,
            "min_duration": 5,
            "max_duration": 9,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Luma Dream Machine",
            "badge": "paid",
            "tag": "High-quality video generation with Ray 2",
            "env_vars": [
                {
                    "key": "LUMAAI_API_KEY",
                    "prompt": "Luma AI API key",
                    "url": "https://lumalabs.ai/dream-machine/api",
                },
            ],
        }

    # -----------------------------------------------------------------------
    # Model resolution
    # -----------------------------------------------------------------------

    def _resolve_model(self, model: Optional[str]) -> str:
        if model and model in LUMAAI_MODELS:
            return model
        env_model = os.environ.get("LUMAAI_VIDEO_MODEL")
        if env_model and env_model in LUMAAI_MODELS:
            return env_model
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            luma_cfg = (cfg or {}).get("video_gen", {}).get("luma", {})
            if isinstance(luma_cfg, dict):
                config_model = luma_cfg.get("model")
                if isinstance(config_model, str) and config_model in LUMAAI_MODELS:
                    return config_model
                generic = (cfg or {}).get("video_gen", {}).get("model")
                if isinstance(generic, str) and generic in LUMAAI_MODELS:
                    return generic
        except Exception:
            pass
        return DEFAULT_MODEL

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate video via Luma Dream Machine API.

        Luma uses a unified ``POST /generations/video`` endpoint with
        ``keyframes`` dict for image-to-video.
        """
        resolved_model = self._resolve_model(model)

        # Map duration to Luma's string format ("5s" or "9s")
        if duration is None:
            duration_str = "5s"
        else:
            duration_str = "9s" if duration >= 7 else "5s"

        # Validate aspect ratio
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            aspect_ratio = DEFAULT_ASPECT_RATIO

        # Validate resolution
        if resolution not in VALID_RESOLUTIONS:
            resolution = "720p"

        api_key = os.environ.get("LUMAAI_API_KEY")
        if not api_key:
            return error_response(
                error="LUMAAI_API_KEY not set. Set it via `hermes tools` → Video Generation.",
                error_type="missing_api_key",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        base_url = "https://api.lumalabs.ai/dream-machine/v1"

        # Build payload
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration_str,
            "resolution": resolution,
        }

        # Image-to-video: use keyframes parameter
        if image_url and image_url.strip():
            payload["keyframes"] = {
                "frame0": {"type": "image", "url": image_url.strip()}
            }

        if seed is not None:
            payload["seed"] = seed

        try:
            logger.info("Submitting Luma generation: %s", prompt[:80])

            resp = requests.post(
                f"{base_url}/generations",
                json=payload,
                headers=headers,
                timeout=30,
            )
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"Submit failed: HTTP {resp.status_code} — {resp.text}")

            job_data = resp.json()
            generation_id = job_data.get("id")
            if not generation_id:
                raise RuntimeError(f"No generation ID in response: {job_data}")

            logger.info("Luma generation submitted: %s", generation_id)

            # Poll
            max_wait = 600  # 10 minutes
            poll_interval = 10
            elapsed = 0

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                status_resp = requests.get(
                    f"{base_url}/generations/{generation_id}",
                    headers=headers,
                    timeout=15,
                )
                if status_resp.status_code != 200:
                    raise RuntimeError(f"Status check failed: HTTP {status_resp.status_code}")

                status_data = status_resp.json()
                state = status_data.get("state", "").lower()

                if state in ("completed", "finished"):
                    video_url = status_data.get("assets", {}).get("video")
                    if video_url:
                        modality = "image" if image_url else "text"
                        return success_response(
                            video=video_url,
                            model=resolved_model,
                            prompt=prompt,
                            modality=modality,
                            aspect_ratio=aspect_ratio,
                            duration=int(duration_str.replace("s", "")),
                            provider=self.name,
                        )
                    raise RuntimeError(f"No video URL in completed generation: {status_data}")

                elif state in ("failed", "error"):
                    error_msg = status_data.get("failure_reason", "Unknown error")
                    raise RuntimeError(f"Generation failed: {error_msg}")

                logger.info(
                    "Luma generation in progress: %s (%ds elapsed)",
                    state,
                    elapsed,
                )

            return error_response(
                error=f"Luma generation timed out after {max_wait}s. Generation ID: {generation_id}",
                error_type="timeout",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        except requests.exceptions.RequestException as exc:
            logger.error("Luma API request failed: %s", exc)
            return error_response(
                error=f"Luma API error: {exc}",
                error_type="api_error",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error("Luma generation failed: %s", exc)
            return error_response(
                error=f"Generation error: {exc}",
                error_type="generation_error",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )


def register(ctx) -> None:
    """Register the Luma Dream Machine video generation provider."""
    ctx.register_video_gen_provider(LumaProvider())
