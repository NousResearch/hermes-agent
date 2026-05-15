"""RunwayML video generation backend.

User-facing surface: pick **gen4.5** model.
The plugin auto-routes to Runway's ``/v1/tasks/text_to_video`` (text-to-video)
or ``/v1/tasks/image_to_video`` (image-to-video) endpoints.

Model:
  gen4.5  — Latest Runway model, cinematic quality

Selection precedence:
    1. ``model=`` arg from the tool call
    2. ``RUNWAYML_VIDEO_MODEL`` env var
    3. ``video_gen.runwayml.model`` in ``config.yaml``
    4. ``video_gen.model`` in ``config.yaml`` (when it's one of our IDs)
    5. ``DEFAULT_MODEL`` (gen4.5)

Authentication via ``RUNWAYML_API_KEY`` (Bearer token). Output is an HTTPS URL
from Runway's CDN; the gateway downloads and delivers it.
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

RUNWAYML_MODELS: Dict[str, Dict[str, Any]] = {
    "gen4.5": {
        "display": "Gen-4.5",
        "model_id": "gen4.5",
        "speed": "~2-5 min",
        "price": "$0.15/s",
        "strengths": "Latest cinematic model, best quality",
        "modalities": ["text", "image"],
    },
}

DEFAULT_MODEL = "gen4.5"

# Runway ratio format (pixel dimensions, not aspect ratio strings)
RATIO_MAP = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "4:3": "1104:832",
    "1:1": "960:960",
    "3:4": "832:1104",
    "21:9": "1584:672",
}


class RunwayMLProvider(VideoGenProvider):
    """RunwayML Gen-4.5 backend."""

    @property
    def name(self) -> str:
        return "runwayml"

    @property
    def display_name(self) -> str:
        return "RunwayML"

    def is_available(self) -> bool:
        return bool(os.environ.get("RUNWAYML_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for mid, meta in RUNWAYML_MODELS.items():
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
            "aspect_ratios": list(RATIO_MAP.keys()),
            "resolutions": ["720p"],  # Runway uses fixed resolution per ratio
            "min_duration": 2,
            "max_duration": 10,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "RunwayML",
            "badge": "paid",
            "tag": "Cinematic video generation with Gen-4.5",
            "env_vars": [
                {
                    "key": "RUNWAYML_API_KEY",
                    "prompt": "RunwayML API key",
                    "url": "https://runwayml.com/settings/api",
                },
            ],
        }

    # -----------------------------------------------------------------------
    # Model resolution
    # -----------------------------------------------------------------------

    def _resolve_model(self, model: Optional[str]) -> str:
        if model and model in RUNWAYML_MODELS:
            return model
        env_model = os.environ.get("RUNWAYML_VIDEO_MODEL")
        if env_model and env_model in RUNWAYML_MODELS:
            return env_model
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            runway_cfg = (cfg or {}).get("video_gen", {}).get("runwayml", {})
            if isinstance(runway_cfg, dict):
                config_model = runway_cfg.get("model")
                if isinstance(config_model, str) and config_model in RUNWAYML_MODELS:
                    return config_model
                generic = (cfg or {}).get("video_gen", {}).get("model")
                if isinstance(generic, str) and generic in RUNWAYML_MODELS:
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
        """Generate video via RunwayML API.

        Routing: image_url presence picks image-to-video vs text-to-video.
        Runway uses separate endpoints for each modality.
        """
        resolved_model = self._resolve_model(model)

        # Clamp duration: Runway supports 2-10 seconds
        if duration is None:
            duration = 5
        duration = max(2, min(10, duration))

        # Map aspect ratio to Runway's pixel ratio format
        ratio = RATIO_MAP.get(aspect_ratio, "1280:720")

        api_key = os.environ.get("RUNWAYML_API_KEY")
        if not api_key:
            return error_response(
                error="RUNWAYML_API_KEY not set. Set it via `hermes tools` → Video Generation.",
                error_type="missing_api_key",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-09-25",
        }

        try:
            if image_url and image_url.strip():
                return self._generate_image_to_video(
                    prompt=prompt,
                    image_url=image_url.strip(),
                    duration=duration,
                    ratio=ratio,
                    seed=seed,
                    headers=headers,
                    resolved_model=resolved_model,
                    aspect_ratio=aspect_ratio,
                )
            else:
                return self._generate_text_to_video(
                    prompt=prompt,
                    duration=duration,
                    ratio=ratio,
                    seed=seed,
                    headers=headers,
                    resolved_model=resolved_model,
                    aspect_ratio=aspect_ratio,
                )
        except requests.exceptions.RequestException as exc:
            logger.error("RunwayML API request failed: %s", exc)
            return error_response(
                error=f"RunwayML API error: {exc}",
                error_type="api_error",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error("RunwayML generation failed: %s", exc)
            return error_response(
                error=f"Generation error: {exc}",
                error_type="generation_error",
                provider=self.name,
                model=resolved_model,
                prompt=prompt,
            )

    def _generate_text_to_video(
        self, prompt, duration, ratio, seed, headers, resolved_model, aspect_ratio
    ) -> Dict[str, Any]:
        """Text-to-video generation."""
        submit_url = "https://api.runwayml.com/v1/tasks/text_to_video"
        payload = {
            "model": "gen4.5",
            "prompt_text": prompt,
            "duration": duration,
            "ratio": ratio,
        }
        if seed is not None:
            payload["seed"] = seed

        return self._submit_and_poll(
            submit_url=submit_url,
            payload=payload,
            headers=headers,
            resolved_model=resolved_model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            duration=duration,
            modality="text",
        )

    def _generate_image_to_video(
        self, prompt, image_url, duration, ratio, seed, headers, resolved_model, aspect_ratio
    ) -> Dict[str, Any]:
        """Image-to-video generation."""
        submit_url = "https://api.runwayml.com/v1/tasks/image_to_video"
        payload = {
            "model": "gen4.5",
            "prompt_text": prompt,
            "prompt_image": {"url": image_url},
            "duration": duration,
            "ratio": ratio,
        }
        if seed is not None:
            payload["seed"] = seed

        return self._submit_and_poll(
            submit_url=submit_url,
            payload=payload,
            headers=headers,
            resolved_model=resolved_model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            duration=duration,
            modality="image",
        )

    def _submit_and_poll(
        self,
        submit_url,
        payload,
        headers,
        resolved_model,
        prompt,
        aspect_ratio,
        duration,
        modality,
    ) -> Dict[str, Any]:
        """Submit generation job and poll until complete."""
        logger.info("Submitting RunwayML generation: %s", payload.get("prompt_text", "")[:80])

        resp = requests.post(submit_url, json=payload, headers=headers, timeout=30)
        if resp.status_code not in (200, 201, 202):
            raise RuntimeError(f"Submit failed: HTTP {resp.status_code} — {resp.text}")

        job_data = resp.json()
        task_id = job_data.get("id")
        if not task_id:
            raise RuntimeError(f"No task_id in response: {job_data}")

        logger.info("RunwayML task submitted: %s", task_id)

        # Poll
        max_wait = 600  # 10 minutes
        poll_interval = 10
        elapsed = 0

        status_url = f"https://api.runwayml.com/v1/tasks/{task_id}"

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_resp = requests.get(status_url, headers=headers, timeout=15)
            if status_resp.status_code != 200:
                raise RuntimeError(f"Status check failed: HTTP {status_resp.status_code}")

            status_data = status_resp.json()
            state = status_data.get("status", "").lower()

            if state in ("succeeded", "completed", "finished"):
                # Extract video URL from output
                outputs = status_data.get("output", [])
                if outputs:
                    video_url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("url")
                    if video_url:
                        return success_response(
                            video=video_url,
                            model=resolved_model,
                            prompt=prompt,
                            modality=modality,
                            aspect_ratio=aspect_ratio,
                            duration=duration,
                            provider=self.name,
                        )
                raise RuntimeError(f"No video URL in completed task: {status_data}")

            elif state in ("failed", "error"):
                error_msg = status_data.get("error", {}).get("message", status_data.get("failure_reason", "Unknown error"))
                raise RuntimeError(f"Generation failed: {error_msg}")

            logger.info(
                "RunwayML generation in progress: %s%% (%ds elapsed)",
                status_data.get("progress", "?"),
                elapsed,
            )

        return error_response(
            error=f"RunwayML generation timed out after {max_wait}s. Task ID: {task_id}",
            error_type="timeout",
            provider=self.name,
            model=resolved_model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
        )


def register(ctx) -> None:
    """Register the RunwayML video generation provider."""
    ctx.register_video_gen_provider(RunwayMLProvider())
