"""Generic OpenAI-compatible video generation for Custom Endpoints.

This plugin reads the ``providers:`` section of ``config.yaml`` and registers
a :class:`VideoGenProvider` for each named provider that declares a
``capabilities.video_gen`` block. This lets users point a single Custom
Endpoint at an OpenAI-compatible video API (e.g. ``grok-imagine-video``,
``grok-imagine-video-1.5``) and select it via ``video_gen.provider`` — without
duplicating credentials into provider-specific env vars.

Config example::

    providers:
      my-gateway:
        name: My Gateway
        base_url: https://gateway.example.com/v1
        key_env: HERMES_CUSTOM_MY_GATEWAY_API_KEY
        capabilities:
          video_gen:
            models:
              - grok-imagine-video
              - grok-imagine-video-1.5
            default_model: grok-imagine-video

The generated provider reuses the endpoint's ``base_url`` and ``key_env``.
Requests follow the standard OpenAI ``POST /videos`` async shape (create →
poll → download).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_url_video,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _load_provider_configs() -> Dict[str, Dict[str, Any]]:
    """Read the ``providers:`` section from config.yaml and return entries
    that declare an ``capabilities.video_gen`` block.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    except Exception as exc:
        logger.debug("Could not load config for custom-endpoint video_gen: %s", exc)
        return {}

    providers = cfg.get("providers") if isinstance(cfg, dict) else None
    if not isinstance(providers, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for key, entry in providers.items():
        if not isinstance(entry, dict):
            continue
        caps = entry.get("capabilities")
        if not isinstance(caps, dict):
            continue
        video_gen_caps = caps.get("video_gen")
        if not isinstance(video_gen_caps, dict):
            continue
        models = video_gen_caps.get("models")
        if not isinstance(models, list) or not models:
            continue
        result[key] = entry
    return result


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

# Polling cadence for the async video job. Video generation is slow — these
# defaults give a reasonable balance between responsiveness and not hammering
# the endpoint.
_POLL_INTERVAL_S: float = 5.0
_POLL_DEADLINE_S: float = 900.0  # 15 minutes max

# Terminal statuses across OpenAI-compatible video APIs.
_TERMINAL_STATUSES = frozenset({
    "completed", "succeeded", "failed", "error", "cancelled", "canceled",
})


class CustomEndpointVideoGenProvider(VideoGenProvider):
    """Video generation provider backed by an OpenAI-compatible Custom Endpoint.

    Each instance corresponds to one ``providers:`` entry that declares
    ``capabilities.video_gen``. The provider name is ``custom:<config-key>``
    so it can be selected via ``video_gen.provider: custom:my-gateway``.
    """

    def __init__(self, config_key: str, entry: Dict[str, Any]) -> None:
        self._config_key = config_key
        self._entry = entry
        self._caps = entry["capabilities"]["video_gen"]

        # Build the model catalog from config
        self._models: List[Dict[str, Any]] = []
        raw_models = self._caps.get("models", [])
        for m in raw_models:
            if isinstance(m, str) and m.strip():
                self._models.append({"id": m.strip(), "display": m.strip()})
            elif isinstance(m, dict) and m.get("id"):
                self._models.append(m)

        self._default_model = self._caps.get("default_model")
        if not self._default_model and self._models:
            self._default_model = self._models[0].get("id")

    # -- VideoGenProvider interface -----------------------------------------

    @property
    def name(self) -> str:
        return f"custom:{self._config_key}"

    @property
    def display_name(self) -> str:
        return self._entry.get("name") or self._config_key

    def is_available(self) -> bool:
        """True when the endpoint's ``key_env`` is set in the environment."""
        key_env = self._entry.get("key_env", "")
        if not key_env:
            return True
        return bool(os.environ.get(key_env, "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return list(self._models)

    def default_model(self) -> Optional[str]:
        return self._default_model

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text"],
            "aspect_ratios": list(self._caps.get("aspect_ratios", ["16:9", "9:16", "1:1"])),
            "resolutions": list(self._caps.get("resolutions", ["720p", "1080p"])),
            "max_duration": self._caps.get("max_duration", 10),
            "min_duration": self._caps.get("min_duration", 1),
            "supports_audio": self._caps.get("supports_audio", False),
            "supports_negative_prompt": self._caps.get("supports_negative_prompt", False),
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        key_env = self._entry.get("key_env", "")
        env_vars = []
        if key_env:
            env_vars.append({
                "key": key_env,
                "prompt": f"API key for {self.display_name}",
                "url": "",
            })
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": f"Custom Endpoint ({self._entry.get('base_url', '')})",
            "env_vars": env_vars,
        }

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
        """Generate a video via the OpenAI-compatible /videos async endpoint."""
        if not prompt or not prompt.strip():
            return error_response(
                error="prompt is required",
                error_type="invalid_request",
                provider=self.name,
            )

        # Resolve model
        model_id = model or self._default_model
        if not model_id:
            return error_response(
                error="No model configured for this Custom Endpoint. "
                      "Set capabilities.video_gen.default_model or pass model=.",
                provider=self.name,
                prompt=prompt,
            )

        # Validate model against catalog
        valid_ids = {m["id"] for m in self._models}
        if model_id not in valid_ids:
            return error_response(
                error=f"Model '{model_id}' is not in the configured model list "
                      f"for {self.display_name}. Available: {', '.join(sorted(valid_ids))}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
            )

        # Resolve credentials
        key_env = self._entry.get("key_env", "")
        api_key = os.environ.get(key_env, "").strip() if key_env else ""

        # Resolve base URL
        base_url = (
            self._entry.get("base_url")
            or self._entry.get("api")
            or self._entry.get("url")
            or ""
        ).strip().rstrip("/")
        if not base_url:
            return error_response(
                error=f"Custom Endpoint '{self._config_key}' has no base_url configured.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
            )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{base_url}/videos"

        # Build the request payload (OpenAI-compatible /videos shape)
        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
        }
        if image_url:
            payload["image_url"] = image_url
        if duration:
            payload["duration"] = duration
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if audio is not None:
            payload["audio"] = audio
        if seed is not None:
            payload["seed"] = seed

        import requests

        # Step 1: Create the video job
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            return error_response(
                error=f"Video generation request failed: {exc}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
            )

        try:
            body = resp.json()
        except Exception as exc:
            return error_response(
                error=f"Video generation returned non-JSON response: {exc}",
                provider=self.name,
                model=model_id,
                prompt=prompt,
            )

        # The response may be synchronous (video URL directly) or async (job ID
        # requiring polling). OpenAI-compatible APIs typically return:
        #   {"id": "vid-xxx", "status": "pending", ...}
        # or for sync endpoints:
        #   {"data": [{"url": "https://..."}]}
        video_url = _extract_video_url(body)

        # If we got a direct URL, we're done
        if video_url:
            return _materialize_video(
                video_url, model_id, prompt, "image" if image_url else "text",
                aspect_ratio, duration or 0, self.name,
            )

        # Async path: poll for completion
        job_id = body.get("id") if isinstance(body, dict) else None
        if not job_id:
            return error_response(
                error="Video generation response contains neither a video URL "
                      "nor a job ID for polling.",
                provider=self.name,
                model=model_id,
                prompt=prompt,
            )

        poll_url = f"{base_url}/videos/{job_id}"
        deadline = time.monotonic() + _POLL_DEADLINE_S

        while True:
            if time.monotonic() >= deadline:
                return error_response(
                    error=f"Video job {job_id} did not complete within "
                          f"{int(_POLL_DEADLINE_S)}s.",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                )

            time.sleep(_POLL_INTERVAL_S)

            try:
                poll_resp = requests.get(poll_url, headers=headers, timeout=60)
                poll_resp.raise_for_status()
            except Exception as exc:
                logger.debug("Video poll for %s failed: %s", job_id, exc)
                continue

            try:
                poll_body = poll_resp.json()
            except Exception:
                continue

            status = (poll_body.get("status") or "").lower() if isinstance(poll_body, dict) else ""
            if status in _TERMINAL_STATUSES:
                if status in ("failed", "error", "cancelled", "canceled"):
                    error_msg = poll_body.get("error", {})
                    error_text = error_msg.get("message", "") if isinstance(error_msg, dict) else str(error_msg)
                    return error_response(
                        error=f"Video generation failed: {error_text or status}",
                        provider=self.name,
                        model=model_id,
                        prompt=prompt,
                    )

                video_url = _extract_video_url(poll_body)
                if video_url:
                    return _materialize_video(
                        video_url, model_id, prompt,
                        "image" if image_url else "text",
                        aspect_ratio, duration or 0, self.name,
                    )

                # Completed but no URL — error
                return error_response(
                    error=f"Video job {job_id} completed but no video URL was returned.",
                    provider=self.name,
                    model=model_id,
                    prompt=prompt,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_video_url(body: Any) -> Optional[str]:
    """Extract a video URL from an OpenAI-compatible response.

    Handles these shapes:
    - {"data": [{"url": "https://..."}]}
    - {"url": "https://..."}
    - {"output": "https://..."}
    - {"video": {"url": "https://..."}}
    """
    if not isinstance(body, dict):
        return None

    # OpenAI shape: {"data": [{"url": "..."}]}
    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            url = first.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()

    # Direct URL
    for key in ("url", "output", "video_url"):
        val = body.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Nested: {"video": {"url": "..."}}
    video_obj = body.get("video")
    if isinstance(video_obj, dict):
        url = video_obj.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()

    return None


def _materialize_video(
    video_url: str,
    model_id: str,
    prompt: str,
    modality: str,
    aspect_ratio: str,
    duration: int,
    provider_name: str,
) -> Dict[str, Any]:
    """Download the video to the local cache and return a success response."""
    try:
        path = save_url_video(video_url, prefix="custom_endpoint")
        video_str = str(path)
    except Exception as exc:
        # Fall back to returning the URL directly
        video_str = video_url
        logger.warning(
            "Could not cache video from %s: %s — returning URL directly",
            video_url, exc,
        )

    return success_response(
        video=video_str,
        model=model_id,
        prompt=prompt,
        modality=modality,
        aspect_ratio=aspect_ratio,
        duration=duration,
        provider=provider_name,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Plugin entry point — called by the plugin loader.

    Reads ``providers:`` from config.yaml, finds entries with
    ``capabilities.video_gen``, and registers a
    :class:`CustomEndpointVideoGenProvider` for each.
    """
    configs = _load_provider_configs()
    if not configs:
        logger.debug("No Custom Endpoints with video_gen capabilities found.")
        return

    for config_key, entry in configs.items():
        try:
            provider = CustomEndpointVideoGenProvider(config_key, entry)
            ctx.register_video_gen_provider(provider)
        except Exception as exc:
            logger.warning(
                "Failed to register Custom Endpoint video_gen provider '%s': %s",
                config_key, exc,
            )