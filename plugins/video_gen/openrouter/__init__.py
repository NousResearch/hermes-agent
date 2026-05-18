"""OpenRouter video generation backend.

Uses OpenRouter's dedicated asynchronous video API:

1. ``POST /api/v1/videos`` with prompt/model/options.
2. Poll ``polling_url`` (or ``GET /api/v1/videos/{id}``) until terminal.
3. Return the first URL from ``unsigned_urls``. The gateway can download and
   deliver that URL; callers may also fetch it directly.

Selection precedence:
1. ``model=`` passed by the tool wrapper (usually ``video_gen.model``)
2. ``OPENROUTER_VIDEO_MODEL`` env var
3. ``video_gen.openrouter.model`` in config.yaml
4. ``video_gen.model`` in config.yaml
5. ``DEFAULT_MODEL``

Any configured model string is accepted so newly-added OpenRouter video models
work before Hermes updates its local picker catalog.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/veo-3.1"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_POLL_INTERVAL_SECONDS = 30

_MODELS: Dict[str, Dict[str, Any]] = {
    "google/veo-3.1": {
        "display": "Google Veo 3.1",
        "speed": "~1-5min",
        "strengths": "High-quality video generation; supports image inputs on compatible routes",
        "price": "see OpenRouter",
        "modalities": ["text", "image"],
    },
    "alibaba/wan-2.7": {
        "display": "Alibaba WAN 2.7",
        "speed": "varies",
        "strengths": "Text-to-video and image/reference guided video",
        "price": "see OpenRouter",
        "modalities": ["text", "image"],
    },
}

_COMMON_ASPECT_RATIOS = {"16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9", "9:21"}
_COMMON_RESOLUTIONS = {"480p", "720p", "1080p", "1K", "2K", "4K"}


def _load_video_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen config: %s", exc)
        return {}


def _resolve_model(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    candidates: List[Optional[str]] = [explicit, os.environ.get("OPENROUTER_VIDEO_MODEL")]
    cfg = _load_video_gen_section()
    or_cfg = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}
    if isinstance(or_cfg, dict):
        candidates.append(or_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            model_id = candidate.strip()
            return model_id, _MODELS.get(model_id, {
                "display": model_id,
                "speed": "unknown",
                "strengths": "Custom OpenRouter video-output model",
                "price": "see OpenRouter",
                "modalities": ["text", "image"],
            })
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def _resolve_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/") or DEFAULT_BASE_URL


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
        "X-Title": "Hermes Agent",
    }


def _normalize_aspect_ratio(value: str) -> str:
    value = (value or DEFAULT_ASPECT_RATIO).strip()
    return value if value in _COMMON_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO


def _normalize_resolution(value: str) -> str:
    value = (value or DEFAULT_RESOLUTION).strip()
    return value if value in _COMMON_RESOLUTIONS else DEFAULT_RESOLUTION


def _normalize_refs(reference_image_urls: Optional[List[str]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for url in reference_image_urls or []:
        normalized = (url or "").strip()
        if normalized:
            refs.append({"type": "image_url", "image_url": {"url": normalized}})
    return refs


def _frame_image(url: str) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": url},
        "frame_type": "first_frame",
    }


class OpenRouterVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "paid",
            "tag": "OpenRouter /api/v1/videos — text-to-video, image-to-video, references",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/settings/keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(_COMMON_ASPECT_RATIOS),
            "resolutions": sorted(_COMMON_RESOLUTIONS),
            "max_duration": 30,
            "min_duration": 1,
            "supports_audio": True,
            "supports_negative_prompt": False,
            "max_reference_images": 8,
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
        prompt = (prompt or "").strip()
        model_id, _meta = _resolve_model(model)
        normalized_aspect = _normalize_aspect_ratio(aspect_ratio)
        normalized_resolution = _normalize_resolution(resolution)
        image_url_norm = (image_url or "").strip() or None
        modality_used = "image" if image_url_norm else "text"

        if not prompt:
            return error_response(
                error="prompt is required for OpenRouter video generation",
                error_type="missing_prompt",
                provider="openrouter",
                model=model_id,
                aspect_ratio=normalized_aspect,
            )

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "OPENROUTER_API_KEY not set. Run `hermes tools` → Video "
                    "Generation → OpenRouter to configure, or set OPENROUTER_API_KEY."
                ),
                error_type="auth_required",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "resolution": normalized_resolution,
            "aspect_ratio": normalized_aspect,
        }
        if duration is not None:
            payload["duration"] = int(duration)
        if seed is not None:
            payload["seed"] = int(seed)
        if audio is not None:
            payload["generate_audio"] = bool(audio)
        if image_url_norm:
            payload["frame_images"] = [_frame_image(image_url_norm)]
        refs = _normalize_refs(reference_image_urls)
        if refs and not image_url_norm:
            payload["input_references"] = refs

        # OpenRouter exposes provider-specific passthrough via provider.options,
        # but the provider slug varies by route. The unified Hermes surface keeps
        # negative_prompt advisory-only for OpenRouter instead of guessing a slug.

        timeout_seconds = int(os.environ.get("OPENROUTER_VIDEO_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        poll_interval = int(os.environ.get("OPENROUTER_VIDEO_POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS))
        base_url = _resolve_base_url()

        try:
            with httpx.Client(timeout=60) as client:
                submit = client.post(
                    f"{base_url}/videos",
                    headers=_headers(api_key),
                    json=payload,
                )
                submit.raise_for_status()
                job = submit.json()

                job_id = job.get("id")
                polling_url = job.get("polling_url") or (f"{base_url}/videos/{job_id}" if job_id else None)
                if not job_id or not polling_url:
                    return error_response(
                        error="OpenRouter video submit response did not include id/polling_url",
                        error_type="empty_response",
                        provider="openrouter",
                        model=model_id,
                        prompt=prompt,
                        aspect_ratio=normalized_aspect,
                    )

                deadline = time.monotonic() + timeout_seconds
                status_body: Dict[str, Any] = job
                while time.monotonic() < deadline:
                    status = str(status_body.get("status") or "").lower()
                    if status in {"completed", "failed", "cancelled", "expired"}:
                        break
                    time.sleep(poll_interval)
                    poll = client.get(polling_url, headers=_headers(api_key), timeout=60)
                    poll.raise_for_status()
                    status_body = poll.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            return error_response(
                error=f"OpenRouter video generation failed ({exc.response.status_code}): {detail}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )
        except Exception as exc:
            logger.debug("OpenRouter video generation failed", exc_info=True)
            return error_response(
                error=f"OpenRouter video generation failed: {exc}",
                error_type="api_error",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        status = str(status_body.get("status") or "").lower()
        if status != "completed":
            if time.monotonic() >= deadline and status not in {"failed", "cancelled", "expired"}:
                return error_response(
                    error=f"Timed out waiting for OpenRouter video after {timeout_seconds}s",
                    error_type="timeout",
                    provider="openrouter",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=normalized_aspect,
                )
            message = status_body.get("error") or status_body.get("message") or f"status={status}"
            return error_response(
                error=f"OpenRouter video generation did not complete: {message}",
                error_type=f"openrouter_{status or 'unknown'}",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        urls = status_body.get("unsigned_urls") or []
        if not urls:
            return error_response(
                error="OpenRouter video completed without unsigned_urls",
                error_type="empty_response",
                provider="openrouter",
                model=model_id,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        extra: Dict[str, Any] = {
            "job_id": status_body.get("id") or job_id,
            "polling_url": status_body.get("polling_url") or polling_url,
            "resolution": normalized_resolution,
        }
        if status_body.get("generation_id"):
            extra["generation_id"] = status_body["generation_id"]
        if status_body.get("usage"):
            extra["usage"] = status_body["usage"]

        return success_response(
            video=urls[0],
            model=model_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=normalized_aspect,
            duration=int(duration) if duration else 0,
            provider="openrouter",
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(OpenRouterVideoGenProvider())
