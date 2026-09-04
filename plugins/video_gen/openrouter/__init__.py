"""OpenRouter MiniMax H3 Max video generation backend.

Uses OpenRouter's dedicated asynchronous video API rather than the chat
completions API: submit ``POST /api/v1/videos``, poll the returned job, then
materialize the completed clip under Hermes' video cache.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    save_url_video,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "minimax/hailuo-3-max"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DURATION = 5
DEFAULT_RESOLUTION = "768p"
DEFAULT_ASPECT_RATIO = "16:9"
SUPPORTED_DURATIONS = tuple(range(5, 16))
SUPPORTED_RESOLUTIONS = ("480p", "768p")
SUPPORTED_ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled", "expired"})


def _is_public_first_frame_url(value: str) -> bool:
    """Accept provider-fetchable HTTPS URLs, never local/private inputs."""
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").strip().lower().rstrip(".")
        if parsed.scheme.lower() != "https" or not host:
            return False
        if host == "localhost" or host.endswith((".localhost", ".local", ".lan", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return True
    except (TypeError, ValueError):
        return False


def _nearest(value: int, supported: tuple[int, ...]) -> int:
    return min(supported, key=lambda candidate: (abs(candidate - value), candidate))


def _build_payload(
    *,
    prompt: str,
    image_url: Optional[str],
    duration: Optional[int],
    aspect_ratio: str,
    resolution: str,
) -> Dict[str, Any]:
    """Translate the unified Hermes inputs to OpenRouter's video API."""
    requested_duration = duration if duration is not None else DEFAULT_DURATION
    try:
        requested_duration = int(requested_duration)
    except (TypeError, ValueError):
        requested_duration = DEFAULT_DURATION

    payload: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "duration": _nearest(requested_duration, SUPPORTED_DURATIONS),
        "resolution": resolution if resolution in SUPPORTED_RESOLUTIONS else DEFAULT_RESOLUTION,
        "aspect_ratio": (
            aspect_ratio if aspect_ratio in SUPPORTED_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO
        ),
    }
    if image_url:
        payload["frame_images"] = [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "frame_type": "first_frame",
            }
        ]
    return payload


class OpenRouterVideoGenProvider(VideoGenProvider):
    """MiniMax H3 Max text-to-video and first-frame image-to-video."""

    _IGNORES_SEED = True  # Unified ABC input; H3 Max's OpenRouter SKU rejects it.
    _poll_interval_s = 5.0
    _poll_deadline_s = 900.0
    _request_timeout_s = 60.0


    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    def _api_key(self) -> str:
        return os.getenv("OPENROUTER_API_KEY", "").strip()

    def _base_url(self) -> str:
        return os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")

    def _session(self):
        import requests

        return requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent/video_gen",
        }

    def is_available(self) -> bool:
        return bool(self._api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "MiniMax H3 Max",
                "speed": "~20-60s",
                "strengths": "Fast text-to-video and first-frame image-to-video.",
                "price": "$0.05/s (480p), $0.08/s (768p)",
                "modalities": ["text", "image"],
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": list(SUPPORTED_ASPECT_RATIOS),
            "resolutions": list(SUPPORTED_RESOLUTIONS),
            "max_duration": max(SUPPORTED_DURATIONS),
            "min_duration": min(SUPPORTED_DURATIONS),
            "supports_audio": False,
            "supports_negative_prompt": False,
            "supports_seed": False,
            "supports_upscale": False,
            "max_reference_images": 0,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenRouter",
            "badge": "paid",
            "tag": "MiniMax H3 Max — text-to-video and first-frame image-to-video",
            "env_vars": [
                {
                    "key": "OPENROUTER_API_KEY",
                    "prompt": "OpenRouter API key",
                    "url": "https://openrouter.ai/settings/keys",
                }
            ],
        }

    def _poll(self, session: Any, job_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self._poll_deadline_s
        url = f"{self._base_url()}/videos/{job_id}"
        last_status = "unknown"

        def raise_timeout() -> None:
            raise TimeoutError(
                f"video job {job_id} did not finish within {int(self._poll_deadline_s)}s "
                f"(last status={last_status})"
            )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise_timeout()
            response = session.get(
                url,
                headers=self._headers(),
                timeout=max(0.001, min(self._request_timeout_s, remaining)),
            )
            response.raise_for_status()
            payload = response.json()
            last_status = str(payload.get("status") or "").lower() or "unknown"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise_timeout()
            if last_status in _TERMINAL_STATUSES:
                return payload
            time.sleep(min(self._poll_interval_s, remaining))

    def _save_completed_video(self, job_id: str) -> str:
        """Stream the authenticated content endpoint through the shared size cap.

        Do not follow a provider-supplied ``unsigned_urls`` destination: deriving
        the endpoint from the configured API origin keeps the bearer request on
        the same operator-selected host and avoids a second unbounded code path.
        """
        return str(
            save_url_video(
                f"{self._base_url()}/videos/{job_id}/content",
                prefix="openrouter-hailuo",
                headers=self._headers(),
                require_video_content_type=True,
            )
        )

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
        del negative_prompt, audio, seed, kwargs
        cleaned_prompt = (prompt or "").strip()
        if not cleaned_prompt:
            return error_response(
                error="prompt is required",
                error_type="invalid_request",
                provider=self.name,
            )
        model_id = (model or DEFAULT_MODEL).strip()
        if model_id != DEFAULT_MODEL:
            return error_response(
                error=f"OpenRouter video currently supports only {DEFAULT_MODEL}",
                error_type="invalid_model",
                provider=self.name,
                model=model_id,
                prompt=cleaned_prompt,
            )
        if reference_image_urls:
            return error_response(
                error=f"{DEFAULT_MODEL} does not support reference_image_urls",
                error_type="unsupported_input",
                provider=self.name,
                model=model_id,
                prompt=cleaned_prompt,
            )
        cleaned_image_url = (image_url or "").strip() or None
        if cleaned_image_url and not _is_public_first_frame_url(cleaned_image_url):
            return error_response(
                error="image_url must be a public HTTPS URL",
                error_type="invalid_request",
                provider=self.name,
                model=model_id,
                prompt=cleaned_prompt,
            )
        if not self._api_key():
            return error_response(
                error="OPENROUTER_API_KEY is not set",
                error_type="missing_credentials",
                provider=self.name,
                model=model_id,
                prompt=cleaned_prompt,
            )

        payload = _build_payload(
            prompt=cleaned_prompt,
            image_url=cleaned_image_url,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )
        session = self._session()
        try:
            submitted = session.post(
                f"{self._base_url()}/videos",
                headers=self._headers(),
                json=payload,
                timeout=self._request_timeout_s,
            )
            submitted.raise_for_status()
            submission = submitted.json()
            job_id = str(submission.get("id") or "").strip()
            if not job_id:
                raise ValueError("OpenRouter submit response did not contain a job id")
            job = self._poll(session, job_id)
            status = str(job.get("status") or "").lower()
            if status != "completed":
                return error_response(
                    error=str(job.get("error") or f"video job ended with status={status!r}"),
                    error_type="job_failed",
                    provider=self.name,
                    model=model_id,
                    prompt=cleaned_prompt,
                    aspect_ratio=payload["aspect_ratio"],
                )
            video_path = self._save_completed_video(job_id)
        except Exception as exc:  # noqa: BLE001 - normalize transport/API failures for tool callers
            logger.debug("OpenRouter H3 Max video generation failed", exc_info=True)
            return error_response(
                error=f"OpenRouter video generation failed: {exc}",
                error_type="api_error",
                provider=self.name,
                model=model_id,
                prompt=cleaned_prompt,
                aspect_ratio=payload["aspect_ratio"],
            )

        raw_usage = job.get("usage")
        usage: Dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        extra: Dict[str, Any] = {"job_id": job_id}
        if usage.get("cost") is not None:
            extra["cost"] = usage["cost"]
        return success_response(
            video=video_path,
            model=model_id,
            prompt=cleaned_prompt,
            modality="image" if cleaned_image_url else "text",
            aspect_ratio=payload["aspect_ratio"],
            duration=payload["duration"],
            provider=self.name,
            extra=extra,
        )


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_video_gen_provider(OpenRouterVideoGenProvider())
