"""xAI Grok-Imagine video generation backend.

Surface: text-to-video, image-to-video, and reference-to-video through the
unified video provider. xAI edit/extend are exposed through separate tools.

Originally salvaged from PR #10600 by @Jaaneek; reshaped into the
:class:`VideoGenProvider` plugin interface and trimmed to the
generate-only surface.

Authentication: xAI Grok OAuth tokens (preferred — billed against the
user's SuperGrok or X Premium+ subscription) or ``XAI_API_KEY``. Both routes are
resolved through ``tools.xai_http.resolve_xai_http_credentials`` so a
single login covers chat + TTS + image gen + video gen + transcription.
When xAI storage is enabled, the primary ``video`` / ``public_url`` fields are the
stored files-cdn HTTPS link. Pass that public MP4 URL as ``video_url`` for
edit/extend; it is sent to xAI as ``video.url``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_TEXT_TO_VIDEO_MODEL = "grok-imagine-video"
DEFAULT_IMAGE_TO_VIDEO_MODEL = "grok-imagine-video-1.5"
DEFAULT_MODEL = DEFAULT_TEXT_TO_VIDEO_MODEL
DEFAULT_DURATION = 8
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_RESOLUTION = "720p"
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_EXTEND_DURATION = 6

VALID_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"}
VALID_RESOLUTIONS = {"480p", "720p"}
MAX_REFERENCE_IMAGES = 7


_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-imagine-video": {
        "display": "Grok Imagine Video",
        "speed": "~60-240s",
        "strengths": "Text-to-video; legacy image-to-video fallback.",
        "price": "see https://docs.x.ai/developers/models/grok-imagine-video",
        "modalities": ["text", "image"],
    },
    "grok-imagine-video-1.5": {
        "display": "Grok Imagine Video 1.5",
        "speed": "~60-240s",
        "strengths": "Latest xAI image-to-video model.",
        "price": "see https://docs.x.ai/developers/pricing",
        "modalities": ["image"],
    },
}

_IMAGE_TO_VIDEO_COMPAT_MODEL_IDS = {
    "grok-imagine-video-1.5-preview",
    "grok-imagine-video-1.5-2026-05-30",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _resolve_xai_credentials() -> Tuple[str, str]:
    """Return ``(api_key, base_url)`` from the shared xAI credential resolver.

    Order: runtime provider (xai-oauth pool entry) → singleton ``auth.json``
    OAuth tokens → ``XAI_API_KEY`` env var. ``api_key`` is empty when no
    credential source is available; callers must check before using it.
    """
    try:
        from tools.xai_http import resolve_xai_http_credentials

        creds = resolve_xai_http_credentials() or {}
    except Exception as exc:
        logger.debug("xAI credential resolver failed: %s", exc)
        creds = {}

    api_key = str(creds.get("api_key") or os.getenv("XAI_API_KEY", "")).strip()
    base_url = str(
        creds.get("base_url")
        or os.getenv("XAI_BASE_URL")
        or DEFAULT_XAI_BASE_URL
    ).strip().rstrip("/")
    return api_key, base_url


def _xai_user_agent() -> str:
    try:
        from tools.xai_http import hermes_xai_user_agent

        return hermes_xai_user_agent()
    except Exception:
        return "hermes-agent/video_gen"


def _xai_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _xai_user_agent(),
    }


def _raise_if_blocked_local_input(ref: str) -> None:
    """Refuse to read a local media path that Hermes' read deny-list blocks.

    Thin wrapper over the shared ``agent.file_safety.raise_if_read_blocked``
    chokepoint so xAI video inputs enforce the same credential-store guard as
    the image providers. Fails open if the guard machinery is unavailable
    (defense-in-depth, per the denylist's own framing).
    """
    try:
        from agent.file_safety import raise_if_read_blocked
    except Exception as exc:  # noqa: BLE001 - guard must never break loading
        logger.debug("xAI media input read guard unavailable: %s", exc)
        return
    raise_if_read_blocked(ref)


def _image_ref_to_xai_url(value: str) -> str:
    """Return a URL/data URI accepted by xAI for image inputs."""
    ref = (value or "").strip()
    if not ref:
        return ""
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:image/")):
        return ref

    path = Path(ref).expanduser()
    if not path.is_file():
        return ref

    _raise_if_blocked_local_input(ref)

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        return ref

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_ref_to_xai_input(value: str) -> Optional[Dict[str, str]]:
    ref = _image_ref_to_xai_url(value)
    if not ref:
        return None
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:image/")):
        return {"url": ref}
    return None


def _xai_video_output_urls(
    video: Dict[str, Any],
) -> Tuple[str, Optional[str], Optional[str]]:
    """Return ``(public_video_url, temporary_url, stored_public_url)``.

    ``public_video_url`` is the stored files-cdn HTTPS MP4 (``public_url``) when
    storage is enabled; otherwise xAI's temporary ``video.url``. Pass this value
    as ``video_url`` for edit/extend chaining.
    """
    file_output = video.get("file_output") if isinstance(video.get("file_output"), dict) else {}
    file_output = file_output or {}
    stored_public = file_output.get("public_url")
    stored_public = stored_public.strip() if isinstance(stored_public, str) else None
    temporary = video.get("url")
    temporary = temporary.strip() if isinstance(temporary, str) else None
    public_video_url = stored_public or temporary or ""
    temporary_out = (
        temporary
        if temporary and stored_public and temporary != stored_public
        else None
    )
    return public_video_url, temporary_out, stored_public


def _video_ref_to_xai_url(value: str) -> str:
    """Return a URL/data URI accepted by xAI for video inputs."""
    ref = (value or "").strip()
    if not ref:
        return ""
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:video/")):
        return ref

    path = Path(ref).expanduser()
    if not path.is_file():
        return ref

    _raise_if_blocked_local_input(ref)

    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    if not mime.startswith("video/"):
        return ref

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def _video_input_from_public_url(
    value: str,
    *,
    api_key: str,
    base_url: str,
) -> Optional[Dict[str, str]]:
    """Build xAI ``video`` input using a public HTTPS URL (``url`` field only)."""
    ref = (value or "").strip()
    if not ref:
        return None

    path = Path(ref).expanduser()
    if path.is_file():
        data_ref = _video_ref_to_xai_url(ref)
        return {"url": data_ref} if data_ref else None

    lower = ref.lower()
    if not lower.startswith(("http://", "https://")):
        return None

    return {"url": ref}


def _normalize_reference_images(
    reference_image_urls: Optional[List[str]],
) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    refs: List[Dict[str, str]] = []
    for url in reference_image_urls or []:
        cleaned = (url or "").strip()
        if not cleaned:
            continue
        normalized = _image_ref_to_xai_input(cleaned)
        if not normalized:
            return None, (
                "reference_image_urls must be public HTTPS URLs or data URIs "
                "(e.g. the `image`/`public_url` from a prior Imagine result)"
            )
        refs.append(normalized)
    return (refs if refs else None), None


def _clamp_duration(
    duration: Optional[int],
    *,
    has_reference_images: bool = False,
    max_seconds: int = 15,
    default: int = DEFAULT_DURATION,
) -> int:
    value = duration if duration is not None else default
    if value < 1:
        value = 1
    if value > max_seconds:
        value = max_seconds
    if has_reference_images and value > 10:
        value = 10
    return value


def _resolve_model_for_modality(
    model: Optional[str],
    *,
    modality: str,
    explicit_model: bool,
) -> str:
    """Select xAI's text/video model without treating config as a prompt override.

    ``grok-imagine-video-1.5`` currently rejects text-only video
    generation, but it is the desired image-to-video backend. Explicit tool
    ``model=`` still wins for users who intentionally request another model.
    """
    requested = (model or "").strip()
    if explicit_model and requested:
        return requested
    if modality == "image":
        return DEFAULT_IMAGE_TO_VIDEO_MODEL
    if requested == DEFAULT_IMAGE_TO_VIDEO_MODEL or requested in _IMAGE_TO_VIDEO_COMPAT_MODEL_IDS:
        return DEFAULT_TEXT_TO_VIDEO_MODEL
    return requested or DEFAULT_TEXT_TO_VIDEO_MODEL


async def _submit(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    endpoint: str = "generations",
) -> str:
    """POST to one of xAI's async video endpoints and return request_id."""
    response = await client.post(
        f"{base_url}/videos/{endpoint}",
        headers={**_xai_headers(api_key), "x-idempotency-key": str(uuid.uuid4())},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    request_id = body.get("request_id")
    if not request_id:
        raise RuntimeError("xAI video response did not include request_id")
    return request_id


async def _poll(
    client: httpx.AsyncClient,
    request_id: str,
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
    poll_interval: int,
) -> Dict[str, Any]:
    elapsed = 0.0
    last_status = "queued"
    while elapsed < timeout_seconds:
        response = await client.get(
            f"{base_url}/videos/{request_id}",
            headers=_xai_headers(api_key),
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        last_status = (body.get("status") or "").lower()

        if last_status == "done":
            return {"status": "done", "body": body}
        if last_status in {"failed", "error", "expired", "cancelled"}:
            return {"status": last_status, "body": body}

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    return {"status": "timeout", "body": {"status": last_status}}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIVideoGenProvider(VideoGenProvider):
    """xAI Grok Imagine video backend."""

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI"

    def is_available(self) -> bool:
        api_key, _ = _resolve_xai_credentials()
        return bool(api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": mid, **meta} for mid, meta in _MODELS.items()]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution lives entirely in the shared ``xai_grok`` post_setup
        # hook (``hermes_cli/tools_config.py``) so the picker doesn't blindly
        # prompt for an API key when the user is already signed in via xAI
        # Grok OAuth (SuperGrok / Premium+) — TTS / image gen / video gen
        # all share the same credential resolver. The hook offers an
        # OAuth-vs-API-key choice when neither is configured.
        try:
            from tools.xai_http import xai_storage_notice_text

            storage_notice = xai_storage_notice_text("video_gen")
        except Exception:
            storage_notice = ""
        tag = (
            "grok-imagine-video for text/reference; "
            "grok-imagine-video-1.5 for image-to-video; "
            "edit/extend: pass the stored public HTTPS MP4 (`video` / "
            "`public_url` from a prior Imagine result); uses xAI Grok OAuth "
            "or XAI_API_KEY"
        )
        if storage_notice:
            tag += f". {storage_notice}"
        return {
            "name": "xAI Grok Imagine",
            "badge": "paid",
            "tag": tag,
            "env_vars": [],
            "post_setup": "xai_grok",
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(VALID_ASPECT_RATIOS),
            "resolutions": sorted(VALID_RESOLUTIONS),
            "max_duration": 15,
            "min_duration": 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": MAX_REFERENCE_IMAGES,
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
        return run_xai_video_generation(
            prompt=prompt,
            model=model,
            explicit_model=bool(kwargs.get("_model_override_explicit")),
            image_url=image_url,
            reference_image_urls=reference_image_urls,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )


def has_xai_video_credentials() -> bool:
    api_key, _ = _resolve_xai_credentials()
    return bool(api_key)


def run_xai_video_generation(
    *,
    prompt: str,
    model: Optional[str],
    explicit_model: bool,
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
    duration: Optional[int],
    aspect_ratio: str,
    resolution: str,
) -> Dict[str, Any]:
    return _run_xai_video_coroutine(
        _generate_xai_video_async(
            prompt=prompt,
            model=model,
            explicit_model=explicit_model,
            image_url=image_url,
            reference_image_urls=reference_image_urls,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        ),
        operation_label="generation",
        model=model,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
    )


def run_xai_video_edit(
    *,
    prompt: str,
    video_url: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    return _run_xai_video_coroutine(
        _edit_xai_video_async(prompt=prompt, video_url=video_url, model=model),
        operation_label="edit",
        model=model,
        prompt=prompt,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
    )


def run_xai_video_extend(
    *,
    prompt: str,
    video_url: str,
    duration: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    return _run_xai_video_coroutine(
        _extend_xai_video_async(
            prompt=prompt,
            video_url=video_url,
            duration=duration,
            model=model,
        ),
        operation_label="extend",
        model=model,
        prompt=prompt,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
    )


def _run_xai_video_coroutine(
    coro,
    *,
    operation_label: str,
    model: Optional[str],
    prompt: str,
    aspect_ratio: str,
) -> Dict[str, Any]:
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("xAI video %s unexpected failure: %s", operation_label, exc, exc_info=True)
        return error_response(
            error=f"xAI video {operation_label} failed: {exc}",
            error_type="api_error",
            provider="xai",
            model=model or DEFAULT_MODEL,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
        )


async def _generate_xai_video_async(
    *,
    prompt: str,
    model: Optional[str],
    explicit_model: bool,
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
    duration: Optional[int],
    aspect_ratio: str,
    resolution: str,
) -> Dict[str, Any]:
    api_key, base_url = _resolve_xai_credentials()
    if not api_key:
        return _auth_required_response(prompt)

    prompt = (prompt or "").strip()
    image_input = None
    if (image_url or "").strip():
        image_input = _image_ref_to_xai_input(image_url)
        if not image_input:
            return error_response(
                error=(
                    "image_url must be a public HTTPS URL or data URI "
                    "(e.g. the `image`/`public_url` from a prior Imagine result)"
                ),
                error_type="invalid_image_url",
                provider="xai",
                prompt=prompt,
            )
    normalized_aspect_ratio = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip()
    normalized_resolution = (resolution or DEFAULT_RESOLUTION).strip().lower()
    refs, refs_error = _normalize_reference_images(reference_image_urls)
    if refs_error:
        return error_response(
            error=refs_error,
            error_type="invalid_reference_image_urls",
            provider="xai",
            prompt=prompt,
        )

    if not prompt:
        return error_response(
            error="prompt is required for xAI video generation",
            error_type="missing_prompt",
            provider="xai", prompt=prompt,
        )
    if refs and len(refs) > MAX_REFERENCE_IMAGES:
        return error_response(
            error=f"reference_image_urls supports at most {MAX_REFERENCE_IMAGES} images on xAI",
            error_type="too_many_references",
            provider="xai", prompt=prompt,
        )
    if image_input and refs:
        return error_response(
            error="image_url and reference_image_urls cannot be combined on xAI",
            error_type="conflicting_inputs",
            provider="xai", prompt=prompt,
        )

    if normalized_aspect_ratio not in VALID_ASPECT_RATIOS:
        normalized_aspect_ratio = DEFAULT_ASPECT_RATIO
    if normalized_resolution not in VALID_RESOLUTIONS:
        normalized_resolution = DEFAULT_RESOLUTION

    modality_used = "reference" if refs else ("image" if image_input else "text")
    resolved_model = _resolve_model_for_modality(
        model,
        modality=modality_used,
        explicit_model=explicit_model,
    )
    if refs and resolved_model != DEFAULT_TEXT_TO_VIDEO_MODEL:
        if explicit_model:
            return error_response(
                error=(
                    "xAI reference-to-video requires "
                    f"{DEFAULT_TEXT_TO_VIDEO_MODEL}; got {resolved_model}"
                ),
                error_type="unsupported_model",
                provider="xai",
                model=resolved_model,
                prompt=prompt,
            )
        resolved_model = DEFAULT_TEXT_TO_VIDEO_MODEL

    clamped_duration = _clamp_duration(duration, has_reference_images=bool(refs))
    payload = {
        "model": resolved_model,
        "prompt": prompt,
        "duration": clamped_duration,
        "aspect_ratio": normalized_aspect_ratio,
        "resolution": normalized_resolution,
    }
    if image_input:
        payload["image"] = image_input
    if refs:
        payload["reference_images"] = refs

    return await _submit_xai_video_payload(
        api_key=api_key,
        base_url=base_url,
        endpoint="generations",
        payload=payload,
        prompt=prompt,
        resolved_model=resolved_model,
        modality=modality_used,
        aspect_ratio=normalized_aspect_ratio,
        duration=clamped_duration,
        operation="generate",
        resolution=normalized_resolution,
    )


async def _run_xai_video_mutation(
    *,
    prompt: str,
    video_url: str,
    model: Optional[str],
    endpoint: str,
    operation: str,
    duration: int,
) -> Dict[str, Any]:
    """Edit or extend using a public HTTPS ``video_url`` input (``url`` on the wire)."""
    api_key, base_url = _resolve_xai_credentials()
    if not api_key:
        return _auth_required_response(prompt)

    prompt = (prompt or "").strip()
    video_input = await _video_input_from_public_url(
        video_url or "",
        api_key=api_key,
        base_url=base_url,
    )
    if not prompt:
        return error_response(
            error="prompt is required for xAI video edit/extend",
            error_type="missing_prompt",
            provider="xai",
            prompt=prompt,
        )
    if not video_input:
        return error_response(
            error=(
                "video_url must be a public HTTPS MP4 URL "
                "(the `video`/`public_url` from a prior Imagine result)"
            ),
            error_type="missing_video",
            provider="xai",
            prompt=prompt,
        )

    resolved_model = _resolve_model_for_modality(
        model,
        modality="text",
        explicit_model=bool(model),
    )
    payload: Dict[str, Any] = {
        "model": resolved_model,
        "prompt": prompt,
        "video": video_input,
    }
    if endpoint == "extensions":
        payload["duration"] = duration

    return await _submit_xai_video_payload(
        api_key=api_key,
        base_url=base_url,
        endpoint=endpoint,
        payload=payload,
        prompt=prompt,
        resolved_model=resolved_model,
        modality=operation,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        duration=duration,
        operation=operation,
    )


async def _edit_xai_video_async(
    *,
    prompt: str,
    video_url: str,
    model: Optional[str],
) -> Dict[str, Any]:
    return await _run_xai_video_mutation(
        prompt=prompt,
        video_url=video_url,
        model=model,
        endpoint="edits",
        operation="edit",
        duration=DEFAULT_DURATION,
    )


async def _extend_xai_video_async(
    *,
    prompt: str,
    video_url: str,
    duration: Optional[int],
    model: Optional[str],
) -> Dict[str, Any]:
    clamped_duration = _clamp_duration(
        duration,
        max_seconds=10,
        default=DEFAULT_EXTEND_DURATION,
    )
    return await _run_xai_video_mutation(
        prompt=prompt,
        video_url=video_url,
        model=model,
        endpoint="extensions",
        operation="extend",
        duration=clamped_duration,
    )


def _auth_required_response(prompt: str) -> Dict[str, Any]:
    return error_response(
        error=(
            "No xAI credentials found. Sign in via `hermes auth add xai-oauth` "
            "(SuperGrok / Premium+) or set XAI_API_KEY from "
            "https://console.x.ai/."
        ),
        error_type="auth_required",
        provider="xai", prompt=prompt,
    )


async def _submit_xai_video_payload(
    *,
    api_key: str,
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    prompt: str,
    resolved_model: str,
    modality: str,
    aspect_ratio: str,
    duration: int,
    operation: str,
    resolution: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        from tools.xai_http import (
            build_xai_storage_options,
            maybe_mark_xai_storage_notice_seen,
            read_xai_imagine_storage_config,
        )

        storage_options = build_xai_storage_options(
            "video_gen",
            filename_prefix="hermes-xai-video",
            extension="mp4",
        )
        storage_notice = maybe_mark_xai_storage_notice_seen("video_gen")
        storage_cfg = read_xai_imagine_storage_config("video_gen")
    except Exception:
        storage_options = None
        storage_notice = None
        storage_cfg = {"enabled": False}
    if storage_options is not None:
        payload["storage_options"] = storage_options

    async with httpx.AsyncClient() as client:
        try:
            request_id = await _submit(
                client, payload, api_key=api_key, base_url=base_url,
                endpoint=endpoint,
            )
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:
                logger.debug("Suppressed exception", exc_info=True)
            return error_response(
                error=f"xAI submit failed ({exc.response.status_code}): {detail or exc}",
                error_type="api_error",
                provider="xai",
                model=resolved_model,
                prompt=prompt,
            )

        poll_result = await _poll(
            client, request_id,
            api_key=api_key, base_url=base_url,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            poll_interval=DEFAULT_POLL_INTERVAL_SECONDS,
        )

    status = poll_result["status"]
    body = poll_result["body"]

    if status == "done":
        video = body.get("video") or {}
        if not isinstance(video, dict):
            video = {}
        file_output = video.get("file_output") if isinstance(video.get("file_output"), dict) else {}
        file_output = file_output or {}
        public_video_url, temporary_url, stored_public_url = _xai_video_output_urls(video)
        if not public_video_url:
            return error_response(
                error="xAI video request completed without a video URL",
                error_type="empty_response",
                provider="xai",
                model=body.get("model") or resolved_model,
                prompt=prompt,
            )
        extra: Dict[str, Any] = {
            "request_id": request_id,
            "operation": operation,
            "storage_enabled": bool(storage_cfg.get("enabled")),
        }
        if resolution:
            extra["resolution"] = resolution
        if storage_notice:
            extra["storage_notice"] = storage_notice
        if stored_public_url:
            extra["public_url"] = stored_public_url
        if temporary_url:
            extra["temporary_url"] = temporary_url
        if file_output:
            for key in (
                "filename",
                "expires_at",
                "public_url_expires_at",
                "public_url_error",
                "storage_error",
            ):
                if key in file_output:
                    extra[key] = file_output[key]
        if body.get("usage"):
            extra["usage"] = body["usage"]
        return success_response(
            video=public_video_url,
            model=body.get("model") or resolved_model,
            prompt=prompt,
            modality=modality,
            aspect_ratio=aspect_ratio,
            duration=video.get("duration") or duration,
            provider="xai",
            extra=extra,
        )

    if status == "timeout":
        return error_response(
            error=f"Timed out waiting for xAI video request after {DEFAULT_TIMEOUT_SECONDS}s",
            error_type="timeout",
            provider="xai",
            model=resolved_model,
            prompt=prompt,
        )

    message = (
        (body.get("error", {}) or {}).get("message")
        or body.get("message")
        or f"xAI video request ended with status '{status}'"
    )
    return error_response(
        error=message,
        error_type=f"xai_{status}",
        provider="xai",
        model=resolved_model,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``XAIVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(XAIVideoGenProvider())
