"""inference.sh video generation backend.

User-facing surface: pick a **model family** (e.g. "Veo 3.1",
"Seedance 2.0", "Wan 2.5", "HappyHorse 1.0", "Grok Video", "P-Video").
The plugin auto-routes to the family's text-to-video app when called
without ``image_url``, and to its image-to-video app when ``image_url``
is provided.

Model families:

  Cheap tier:
    p-video       p-video (text-to-video only, fastest + cheapest)

  Premium tier:
    veo-3.1       veo/3.1 (text-to-video + image-to-video)
    seedance-2.0  seedance/2.0 (text-to-video + image-to-video, native audio)
    wan-2.5       wan/2.5-t2v / wan/2.5-i2v (text + image)
    happyhorse    happyhorse/1.0 (text-to-video + image-to-video)
    grok-video    grok-video (text-to-video only)

Selection precedence for the active family:
    1. ``model=`` arg from the tool call
    2. ``INFERENCE_VIDEO_MODEL`` env var
    3. ``video_gen.inference_sh.model`` in ``config.yaml``
    4. ``video_gen.model`` in ``config.yaml`` (when it's one of our family IDs)
    5. ``DEFAULT_MODEL``

Authentication via ``INFERENCE_API_KEY``. Output is an HTTPS URL;
the gateway downloads and delivers it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "inference-sh"


# ---------------------------------------------------------------------------
# Family catalog
# ---------------------------------------------------------------------------
#
# Each family declares both app IDs (when available) plus a per-family
# capability sheet. Capability flags drive which keys get added to the
# request payload — keys a family doesn't advertise are dropped.
#
# Capabilities:
#   aspect_ratios  : tuple of supported ratios (None = app decides)
#   resolutions    : tuple of supported resolutions (None = app decides)
#   durations      : tuple of supported durations OR (min, max) range
#                    (heuristic: 2-element with gap > 1 is a range)
#   audio          : True if generate_audio is supported
#   negative       : True if negative_prompt is supported

FAMILIES: Dict[str, Dict[str, Any]] = {
    # --- Cheap / fast tier -------------------------------------------------
    "pruna/p-video": {
        "display": "P-Video",
        "speed": "~15-30s",
        "price": "cheap",
        "strengths": "Fastest and cheapest video generation. Text-to-video and image-to-video.",
        "tier": "cheap",
        "text_app": "pruna/p-video",
        "image_app": "pruna/p-video",
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": False,
        "negative": False,
    },
    # --- Premium tier ------------------------------------------------------
    "google/veo-3-1": {
        "display": "Veo 3.1",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Google DeepMind. Cinematic quality, native audio, strong prompt adherence.",
        "tier": "premium",
        "text_app": "google/veo-3-1",
        "image_app": "google/veo-3-1",
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": ("720p", "1080p"),
        "durations": (4, 6, 8),
        "audio": True,
        "negative": True,
    },
    "bytedance/seedance-2-0": {
        "display": "Seedance 2.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "ByteDance. Cinematic quality, synchronized audio + lip-sync, 4-15s.",
        "tier": "premium",
        "text_app": "bytedance/seedance-2-0",
        "image_app": "bytedance/seedance-2-0",
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (4, 15),
        "audio": True,
        "negative": False,
    },
    "alibaba/wan-2-7-t2v": {
        "display": "Wan 2.7",
        "speed": "~60-90s",
        "price": "premium",
        "strengths": "Alibaba. Strong image-to-video animation, consistent motion.",
        "tier": "premium",
        "text_app": "alibaba/wan-2-7-t2v",
        "image_app": "alibaba/wan-2-7-i2v",
        "aspect_ratios": ("16:9", "9:16", "1:1"),
        "resolutions": ("480p", "720p"),
        "durations": (3, 10),
        "audio": False,
        "negative": True,
    },
    "alibaba/happyhorse-1-0-t2v": {
        "display": "HappyHorse 1.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Alibaba. Physically realistic motion, strong dynamics.",
        "tier": "premium",
        "text_app": "alibaba/happyhorse-1-0-t2v",
        "image_app": "alibaba/happyhorse-1-0-i2v",
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": False,
        "negative": False,
    },
    "xai/grok-imagine-video": {
        "display": "Grok Video",
        "speed": "~30-60s",
        "price": "premium",
        "strengths": "xAI. Fast creative video generation from text prompts.",
        "tier": "premium",
        "text_app": "xai/grok-imagine-video",
        "image_app": None,
        "aspect_ratios": ("16:9", "9:16", "1:1"),
        "resolutions": None,
        "durations": None,
        "audio": False,
        "negative": False,
    },
}

DEFAULT_MODEL = "bytedance/seedance-2-0"


def _is_duration_range(durations: Any) -> bool:
    """Heuristic: a 2-tuple of ints with a gap > 1 is treated as ``(min, max)``."""
    if not isinstance(durations, tuple) or len(durations) != 2:
        return False
    if not all(isinstance(d, int) for d in durations):
        return False
    return durations[1] - durations[0] > 1


def _clamp_duration(family: Dict[str, Any], duration: Optional[int]) -> Optional[int]:
    durations = family.get("durations")
    if not durations:
        return duration
    if duration is None:
        return durations[0]
    if _is_duration_range(durations):
        lo, hi = durations
        return max(lo, min(hi, duration))
    # enum
    if duration in durations:
        return duration
    return min(durations, key=lambda d: abs(d - duration))


# ---------------------------------------------------------------------------
# Config / model resolution
# ---------------------------------------------------------------------------


def _load_video_gen_section() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen config: %s", exc)
        return {}


def _resolve_family(explicit: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """Decide which model family to use. Returns ``(family_id, meta)``."""
    candidates: List[Optional[str]] = []
    candidates.append(explicit)
    candidates.append(os.environ.get("INFERENCE_VIDEO_MODEL"))

    cfg = _load_video_gen_section()
    infsh_cfg = cfg.get("inference_sh") if isinstance(cfg.get("inference_sh"), dict) else {}
    if isinstance(infsh_cfg, dict):
        candidates.append(infsh_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for c in candidates:
        if isinstance(c, str) and c.strip() and c.strip() in FAMILIES:
            fid = c.strip()
            return fid, FAMILIES[fid]

    return DEFAULT_MODEL, FAMILIES[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# SDK lazy import
# ---------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    """Return a cached ``inferencesh.inference`` client instance."""
    global _client
    if _client is not None:
        return _client
    from inferencesh import inference  # type: ignore

    api_key = os.environ.get("INFERENCE_API_KEY", "").strip()
    _client = inference(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class InferenceShVideoGenProvider(VideoGenProvider):
    """inference.sh multi-family video generation backend.

    Routes between text-to-video and image-to-video apps automatically
    based on whether ``image_url`` was provided. Uses the ``inferencesh``
    Python SDK to call apps on the inference.sh cloud.
    """

    @property
    def name(self) -> str:
        return "inference-sh"

    @property
    def display_name(self) -> str:
        return "inference.sh"

    def is_available(self) -> bool:
        # Only check for the API key — the inferencesh SDK is lazy-installed
        # on first generate() call via tools/lazy_deps.py.
        return bool(os.environ.get("INFERENCE_API_KEY", "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fid, meta in FAMILIES.items():
            modalities: List[str] = []
            if meta.get("text_app"):
                modalities.append("text")
            if meta.get("image_app"):
                modalities.append("image")
            out.append({
                "id": fid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
                "tier": meta.get("tier", "premium"),
                "modalities": modalities,
            })
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "inference.sh",
            "badge": "paid",
            "tag": (
                "one key for any model — Veo 3.1, Seedance 2.0, Wan 2.7, "
                "HappyHorse, Grok Video, P-Video (inference.sh)"
            ),
            "env_vars": [
                {
                    "key": "INFERENCE_API_KEY",
                    "prompt": "inference.sh API key",
                    "url": "https://app.inference.sh/settings/keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
            "resolutions": ["480p", "720p", "1080p"],
            "max_duration": 15,
            "min_duration": 3,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not os.environ.get("INFERENCE_API_KEY", "").strip():
            return error_response(
                error=(
                    "INFERENCE_API_KEY not set. Run `hermes tools` -> Video "
                    "Generation -> inference.sh to configure, or sign up at "
                    "https://inference.sh"
                ),
                error_type="auth_required",
                provider=PROVIDER_NAME,
                prompt=prompt,
            )

        try:
            from tools.lazy_deps import ensure
            ensure("video.inference_sh", prompt=False)
        except Exception:
            pass  # Best effort — SDK may already be installed

        try:
            client = _get_client()
        except ImportError:
            return error_response(
                error=(
                    "inferencesh Python package not installed "
                    "(pip install inferencesh)"
                ),
                error_type="missing_dependency",
                provider=PROVIDER_NAME,
                prompt=prompt,
            )

        prompt = (prompt or "").strip()
        family_id, family = _resolve_family(model)

        # Route: image_url -> image-to-video app; else -> text-to-video.
        image_url_norm = (image_url or "").strip() or None
        if image_url_norm:
            app_id = family.get("image_app")
            modality_used = "image"
            if not app_id:
                return error_response(
                    error=(
                        f"Model {family_id} has no image-to-video support. "
                        f"Pick a family with image-to-video support "
                        f"via `hermes tools` -> Video Generation."
                    ),
                    error_type="modality_unsupported",
                    provider=PROVIDER_NAME, model=family_id, prompt=prompt,
                )
        else:
            app_id = family.get("text_app")
            modality_used = "text"
            if not app_id:
                return error_response(
                    error=(
                        f"Model {family_id} has no text-to-video support. "
                        f"Pass an image_url to use image-to-video, or pick "
                        f"a different model."
                    ),
                    error_type="modality_unsupported",
                    provider=PROVIDER_NAME, model=family_id, prompt=prompt,
                )

        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider=PROVIDER_NAME, model=family_id, prompt=prompt,
            )

        # Build input payload — only include keys the family declares
        input_data: Dict[str, Any] = {"prompt": prompt}

        if image_url_norm:
            input_data["image"] = image_url_norm

        if seed is not None:
            input_data["seed"] = seed

        if family.get("aspect_ratios"):
            if aspect_ratio in family["aspect_ratios"]:
                input_data["aspect_ratio"] = aspect_ratio

        if family.get("resolutions"):
            if resolution in family["resolutions"]:
                input_data["resolution"] = resolution

        clamped = _clamp_duration(family, duration)
        if clamped is not None and family.get("durations"):
            input_data["duration"] = clamped

        if family.get("audio") and audio is not None:
            input_data["generate_audio"] = bool(audio)
        elif family.get("audio") and audio is None:
            # Default to audio on for families that support it
            input_data["generate_audio"] = True

        if family.get("negative") and negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        try:
            result = client.tasks.run({
                "app": app_id,
                "input": input_data,
            })
        except Exception as exc:
            logger.warning(
                "inference.sh video gen failed (family=%s, app=%s): %s",
                family_id, app_id, exc, exc_info=True,
            )
            return error_response(
                error=f"inference.sh video generation failed: {exc}",
                error_type="api_error",
                provider=PROVIDER_NAME, model=family_id, prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        # Extract video URL from task output
        output = result.get("output") if isinstance(result, dict) else None
        if not isinstance(output, dict):
            output = result if isinstance(result, dict) else {}

        # inference.sh apps return output URLs in various shapes
        url: Optional[str] = None
        for key in ("url", "video_url", "video", "output_url"):
            val = output.get(key)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break
            if isinstance(val, dict):
                url = val.get("url")
                if url:
                    break
        # Check for list-of-files pattern
        if not url:
            files = output.get("files") or output.get("outputs")
            if isinstance(files, list) and files:
                first = files[0]
                if isinstance(first, str) and first.startswith("http"):
                    url = first
                elif isinstance(first, dict):
                    url = first.get("url")

        if not url:
            return error_response(
                error="inference.sh returned no video URL in response",
                error_type="empty_response",
                provider=PROVIDER_NAME, model=family_id, prompt=prompt,
            )

        extra: Dict[str, Any] = {"app": app_id}
        task_id = result.get("id") if isinstance(result, dict) else None
        if task_id:
            extra["task_id"] = task_id

        return success_response(
            video=url,
            model=family_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio if "aspect_ratio" in input_data else "",
            duration=int(input_data["duration"]) if "duration" in input_data else 0,
            provider=PROVIDER_NAME,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``InferenceShVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(InferenceShVideoGenProvider())
