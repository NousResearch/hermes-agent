"""KIE (kie.ai) video generation backend for EasyHermes.

Wraps kie.ai's ByteDance Seedance 2.0 (``bytedance/seedance-2``) as a
:class:`VideoGenProvider`. This is EasyHermes' native video backend —
install-and-use with a single backend ``KIE_API_KEY``, replacing the FAL
pipeline.

Unified routing (matching the ``video_generate`` tool surface):
    - no ``image_url``            -> text-to-video
    - ``image_url``               -> image-to-video (used as Seedance *first frame*)
    - ``image_url`` + last frame  -> first+last-frame interpolation. The last
      frame is taken from ``last_frame_url`` (kwarg) or ``reference_image_urls[0]``.

All kie.ai HTTP lives in :mod:`tools.kie_common`.

Selection: set ``video_gen.provider: kie`` in ``config.yaml`` (EasyHermes ships
this default).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Seedance 2.0 capability sheet (mirrors kari_seedance_video.py).
_KIE_MODEL = "bytedance/seedance-2"
_FAMILY_ID = "seedance-2"
_SEEDANCE_ASPECT_RATIOS = ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive")
_SEEDANCE_RESOLUTIONS = ("480p", "720p", "1080p")
_MIN_DURATION = 4
_MAX_DURATION = 15
_DEFAULT_DURATION = 5


def _clamp_duration(value: Optional[int]) -> int:
    try:
        seconds = int(value) if value is not None else _DEFAULT_DURATION
    except (TypeError, ValueError):
        seconds = _DEFAULT_DURATION
    return max(_MIN_DURATION, min(_MAX_DURATION, seconds))


class KieVideoGenProvider(VideoGenProvider):
    """kie.ai video generation backend (Seedance 2.0)."""

    @property
    def name(self) -> str:
        return "kie"

    @property
    def display_name(self) -> str:
        return "KIE (kie.ai)"

    def is_available(self) -> bool:
        from tools.kie_common import kie_key_is_configured

        return kie_key_is_configured()

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": _FAMILY_ID,
                "display": "Seedance 2.0",
                "speed": "~60-120s",
                "strengths": "ByteDance. Text/image/first+last-frame, synced audio, 4-15s.",
                "price": "premium",
                "modalities": ["text", "image"],
            }
        ]

    def default_model(self) -> Optional[str]:
        return _FAMILY_ID

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "KIE (kie.ai)",
            "badge": "paid",
            "tag": "EasyHermes native video gen — Seedance 2.0 via kie.ai",
            "env_vars": [
                {
                    "key": "KIE_API_KEY",
                    "prompt": "KIE (kie.ai) API key",
                    "url": "https://kie.ai",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": list(_SEEDANCE_ASPECT_RATIOS),
            "resolutions": list(_SEEDANCE_RESOLUTIONS),
            "max_duration": _MAX_DURATION,
            "min_duration": _MIN_DURATION,
            "supports_audio": True,
            "supports_negative_prompt": False,
            "max_reference_images": 1,
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
        from tools import kie_billing as kb
        from tools.kie_common import KieError, as_url_list, run

        prompt = (prompt or "").strip()
        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider="kie",
                model=_FAMILY_ID,
            )

        first_frame = (image_url or "").strip() or None
        # Last frame (for first+last interpolation): explicit kwarg, else the
        # first reference image. Only meaningful when a first frame is present.
        last_frame = (kwargs.get("last_frame_url") or "").strip() or None
        if not last_frame:
            refs = as_url_list(reference_image_urls)
            last_frame = refs[0] if refs else None
        if last_frame and not first_frame:
            return error_response(
                error="A last frame requires a first frame (pass image_url too).",
                error_type="invalid_input",
                provider="kie",
                model=_FAMILY_ID,
                prompt=prompt,
            )

        seconds = _clamp_duration(duration)
        kie_aspect = aspect_ratio if aspect_ratio in _SEEDANCE_ASPECT_RATIOS else "16:9"
        kie_resolution = resolution if resolution in _SEEDANCE_RESOLUTIONS else "720p"

        # Billing pre-check (opt-in). Seedance is priced per second by resolution.
        cost = kb.credits_for_video("seedance", kie_resolution, seconds)
        billing_on = kb.billing_enabled()
        if billing_on:
            try:
                kb.ensure_balance(cost)
            except kb.InsufficientCreditsError as exc:
                return error_response(
                    error=str(exc),
                    error_type="insufficient_credits",
                    provider="kie",
                    model=_FAMILY_ID,
                    prompt=prompt,
                    aspect_ratio=kie_aspect,
                )

        kie_input: Dict[str, Any] = {
            "prompt": prompt,
            "resolution": kie_resolution,
            "aspect_ratio": kie_aspect,
            "duration": seconds,
            "generate_audio": True if audio is None else bool(audio),
        }
        if first_frame:
            kie_input["first_frame_url"] = first_frame
        if last_frame:
            kie_input["last_frame_url"] = last_frame

        modality = "image" if first_frame else "text"
        logger.info(
            "kie video generate model=%s mode=%s dur=%ds",
            _KIE_MODEL,
            "first+last" if last_frame else modality,
            seconds,
        )
        try:
            urls = run(_KIE_MODEL, kie_input, max_wait=600, interval=4.0)
        except KieError as exc:
            return error_response(
                error=str(exc),
                error_type="provider_error",
                provider="kie",
                model=_FAMILY_ID,
                prompt=prompt,
                aspect_ratio=kie_aspect,
            )
        except Exception as exc:  # noqa: BLE001 — never raise out of generate()
            logger.warning("kie video generate raised: %s", exc, exc_info=True)
            return error_response(
                error=f"KIE video generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="kie",
                model=_FAMILY_ID,
                prompt=prompt,
                aspect_ratio=kie_aspect,
            )

        if not urls:
            return error_response(
                error="KIE returned no result URLs",
                error_type="empty_result",
                provider="kie",
                model=_FAMILY_ID,
                prompt=prompt,
                aspect_ratio=kie_aspect,
            )

        extra: Dict[str, Any] = {}
        if len(urls) > 1:
            extra["all_videos"] = urls
        if billing_on:
            try:
                balance = kb.charge(cost, "video", f"KIE Seedance {kie_resolution} {seconds}s")
                extra["credits_charged"] = cost
                extra["balance"] = round(balance, 2)
            except Exception as exc:  # noqa: BLE001 — generation already succeeded
                logger.warning("kie video charge failed (video kept): %s", exc)
        return success_response(
            video=urls[0],
            model=_FAMILY_ID,
            prompt=prompt,
            modality=modality,
            aspect_ratio=kie_aspect,
            duration=seconds,
            provider="kie",
            extra=extra or None,
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``KieVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(KieVideoGenProvider())
