"""KIE (kie.ai) image generation backend for EasyHermes.

Wraps kie.ai's ``gpt-image-2`` family as an :class:`ImageGenProvider`. This is
EasyHermes' native image backend — install-and-use with a single backend
``KIE_API_KEY``, replacing the FAL pipeline.

All kie.ai HTTP lives in :mod:`tools.kie_common` (the single source of truth
shared with the video provider and the website/slides/web-video tools).

Selection: set ``image_gen.provider: kie`` in ``config.yaml`` (EasyHermes ships
this default). The agent calls ``image_generate(prompt, aspect_ratio)``; this
provider maps the unified aspect ratio to kie's enum and runs a text-to-image
job. Image-to-image (reference images) is available when a caller passes
``image_urls`` via kwargs — the standard tool schema is text-to-image only for
now.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

# Unified (landscape/square/portrait) -> kie.ai aspect_ratio enum.
_ASPECT_TO_KIE = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

_TEXT_MODEL = "gpt-image-2-text-to-image"
_IMAGE_MODEL = "gpt-image-2-image-to-image"
_DEFAULT_RESOLUTION = "2K"


class KieImageGenProvider(ImageGenProvider):
    """kie.ai image generation backend (gpt-image-2)."""

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
                "id": _TEXT_MODEL,
                "display": "GPT Image 2 (text-to-image)",
                "strengths": "Text-to-image, 1K/2K/4K",
            },
            {
                "id": _IMAGE_MODEL,
                "display": "GPT Image 2 (image-to-image)",
                "strengths": "Edit / image-to-image with up to 16 reference images",
            },
        ]

    def default_model(self) -> Optional[str]:
        return _TEXT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "KIE (kie.ai)",
            "badge": "paid",
            "tag": "EasyHermes native image gen — gpt-image-2 via kie.ai",
            "env_vars": [
                {
                    "key": "KIE_API_KEY",
                    "prompt": "KIE (kie.ai) API key",
                    "url": "https://kie.ai",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from tools import kie_billing as kb
        from tools.kie_common import KieError, as_url_list, run

        unified_aspect = resolve_aspect_ratio(aspect_ratio)
        resolution = str(kwargs.get("resolution") or _DEFAULT_RESOLUTION)
        refs = as_url_list(kwargs.get("image_urls"))[:16]
        kind = "image_to_image" if refs else "text_to_image"

        kie_input: Dict[str, Any] = {"prompt": prompt, "resolution": resolution}
        kie_ratio = _ASPECT_TO_KIE.get(unified_aspect)
        if kie_ratio:
            kie_input["aspect_ratio"] = kie_ratio

        if refs:
            model = _IMAGE_MODEL
            kie_input["input_urls"] = refs
        else:
            model = _TEXT_MODEL

        # Allow an explicit model override from image_gen.model config.
        configured_model = kwargs.get("model")
        if isinstance(configured_model, str) and configured_model.strip():
            model = configured_model.strip()

        # Billing pre-check (opt-in). Reserve nothing on disk when disabled.
        cost = kb.credits_for_image(resolution, kind)
        billing_on = kb.billing_enabled()
        if billing_on:
            try:
                kb.ensure_balance(cost)
            except kb.InsufficientCreditsError as exc:
                return error_response(
                    error=str(exc),
                    error_type="insufficient_credits",
                    provider="kie",
                    model=model,
                    prompt=prompt,
                    aspect_ratio=unified_aspect,
                )

        logger.info("kie image generate model=%s refs=%d", model, len(refs))
        try:
            urls = run(model, kie_input)
        except KieError as exc:
            return error_response(
                error=str(exc),
                error_type="provider_error",
                provider="kie",
                model=model,
                prompt=prompt,
                aspect_ratio=unified_aspect,
            )
        except Exception as exc:  # noqa: BLE001 — never raise out of generate()
            logger.warning("kie image generate raised: %s", exc, exc_info=True)
            return error_response(
                error=f"KIE image generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="kie",
                model=model,
                prompt=prompt,
                aspect_ratio=unified_aspect,
            )

        if not urls:
            return error_response(
                error="KIE returned no result URLs",
                error_type="empty_result",
                provider="kie",
                model=model,
                prompt=prompt,
                aspect_ratio=unified_aspect,
            )

        extra: Dict[str, Any] = {}
        if len(urls) > 1:
            extra["all_images"] = urls
        if billing_on:
            try:
                balance = kb.charge(cost, "image", f"KIE {kind} {resolution}")
                extra["credits_charged"] = cost
                extra["balance"] = round(balance, 2)
            except Exception as exc:  # noqa: BLE001 — generation already succeeded
                logger.warning("kie image charge failed (image kept): %s", exc)
        return success_response(
            image=urls[0],
            model=model,
            prompt=prompt,
            aspect_ratio=unified_aspect,
            provider="kie",
            extra=extra or None,
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``KieImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(KieImageGenProvider())
