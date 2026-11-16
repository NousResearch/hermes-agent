"""MiniMax image generation backend.

Exposes MiniMax's ``image-01`` model as an :class:`ImageGenProvider`.
Text-to-image is the primary mode. Image-to-image is narrower than most
other backends here: MiniMax's API only supports a single
``subject_reference`` (``type: "character"``) rather than a general
image-edit endpoint, so only one source image (``image_url``, or the first
of ``reference_image_urls`` if ``image_url`` is omitted) is honored — see
:meth:`MiniMaxImageGenProvider.capabilities`.

API reference: https://platform.minimax.io/docs/api-reference/image-generation-t2i
Auth: the ``MINIMAX_API_KEY`` environment variable is sent as the API token
(same key as the MiniMax chat model provider — no separate credential needed).

The API returns an ``image_urls`` entry that expires in 24h, so results are
downloaded and cached locally immediately, mirroring the xAI/OpenAI backends
in this tree.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_url_image,
    success_response,
)
from agent.secret_scope import get_secret
from plugins.image_gen._common import post_json

logger = logging.getLogger(__name__)

API_URL = "https://api.minimax.io/v1/image_generation"
MODEL = "image-01"
_REQUEST_TIMEOUT = 120


def _api_key() -> str:
    """Resolve the MiniMax credential — ``MINIMAX_API_KEY`` and nothing else.

    The secret scope is consulted first so a profile-bound key wins, then the
    process environment (single-profile deployments, cron, systemd). A scope
    lookup can legitimately fail when no profile is installed, hence the guard.
    """
    try:
        scoped = get_secret("MINIMAX_API_KEY")
    except Exception:  # no scope available — fall back to the process environment
        scoped = None
    value = scoped or os.environ.get("MINIMAX_API_KEY")
    return value.strip() if isinstance(value, str) else ""


# Mirrors the openrouter backend's mapping — MiniMax takes the same
# "W:H" aspect-ratio strings.
_ASPECT_RATIOS = {
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
}


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax ``image-01`` backend — text-to-image + single-reference edit."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        return bool(_api_key())

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": MODEL,
                "display": "MiniMax Image-01",
                "speed": "~10-20s",
                "strengths": "Text-to-image; single-subject character-reference editing",
                "price": "varies",
            }
        ]

    def default_model(self) -> Optional[str]:
        return MODEL

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "image-01 — text-to-image & single-reference character editing",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key",
                    "url": "https://platform.minimax.io/",
                },
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        # subject_reference accepts one character-reference image, not a
        # general multi-image edit — cap at 1 so the tool schema doesn't
        # promise more than the API delivers.
        return {"modalities": ["text", "image"], "max_reference_images": 1}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=aspect,
            )

        api_key = _api_key()
        if not api_key:
            return error_response(
                error=(
                    "MINIMAX_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → MiniMax to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="minimax",
                aspect_ratio=aspect,
            )

        # Single reference image only — image_url wins over reference_image_urls.
        source: Optional[str] = None
        if isinstance(image_url, str) and image_url.strip():
            source = image_url.strip()
        else:
            refs = normalize_reference_images(reference_image_urls) or []
            if refs:
                source = refs[0]
        modality = "image" if source else "text"

        payload: dict[str, Any] = {
            "model": MODEL,
            "prompt": prompt,
            "aspect_ratio": _ASPECT_RATIOS.get(aspect, "1:1"),
            "response_format": "url",
            "n": 1,
        }
        if source:
            payload["subject_reference"] = [
                {"type": "character", "image_file": source}
            ]

        try:
            body, failure = post_json(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
                timeout=_REQUEST_TIMEOUT,
                label="MiniMax",
            )
        except Exception as exc:
            logger.debug("MiniMax image generation request failed", exc_info=True)
            return error_response(
                error=f"MiniMax image generation request failed: {exc}",
                error_type="api_error",
                provider="minimax",
                model=MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        if failure:
            if failure.kind == "http":
                logger.error("MiniMax image gen failed (%d): %s", failure.status, failure.message)
            return error_response(
                error=failure.error,
                error_type=failure.error_type,
                provider="minimax",
                model=MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        body = body or {}

        base_resp = body.get("base_resp") or {}
        if base_resp.get("status_code") not in (0, None):
            return error_response(
                error=f"MiniMax API error: {base_resp.get('status_msg') or base_resp}",
                error_type="api_error",
                provider="minimax",
                model=MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_urls = ((body.get("data") or {}).get("image_urls")) or []
        if not image_urls:
            return error_response(
                error="MiniMax response contained no image_urls",
                error_type="empty_response",
                provider="minimax",
                model=MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_url_image(image_urls[0], prefix="minimax_image-01")
        except Exception as exc:
            logger.warning(
                "MiniMax image URL could not be cached (%s); falling back to bare URL.",
                exc,
            )
            image_ref = image_urls[0]
        else:
            image_ref = str(saved_path)

        return success_response(
            image=image_ref,
            model=MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="minimax",
            modality=modality,
            extra={"minimax_id": body.get("id")},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
