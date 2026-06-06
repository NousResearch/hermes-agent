"""
MiniMax image-01 generation backend.

Exposes MiniMax's ``image-01`` model as an :class:`ImageGenProvider`
implementation. Uses the existing ``MINIMAX_API_KEY`` env var and the
``https://api.minimax.io/v1/image_generation`` endpoint.

Plugin entry point: ``register(ctx)``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

API_ENDPOINT = "https://api.minimax.io/v1/image_generation"
API_MODEL = "image-01"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}

DEFAULT_MODEL = "image-01"


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax ``image-01`` backend."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        return bool(os.environ.get("MINIMAX_API_KEY"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "image-01",
                "display": "MiniMax Image-01",
                "speed": "~15-30s",
                "strengths": "Fast, high quality, zero extra cost on Token Plan",
                "price": "Free (Token Plan quota)",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "★ free · Token Plan",
            "tag": "image-01 via api.minimax.io — uses your existing Token Plan quota",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key (sk-cp-...)",
                    "url": "https://platform.minimax.io",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            return error_response(
                error="MINIMAX_API_KEY not set. Add it to ~/.hermes/.env",
                error_type="auth_required",
                provider="minimax",
                aspect_ratio=aspect,
            )

        size = _SIZES.get(aspect, _SIZES["square"])

        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }

        try:
            response = requests.post(
                API_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
        except Exception as exc:
            logger.debug("MiniMax image request failed", exc_info=True)
            return error_response(
                error=f"MiniMax API request failed: {exc}",
                error_type="network_error",
                provider="minimax",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not response.ok:
            try:
                err_body = response.json()
            except Exception:
                err_body = {"detail": response.text}
            err_msg = err_body.get("detail", str(err_body))
            return error_response(
                error=f"MiniMax API error ({response.status_code}): {err_msg}",
                error_type="api_error",
                provider="minimax",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            data = response.json()
        except Exception as exc:
            return error_response(
                error=f"Could not parse MiniMax response: {exc}",
                error_type="parse_error",
                provider="minimax",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # MiniMax returns {id, data: {image_urls: [...], ...}, metadata: {...}}
        image_urls = data.get("data", {}).get("image_urls") or []
        if not image_urls:
            return error_response(
                error=f"MiniMax returned no image URLs. Response: {data}",
                error_type="empty_response",
                provider="minimax",
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        image_url = image_urls[0]

        return success_response(
            image=image_url,
            model=DEFAULT_MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="minimax",
            extra={"size": size},
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
