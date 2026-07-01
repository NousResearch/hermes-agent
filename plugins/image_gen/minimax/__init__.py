"""MiniMax image generation backend.

Exposes the MiniMax ``image-01`` model under three virtual model IDs that
match the existing image_gen picker conventions. MiniMax's image API is
synchronous — POST returns a JSON body with a URL (or base64) for the
generated image, no job polling required.

    minimax-image-01            — full quality (default)
    minimax-image-01-live       — lower latency variant
    minimax-image-01-square     — alias that forces a square aspect ratio

All three hit the same native endpoint with a different ``model`` field.
Output is saved under ``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``MINIMAX_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.minimax.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL` — ``minimax-image-01``

Endpoint: ``POST https://api.minimax.io/v1/image_generation``
Auth: ``Authorization: Bearer $MINIMAX_API_KEY`` (always) and an optional
``GroupId`` query parameter (``$MINIMAX_GROUP_ID``) for accounts that use
group-scoped billing. The native endpoint is **not** OpenAI-compatible —
``/v1/images/generations`` returns 404.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.minimax.io/v1/image_generation"
TIMEOUT_SECONDS = 120.0

# Each virtual model ID maps to a MiniMax ``model`` field. We expose
# three IDs so the picker UX matches other backends (which all advertise
# 2-3 quality tiers) and so the ``-live`` suffix hints at the live variant.
_MODELS: Dict[str, Dict[str, Any]] = {
    "minimax-image-01": {
        "display": "MiniMax Image 01",
        "speed": "~10-30s",
        "strengths": "Default. Strong prompt adherence, 1024x1024 default.",
        "price": "varies (see MiniMax pricing)",
        "api_model": "image-01",
    },
    "minimax-image-01-live": {
        "display": "MiniMax Image 01 (Live)",
        "speed": "~5-15s",
        "strengths": "Lower latency variant, same quality ceiling.",
        "price": "varies (see MiniMax pricing)",
        "api_model": "image-01-live",
    },
    "minimax-image-01-square": {
        "display": "MiniMax Image 01 (Square)",
        "speed": "~10-30s",
        "strengths": "Alias that forces 1024x1024 — convenience for square output.",
        "price": "varies (see MiniMax pricing)",
        "api_model": "image-01",
    },
}

DEFAULT_MODEL = "minimax-image-01"

# Map hermes 3-ratio abstraction to MiniMax's aspect ratio strings.
# MiniMax accepts "1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9".
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


# ---------------------------------------------------------------------------
# Config + resolution
# ---------------------------------------------------------------------------


def _load_minimax_config() -> Dict[str, Any]:
    """Read the ``image_gen`` section from config.yaml.

    Returns the full ``image_gen`` dict (callers extract their own sub-keys);
    returns an empty dict on any read or parse failure so callers can use
    ``.get()`` chains without guarding for None.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``.

    Precedence: env var > config (``image_gen.minimax.model``) > default.
    """
    env_override = os.environ.get("MINIMAX_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_minimax_config()
    minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(minimax_cfg, dict):
        value = minimax_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_aspect(model_id: str, aspect: str) -> str:
    """Map hermes's 3-ratio abstraction to MiniMax's aspect strings.

    The ``-square`` virtual model always forces 1:1 regardless of the
    aspect_ratio kwarg. The other two honor the kwarg.
    """
    if model_id == "minimax-image-01-square":
        return "1:1"
    return _ASPECT_MAP.get(aspect, "1:1")


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax image generation backend — image-01 / image-01-live."""

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
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for model_id, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "MiniMax image-01 / image-01-live. Native endpoint, not OpenAI-compat.",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key",
                    "url": "https://platform.minimax.io/user-center/basic-information/interface-key",
                },
                {
                    "key": "MINIMAX_GROUP_ID",
                    "prompt": "MiniMax Group ID (required for group-scoped accounts; optional otherwise)",
                    "url": "https://platform.minimax.io/user-center/basic-information/interface-key",
                    "required": False,
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
        provider_name = self.name

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=provider_name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        api_key = os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "MINIMAX_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → MiniMax to configure, or get a key at "
                    "https://platform.minimax.io/"
                ),
                error_type="auth_required",
                provider=provider_name,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        api_model = meta["api_model"]
        minmax_aspect = _resolve_aspect(model_id, aspect)

        # Forward-compat passthroughs — MiniMax accepts these but the agent
        # rarely supplies them. Default to a single image.
        n = int(kwargs.get("n", 1))
        if n < 1 or n > 4:
            return error_response(
                error=f"n must be 1-4, got {n}",
                error_type="invalid_argument",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        response_format = str(kwargs.get("response_format", "url")).lower()
        if response_format not in {"url", "base64"}:
            return error_response(
                error=f"response_format must be 'url' or 'base64', got {response_format!r}",
                error_type="invalid_argument",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        payload: Dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "aspect_ratio": minmax_aspect,
            "n": n,
            "response_format": response_format,
        }
        if "seed" in kwargs:
            payload["seed"] = int(kwargs["seed"])
        if kwargs.get("prompt_optimizer"):
            payload["prompt_optimizer"] = True
        if kwargs.get("aigc_watermark"):
            payload["aigc_watermark"] = True

        # Build the URL — GroupId is a query string parameter when present
        # (some MiniMax accounts use group-scoped billing; without it the
        # server returns 401 "GroupId is required").
        group_id = (os.environ.get("MINIMAX_GROUP_ID") or "").strip()
        url = BASE_URL
        if group_id:
            url = f"{url}?GroupId={group_id}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            return error_response(
                error=f"MiniMax image generation timed out after {TIMEOUT_SECONDS}s",
                error_type="timeout",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            logger.debug("MiniMax image generation transport error", exc_info=True)
            return error_response(
                error=f"MiniMax image generation failed: {exc}",
                error_type="api_error",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # MiniMax returns 401/403/404 with a JSON body — surface its message
        # to the user verbatim so they can fix the auth or pick a different
        # model. 200 with no data is also a real failure mode.
        if response.status_code != 200:
            server_msg = ""
            try:
                body = response.json()
                server_msg = (
                    body.get("message")
                    or body.get("error", {}).get("message")
                    or body.get("base_resp", {}).get("status_msg")
                    or ""
                )
            except Exception:  # noqa: BLE001
                server_msg = response.text[:200]
            return error_response(
                error=(
                    f"MiniMax image generation HTTP {response.status_code}: "
                    f"{server_msg or response.reason}"
                ),
                error_type="api_error",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            body = response.json()
        except ValueError as exc:
            return error_response(
                error=f"MiniMax returned non-JSON body: {exc}",
                error_type="invalid_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # The native endpoint returns ``data.image_urls`` — a list of
        # signed CDN URLs (each ~30-min TTL) when response_format=url, or
        # ``data.b64_json`` (also a list) when response_format=base64.
        # Unlike the OpenAI images API, ``data`` is a DICT not a list.
        data = body.get("data") or {}
        if not data:
            base_resp = body.get("base_resp") or {}
            return error_response(
                error=(
                    f"MiniMax returned no image data: "
                    f"{base_resp.get('status_msg') or 'empty data object'}"
                ),
                error_type="empty_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Prefer b64_json if the caller asked for it; fall back to image_urls.
        image_url: Optional[str] = None
        b64: Optional[str] = None
        b64_list = data.get("b64_json")
        if isinstance(b64_list, list) and b64_list:
            b64 = b64_list[0]
        if b64 is None:
            url_list = data.get("image_urls")
            if isinstance(url_list, list) and url_list:
                image_url = url_list[0]

        image_ref: Optional[str] = None
        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"minimax_{model_id}")
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider=provider_name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif image_url:
            # Cache the bytes locally so the gateway never tries to fetch
            # an expired signed URL after the 30-min TTL. Same pattern as
            # the xAI provider (see test_xai_provider for the original).
            try:
                saved_path = save_url_image(image_url, prefix=f"minimax_{model_id}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MiniMax image URL %s could not be cached (%s); falling back to bare URL.",
                    image_url,
                    exc,
                )
                image_ref = image_url
            else:
                image_ref = str(saved_path)

        if not image_ref:
            return error_response(
                error="MiniMax response contained neither b64_json nor image_urls",
                error_type="empty_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=provider_name,
            extra={"minimax_aspect": minmax_aspect, "api_model": api_model},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
