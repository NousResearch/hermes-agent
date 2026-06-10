"""MiniMax image generation backend.

Wraps the MiniMax Text-to-Image API as an :class:`ImageGenProvider`
implementation. Single model: ``image-01``.

API surface
-----------
- Endpoint: ``POST https://api.minimax.io/v1/image_generation``
- Auth: ``Authorization: Bearer $MINIMAX_API_KEY``
- Request: ``{model, prompt, aspect_ratio, response_format, n, seed,
  prompt_optimizer}``
- Response (response_format=base64): ``{base_resp: {status_code,
  status_msg}, data: {image_base64: [...]}, metadata: {...}, id}``
- Response (response_format=url): ``data.image_urls`` (24-hour expiry)

We default to ``response_format=base64`` so the gateway can hand back a
persistent file path instead of a 24-hour-expiring URL — same rationale
as the OpenAI provider caching the b64 bytes locally.

Aspect ratio mapping (Hermes 3-value enum → MiniMax 8-value enum):

  landscape → 16:9
  square    → 1:1
  portrait  → 9:16

Other MiniMax ratios (``4:3``, ``3:2``, ``2:3``, ``3:4``, ``21:9``) are
deliberately not exposed — the Hermes ``aspect_ratio`` contract is the
3-value landscape/square/portrait tuple, and silent coercion is
forgiving enough that we don't need an escape hatch here.

Image URLs from MiniMax (when ``response_format=url`` is set) expire
in 24 hours per the API docs. We always set ``response_format=base64``
in the request so the saved image path is durable.

Configuration precedence for the active model (first hit wins):

1. ``MINIMAX_IMAGE_MODEL`` env var
2. ``image_gen.minimax.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's ``image-01``)
4. :data:`DEFAULT_MODEL` — ``image-01``
"""

from __future__ import annotations

import base64
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
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_MODEL = "image-01"
DEFAULT_MODEL = "image-01"

# MiniMax hard caps per the API docs.
MAX_PROMPT_LENGTH = 1500
# n=1 is what the Hermes provider contract returns; multiple-image
# fan-out is not in the ImageGenProvider ABC. MiniMax allows [1, 9].
NUM_IMAGES = 1

# Endpoint
DEFAULT_BASE_URL = "https://api.minimax.io/v1/image_generation"
DEFAULT_TIMEOUT_SECONDS = 120

# Map Hermes' 3-value aspect_ratio enum to MiniMax's 8-value enum.
# MiniMax offers 8 ratios (1:1, 16:9, 4:3, 3:2, 2:3, 3:4, 9:16, 21:9)
# but Hermes' provider ABC only exposes landscape/square/portrait.
_ASPECT_RATIO_MAP: Dict[str, str] = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


def _sniff_image_extension(raw_bytes: bytes) -> str:
    """Sniff the image format from magic bytes so the saved file lands
    on disk with the right extension.

    MiniMax's image-01 model returns JPEG by default, but the API
    doesn't expose a format knob — we sniff rather than hard-code so
    we keep working if they change defaults. Falls back to ``png`` for
    anything we don't recognize (matches ``save_b64_image``'s default
    so unknown formats still get a working file).
    """
    if not raw_bytes:
        return "png"
    # JPEG: FF D8 FF
    if raw_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    # WebP: RIFF .... WEBP
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "webp"
    # GIF: GIF87a or GIF89a
    if raw_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


# ---------------------------------------------------------------------------
# Config / model resolution
# ---------------------------------------------------------------------------


def _load_image_gen_section() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use. Returns ``(model_id, meta)``.

    Currently only one model ships (``image-01``) so the resolution is
    mostly a sanity check on the configured value — we keep the same
    precedence order as the other plugins so a future model addition
    (e.g. ``image-02``) plugs in without touching the dispatcher.
    """
    candidates: List[Optional[str]] = []
    candidates.append(os.environ.get("MINIMAX_IMAGE_MODEL"))

    cfg = _load_image_gen_section()
    minimax_cfg = cfg.get("minimax") if isinstance(cfg.get("minimax"), dict) else {}
    if isinstance(minimax_cfg, dict):
        candidates.append(minimax_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for c in candidates:
        if isinstance(c, str) and c.strip() == API_MODEL:
            return API_MODEL, {"display": "MiniMax Image-01", "id": API_MODEL}

    return DEFAULT_MODEL, {"display": "MiniMax Image-01", "id": DEFAULT_MODEL}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MiniMaxImageGenProvider(ImageGenProvider):
    """MiniMax Text-to-Image backend (image-01 model)."""

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        # Single check: do we have the API key? No SDK to install —
        # we use the stdlib ``requests`` directly.
        return bool(str(os.environ.get("MINIMAX_API_KEY") or "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": API_MODEL,
                "display": "MiniMax Image-01",
                "speed": "~10-30s",
                "strengths": "Photorealistic, 8 aspect ratios, batch up to 9 images",
                "price": "paid",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "paid",
            "tag": "MiniMax image-01 — photorealistic, 8 aspect ratios",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API key",
                    "url": "https://platform.minimax.io/user-center/basic-information/interface-key",
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
        mm_aspect = _ASPECT_RATIO_MAP.get(aspect, _ASPECT_RATIO_MAP["landscape"])

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="minimax",
                aspect_ratio=aspect,
            )

        api_key = str(os.environ.get("MINIMAX_API_KEY") or "").strip()
        if not api_key:
            return error_response(
                error=(
                    "MINIMAX_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → MiniMax to configure, or set the env var."
                ),
                error_type="auth_required",
                provider="minimax",
                aspect_ratio=aspect,
            )

        # Truncate to the MiniMax hard cap. The API rejects longer
        # prompts with a 1004-style error; we clamp here for a cleaner
        # agent-side failure mode.
        truncated = False
        if len(prompt) > MAX_PROMPT_LENGTH:
            prompt = prompt[:MAX_PROMPT_LENGTH]
            truncated = True

        model_id, _meta = _resolve_model()

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "aspect_ratio": mm_aspect,
            "response_format": "base64",
            "n": NUM_IMAGES,
        }

        # Forward optional knobs the docs expose. The ABC doesn't
        # declare these but the dispatcher passes **kwargs through, so
        # power users can hit them via config or a future tool schema.
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            payload["seed"] = seed
        prompt_optimizer = kwargs.get("prompt_optimizer")
        if isinstance(prompt_optimizer, bool):
            payload["prompt_optimizer"] = prompt_optimizer

        # Allow config-level base_url override (escape hatch for
        # self-hosted / proxied MiniMax-compatible gateways).
        base_url = str(
            kwargs.get("base_url")
            or os.environ.get("MINIMAX_IMAGE_BASE_URL")
            or DEFAULT_BASE_URL
        ).strip().rstrip("/") or DEFAULT_BASE_URL

        try:
            response = requests.post(
                base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            logger.debug("MiniMax image generation network error", exc_info=True)
            return error_response(
                error=f"MiniMax image generation request failed: {exc}",
                error_type="network_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # 4xx/5xx — surface body if it looks JSON, else status text.
        if response.status_code >= 400:
            detail = (response.text or "").strip()[:500]
            return error_response(
                error=(
                    f"MiniMax image generation failed "
                    f"(HTTP {response.status_code}): {detail or 'no response body'}"
                ),
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            data = response.json()
        except ValueError:
            return error_response(
                error=(
                    f"MiniMax returned non-JSON response "
                    f"(HTTP {response.status_code}): "
                    f"{(response.text or '')[:200]}"
                ),
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # MiniMax uses a base_resp.status_code envelope like the TTS
        # API. status_code != 0 is the failure path.
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code", -1)
        if status_code != 0:
            status_msg = base_resp.get("status_msg") or "unknown error"
            return error_response(
                error=f"MiniMax image generation failed ({status_code}): {status_msg}",
                error_type="api_error",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        payload_data = data.get("data") or {}

        # response_format=base64 → data.image_base64 (list of strings)
        # response_format=url    → data.image_urls   (list of strings)
        # We sent base64 but be defensive in case the API ever returns
        # both or neither.
        b64_list = payload_data.get("image_base64") or []
        url_list = payload_data.get("image_urls") or []

        if b64_list:
            try:
                # MiniMax doesn't expose an image-format knob — the
                # API returns whatever its pipeline produces (typically
                # JPEG for image-01). Decode first, sniff the magic
                # bytes, then save with the right extension. This
                # matters because the gateway and downstream tools
                # key off the file suffix.
                raw_bytes = base64.b64decode(b64_list[0])
                extension = _sniff_image_extension(raw_bytes)
                saved_path = save_b64_image(
                    b64_list[0],
                    prefix=f"minimax_{model_id}",
                    extension=extension,
                )
                image_ref = str(saved_path)
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="minimax",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif url_list:
            # Defensive: if we ever flip response_format=url, we still
            # need a durable handle. Fetch via the helper (which
            # streams + caps + infers extension) instead of handing the
            # bare 24h-expire URL to the gateway.
            try:
                from agent.image_gen_provider import save_url_image

                saved_path = save_url_image(
                    url_list[0], prefix=f"minimax_{model_id}"
                )
                image_ref = str(saved_path)
            except Exception as exc:
                # Fall back to the bare URL — caller will see the
                # 24h-expiry footgun in the worst case but we don't
                # want to mask a transient save error as a fatal
                # provider failure.
                logger.warning(
                    "MiniMax returned a URL but it could not be cached (%s); "
                    "passing bare URL through (expires in 24h).",
                    exc,
                )
                image_ref = url_list[0]
        else:
            return error_response(
                error="MiniMax response contained neither image_base64 nor image_urls",
                error_type="empty_response",
                provider="minimax",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # image_ref is guaranteed non-None here: every code path above
        # either assigns it or returns. The assert keeps mypy / pyright
        # happy and doubles as a runtime guard.
        assert image_ref is not None

        extra: Dict[str, Any] = {
            "minimax_aspect_ratio": mm_aspect,
            "request_id": data.get("id", ""),
        }
        metadata = data.get("metadata") or {}
        if metadata:
            # Surface success/failure counts for batch debugging even
            # though we only requested n=1 today.
            extra["metadata"] = metadata
        if truncated:
            extra["prompt_truncated"] = True
            extra["prompt_truncated_to"] = MAX_PROMPT_LENGTH

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="minimax",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``MiniMaxImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(MiniMaxImageGenProvider())
