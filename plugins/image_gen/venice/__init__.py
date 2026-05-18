"""Venice image generation backend.

Exposes Venice's ``/images/generate`` endpoint as an
:class:`ImageGenProvider` implementation.

Venice's catalog mixes three sizing families — pixel-based
(``venice-sd35``), aspect-ratio (``qwen-image-2``, ``grok-imagine-image``),
and resolution-tier (``nano-banana-pro``, ``gpt-image-2``). The plugin
hides the family difference behind the unified
``aspect_ratio = landscape|square|portrait`` agent input, and only sends
the keys each model actually accepts.

Selection precedence (first hit wins):
    1. ``model=`` arg from the tool call
    2. ``VENICE_IMAGE_MODEL`` env var
    3. ``image_gen.venice.model`` in ``config.yaml``
    4. :data:`DEFAULT_MODEL` (``qwen-image-2``)

Authentication: ``VENICE_API_KEY``. Base URL is overridable via
``VENICE_BASE_URL`` so the same plugin can target either Venice direct
(``https://api.venice.ai/api/v1``) or the HermesOS managed-Venice proxy
(``https://hermesos.cloud/api/managed-venice/v1``) with no code change.
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
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VENICE_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 180


# Sizing family — each Venice model takes a different shape of size hint:
#   "aspect"     → ``aspect_ratio`` enum (1:1 / 16:9 / 9:16 / ...)
#   "pixels"     → ``width`` + ``height`` integers (max 1280)
#   "resolution" → ``aspect_ratio`` + ``resolution`` ("1K" / "2K" / "4K")
_FAMILY_ASPECT = "aspect"
_FAMILY_PIXELS = "pixels"
_FAMILY_RESOLUTION = "resolution"


# Curated Venice catalog. Not exhaustive — Venice ships many more models
# (and renames them frequently). Picking 5 well-known heros keeps the
# picker tight and the plugin maintainable. Users who want a model not
# in this list can set ``VENICE_IMAGE_MODEL`` directly; unknown ids are
# accepted and forwarded with aspect-ratio defaults.
_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-image-2": {
        "display": "Qwen Image 2",
        "speed": "~5-15s",
        "strengths": "Newer Qwen text-to-image. Good general-purpose default.",
        "family": _FAMILY_ASPECT,
    },
    "nano-banana-pro": {
        "display": "Nano Banana Pro",
        "speed": "~10-20s",
        "strengths": "Gemini-based, strong on photoreal + composition.",
        "family": _FAMILY_RESOLUTION,
    },
    "gpt-image-2": {
        "display": "GPT Image 2",
        "speed": "~10-25s",
        "strengths": "OpenAI gpt-image-2 via Venice. High prompt adherence.",
        "family": _FAMILY_RESOLUTION,
    },
    "venice-sd35": {
        "display": "Venice SD 3.5",
        "speed": "~3-8s",
        "strengths": "Venice's house SD 3.5. Cheap, fast, uncensored.",
        "family": _FAMILY_PIXELS,
    },
    "grok-imagine-image": {
        "display": "Grok Imagine (via Venice)",
        "speed": "~5-10s",
        "strengths": "Grok Imagine routed through Venice (single key).",
        "family": _FAMILY_ASPECT,
    },
}

DEFAULT_MODEL = "qwen-image-2"


# Agent unified aspect_ratio -> Venice native enum
_VENICE_ASPECT_RATIOS = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Pixel sizing for the pixels family. Venice's documented max is 1280 per axis.
_PIXEL_DIMENSIONS = {
    "landscape": (1280, 720),
    "square": (1024, 1024),
    "portrait": (720, 1280),
}

_VALID_RESOLUTIONS = {"1k", "2k", "4k"}
DEFAULT_RESOLUTION = "1k"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_venice_config() -> Dict[str, Any]:
    """Read ``image_gen.venice`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        venice_section = section.get("venice") if isinstance(section, dict) else None
        return venice_section if isinstance(venice_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.venice config: %s", exc)
        return {}


def _resolve_model(call_override: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``.

    Unknown ids are honored — they pass through to Venice with
    aspect-ratio sizing defaults. This keeps the plugin future-proof
    as Venice adds models faster than we can update the catalog.
    """
    candidate = (call_override or "").strip()
    if candidate:
        meta = _MODELS.get(candidate) or {"family": _FAMILY_ASPECT, "display": candidate}
        return candidate, meta

    env_override = os.environ.get("VENICE_IMAGE_MODEL")
    if env_override:
        meta = _MODELS.get(env_override) or {"family": _FAMILY_ASPECT, "display": env_override}
        return env_override, meta

    cfg = _load_venice_config()
    cfg_model = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if cfg_model:
        meta = _MODELS.get(cfg_model) or {"family": _FAMILY_ASPECT, "display": cfg_model}
        return cfg_model, meta

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_base_url() -> str:
    """Resolve Venice base URL.

    Order: ``VENICE_BASE_URL`` env (used by the managed-venice proxy) →
    ``image_gen.venice.base_url`` config → public Venice default.
    """
    env_url = os.environ.get("VENICE_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    cfg = _load_venice_config()
    cfg_url = cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else None
    if cfg_url:
        return cfg_url.strip().rstrip("/")

    return DEFAULT_VENICE_BASE_URL


def _resolve_resolution() -> str:
    cfg = _load_venice_config()
    res = cfg.get("resolution") if isinstance(cfg.get("resolution"), str) else None
    if res and res.lower() in _VALID_RESOLUTIONS:
        return res.lower()
    return DEFAULT_RESOLUTION


def _build_payload(
    *,
    prompt: str,
    model_id: str,
    family: str,
    aspect: str,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Venice request body. Only keys the family supports are sent."""
    payload: Dict[str, Any] = {
        "model": model_id,
        "prompt": prompt,
        "safe_mode": bool(extra_kwargs.get("safe_mode", True)),
        "format": str(extra_kwargs.get("format", "webp")),
        "return_binary": False,
    }

    if family == _FAMILY_ASPECT:
        payload["aspect_ratio"] = _VENICE_ASPECT_RATIOS.get(aspect, "1:1")
    elif family == _FAMILY_PIXELS:
        w, h = _PIXEL_DIMENSIONS.get(aspect, _PIXEL_DIMENSIONS["square"])
        payload["width"] = w
        payload["height"] = h
    elif family == _FAMILY_RESOLUTION:
        payload["aspect_ratio"] = _VENICE_ASPECT_RATIOS.get(aspect, "1:1")
        payload["resolution"] = _resolve_resolution().upper()

    neg = extra_kwargs.get("negative_prompt")
    if isinstance(neg, str) and neg.strip():
        payload["negative_prompt"] = neg.strip()

    style = extra_kwargs.get("style_preset")
    if isinstance(style, str) and style.strip():
        payload["style_preset"] = style.strip()

    seed = extra_kwargs.get("seed")
    if isinstance(seed, int):
        payload["seed"] = seed

    return payload


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class VeniceImageGenProvider(ImageGenProvider):
    """Venice ``/images/generate`` backend."""

    @property
    def name(self) -> str:
        return "venice"

    @property
    def display_name(self) -> str:
        return "Venice"

    def is_available(self) -> bool:
        return bool(os.environ.get("VENICE_API_KEY", "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Venice",
            "badge": "paid",
            "tag": "qwen-image-2 / nano-banana-pro / gpt-image-2 / venice-sd35 — uses VENICE_API_KEY (same key as Venice chat)",
            "env_vars": [
                {
                    "key": "VENICE_API_KEY",
                    "prompt": "Venice API key",
                    "url": "https://venice.ai/settings/api",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = os.environ.get("VENICE_API_KEY", "").strip()
        if not api_key:
            return error_response(
                error="No Venice credentials found. Set VENICE_API_KEY (same key as Venice chat).",
                error_type="missing_api_key",
                provider="venice",
                aspect_ratio=aspect_ratio,
            )

        model_id, meta = _resolve_model(kwargs.get("model"))
        family = str(meta.get("family", _FAMILY_ASPECT))
        aspect = resolve_aspect_ratio(aspect_ratio)
        base_url = _resolve_base_url()

        payload = _build_payload(
            prompt=prompt,
            model_id=model_id,
            family=family,
            aspect=aspect,
            extra_kwargs=kwargs,
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent/image_gen-venice",
        }

        try:
            response = requests.post(
                f"{base_url}/images/generate",
                headers=headers,
                json=payload,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                err_msg = resp.json().get("error", {}).get("message") if resp is not None else str(exc)
                if not err_msg:
                    err_msg = (resp.text[:300] if resp is not None else str(exc))
            except Exception:
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("Venice image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"Venice image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="venice",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error=f"Venice image generation timed out ({DEFAULT_TIMEOUT_SECONDS}s)",
                error_type="timeout",
                provider="venice",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"Venice connection error: {exc}",
                error_type="connection_error",
                provider="venice",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:
            return error_response(
                error=f"Venice returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider="venice",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Venice's native response shape: ``{"images": [<b64>, ...], ...}``.
        # Fall back to OpenAI-compat ``data[].b64_json`` / ``data[].url``
        # in case Venice re-routes a request through their /images/generations
        # path (some model ids do).
        first_b64: Optional[str] = None
        first_url: Optional[str] = None

        images_field = result.get("images")
        if isinstance(images_field, list) and images_field:
            head = images_field[0]
            if isinstance(head, str):
                first_b64 = head
            elif isinstance(head, dict):
                first_b64 = head.get("b64_json") or head.get("image")
                first_url = head.get("url")

        if not first_b64 and not first_url:
            data_field = result.get("data")
            if isinstance(data_field, list) and data_field:
                head = data_field[0]
                if isinstance(head, dict):
                    first_b64 = head.get("b64_json")
                    first_url = head.get("url")

        if not first_b64 and not first_url:
            return error_response(
                error="Venice returned no image data",
                error_type="empty_response",
                provider="venice",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if first_b64:
            try:
                saved_path = save_b64_image(first_b64, prefix=f"venice_{model_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="venice",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        else:
            image_ref = first_url or ""

        extra: Dict[str, Any] = {}
        if family == _FAMILY_RESOLUTION:
            extra["resolution"] = payload.get("resolution")
        if isinstance(result.get("id"), str):
            extra["venice_request_id"] = result["id"]

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="venice",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(VeniceImageGenProvider())
