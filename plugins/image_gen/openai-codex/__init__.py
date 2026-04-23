"""OpenAI image generation backend — ChatGPT/Codex OAuth variant.

Identical model catalog and tier semantics to the ``openai`` image-gen plugin
(``gpt-image-2`` at auto/low/medium/high quality), but routes the request through
the Codex Responses API ``image_generation`` tool instead of the
``images.generate`` REST endpoint. This lets users who are already
authenticated with Codex/ChatGPT generate images without configuring a
separate ``OPENAI_API_KEY``.

Selection precedence for the tier (first hit wins):

1. Per-call ``quality`` override (``auto``, ``low``, ``medium``, ``high``)
2. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
3. ``image_gen.openai-codex.model`` in ``config.yaml``
4. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
5. :data:`DEFAULT_MODEL` — ``gpt-image-2-auto``

Output is saved under ``$HERMES_HOME/cache/images/`` using the requested format
(``png`` by default, or ``jpeg``/``webp`` when selected per call).
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

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
# Model catalog — mirrors the ``openai`` plugin so the picker UX is identical.
# ---------------------------------------------------------------------------

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-auto": {
        "display": "GPT Image 2 (Auto)",
        "speed": "varies",
        "strengths": "Provider-selected balance of quality, latency, and cost",
        "quality": "auto",
    },
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-auto"

_QUALITY_TO_TIER = {
    "low": "gpt-image-2-low",
    "medium": "gpt-image-2-medium",
    "high": "gpt-image-2-high",
}
_VALID_QUALITIES = {"auto", "low", "medium", "high"}
_VALID_OUTPUT_FORMATS = {"png", "jpeg", "webp"}

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}
_SIZE_RE = re.compile(r"^(\d+)x(\d+)$")

# Codex Responses surface used for the request. The chat model itself is only
# the host that calls the ``image_generation`` tool; the actual image work is
# done by ``API_MODEL``.
_CODEX_CHAT_MODEL = "gpt-5.4"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by "
    "using the image_generation tool when provided."
)


# ---------------------------------------------------------------------------
# Config + auth helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
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
    """Decide which tier to use and return ``(model_id, meta)``."""
    import os

    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_image_gen_config()
    sub = cfg.get("openai-codex") if isinstance(cfg.get("openai-codex"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(sub, dict):
        value = sub.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _validate_gpt_image_2_size(size: str) -> Optional[str]:
    """Return an error message when *size* violates gpt-image-2 limits."""
    if size == "auto":
        return None
    match = _SIZE_RE.match(size)
    if not match:
        return "size must be 'auto' or '<width>x<height>'"
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return "width and height must be positive integers"
    pixels = width * height
    if width % 16 != 0 or height % 16 != 0:
        return "both width and height must be multiples of 16"
    if max(width, height) >= 3840:
        return "maximum edge length must be less than 3840px"
    if max(width / height, height / width) > 3:
        return "aspect ratio must not exceed 3:1"
    if pixels < 655_360:
        return "total pixels must be at least 655,360"
    if pixels > 8_294_400:
        return "total pixels must not exceed 8,294,400"
    return None


def _resolve_generation_options(
    *,
    aspect: str,
    quality: Optional[str] = None,
    size: Optional[str] = None,
    output_format: Optional[str] = None,
    output_compression: Optional[Any] = None,
) -> Tuple[
    Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str],
    Optional[str], Optional[int], Optional[str],
]:
    """Resolve tier/model, API quality, size, format, and compression.

    Returns ``(tier_id, meta, api_quality, api_size, output_format,
    output_compression, error)``. ``quality='auto'`` is passed through to the
    API without pinning a low/medium/high tier.
    """
    tier_id, meta = _resolve_model()
    api_quality = meta["quality"]

    if quality is not None:
        quality = str(quality).strip().lower()
        if quality:
            if quality not in _VALID_QUALITIES:
                return None, None, None, None, None, None, (
                    "quality must be one of: auto, low, medium, high"
                )
            api_quality = quality
            if quality == "auto":
                tier_id = "gpt-image-2-auto"
                meta = _MODELS[tier_id]
            else:
                tier_id = _QUALITY_TO_TIER[quality]
                meta = _MODELS[tier_id]

    api_size = _SIZES.get(aspect, _SIZES["square"])
    if size is not None:
        size = str(size).strip().lower()
        if size:
            error = _validate_gpt_image_2_size(size)
            if error:
                return None, None, None, None, None, None, f"Invalid size '{size}': {error}"
            api_size = size

    api_format = "png"
    if output_format is not None:
        api_format = str(output_format).strip().lower()
        if api_format not in _VALID_OUTPUT_FORMATS:
            return None, None, None, None, None, None, (
                "output_format must be one of: png, jpeg, webp"
            )

    api_compression: Optional[int] = None
    if output_compression is not None:
        if api_format not in {"jpeg", "webp"}:
            return None, None, None, None, None, None, (
                "output_compression is only supported with jpeg/webp output_format"
            )
        try:
            api_compression = int(output_compression)
        except (TypeError, ValueError):
            return None, None, None, None, None, None, (
                "output_compression must be an integer from 0 to 100"
            )
        if not 0 <= api_compression <= 100:
            return None, None, None, None, None, None, (
                "output_compression must be an integer from 0 to 100"
            )

    return tier_id, meta, api_quality, api_size, api_format, api_compression, None


def _read_codex_access_token() -> Optional[str]:
    """Return a usable Codex OAuth token, or None.

    Delegates to the canonical reader in ``agent.auxiliary_client`` so token
    expiry, credential pool selection, and JWT decoding stay in one place.
    """
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        if isinstance(token, str) and token.strip():
            return token.strip()
        return None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token: %s", exc)
        return None


def _build_codex_client():
    """Return an OpenAI client pointed at the ChatGPT/Codex backend, or None."""
    token = _read_codex_access_token()
    if not token:
        return None
    try:
        import openai
        from agent.auxiliary_client import _codex_cloudflare_headers

        return openai.OpenAI(
            api_key=token,
            base_url=_CODEX_BASE_URL,
            default_headers=_codex_cloudflare_headers(token),
        )
    except Exception as exc:
        logger.debug("Could not build Codex image client: %s", exc)
        return None


def _collect_image_b64(
    client: Any,
    *,
    prompt: str,
    size: str,
    quality: str,
    output_format: str = "png",
    output_compression: Optional[int] = None,
) -> Optional[str]:
    """Stream a Codex Responses image_generation call and return the b64 image."""
    image_b64: Optional[str] = None
    tool: Dict[str, Any] = {
        "type": "image_generation",
        "model": API_MODEL,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "background": "opaque",
        "partial_images": 0,
    }
    if output_compression is not None:
        tool["output_compression"] = output_compression

    # OpenAI SDK 2.31.0 can emit noisy Pydantic serializer warnings when the
    # API returns gpt-image-2 fields its generated response schema has not yet
    # learned, especially custom sizes such as 1152x2496. The API accepts these
    # values and generation succeeds; keep this quirk scoped to the Codex image
    # stream so CLI users do not see alarming warnings for valid requests.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Pydantic serializer warnings:.*",
            category=UserWarning,
        )
        with client.responses.stream(
            model=_CODEX_CHAT_MODEL,
            store=False,
            instructions=_CODEX_INSTRUCTIONS,
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }],
            tools=[tool],
            tool_choice={
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "image_generation"}],
            },
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) == "image_generation_call":
                        result = getattr(item, "result", None)
                        if isinstance(result, str) and result:
                            image_b64 = result
                elif event_type == "response.image_generation_call.partial_image":
                    partial = getattr(event, "partial_image_b64", None)
                    if isinstance(partial, str) and partial:
                        image_b64 = partial
            final = stream.get_final_response()

    # Final-response sweep covers the case where the stream finished before
    # we observed the ``output_item.done`` event for the image call.
    for item in getattr(final, "output", None) or []:
        if getattr(item, "type", None) == "image_generation_call":
            result = getattr(item, "result", None)
            if isinstance(result, str) and result:
                image_b64 = result

    return image_b64


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAICodexImageGenProvider(ImageGenProvider):
    """gpt-image-2 routed through ChatGPT/Codex OAuth instead of an API key."""

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI (Codex auth)"

    def is_available(self) -> bool:
        if not _read_codex_access_token():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI (Codex auth)",
            "badge": "free",
            "tag": "gpt-image-2 via ChatGPT/Codex OAuth — no API key required",
            "env_vars": [],
            "post_setup_hint": (
                "Sign in with `hermes auth codex` (or `hermes setup` → Codex) "
                "if you haven't already. No API key needed."
            ),
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
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        if not _read_codex_access_token():
            return error_response(
                error=(
                    "No Codex/ChatGPT OAuth credentials available. Run "
                    "`hermes auth codex` (or `hermes setup` → Codex) to sign in."
                ),
                error_type="auth_required",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        try:
            import openai  # noqa: F401
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai-codex",
                aspect_ratio=aspect,
            )

        tier_id, meta, api_quality, size, output_format, output_compression, option_error = _resolve_generation_options(
            aspect=aspect,
            quality=kwargs.get("quality"),
            size=kwargs.get("size"),
            output_format=kwargs.get("output_format"),
            output_compression=kwargs.get("output_compression"),
        )
        if option_error:
            return error_response(
                error=option_error,
                error_type="invalid_argument",
                provider="openai-codex",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        client = _build_codex_client()
        if client is None:
            return error_response(
                error="Could not initialize Codex image client",
                error_type="auth_required",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            b64 = _collect_image_b64(
                client,
                prompt=prompt,
                size=size,
                quality=api_quality,
                output_format=output_format,
                output_compression=output_compression,
            )
        except Exception as exc:
            logger.debug("Codex image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation via Codex auth failed: {exc}",
                error_type="api_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not b64:
            return error_response(
                error="Codex response contained no image_generation_call result",
                error_type="empty_response",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            saved_path = save_b64_image(
                b64,
                prefix=f"openai_codex_{tier_id}",
                extension=output_format,
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai-codex",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai-codex",
            extra={
                "size": size,
                "quality": api_quality,
                "output_format": output_format,
                "output_compression": output_compression,
            },
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — register the Codex-backed image-gen provider."""
    ctx.register_image_gen_provider(OpenAICodexImageGenProvider())
