"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2``) with a
different ``quality`` parameter. Output is base64 JSON → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
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
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
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

DEFAULT_MODEL = "gpt-image-2-medium"

_CODEX_CHAT_MODEL = "gpt-5.4"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_CODEX_INSTRUCTIONS = (
    "You are an assistant that must fulfill image generation requests by using "
    "the image_generation tool when provided."
)

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _read_codex_access_token() -> Optional[str]:
    """Read a usable Codex OAuth token from Hermes auth state."""
    try:
        from agent.auxiliary_client import _read_codex_access_token as _reader

        token = _reader()
        return str(token).strip() if isinstance(token, str) and token.strip() else None
    except Exception as exc:
        logger.debug("Could not resolve Codex access token for image generation: %s", exc)
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


def _collect_codex_image_b64(client: Any, *, prompt: str, size: str, quality: str) -> Optional[str]:
    """Generate via Codex Responses image tool and return the final base64 image."""
    image_b64: Optional[str] = None
    with client.responses.stream(
        model=_CODEX_CHAT_MODEL,
        store=False,
        instructions=_CODEX_INSTRUCTIONS,
        input=[{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}],
        }],
        tools=[{
            "type": "image_generation",
            "model": API_MODEL,
            "size": size,
            "quality": quality,
            "output_format": "png",
            "background": "opaque",
            "partial_images": 1,
        }],
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

    for item in getattr(final, "output", None) or []:
        if getattr(item, "type", None) == "image_generation_call":
            result = getattr(item, "result", None)
            if isinstance(result, str) and result:
                image_b64 = result

    return image_b64


def _load_openai_config() -> Dict[str, Any]:
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
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI ``images.generate`` backend — gpt-image-2 at low/medium/high."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not (os.environ.get("OPENAI_API_KEY") or _read_codex_access_token()):
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
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 at low/medium/high quality tiers (API key or ChatGPT/Codex auth)",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key (optional if ChatGPT/Codex auth is already configured)",
                    "url": "https://platform.openai.com/api-keys",
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
                provider="openai",
                aspect_ratio=aspect,
            )

        api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
        codex_token = _read_codex_access_token()
        use_codex = not api_key_present and bool(codex_token)

        if not api_key_present and not codex_token:
            return error_response(
                error=(
                    "Neither OPENAI_API_KEY nor ChatGPT/Codex OAuth auth is available. "
                    "Run `hermes tools` → Image Generation → OpenAI to configure, "
                    "or authenticate Hermes with OpenAI Codex/ChatGPT first."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        # gpt-image-2 returns b64_json unconditionally and REJECTS
        # ``response_format`` as an unknown parameter. Don't send it.
        payload: Dict[str, Any] = {
            "model": API_MODEL,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "quality": meta["quality"],
        }

        b64 = None
        url = None
        revised_prompt = None

        if use_codex:
            try:
                client = _build_codex_client()
                if client is None:
                    raise RuntimeError("Could not initialize Codex image client")
                b64 = _collect_codex_image_b64(
                    client,
                    prompt=prompt,
                    size=size,
                    quality=meta["quality"],
                )
            except Exception as exc:
                logger.debug("Codex-backed OpenAI image generation failed", exc_info=True)
                return error_response(
                    error=f"OpenAI image generation via Codex auth failed: {exc}",
                    error_type="api_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        else:
            try:
                client = openai.OpenAI()
                response = client.images.generate(**payload)
            except Exception as exc:
                logger.debug("OpenAI image generation failed", exc_info=True)
                return error_response(
                    error=f"OpenAI image generation failed: {exc}",
                    error_type="api_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            data = getattr(response, "data", None) or []
            if not data:
                return error_response(
                    error="OpenAI returned no image data",
                    error_type="empty_response",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            first = data[0]
            b64 = getattr(first, "b64_json", None)
            url = getattr(first, "url", None)
            revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Defensive — gpt-image-2 returns b64 today, but fall back
            # gracefully if the API ever changes.
            image_ref = url
        else:
            return error_response(
                error="OpenAI response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {
            "size": size,
            "quality": meta["quality"],
            "auth_source": "codex" if use_codex else "api_key",
        }
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
