"""DashScope Qwen-Image generation backend for Hermes Agent.

Exposes Alibaba Cloud DashScope (阿里云百炼) Qwen-Image models as
an :class:`ImageGenProvider` implementation.  Calls DashScope's native
``multimodal-generation`` API directly via ``requests`` — no OpenAI SDK needed.

Supported models
----------------
    qwen-image-2.0-pro      Best quality, complex text rendering (recommended)
    qwen-image-2.0          Accelerated — balanced speed / quality
    qwen-image-max          Highest photorealism, fewest AI artifacts
    qwen-image-plus         Diverse artistic styles

API reference
-------------
https://www.alibabacloud.com/help/en/model-studio/qwen-image-api

Model selection (first hit wins)
--------------------------------
1. ``DASHSCOPE_IMAGE_MODEL`` env var (escape hatch for scripts).
2. ``image_gen.dashscope.model`` in ``config.yaml``.
3. ``image_gen.model`` in ``config.yaml`` (when it matches a known ID).
4. :data:`DEFAULT_MODEL` — ``qwen-image-2.0-pro``.

Endpoint resolution
-------------------
DashScope exposes two API surfaces:

* **OpenAI-compatible**  ``/compatible-mode/v1/chat/completions``  (LLM chat)
* **Native**             ``/api/v1/services/aigc/multimodal-generation/...``  (images)

The user's ``image_gen.base_url`` config typically points at the compatible-mode
endpoint.  :func:`_get_native_endpoint` strips ``/compatible-mode/v1`` and
substitutes ``/api/v1/services/...`` so no extra config is required.

Image caching
-------------
DashScope returns ephemeral URLs that expire after 24 h.  The plugin calls
:func:`agent.image_gen_provider.save_url_image` to materialise the bytes
locally under ``$HERMES_HOME/cache/images/`` at generation time.
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
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model catalog
# =============================================================================
# Each entry provides display metadata for the ``hermes tools`` model picker.
# ``id`` is the key — it doubles as the DashScope API model name.

_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen-image-2.0-pro": {
        "display": "Qwen Image 2.0 Pro",
        "speed": "~15s",
        "strengths": (
            "Best quality, complex text rendering, multi-line layouts, "
            "fine-grained detail. Supports 1-6 images per call."
        ),
        "price": "varies",
    },
    "qwen-image-2.0": {
        "display": "Qwen Image 2.0",
        "speed": "~10s",
        "strengths": "Balanced speed/quality, accelerated generation",
        "price": "varies",
    },
    "qwen-image-max": {
        "display": "Qwen Image Max",
        "speed": "~20s",
        "strengths": "Highest realism, fewer AI artifacts, natural look",
        "price": "varies",
    },
    "qwen-image-plus": {
        "display": "Qwen Image Plus",
        "speed": "~12s",
        "strengths": "Diverse artistic styles, good text rendering",
        "price": "varies",
    },
}

DEFAULT_MODEL = "qwen-image-2.0-pro"


# =============================================================================
# Size tables — DashScope uses "width*height" (note: *, not x).
# =============================================================================
# The qwen-image-2.0 series supports a wider pixel range (up to 2048×2048)
# than the older max/plus series.

_QWEN2_SIZES = {
    "landscape": "2688*1536",   # 16:9
    "square":    "2048*2048",   # 1:1
    "portrait":  "1536*2688",   # 9:16
}

_QWEN_MAX_SIZES = {
    "landscape": "1664*928",
    "square":    "1328*1328",
    "portrait":  "928*1664",
}

# Which model IDs belong to which size family.
_QWEN2_SERIES = frozenset({"qwen-image-2.0-pro", "qwen-image-2.0"})
_QWEN_MAX_SERIES = frozenset({"qwen-image-max", "qwen-image-plus"})


def _size_for(aspect: str, model_id: str) -> str:
    """Pick the ``width*height`` string for *aspect* and *model_id*.

    Falls back to square when *aspect* is unrecognised.
    """
    if model_id in _QWEN2_SERIES:
        return _QWEN2_SIZES.get(aspect, _QWEN2_SIZES["square"])
    return _QWEN_MAX_SIZES.get(aspect, _QWEN_MAX_SIZES["square"])


# =============================================================================
# Config helpers
# =============================================================================


def _load_dashscope_config() -> Dict[str, Any]:
    """Read the ``image_gen`` section from ``config.yaml``.

    Returns an empty dict on any failure so callers never crash on a
    malformed or missing config file.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model ID to use, returning ``(id, meta_dict)``.

    Precedence:
    1. ``DASHSCOPE_IMAGE_MODEL`` env var.
    2. ``image_gen.dashscope.model`` in config.yaml.
    3. ``image_gen.model`` in config.yaml (top-level fallback).
    4. :data:`DEFAULT_MODEL`.
    """
    # 1 — env var escape hatch
    env_override = os.environ.get("DASHSCOPE_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_dashscope_config()

    # 2 — provider-specific model key
    dashscope_cfg = (
        cfg.get("dashscope") if isinstance(cfg.get("dashscope"), dict) else {}
    )
    candidate: Optional[str] = None
    if isinstance(dashscope_cfg, dict):
        value = dashscope_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value

    # 3 — top-level image_gen.model
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    # 4 — default
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# =============================================================================
# Endpoint resolution
# =============================================================================
# This is the trickiest part of the plugin.
#
# DashScope's image generation does NOT live under the OpenAI-compatible
# ``/compatible-mode/v1`` path (that's for chat completions only).  It lives
# under the native ``/api/v1/services/aigc/multimodal-generation/generation``
# endpoint.
#
# Most users configure ``image_gen.base_url`` to the compatible-mode URL
# because that's what they use for the LLM text provider.  If we blindly
# appended ``/services/...`` to that, we'd produce:
#
#     https://dashscope.aliyuncs.com/compatible-mode/v1/services/...  ← 404
#
# So we detect and strip the compatible-mode suffix before building the
# native endpoint.


def _get_native_endpoint() -> str:
    """Build the full DashScope multimodal-generation endpoint URL.

    Resolution order
    ----------------
    1. ``DASHSCOPE_BASE_URL`` env var (operator override).
    2. ``image_gen.base_url`` in ``config.yaml``, with ``/compatible-mode/v1``
       stripped if present.
    3. Default: ``https://dashscope.aliyuncs.com/api/v1`` (China domestic).

    All paths strip the compatible-mode suffix and ensure the URL ends with
    ``/api/v1/services/aigc/multimodal-generation/generation``.
    """
    # ---- 1) env var (highest priority) ----
    env_url = os.environ.get("DASHSCOPE_BASE_URL")
    if env_url:
        base = env_url.rstrip("/")
        if base.endswith("/compatible-mode/v1"):
            base = base[: -len("/compatible-mode/v1")]
        if not base.endswith("/api/v1"):
            if base.endswith("/api"):
                base += "/v1"
            else:
                base += "/api/v1"
        return f"{base}/services/aigc/multimodal-generation/generation"

    # ---- 2) config.yaml image_gen.base_url ----
    try:
        cfg = _load_dashscope_config()
        cfg_url = cfg.get("base_url")
        if isinstance(cfg_url, str) and cfg_url.strip():
            base = cfg_url.strip().rstrip("/")
            if base.endswith("/compatible-mode/v1"):
                base = base[: -len("/compatible-mode/v1")]
            if not base.endswith("/api/v1"):
                if base.endswith("/api"):
                    base += "/v1"
                else:
                    base += "/api/v1"
            return f"{base}/services/aigc/multimodal-generation/generation"
    except Exception:
        pass

    # ---- 3) default China domestic endpoint ----
    return (
        "https://dashscope.aliyuncs.com/api/v1"
        "/services/aigc/multimodal-generation/generation"
    )


# =============================================================================
# Response parsing
# =============================================================================


def _extract_image_url(response_data: Dict[str, Any]) -> Optional[str]:
    """Pull the first image URL from a DashScope API response.

    Expected shape::

        {
            "output": {
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [{"image": "https://dashscope-result-..."}]
                    }
                }]
            },
            "usage": {"image_count": 1, "width": 2048, "height": 2048},
            "request_id": "..."
        }

    Returns ``None`` when the expected keys are missing — the caller
    surfaces a clean ``error_response``.
    """
    try:
        choices = response_data["output"]["choices"]
        if not choices:
            return None
        content = choices[0]["message"]["content"]
        if not content:
            return None
        return content[0]["image"]
    except (KeyError, IndexError, TypeError):
        return None


# =============================================================================
# ImageGenProvider implementation
# =============================================================================


class DashScopeImageGenProvider(ImageGenProvider):
    """DashScope Qwen-Image backend via native multimodal-generation API.

    Implements the full :class:`ImageGenProvider` contract:
    ``name``, ``is_available``, ``list_models``, ``generate``, etc.
    """

    # ---- Provider identity ----

    @property
    def name(self) -> str:
        """Stable short ID — used as ``image_gen.provider`` in config.yaml."""
        return "dashscope"

    @property
    def display_name(self) -> str:
        """Human-readable label for the ``hermes tools`` provider picker."""
        return "DashScope (Qwen Image)"

    # ---- Availability ----

    def is_available(self) -> bool:
        """Only advertise when ``DASHSCOPE_API_KEY`` is set."""
        if not os.environ.get("DASHSCOPE_API_KEY"):
            return False
        return True

    # ---- Model catalogue ----

    def list_models(self) -> List[Dict[str, Any]]:
        """Return entries for the ``hermes tools`` model picker."""
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta.get("price", "varies"),
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    # ---- Setup UI ----

    def get_setup_schema(self) -> Dict[str, Any]:
        """Metadata for the ``hermes tools`` provider configuration UI."""
        return {
            "name": "DashScope (Qwen Image)",
            "badge": "paid",
            "tag": (
                "qwen-image-2.0-pro, qwen-image-2.0, qwen-image-max, "
                "qwen-image-plus — Alibaba Cloud Qwen-Image generation"
            ),
            "env_vars": [
                {
                    "key": "DASHSCOPE_API_KEY",
                    "prompt": "DashScope API key",
                    "url": "https://bailian.console.aliyun.com/?apiKey=1",
                },
            ],
        }

    # ---- Capabilities ----

    def capabilities(self) -> Dict[str, Any]:
        """Text-to-image only (no image editing via this backend)."""
        return {"modalities": ["text"], "max_reference_images": 0}

    # ---- Core: generate() ----

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image from a text prompt via DashScope.

        Parameters
        ----------
        prompt:
            Text description of the desired image (required).
        aspect_ratio:
            ``"landscape"`` | ``"square"`` | ``"portrait"``.
        image_url, reference_image_urls:
            Ignored — DashScope text-to-image does not support
            image-to-image editing through this endpoint.

        Returns
        -------
        dict
            Standard Hermes ``success_response`` or ``error_response``.
        """
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        # -- guard: empty prompt --
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="dashscope",
                aspect_ratio=aspect,
            )

        # -- guard: missing API key --
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return error_response(
                error=(
                    "DASHSCOPE_API_KEY not set. Set it in ~/.hermes/.env "
                    "or via environment variable. Get a key at "
                    "https://bailian.console.aliyun.com/?apiKey=1"
                ),
                error_type="auth_required",
                provider="dashscope",
                aspect_ratio=aspect,
            )

        # -- guard: missing requests library --
        try:
            import requests
        except ImportError:
            return error_response(
                error="requests Python package not installed",
                error_type="missing_dependency",
                provider="dashscope",
                aspect_ratio=aspect,
            )

        # Resolve model, size, and endpoint
        model_id, _meta = _resolve_model()
        size = _size_for(aspect, model_id)
        endpoint = _get_native_endpoint()

        # Build the DashScope-native request payload.
        # Note: ``input.messages`` uses a multimodal chat shape, NOT
        # the OpenAI ``images.generate`` shape.
        payload: Dict[str, Any] = {
            "model": model_id,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            },
            "parameters": {
                "size": size,
                "n": 1,
                "prompt_extend": True,   # let the model optimise the prompt
                "watermark": False,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # -- HTTP call with explicit timeout --
        try:
            resp = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=120,
            )
        except requests.exceptions.Timeout:
            return error_response(
                error="DashScope image generation timed out (120s)",
                error_type="timeout",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.exceptions.RequestException as exc:
            logger.debug("DashScope request failed", exc_info=True)
            return error_response(
                error=f"DashScope request failed: {exc}",
                error_type="network_error",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # -- non-200: surface the error body --
        if resp.status_code != 200:
            err_detail = resp.text[:500] if resp.text else "(empty body)"
            logger.debug(
                "DashScope returned %d: %s", resp.status_code, err_detail
            )
            return error_response(
                error=(
                    f"DashScope image generation failed "
                    f"({resp.status_code}): {err_detail}"
                ),
                error_type="api_error",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # -- parse JSON --
        try:
            data = resp.json()
        except ValueError:
            return error_response(
                error="DashScope returned non-JSON response",
                error_type="parse_error",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Some DashScope errors return HTTP 200 with an error code in the body.
        if "code" in data and data.get("code") != "":
            code = data.get("code", "unknown")
            message = data.get("message", "Unknown error")
            return error_response(
                error=f"DashScope API error [{code}]: {message}",
                error_type="api_error",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # -- extract image URL from the response --
        image_url_resp = _extract_image_url(data)
        if not image_url_resp:
            return error_response(
                error="DashScope returned no image URL in response",
                error_type="empty_response",
                provider="dashscope",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # -- download and cache locally (DashScope URLs expire in 24 h) --
        try:
            saved_path = save_url_image(
                image_url_resp, prefix=f"dashscope_{model_id}"
            )
            image_ref = str(saved_path)
        except Exception as exc:
            logger.warning(
                "Could not cache DashScope image %s (%s); using bare URL.",
                image_url_resp,
                exc,
            )
            image_ref = image_url_resp

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="dashscope",
            modality="text",
            extra={"size": size},
        )


# =============================================================================
# Plugin entry point — called by Hermes at plugin load time.
# =============================================================================


def register(ctx) -> None:
    """Register this provider with the Hermes image-gen registry.

    Hermes calls this function once when the plugin is loaded.
    ``ctx.register_image_gen_provider()`` makes the provider available
    for ``image_gen.provider: dashscope`` in config.yaml.
    """
    ctx.register_image_gen_provider(DashScopeImageGenProvider())
