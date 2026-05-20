"""inference.sh image generation backend.

Exposes multiple image generation models as an :class:`ImageGenProvider`
implementation — one API key, seven model families:

    pruna/p-image                      ~5s     Fastest and cheapest ($0.0001/image)
    pruna/flux-dev-lora                ~15s    FLUX with LoRA style support
    bytedance/seedream-4-5             ~30s    4K cinematic, text-in-image
    openai/gpt-image-2                 ~30s    OpenAI, editing + inpainting
    google/gemini-3-pro-image-preview  ~30s    Google, high fidelity
    xai/grok-imagine-image             ~15s    xAI, fast creative

Selection precedence (first hit wins):

1. ``model=`` kwarg from the tool call
2. ``INFERENCE_IMAGE_MODEL`` env var
3. ``image_gen.inference_sh.model`` in ``config.yaml``
4. ``image_gen.model`` in ``config.yaml`` (when it's one of our model IDs)
5. :data:`DEFAULT_MODEL` — ``pruna/flux-dev-lora``

Authentication via ``INFERENCE_API_KEY``. Output is an HTTPS URL from the
inference.sh CDN; the gateway downloads and delivers it.
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
    success_response,
)

logger = logging.getLogger(__name__)

PROVIDER_NAME = "inference-sh"


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    "pruna/p-image": {
        "display": "P-Image",
        "speed": "~5s",
        "strengths": "Fastest and cheapest image generation ($0.0001/image)",
        "app": "pruna/p-image",
    },
    "pruna/flux-dev-lora": {
        "display": "FLUX Dev LoRA",
        "speed": "~15s",
        "strengths": "Quality generation with LoRA style support",
        "app": "pruna/flux-dev-lora",
    },
    "bytedance/seedream-4-5": {
        "display": "Seedream 4.5",
        "speed": "~30s",
        "strengths": "4K cinematic quality, reliable text rendering in images",
        "app": "bytedance/seedream-4-5",
    },
    "openai/gpt-image-2": {
        "display": "GPT-Image-2",
        "speed": "~30s",
        "strengths": "Editing, inpainting, multi-reference composition",
        "app": "openai/gpt-image-2",
    },
    "google/gemini-3-pro-image-preview": {
        "display": "Gemini 3 Pro Image",
        "speed": "~30s",
        "strengths": "Google. High fidelity, strong prompt adherence",
        "app": "google/gemini-3-pro-image-preview",
    },
    "xai/grok-imagine-image": {
        "display": "Grok Imagine",
        "speed": "~15s",
        "strengths": "xAI. Fast creative generation",
        "app": "xai/grok-imagine-image",
    },
}

DEFAULT_MODEL = "pruna/flux-dev-lora"

# Map image_gen aspect ratios to aspect_ratio values inference.sh apps accept.
# The ImageGenProvider ABC uses "landscape"/"square"/"portrait" while
# inference.sh apps accept ratio strings like "16:9".
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


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


def _resolve_model(explicit: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    candidates: List[Optional[str]] = []
    candidates.append(explicit)
    candidates.append(os.environ.get("INFERENCE_IMAGE_MODEL"))

    cfg = _load_image_gen_section()
    infsh_cfg = cfg.get("inference_sh") if isinstance(cfg.get("inference_sh"), dict) else {}
    if isinstance(infsh_cfg, dict):
        candidates.append(infsh_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for c in candidates:
        if isinstance(c, str) and c.strip() and c.strip() in _MODELS:
            mid = c.strip()
            return mid, _MODELS[mid]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# SDK lazy import
# ---------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    """Return a cached ``inferencesh.inference`` client instance."""
    global _client
    if _client is not None:
        return _client
    from inferencesh import inference  # type: ignore

    api_key = os.environ.get("INFERENCE_API_KEY", "").strip()
    _client = inference(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class InferenceShImageGenProvider(ImageGenProvider):
    """inference.sh multi-model image generation backend.

    Uses the ``inferencesh`` Python SDK to call image generation apps
    on the inference.sh cloud. One API key covers all models.
    """

    @property
    def name(self) -> str:
        return "inference-sh"

    @property
    def display_name(self) -> str:
        return "inference.sh"

    def is_available(self) -> bool:
        # Only check for the API key — the inferencesh SDK is lazy-installed
        # on first generate() call via tools/lazy_deps.py.
        return bool(os.environ.get("INFERENCE_API_KEY", "").strip())

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
            "name": "inference.sh",
            "badge": "paid",
            "tag": (
                "one key for any model — FLUX, Seedream 4.5, GPT-Image-2, "
                "Gemini, Grok Imagine, P-Image (inference.sh)"
            ),
            "env_vars": [
                {
                    "key": "INFERENCE_API_KEY",
                    "prompt": "inference.sh API key",
                    "url": "https://app.inference.sh/settings/keys",
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
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        if not os.environ.get("INFERENCE_API_KEY", "").strip():
            return error_response(
                error=(
                    "INFERENCE_API_KEY not set. Run `hermes tools` -> Image "
                    "Generation -> inference.sh to configure, or sign up at "
                    "https://inference.sh"
                ),
                error_type="auth_required",
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        try:
            from tools.lazy_deps import ensure
            ensure("image.inference_sh", prompt=False)
        except Exception:
            pass  # Best effort

        try:
            client = _get_client()
        except ImportError:
            return error_response(
                error=(
                    "inferencesh Python package not installed "
                    "(pip install inferencesh)"
                ),
                error_type="missing_dependency",
                provider=PROVIDER_NAME,
                aspect_ratio=aspect,
            )

        # Resolve model from kwargs or config
        explicit_model = kwargs.get("model")
        model_id, meta = _resolve_model(explicit_model)
        app_id = meta["app"]

        # Build input payload
        input_data: Dict[str, Any] = {"prompt": prompt}

        # Map hermes aspect ratio to inference.sh format
        mapped_aspect = _ASPECT_MAP.get(aspect)
        if mapped_aspect:
            input_data["aspect_ratio"] = mapped_aspect

        # Pass through image for editing workflows (e.g. GPT-Image-2, Reve)
        image = kwargs.get("image") or kwargs.get("image_url")
        if image:
            input_data["image"] = image

        try:
            result = client.tasks.run({
                "app": app_id,
                "input": input_data,
            })
        except Exception as exc:
            logger.debug("inference.sh image generation failed", exc_info=True)
            return error_response(
                error=f"inference.sh image generation failed: {exc}",
                error_type="api_error",
                provider=PROVIDER_NAME,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        # Extract image URL from task output
        output = result.get("output") if isinstance(result, dict) else None
        if not isinstance(output, dict):
            output = result if isinstance(result, dict) else {}

        url: Optional[str] = None
        for key in ("url", "image_url", "image", "output_url"):
            val = output.get(key)
            if isinstance(val, str) and val.startswith("http"):
                url = val
                break
            if isinstance(val, dict):
                url = val.get("url")
                if url:
                    break
        # Check for list-of-files pattern
        if not url:
            files = output.get("files") or output.get("outputs") or output.get("images")
            if isinstance(files, list) and files:
                first = files[0]
                if isinstance(first, str) and first.startswith("http"):
                    url = first
                elif isinstance(first, dict):
                    url = first.get("url")

        if not url:
            return error_response(
                error="inference.sh returned no image URL in response",
                error_type="empty_response",
                provider=PROVIDER_NAME,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"app": app_id}
        task_id = result.get("id") if isinstance(result, dict) else None
        if task_id:
            extra["task_id"] = task_id

        return success_response(
            image=url,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=PROVIDER_NAME,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``InferenceShImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(InferenceShImageGenProvider())
