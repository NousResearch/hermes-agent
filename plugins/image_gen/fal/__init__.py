"""FAL.ai registration adapter. Catalog, payload, submission, managed-gateway selection and
Clarity Upscaler chaining live in :mod:`tools.image_generation_tool`; this plugin imports it at
call time so tests keep patching ``image_tool.*`` and there is one FAL code path."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio
from plugins.image_gen._common import StaticImageGenProvider, catalog_rows

logger = logging.getLogger(__name__)

_PASSTHROUGH_KWARGS = (
    "num_inference_steps", "guidance_scale", "num_images", "output_format", "seed", "upscale",
)


class FalImageGenProvider(StaticImageGenProvider):
    """FAL.ai backend delegating to ``tools.image_generation_tool`` at call time."""

    provider_id = "fal"
    label = "FAL.ai"
    setup = dict(
        name="FAL.ai", badge="paid",
        tag="Pick from flux-2-klein, flux-2-pro, gpt-image, nano-banana-2, nano-banana-pro, etc. — text-to-image & image editing",
        key="FAL_KEY", prompt="FAL API key", url="https://fal.ai/dashboard/keys")

    def is_available(self) -> bool:
        # Direct FAL_KEY or a managed Nous fal-queue origin, per the legacy module.
        import tools.image_generation_tool as _it

        try:
            return bool(_it.check_fal_api_key())
        except Exception:  # noqa: BLE001 — never break the picker
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        from tools.image_generation_catalog import FAL_MODELS
        return catalog_rows(FAL_MODELS)

    def default_model(self) -> Optional[str]:
        from tools.image_generation_catalog import DEFAULT_MODEL
        return DEFAULT_MODEL

    def capabilities(self) -> Dict[str, Any]:
        # Image-to-image depends on the selected model (``edit_endpoint``); upscale works for any.
        import tools.image_generation_tool as _it

        try:
            _model_id, meta = _it._resolve_fal_model()
        except Exception:  # noqa: BLE001
            return {"modalities": ["text"], "max_reference_images": 0}
        if meta.get("edit_endpoint"):
            return {
                "modalities": ["text", "image"],
                "max_reference_images": int(meta.get("max_reference_images") or 1),
                "supports_upscale": True,
            }
        return {"modalities": ["text"], "max_reference_images": 0, "supports_upscale": True}

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Forward to ``image_generate_tool`` and reshape its JSON-string response into the ABC dict."""
        import tools.image_generation_tool as _it

        aspect = resolve_aspect_ratio(aspect_ratio)
        passthrough = {key: kwargs[key] for key in _PASSTHROUGH_KWARGS if kwargs.get(key) is not None}
        # Only forward image-to-image inputs when supplied (no noisy None kwargs).
        if image_url is not None:
            passthrough["image_url"] = image_url
        if reference_image_urls is not None:
            passthrough["reference_image_urls"] = reference_image_urls

        try:
            raw = _it.image_generate_tool(prompt=prompt, aspect_ratio=aspect, **passthrough)
        except Exception as exc:  # noqa: BLE001 — never raise out of generate
            logger.warning("FAL image_generate_tool raised: %s", exc, exc_info=True)
            return {
                "success": False, "image": None, "error": f"FAL image generation failed: {exc}",
                "error_type": type(exc).__name__, "provider": "fal", "prompt": prompt, "aspect_ratio": aspect,
            }

        try:
            response = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001
            response = {"success": False, "image": None, "error": "Invalid JSON from FAL pipeline"}

        if not isinstance(response, dict):
            response = {
                "success": False, "image": None, "error": "FAL pipeline returned a non-dict response",
                "error_type": "provider_contract",
            }
        # Stamp the uniform shape; the legacy pipeline resolves the model internally.
        response.setdefault("provider", "fal")
        response.setdefault("prompt", prompt)
        response.setdefault("aspect_ratio", aspect)
        if "model" not in response:
            try:
                response["model"] = _it._resolve_fal_model()[0]
            except Exception:  # noqa: BLE001
                pass
        return response


def register(ctx) -> None:
    """Plugin entry point — wire ``FalImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(FalImageGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'ImageGenProvider': ('agent.image_gen_provider', 'ImageGenProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
