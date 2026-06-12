"""FAL.ai image generation backend.

Wraps the 18-model FAL catalog (FLUX 2, Z-Image, Nano Banana, GPT
Image 1.5, Recraft, Imagen 4, Qwen, Ideogram, …) as an
:class:`ImageGenProvider` implementation.

The heavy lifting — model catalog, payload construction, request
submission, managed-Nous-gateway selection, Clarity Upscaler chaining
— lives in :mod:`tools.image_generation_tool`. This plugin reaches into
that module via call-time indirection (``import tools.image_generation_tool as _it``)
so:

* the existing test suite (``tests/tools/test_image_generation.py``,
  ``tests/tools/test_managed_media_gateways.py``) keeps patching
  ``image_tool._submit_fal_request`` / ``image_tool.fal_client`` /
  ``image_tool._managed_fal_client`` without modification, and
* there's exactly one canonical FAL code path on disk — the plugin is a
  registration adapter, not a parallel implementation.

See issue #26241 for the migration plan and the
``plugin-extraction-test-patch-compatibility.md`` rules this follows.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    resolve_aspect_ratio,
)

logger = logging.getLogger(__name__)


class FalImageGenProvider(ImageGenProvider):
    """FAL.ai image generation backend.

    Delegates to ``tools.image_generation_tool.image_generate_tool`` so
    the in-tree FAL implementation (model catalog, payload builder,
    managed-gateway selection, Clarity Upscaler chaining) is the single
    source of truth. Everything is resolved at call time via the
    ``_it`` indirection so tests can monkey-patch the legacy module.
    """

    @property
    def name(self) -> str:
        return "fal"

    @property
    def display_name(self) -> str:
        return "FAL.ai"

    def is_available(self) -> bool:
        # Available when direct FAL_KEY is set OR the managed Nous
        # gateway resolves a fal-queue origin. Both checks come from the
        # legacy module so this provider tracks whatever logic ships
        # there.
        import tools.image_generation_tool as _it
        try:
            return bool(_it.check_fal_api_key())
        except Exception:  # noqa: BLE001 — defensive; never break the picker
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        import tools.image_generation_tool as _it
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
                "price": meta.get("price", ""),
            }
            for model_id, meta in _it.FAL_MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        import tools.image_generation_tool as _it
        return _it.DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "FAL.ai",
            "badge": "paid",
            "tag": "Pick from flux-2-klein, flux-2-pro, gpt-image, nano-banana, etc.",
            "env_vars": [
                {
                    "key": "FAL_KEY",
                    "prompt": "FAL API key",
                    "url": "https://fal.ai/dashboard/keys",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image via the legacy FAL pipeline.

        Forwards prompt + aspect_ratio (and any forward-compat extras
        the schema supports) into :func:`tools.image_generation_tool.image_generate_tool`,
        then reshapes its JSON-string response into the provider-ABC
        dict format consumed by ``_dispatch_to_plugin_provider``.
        """
        import tools.image_generation_tool as _it

        aspect = resolve_aspect_ratio(aspect_ratio)
        passthrough = {
            key: kwargs[key]
            for key in (
                "num_inference_steps",
                "guidance_scale",
                "num_images",
                "output_format",
                "seed",
            )
            if key in kwargs and kwargs[key] is not None
        }

        try:
            raw = _it.image_generate_tool(
                prompt=prompt,
                aspect_ratio=aspect,
                **passthrough,
            )
        except Exception as exc:  # noqa: BLE001 — never raise out of generate
            logger.warning("FAL image_generate_tool raised: %s", exc, exc_info=True)
            return {
                "success": False,
                "image": None,
                "error": f"FAL image generation failed: {exc}",
                "error_type": type(exc).__name__,
                "provider": "fal",
                "prompt": prompt,
                "aspect_ratio": aspect,
            }

        try:
            response = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:  # noqa: BLE001
            response = {"success": False, "image": None, "error": "Invalid JSON from FAL pipeline"}

        if not isinstance(response, dict):
            response = {
                "success": False,
                "image": None,
                "error": "FAL pipeline returned a non-dict response",
                "error_type": "provider_contract",
            }

        # Stamp provider/prompt/aspect_ratio so downstream consumers see
        # the uniform shape declared in ``agent.image_gen_provider``.
        response.setdefault("provider", "fal")
        response.setdefault("prompt", prompt)
        response.setdefault("aspect_ratio", aspect)
        # Annotate model best-effort — the legacy pipeline resolves it
        # internally, so query it after the fact for the response shape.
        if "model" not in response:
            try:
                model_id, _meta = _it._resolve_fal_model()
                response["model"] = model_id
            except Exception:  # noqa: BLE001
                pass
        return response


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``FalImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(FalImageGenProvider())
