"""DeepInfra image generation (FLUX, Qwen-Image-Edit, …) via the OpenAI-compatible
``/v1/openai/images/generations`` endpoint. The catalog is fully dynamic (``image-gen``-tagged
models from :func:`hermes_cli.models._fetch_deepinfra_models_by_tag`; no ids hardcoded).
Selection: ``DEEPINFRA_IMAGE_MODEL`` → ``image_gen.deepinfra.model`` → first live model;
when all are absent ``generate()`` errors rather than guessing."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.secret_scope import get_secret
from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from plugins.image_gen._common import (
    StaticImageGenProvider, error_factory, import_openai, load_image_gen_config, materialize_image,
    prompt_required_error, size_for)

logger = logging.getLogger(__name__)


def _live_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch ``image-gen``-tagged models from the DeepInfra catalog."""
    try:
        from hermes_cli.models import _fetch_deepinfra_models_by_tag
    except Exception as exc:
        logger.debug("Cannot import _fetch_deepinfra_models_by_tag: %s", exc)
        return None
    return _fetch_deepinfra_models_by_tag("image-gen")


def _format_catalog_row(item: Dict[str, Any]) -> Dict[str, Any]:
    """Picker row for a catalog item."""
    mid = item.get("id", "")
    metadata = item.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    row: Dict[str, Any] = {
        "id": mid, "display": mid.split("/", 1)[-1], "strengths": metadata.get("description", ""),
    }
    pricing = metadata.get("pricing")
    if isinstance(pricing, dict) and pricing.get("per_image_unit") is not None:
        try:
            row["price"] = f"${float(pricing['per_image_unit']):.4f}/image"
        except (TypeError, ValueError):
            pass
    for key in ("default_width", "default_height", "default_iterations"):
        if metadata.get(key) is not None:
            row[key] = metadata[key]
    return row


def _resolve_model(catalog: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[str]:
    """env > config > first live result, else None (``cfg`` = loaded ``image_gen.deepinfra``)."""
    env_override = os.environ.get("DEEPINFRA_IMAGE_MODEL", "").strip()
    if env_override:
        return env_override
    cfg_model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()
    first = catalog[0].get("id") if catalog else None
    return first if isinstance(first, str) and first else None


class DeepInfraImageGenProvider(StaticImageGenProvider):
    """DeepInfra ``images.generations`` backend; catalog discovered live by the ``image-gen`` tag."""

    provider_id = "deepinfra"
    label = "DeepInfra"
    setup = dict(
        name="DeepInfra", badge="paid", tag="FLUX, Qwen-Image, … — live catalog from api.deepinfra.com",
        key="DEEPINFRA_API_KEY", prompt="DeepInfra API key", url="https://deepinfra.com/dash/api_keys",
    )

    def is_available(self) -> bool:
        return bool((get_secret("DEEPINFRA_API_KEY", "") or "").strip())

    def list_models(self) -> List[Dict[str, Any]]:
        return [_format_catalog_row(item) for item in _live_models() or []]

    def default_model(self) -> Optional[str]:
        rows = self.list_models()
        return rows[0].get("id") if rows else None

    def capabilities(self) -> Dict[str, Any]:
        """DeepInfra's OpenAI-compatible generation surface is text-only."""
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, **kwargs: Any) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        fail = error_factory("deepinfra", aspect)
        if kwargs.get("image_url") or kwargs.get("reference_image_urls"):
            return fail(
                "DeepInfra image generation is text-to-image only in this "
                "backend; image_url and reference_image_urls are unsupported.",
                "modality_unsupported", prompt=prompt)
        if not prompt:
            return prompt_required_error("deepinfra", aspect)
        api_key = (get_secret("DEEPINFRA_API_KEY", "") or "").strip()
        if not api_key:
            return fail(
                "DEEPINFRA_API_KEY not set. Run `hermes tools` → Image "
                "Generation → DeepInfra to configure, or `hermes setup` "
                "to add the key.",
                "auth_required")
        di_cfg = load_image_gen_config("deepinfra")
        model_id = _resolve_model(_live_models() or [], di_cfg)
        if not model_id:
            return fail(
                "No DeepInfra image-gen model available. Pin one in "
                "config.yaml under image_gen.deepinfra.model, set "
                "DEEPINFRA_IMAGE_MODEL, or check connectivity to "
                "api.deepinfra.com so the live catalog can be fetched.",
                "no_model_available", prompt=prompt)
        size = size_for(aspect)
        from hermes_cli.models import deepinfra_base_url

        # The openai SDK supplies retry, timeout and error mapping.
        openai, err = import_openai("deepinfra", aspect)
        if err:
            return err
        fail = error_factory("deepinfra", aspect, model=model_id, prompt=prompt)
        client = openai.OpenAI(api_key=api_key, base_url=deepinfra_base_url(di_cfg))
        try:
            response = client.images.generate(model=model_id, prompt=prompt, size=size, n=1)
        except Exception as exc:
            logger.debug("DeepInfra image generation failed", exc_info=True)
            return fail(f"DeepInfra image generation failed: {exc}", "api_error")
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        data = getattr(response, "data", None) or []
        if not data:
            return fail("DeepInfra returned no image data", "empty_response")
        first = data[0]
        # Prefix drops ``vendor/`` and colons (single path component on every OS); delivery URLs
        # are short-lived, so materialise locally best-effort.
        image_ref, err = materialize_image(
            getattr(first, "b64_json", None), getattr(first, "url", None),
            prefix=f"deepinfra_{model_id.split('/', 1)[-1].replace(':', '_')}", label="DeepInfra",
            provider="deepinfra", model=model_id, prompt=prompt, aspect=aspect,
            on_url_fail=lambda exc: logger.debug(
                "DeepInfra: caching delivery URL failed (%s); returning URL", exc))
        if err:
            return err
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="deepinfra",
            extra={"size": size})


def register(ctx) -> None:
    """Plugin entry point — wire ``DeepInfraImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(DeepInfraImageGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'ImageGenProvider': ('agent.image_gen_provider', 'ImageGenProvider'),
    'error_response': ('agent.image_gen_provider', 'error_response'),
    'save_b64_image': ('agent.image_gen_provider', 'save_b64_image'),
    'save_url_image': ('agent.image_gen_provider', 'save_url_image'),
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
