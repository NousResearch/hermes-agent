"""xAI ``grok-imagine-image`` backend: text-to-image (``/v1/images/generations``) and editing
(``/v1/images/edits``), base64 output saved to cache. Selection: ``model`` kwarg →
``XAI_IMAGE_MODEL`` → ``image_gen.xai.model`` → :data:`DEFAULT_MODEL`."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from plugins.image_gen._common import (
    StaticImageGenProvider, catalog_rows, collect_source_images, error_factory,
    load_image_gen_config, materialize_image, post_json)
from tools.xai_http import (
    build_xai_storage_options, hermes_xai_user_agent, maybe_mark_xai_storage_notice_seen,
    read_xai_imagine_storage_config, resolve_xai_http_credentials, xai_storage_notice_text)

logger = logging.getLogger(__name__)

_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-imagine-image": {
        "display": "Grok Imagine Image", "speed": "~5-10s", "strengths": "Fast, high-quality",
    },
    "grok-imagine-image-2.0": {
        "display": "Grok Imagine Image 2.0", "speed": "~10-20s",
        "strengths": "Typography/layout-aware; legible small text; strongest quality.",
    },
    "grok-imagine-image-quality": {
        "display": "Grok Imagine Image (Quality)", "speed": "~10-20s",
        "strengths": "Higher fidelity / detail; slower than the standard model.",
    },
}

DEFAULT_MODEL = "grok-imagine-image"
# xAI documents the quality model as the edit-capable baseline.
_EDIT_FALLBACK_MODEL = "grok-imagine-image-quality"

# Live catalog cache ``(models, fetched_monotonic)``: ``/image-generation-models`` is the source of
# truth (new models need no code change); ``_MODELS`` is the offline fallback + curated text.
_LIVE_CACHE: Optional[Tuple[Dict[str, Dict[str, Any]], float]] = None
_LIVE_CACHE_TTL = 300.0
_LIVE_TIMEOUT = 10.0

_XAI_ASPECT_RATIOS = {
    "landscape": "16:9", "square": "1:1", "portrait": "9:16",
    "4:3": "4:3", "3:4": "3:4", "3:2": "3:2", "2:3": "2:3",
}
_XAI_RESOLUTIONS = {"1k", "2k"}
DEFAULT_RESOLUTION = "1k"
_MAX_SOURCE_IMAGES = 3
_REQUEST_TIMEOUT = 120
_REMOTE_PREFIXES = ("http://", "https://", "data:")
_FILE_OUTPUT_EXTRA_KEYS = (
    "filename", "expires_at", "public_url_expires_at", "public_url_error", "storage_error")


def _base_url(creds: Dict[str, Any]) -> str:
    return str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")


def _fetch_live_models() -> Dict[str, Dict[str, Any]]:
    """``{model_id: {"input_modalities", "aliases"}}`` from the live endpoint; raises on failure."""
    creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("no xAI credentials")
    response = requests.get(
        f"{_base_url(creds)}/image-generation-models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": hermes_xai_user_agent()},
        timeout=_LIVE_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    out: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("models") or payload.get("data") or []:
        model_id = entry.get("id") or entry.get("name") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            out[model_id.strip()] = {
                "input_modalities": entry.get("input_modalities") or [], "aliases": entry.get("aliases") or [],
            }
    return out


def _live_models() -> Dict[str, Dict[str, Any]]:
    """Cached live catalog (``{}`` when unreachable)."""
    global _LIVE_CACHE
    if _LIVE_CACHE is not None and time.monotonic() - _LIVE_CACHE[1] < _LIVE_CACHE_TTL:
        return _LIVE_CACHE[0]
    try:
        live = _fetch_live_models()
    except Exception as exc:  # noqa: BLE001 - offline/unauth → static fallback
        logger.debug("xAI live image model catalog unavailable: %s", exc)
        live = {}
    _LIVE_CACHE = (live, time.monotonic())
    return live


def _catalog() -> Dict[str, Dict[str, Any]]:
    """Live ids + curated metadata (unknown live models get generic text; curated entries the live
    list omits are kept); the static table alone when the API is unreachable."""
    live = _live_models()
    if not live:
        return dict(_MODELS)
    merged: Dict[str, Dict[str, Any]] = {}
    for model_id in live:
        meta = _MODELS.get(model_id) or {
            "display": model_id, "speed": "", "strengths": "New xAI Imagine model (from live xAI catalog)",
        }
        merged[model_id] = {**meta, "input_modalities": live[model_id].get("input_modalities") or []}
    for model_id, meta in _MODELS.items():
        merged.setdefault(model_id, dict(meta))
    return merged


def _configured_model() -> Optional[str]:
    value = load_image_gen_config("xai").get("model")
    return value if isinstance(value, str) else None


def _resolve_model(caller_model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """caller kwarg → ``XAI_IMAGE_MODEL`` → config → default, validated against the merged catalog."""
    catalog = _catalog()
    for candidate in (caller_model, os.environ.get("XAI_IMAGE_MODEL"), _configured_model()):
        if candidate and candidate in catalog:
            return candidate, catalog[candidate]
    return DEFAULT_MODEL, catalog.get(DEFAULT_MODEL, _MODELS[DEFAULT_MODEL])


def _resolve_edit_model(caller_model: Optional[str] = None) -> str:
    """Edit model: an explicit selection that accepts image input, else the documented quality baseline."""
    catalog = _catalog()
    explicit = caller_model or os.environ.get("XAI_IMAGE_MODEL") or _configured_model()
    if explicit and explicit in catalog and "image" in (catalog[explicit].get("input_modalities") or []):
        return explicit
    return _EDIT_FALLBACK_MODEL


def _resolve_resolution() -> str:
    res = load_image_gen_config("xai").get("resolution")
    return res if isinstance(res, str) and res in _XAI_RESOLUTIONS else DEFAULT_RESOLUTION


def _xai_image_field(source: str) -> Dict[str, str]:
    """Edit ``image`` field: URL / data URI pass through; local paths are inlined as ``data:`` URIs."""
    source = source.strip()
    if source.lower().startswith(_REMOTE_PREFIXES):
        return {"url": source, "type": "image_url"}
    import base64

    from agent.file_safety import raise_if_read_blocked  # credential-read guard before local bytes

    raise_if_read_blocked(source)
    with open(os.path.expanduser(source), "rb") as fh:  # windows-footgun: ok
        raw = fh.read()
    ext = (os.path.splitext(source)[1].lstrip(".") or "png").lower()
    if ext == "jpg":
        ext = "jpeg"
    return {"url": f"data:image/{ext};base64,{base64.b64encode(raw).decode('utf-8')}", "type": "image_url"}


def _check_source_images(
    source_images: List[str], image_url: Optional[str], fail: Any
) -> Optional[Dict[str, Any]]:
    """Edit-request guard: at most 3 sources, each a remote URL/data URI or an existing local file."""
    if len(source_images) > _MAX_SOURCE_IMAGES:
        return fail(
            f"xAI image editing supports at most {_MAX_SOURCE_IMAGES} source images", "too_many_references",
        )
    for index, source in enumerate(source_images):
        if source.lower().startswith(_REMOTE_PREFIXES) or Path(source).expanduser().is_file():
            continue
        is_primary = index == 0 and image_url and image_url.strip() == source
        field = "image_url" if is_primary else "reference_image_urls"
        return fail(
            f"{field} must be a public HTTPS URL or data URI "
            "(e.g. the `image`/`public_url` from a prior Imagine result)",
            "invalid_image_url")
    return None


class XAIImageGenProvider(StaticImageGenProvider):
    """xAI ``grok-imagine-image`` backend."""

    provider_id = "xai"
    label = "xAI (Grok)"

    def is_available(self) -> bool:
        return bool(resolve_xai_http_credentials().get("api_key"))

    def list_models(self) -> List[Dict[str, Any]]:
        return catalog_rows(_catalog(), ("display", "speed", "strengths"))

    def default_model(self) -> Optional[str]:
        # First live/static catalog row (inherited ImageGenProvider behaviour).
        return next(iter(_catalog()), None)

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth goes through the shared ``xai_grok`` post_setup hook (same OAuth-or-key choice everywhere).
        storage_notice = xai_storage_notice_text("image_gen")
        tag = "grok-imagine-image - text-to-image & image editing; uses xAI Grok OAuth or XAI_API_KEY"
        if storage_notice:
            tag += f". {storage_notice}"
        return {
            "name": "xAI Grok Imagine (image)", "badge": "paid", "tag": tag, "env_vars": [],
            "post_setup": "xai_grok",
        }

    def capabilities(self) -> Dict[str, Any]:
        # /v1/images/edits accepts up to 3 total source images.
        return {
            "modalities": ["text", "image"], "max_reference_images": 2,
            "max_source_images": _MAX_SOURCE_IMAGES,
        }

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Text-to-image, or editing via ``/v1/images/edits`` (JSON body — xAI does not support the
        SDK's multipart ``images.edit()``) when source images are supplied."""
        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        provider_name = str(creds.get("provider") or "xai").strip() or "xai"
        if not api_key:
            return error_factory(provider_name, aspect_ratio)(
                "No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.",
                "missing_api_key")

        model_id, meta = _resolve_model(kwargs.get("model"))
        aspect = resolve_aspect_ratio(aspect_ratio)
        xai_res = _resolve_resolution()
        source_images = collect_source_images(image_url, reference_image_urls)
        edit_fail = error_factory(provider_name, aspect, model=_EDIT_FALLBACK_MODEL, prompt=prompt)
        err = _check_source_images(source_images, image_url, edit_fail)
        if err:
            return err
        is_edit = bool(source_images)

        headers = {
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        }
        base_url = _base_url(creds)
        storage_options = build_xai_storage_options(
            "image_gen", filename_prefix="hermes-xai-image", extension="png")
        storage_notice = maybe_mark_xai_storage_notice_seen("image_gen")
        storage_cfg = read_xai_imagine_storage_config("image_gen")

        if is_edit:
            model_id = _resolve_edit_model(kwargs.get("model"))
            try:
                image_fields = [_xai_image_field(source) for source in source_images]
            except Exception as exc:
                return edit_fail(f"Could not load source image for editing: {exc}", "io_error", model=model_id)
            payload: Dict[str, Any] = {"model": model_id, "prompt": prompt}
            if len(image_fields) == 1:
                payload["image"] = image_fields[0]
            else:
                payload["images"] = image_fields
            endpoint_url = f"{base_url}/images/edits"
        else:
            payload = {
                "model": model_id, "prompt": prompt, "aspect_ratio": _XAI_ASPECT_RATIOS.get(aspect, "1:1"),
                "resolution": xai_res,
            }
            endpoint_url = f"{base_url}/images/generations"
        if storage_options is not None:
            payload["storage_options"] = storage_options

        fail = error_factory(provider_name, aspect, model=model_id, prompt=prompt)
        result, failure = post_json(
            endpoint_url, headers=headers, payload=payload, timeout=_REQUEST_TIMEOUT, label="xAI")
        if failure:
            if failure.kind == "http":
                logger.error("xAI image gen failed (%d): %s", failure.status, failure.message)
            return fail(failure.error, failure.error_type)

        # data[0] carries b64_json / url, plus file_output when storage_options were requested.
        data = result.get("data", [])
        if not data:
            return fail("xAI returned no image data", "empty_response")
        first = data[0]
        file_output = first.get("file_output") if isinstance(first, dict) else None
        file_output = file_output if isinstance(file_output, dict) else {}
        public_url = file_output.get("public_url")
        public_url = public_url if isinstance(public_url, str) else None
        if public_url:
            image_ref = public_url
        else:
            # ``imgen.x.ai/xai-tmp-*`` URLs 404 within minutes; materialise locally for a stable path.
            image_ref, err = materialize_image(
                first.get("b64_json"), first.get("url"), prefix=f"xai_{model_id}", label="xAI", provider="xai",
                model=model_id, prompt=prompt, aspect=aspect, log=logger)
            if err:
                return err

        extra: Dict[str, Any] = {"storage_enabled": bool(storage_cfg["enabled"])}
        if not is_edit:
            extra["resolution"] = xai_res
        if storage_notice:
            extra["storage_notice"] = storage_notice
        if public_url:
            extra["public_url"] = public_url
        extra.update({key: file_output[key] for key in _FILE_OUTPUT_EXTRA_KEYS if key in file_output})
        if result.get("usage"):
            extra["usage"] = result["usage"]
        return success_response(
            image=image_ref, model=model_id, prompt=prompt, aspect_ratio=aspect, provider="xai",
            modality="image" if is_edit else "text", extra=extra)


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(XAIImageGenProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'ImageGenProvider': ('agent.image_gen_provider', 'ImageGenProvider'),
    'error_response': ('agent.image_gen_provider', 'error_response'),
    'normalize_reference_images': ('agent.image_gen_provider', 'normalize_reference_images'),
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
