"""Capability answers for models served by the managed runtime.

Capability lookups (vision, and whatever comes next) consult cloud-shaped
catalogs that have never heard of a local GGUF, so a vision-capable local
model reads as text-only and images detour to an auxiliary cloud model —
the wrong behavior twice over for a local-first user (broken feature, and
a screenshot silently leaving the machine).

The managed runtime can answer from ground truth instead, best source
first:

1. The RUNNING child's /props: llama-server reports a ``modalities`` block
   when a vision projector is loaded. The server that will receive the
   image says whether it can see — no inference, no catalog.
2. The catalog entry's declared capability (the ``vision`` tag + mmproj
   asset) for staged-but-unloaded models: what the model WILL support once
   its projector loads beside it.
3. None — not one of ours, or nothing known; the caller falls through to
   its other sources.
"""

from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

_LLAMACPP_ALIASES = frozenset({"llamacpp", "llama.cpp", "llama-cpp"})

# Image formats the managed server's decoder actually handles. llama.cpp
# decodes with stb_image: PNG/JPEG/GIF/BMP yes, WebP NO — and a WebP part
# fails SILENTLY (no HTTP error, no log line; the model just never sees an
# image and confabulates a description). Anything outside this set must be
# transcoded before the request. Measured against the live server: the
# same red square answered 'Red' as PNG and 'Unseen' as WebP.
ACCEPTED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg"})


def is_managed_provider(provider: str, base_url: str = "") -> bool:
    """True when this provider/base_url pair points at the managed server.
    ``custom`` only counts when the base_url IS the managed endpoint —
    background lookups must never claim someone else's custom server."""
    p = (provider or "").strip().lower()
    if p in _LLAMACPP_ALIASES:
        return True
    if p == "custom" and base_url:
        try:
            from hermes_cli.local_runtime.growth import is_managed_endpoint

            return is_managed_endpoint(base_url)
        except Exception:  # noqa: BLE001
            return False
    return False


def _props_modalities(model_id: str) -> "bool | None":
    """Ask the running server whether this loaded child sees images.
    None when the server is down, the model isn't loaded, or the build
    doesn't report modalities."""
    try:
        from hermes_cli.local_runtime.endpoint import _state_endpoint

        state = _state_endpoint()
        if state is None:
            return None
        base = state["base_url"].rsplit("/v1", 1)[0]
        req = urllib.request.Request(
            f"{base}/props?model={model_id}",
            headers={"Authorization": f"Bearer {state.get('api_key', '')}"})
        with urllib.request.urlopen(req, timeout=3) as r:
            props = json.load(r)
        modalities = props.get("modalities")
        if isinstance(modalities, dict) and "vision" in modalities:
            return bool(modalities["vision"])
        return None
    except Exception:  # noqa: BLE001
        return None


def managed_model_supports_vision(model_id: str) -> "bool | None":
    """Ground-truth vision capability for a staged model, or None when the
    model isn't ours / nothing is known (caller keeps falling through)."""
    if not model_id:
        return None

    # Only answer for models actually staged with us.
    try:
        from hermes_cli.local_runtime.bootstrap import staged_model_ids

        if model_id not in staged_model_ids():
            return None
    except Exception:  # noqa: BLE001
        return None

    live = _props_modalities(model_id)
    if live is not None:
        return live

    # Staged but not loaded (or an older server build): the catalog knows
    # whether this model ships a vision projector.
    try:
        from hermes_cli.local_runtime.bootstrap import assets_dir
        from hermes_cli.local_runtime.catalog import find_entry_for_model

        hit = find_entry_for_model(model_id)
        if hit is None:
            return None
        entry = hit[0]
        if entry.mmproj is None:
            return False
        # Capability requires the projector to actually be on disk — a
        # model downloaded before its mmproj (partial delete, old layout)
        # genuinely cannot see.
        return (assets_dir() / entry.mmproj.local_name).exists()
    except Exception:  # noqa: BLE001
        return None
