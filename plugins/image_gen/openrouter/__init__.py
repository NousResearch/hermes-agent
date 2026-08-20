"""OpenRouter-compatible image generation backend (OpenRouter + Nous Portal).

Both OpenRouter and the Nous Portal inference endpoint speak the same
OpenAI-style ``/chat/completions`` image-generation protocol: send
``modalities: ["image", "text"]`` with an image-output model (e.g.
``google/gemini-3-pro-image``), pass reference images as ``image_url``
content parts for grounding, and read the generated images back from
``choices[0].message.images[].image_url.url`` (a ``data:image/...;base64`` URI).

Nous Portal proxies OpenRouter, so one implementation services both — we only
swap the resolved ``(base_url, api_key)``. Credentials are resolved through the
agent's existing :func:`~hermes_cli.runtime_provider.resolve_runtime_provider`,
which already understands OpenRouter's key pool and the Nous OAuth device-code
token, so this plugin never reinvents auth.

Reference grounding is the reason pet sprite generation cares about this
backend: each animation row must stay the same character as the chosen base
frame, which only works on models that accept image input. Gemini Flash Image
("nano-banana") does, so both providers advertise image-to-image support.

Two request surfaces
--------------------
OpenRouter ships a *second*, entirely separate image surface: the **Dedicated
Image API** at ``POST /api/v1/images/generations``, with its own catalog
(``GET /api/v1/images/models``) and the full OpenAI-style parameter set.
``openai/gpt-image-2``, ``openai/gpt-image-1-mini``, ``krea/krea-2-*``,
``qwen/qwen-image-3-pro``, ``microsoft/mai-image-2.5*`` and
``x-ai/grok-imagine-image-quality`` are reachable *only* there.

Why routing is not "is it in the image catalog?"
    The two catalogs **overlap**. As of 2026-08 ``GET /images/models`` lists 40
    models, and that list includes this backend's own chat-completions defaults
    (``openai/gpt-5.4-image-2`` and ``google/gemini-3-pro-image``). Routing on
    catalog membership alone would therefore move every existing default call
    off ``/chat/completions`` — a silent behaviour change for setups that work
    today, and one that changes how reference images are passed (content parts
    vs ``input_references``).

So the surface is chosen conservatively, per model, by
:func:`_select_surface`:

* ``image_gen.openrouter.surface`` / ``OPENROUTER_IMAGE_API_SURFACE`` forces
  ``images`` or ``chat`` when an operator wants to decide themselves;
* otherwise (``auto``, the default) only ids in :data:`_IMAGE_API_MODELS` —
  curated, and none of them reachable over chat-completions — take the new
  path. The chat defaults stay pinned to chat-completions no matter what the
  catalog says;
* an id in neither group keeps today's behaviour, and the cached, best-effort
  ``GET /images/models`` probe is used only to log a one-line hint that the
  model is available on the Image API and how to switch. Nothing reroutes
  itself behind the operator's back, and a failed probe costs nothing.

On the Image API path the request gains exact per-model aspect ratios plus
``resolution`` / ``quality`` / ``background`` / ``seed`` / ``n`` /
``output_compression``, and up to 16 reference images instead of 3.

The Image API is OpenRouter-only: Nous Portal proxies the chat-completions
protocol and has no ``/images/generations`` route, so the second surface is
enabled per provider (see ``supports_image_api``) and stays off for Nous.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

# Quality-first model chain for OpenRouter-compatible endpoints.
#
# Default behavior (no env/config override): try the highest-fidelity OpenAI
# image model first, then fall back to Gemini 3 Pro Image if the OpenAI model
# is access-gated / unavailable / times out on this endpoint.
#
# Explicit override (OPENROUTER_IMAGE_MODEL, image_gen.<provider>.model, or
# image_gen.model from ``hermes tools``): use exactly that model (no auto
# fallback), so power users keep full control.
DEFAULT_MODEL = "openai/gpt-5.4-image-2"
_FALLBACK_MODEL = "google/gemini-3-pro-image"
_DEFAULT_MODEL_CHAIN = (DEFAULT_MODEL, _FALLBACK_MODEL)

# Semantic aspect ratio (the image_gen contract) → OpenRouter's image_config
# aspect_ratio strings.
_ASPECT_RATIOS = {
    "square": "1:1",
    "landscape": "16:9",
    "portrait": "9:16",
}

# Gemini Flash Image accepts up to 3 input images per prompt; clamp references
# so we never overflow the model's limit.
_MAX_REFERENCE_IMAGES = 3

# Per single image call. The quality-first default (OpenAI image via OpenRouter)
# is genuinely slow — a single cold row can run well past 3 minutes — so give
# each call real headroom before we treat it as hung and fall back / retry.
_REQUEST_TIMEOUT = 300.0


def _load_image_gen_config() -> Dict[str, Any]:
    """Read the ``image_gen`` section from config.yaml (``{}`` on failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 - config is best-effort
        logger.debug("could not load image_gen config: %s", exc)
        return {}


def _to_image_url_part(ref: str) -> Optional[str]:
    """Turn a reference (local path or http URL) into an ``image_url`` value.

    Remote URLs pass through unchanged; local files are inlined as base64 data
    URIs so the request is self-contained (the provider endpoint can't reach a
    path on our disk). Returns ``None`` when the reference can't be read.
    """
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    # Enforce the shared credential-read guard before inlining local bytes.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.debug("could not read reference image %s: %s", ref, exc)
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_images(payload: Dict[str, Any]) -> List[str]:
    """Pull generated image URLs from a chat-completions response.

    OpenRouter returns generated images under
    ``choices[0].message.images[].image_url.url`` (typically a base64 data URI).
    """
    out: List[str] = []
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        return out
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        images = message.get("images") if isinstance(message, dict) else None
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            image_url = image.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if isinstance(url, str) and url.strip():
                out.append(url.strip())
    return out


def _access_error_hint(
    display: str, model_id: str, env_var: str, status: int, err_msg: str
) -> Optional[str]:
    """A targeted hint when an access-gated OpenAI image model can't be reached.

    Some OpenAI image models on OpenRouter need account enablement / BYOK, so the
    failure isn't a missing key (the key is valid) — the *model* is unreachable.
    The generic "check your key" message is misleading there, so we detect that
    case and point the user at the real fix. Returns one actionable line, or
    ``None`` when this isn't the access-gated case.
    """
    if not model_id.startswith("openai/"):
        return None
    low = (err_msg or "").lower()
    gated = status in (402, 403, 404) or any(
        s in low for s in ("no endpoints", "no allowed", "not a valid model", "data policy")
    )
    if not gated:
        return None
    return (
        f"{display} can't reach image model '{model_id}' ({status}) — enable OpenAI "
        f"image access in your {display} account, or set {env_var}={_FALLBACK_MODEL}."
    )


def _dedupe_models(models: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for model in models:
        m = (model or "").strip()
        if not m or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


# Curated metadata for well-known image models; anything else discovered via
# the live catalog gets a generic strengths line.
_KNOWN_MODEL_META = {
    DEFAULT_MODEL: {
        "display": "OpenAI GPT-5.4 Image 2",
        "strengths": "Highest fidelity; best prompt adherence; slower on OpenRouter",
    },
    _FALLBACK_MODEL: {
        "display": "Gemini 3 Pro Image",
        "strengths": "Fast, reliable fallback with good layout adherence",
    },
}

# Router pseudo-models advertise image output but are not image models.
_EXCLUDED_MODEL_PREFIXES = ("openrouter/auto",)

_LIVE_CACHE_TTL = 300.0
_LIVE_TIMEOUT = 10.0


def _fetch_live_image_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    """List image-output models from the endpoint's ``/models`` catalog.

    Filters on ``architecture.output_modalities`` containing ``image`` and
    drops router pseudo-models (``openrouter/auto*``). Raises on failure —
    callers fall back to the static default chain.
    """
    import requests

    response = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=_LIVE_TIMEOUT,
    )
    response.raise_for_status()
    entries = response.json().get("data") or []
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        model_id = model_id.strip()
        if model_id.startswith(_EXCLUDED_MODEL_PREFIXES):
            continue
        arch_raw = entry.get("architecture")
        arch: Dict[str, Any] = arch_raw if isinstance(arch_raw, dict) else {}
        if "image" not in (arch.get("output_modalities") or []):
            continue
        meta = _KNOWN_MODEL_META.get(model_id, {})
        out.append(
            {
                "id": model_id,
                "display": meta.get("display", entry.get("name") or model_id),
                "strengths": meta.get(
                    "strengths", "Image-output model (from live OpenRouter catalog)"
                ),
                "input_modalities": arch.get("input_modalities") or [],
            }
        )
    # Stable order: defaults first, then alphabetical.
    priority = {DEFAULT_MODEL: 0, _FALLBACK_MODEL: 1}
    out.sort(key=lambda m: (priority.get(m["id"], 2), m["id"]))
    return out


# ===========================================================================
# OpenRouter Dedicated Image API (POST /images/generations)
# ===========================================================================
#
# Everything below serves the second surface described in the module docstring.
# The chat-completions path above is untouched.

#: Env prefix for the Image API knobs (``OPENROUTER_IMAGE_API_QUALITY`` …).
#: Deliberately distinct from ``OPENROUTER_IMAGE_MODEL``, which selects the
#: model on either surface.
_IMAGE_API_ENV_PREFIX = "OPENROUTER_IMAGE_API_"

#: Connect budget, separate from the read budget. If DNS + TCP + TLS hasn't
#: completed in 20s the endpoint is effectively down, and waiting out the full
#: generation timeout only delays the error.
_IMAGE_API_CONNECT_TIMEOUT = 20.0

#: How long a fetched ``/images/models`` catalog stays usable. The catalog only
#: grows, and a stale entry costs one wasted 404 at worst.
_CATALOG_TTL_SECONDS = 900.0

#: Cached catalog probes, keyed by base URL: ``(fetched_at, model_ids)``. An
#: empty set is cached too, so a failing endpoint is probed once per TTL rather
#: than on every call.
_CATALOG_CACHE: Dict[str, Tuple[float, frozenset]] = {}

_GEMINI_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3",
    "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
)
_MAI_RATIOS = ("1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "auto")
_KREA_RATIOS = ("1:1", "4:3", "3:2", "16:9", "4:5", "2:3", "9:16")

#: Curated Image API models and the parameters each one declares in
#: ``GET /images/models`` (catalog snapshot 2026-08). The catalog is larger and
#: keeps growing; an id missing from here still works — it just gets no
#: per-model parameter filtering, and it costs one cached catalog probe to
#: recognise. Keys mirror the payload field they gate; an empty tuple means the
#: model has no such knob.
_IMAGE_API_MODELS: Dict[str, Dict[str, Any]] = {
    "google/gemini-3.1-flash-lite-image": {
        "display": "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "strengths": "Cheap and fast; 14 exact aspect ratios; 14 reference images",
        "aspect_ratios": _GEMINI_RATIOS,
        "resolutions": ("1K",),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": False, "max_n": 1, "max_refs": 14,
    },
    "google/gemini-3.1-flash-image": {
        "display": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "strengths": "Same ratios as Lite plus resolution control (512/1K/2K/4K)",
        "aspect_ratios": _GEMINI_RATIOS,
        "resolutions": ("512", "1K", "2K", "4K"),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": False, "max_n": 1, "max_refs": 14,
    },
    "openai/gpt-image-2": {
        "display": "OpenAI GPT Image 2",
        "strengths": "Best editing fidelity; up to 16 references; strongest prompt adherence",
        "aspect_ratios": ("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "auto"),
        "resolutions": (),
        "quality": ("auto", "low", "medium", "high"),
        "background": ("auto", "opaque"), "output_format": (),
        "compression": True, "seed": False, "max_n": 10, "max_refs": 16,
    },
    "openai/gpt-image-1-mini": {
        "display": "OpenAI GPT Image 1 Mini",
        "strengths": "The only model here with background=transparent (cut-out PNG)",
        "aspect_ratios": ("1:1", "3:2", "2:3", "auto"),
        "resolutions": (),
        "quality": ("auto", "low", "medium", "high"),
        "background": ("auto", "transparent", "opaque"), "output_format": (),
        "compression": True, "seed": False, "max_n": 10, "max_refs": 16,
    },
    "microsoft/mai-image-2.5": {
        "display": "Microsoft MAI-Image-2.5",
        "strengths": "Standard ratios; a good second opinion next to Gemini",
        "aspect_ratios": _MAI_RATIOS, "resolutions": (),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": False, "max_n": 1, "max_refs": 1,
    },
    "microsoft/mai-image-2.5-pro": {
        "display": "Microsoft MAI-Image-2.5 Pro",
        "strengths": "Reach for it when gpt-image-2 misses the brief",
        "aspect_ratios": _MAI_RATIOS, "resolutions": (),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": False, "max_n": 1, "max_refs": 1,
    },
    "x-ai/grok-imagine-image-quality": {
        "display": "Grok Imagine (Image Quality)",
        "strengths": "Photoreal; widest exotic-ratio set (9:19.5, 20:9, 2:1 …); 1K/2K",
        "aspect_ratios": (
            "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2",
            "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto",
        ),
        "resolutions": ("1K", "2K"),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": False, "max_n": 1, "max_refs": 3,
    },
    "krea/krea-2-medium": {
        "display": "Krea 2 Medium",
        "strengths": "Realistic, expressive styles; deterministic via seed",
        "aspect_ratios": _KREA_RATIOS, "resolutions": ("1K",),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": True, "max_n": 1, "max_refs": 1,
    },
    "krea/krea-2-medium-turbo": {
        "display": "Krea 2 Medium Turbo",
        "strengths": "Cheapest here — bulk content, cards, thumbnails; seed support",
        "aspect_ratios": _KREA_RATIOS, "resolutions": ("1K",),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": True, "max_n": 1, "max_refs": 1,
    },
    "qwen/qwen-image-3-pro": {
        "display": "Qwen Image 3 Pro",
        "strengths": "Precise small text and detail rendering; n up to 6; 1K/2K; seed",
        "aspect_ratios": (
            "1:1", "1:2", "1:4", "2:1", "2:3", "3:2", "3:4",
            "4:1", "4:3", "4:5", "5:4", "9:16", "16:9",
        ),
        "resolutions": ("1K", "2K"),
        "quality": (), "background": (), "output_format": (),
        "compression": False, "seed": True, "max_n": 6, "max_refs": 4,
    },
}

#: Applied to a catalog model this table doesn't describe, so a newly released
#: id still generates instead of erroring out. Empty ``aspect_ratios`` means
#: "enum unknown" and the field is omitted rather than guessed.
_UNKNOWN_IMAGE_API_MODEL: Dict[str, Any] = {
    "display": "", "strengths": "",
    "aspect_ratios": (), "resolutions": (),
    "quality": (), "background": (), "output_format": (),
    "compression": False, "seed": False, "max_n": 1, "max_refs": 16,
}

#: The union of exact ratios the endpoint's validator accepts across all
#: models. Used to sanity-check an override aimed at a model whose own enum we
#: don't know.
_ENDPOINT_ASPECT_RATIOS = frozenset({
    "1:1", "1:2", "1:4", "1:8", "2:1", "2:3", "3:2", "3:4", "4:1", "4:3",
    "4:5", "5:4", "8:1", "9:16", "16:9", "9:19.5", "19.5:9", "9:20", "20:9",
    "9:21", "21:9", "auto",
})

#: Semantic ratio → exact ratios, best first. The first one the model supports
#: wins, so ``landscape`` lands on 16:9 where it exists and degrades to 3:2 on
#: ``gpt-image-1-mini``, which has no 16:9 at all.
_ASPECT_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "landscape": ("16:9", "3:2", "4:3", "5:4", "21:9", "2:1", "19.5:9", "20:9", "4:1", "8:1"),
    "portrait": ("9:16", "2:3", "3:4", "4:5", "9:21", "1:2", "9:19.5", "9:20", "1:4", "1:8"),
    "square": ("1:1",),
}

_MEDIA_TYPE_EXTENSIONS = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/svg+xml": "svg",
}

#: HTTP statuses worth retrying on the next model of the chain. 400 is our own
#: payload being wrong and would repeat; 401/403 are account-level and are
#: answered before this set is consulted. 502 matters because the Image API
#: bills all-or-nothing and reports a failed (unbilled) generation that way.
_IMAGE_API_FALLBACK_STATUSES = frozenset({402, 404, 408, 409, 425, 429, 500, 502, 503, 504})


def _image_api_model_meta(model_id: str) -> Dict[str, Any]:
    """Catalog metadata for *model_id*, or permissive defaults when unknown."""
    return _IMAGE_API_MODELS.get(model_id, _UNKNOWN_IMAGE_API_MODEL)


def _fetch_image_api_catalog(base_url: str, api_key: str) -> frozenset:
    """Model ids served by ``GET {base_url}/images/models``, cached per base URL.

    Best-effort by design: any failure caches and returns an empty set, which
    routes the call to chat-completions. Guessing the other way would send a
    chat-completions id to ``/images/generations`` and turn a working setup
    into a 404.
    """
    import requests

    cached = _CATALOG_CACHE.get(base_url)
    if cached and (time.monotonic() - cached[0]) < _CATALOG_TTL_SECONDS:
        return cached[1]

    ids: set = set()
    try:
        response = requests.get(
            f"{base_url}/images/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=(_IMAGE_API_CONNECT_TIMEOUT, 30.0),
        )
        response.raise_for_status()
        body = response.json()
        entries = body.get("data") if isinstance(body, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            model_id = entry.get("id") if isinstance(entry, dict) else None
            if isinstance(model_id, str) and model_id.strip():
                ids.add(model_id.strip())
    except Exception as exc:  # noqa: BLE001 - probe must never break generation
        logger.debug("image API catalog probe failed for %s: %s", base_url, exc)

    resolved = frozenset(ids)
    _CATALOG_CACHE[base_url] = (time.monotonic(), resolved)
    return resolved


#: Ids that stay on ``/chat/completions`` whatever the image catalog says.
#: Both are *in* that catalog, but they are this backend's tested defaults and
#: rerouting them would be a silent behaviour change — see the module docstring.
_CHAT_ONLY_MODELS = frozenset({DEFAULT_MODEL, _FALLBACK_MODEL})



def _select_surface(model_id: str, base_url: str, api_key: str, config_key: str) -> str:
    """Return ``"images"`` or ``"chat"`` for *model_id*.

    Deterministic and offline for curated ids: the decision comes from
    :data:`_IMAGE_API_MODELS` and :data:`_CHAT_ONLY_MODELS`. An id in neither
    table is checked against the (cached) live ``/images/models`` catalog —
    models the dedicated API serves route there, so a model picked from the
    live picker works even when it postdates the curated snapshot. Offline
    the probe returns empty and unknown ids stay on chat-completions.
    """
    if not model_id:
        return "chat"

    forced = _image_api_setting("surface", None, config_key)
    if isinstance(forced, str) and forced.strip().lower() in {"images", "chat"}:
        return forced.strip().lower()

    if model_id in _CHAT_ONLY_MODELS:
        return "chat"
    if model_id in _IMAGE_API_MODELS:
        return "images"

    # Unknown id: the live catalog decides. Most of the endpoint's image
    # models exist only on the dedicated API, so a positive probe must route
    # there — staying on chat would 404/400 a model the live picker offered.
    if model_id in _fetch_image_api_catalog(base_url, api_key):
        return "images"
    return "chat"


def _image_api_setting(name: str, explicit: Any, config_key: str) -> Any:
    """Resolve one Image API knob: call kwarg → env → scoped config.

    ``name`` is the payload field (``quality``); the env var checked is
    ``OPENROUTER_IMAGE_API_QUALITY``. Blank strings count as unset, so an empty
    env var doesn't shadow config.
    """
    if explicit is not None and not (isinstance(explicit, str) and not explicit.strip()):
        return explicit
    env_value = os.environ.get(f"{_IMAGE_API_ENV_PREFIX}{name.upper()}", "").strip()
    if env_value:
        return env_value
    cfg = _load_image_gen_config()
    scoped = cfg.get(config_key)
    value = scoped.get(name) if isinstance(scoped, dict) else None
    if isinstance(value, str):
        return value.strip() or None
    return value


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort int (env vars arrive as strings); ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _pick_exact_aspect_ratio(
    semantic: str,
    meta: Dict[str, Any],
    forced: Optional[str],
    notes: List[str],
) -> Optional[str]:
    """Choose the exact ``aspect_ratio`` to send, or ``None`` to omit it.

    *forced* is an exact ratio from config/env/kwarg. It wins when the model
    supports it; otherwise the downgrade is noted and the per-model mapping of
    *semantic* applies.
    """
    supported: Tuple[str, ...] = tuple(meta.get("aspect_ratios") or ())

    if isinstance(forced, str) and forced.strip():
        value = forced.strip()
        if (supported and value in supported) or (not supported and value in _ENDPOINT_ASPECT_RATIOS):
            return value
        notes.append(
            f"requested aspect_ratio '{value}' is unsupported by this model; "
            f"used the '{semantic}' mapping instead"
        )

    if not supported:
        # A model outside the table: we don't know its enum, and an
        # out-of-enum aspect_ratio is a hard 400 (unlike an unknown
        # *parameter*, which the endpoint ignores). Omitting the field lets the
        # model apply its own default instead of failing every call.
        notes.append(
            "model is not in this backend's catalog, so its aspect_ratio enum is "
            f"unknown; the field was omitted and '{semantic}' was not applied"
        )
        return None

    for candidate in _ASPECT_PREFERENCES.get(semantic, ()):
        if candidate in supported:
            return candidate

    if "auto" in supported:
        return "auto"
    return supported[0] if supported else None


def _image_api_enum(
    name: str,
    explicit: Any,
    meta: Dict[str, Any],
    config_key: str,
    notes: List[str],
) -> Optional[str]:
    """Resolve an enum knob and drop it when this model doesn't accept it."""
    value = _image_api_setting(name, explicit, config_key)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    allowed: Tuple[str, ...] = tuple(meta.get(name) or ())
    if not allowed:
        notes.append(f"'{name}' is not supported by this model; dropped")
        return None
    if value not in allowed:
        notes.append(
            f"'{name}={value}' is not valid for this model "
            f"(accepts {', '.join(allowed)}); dropped"
        )
        return None
    return value


def _build_image_api_payload(
    *,
    model_id: str,
    prompt: str,
    semantic_aspect: str,
    references: List[str],
    config_key: str,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Assemble the ``/images/generations`` body. Returns ``(payload, notes)``.

    Every optional knob is filtered against what the model declares, because a
    parameter the model doesn't know is silently *ignored* by the endpoint —
    which would otherwise let a caller believe ``background=transparent`` took
    effect on a model that has no transparency.
    """
    meta = _image_api_model_meta(model_id)
    notes: List[str] = []
    payload: Dict[str, Any] = {"model": model_id, "prompt": prompt}

    forced_ratio = _image_api_setting(
        "aspect_ratio", kwargs.get("aspect_ratio_exact"), config_key
    )
    ratio = _pick_exact_aspect_ratio(semantic_aspect, meta, forced_ratio, notes)
    if ratio:
        payload["aspect_ratio"] = ratio

    resolution = _image_api_setting("resolution", kwargs.get("resolution"), config_key)
    if isinstance(resolution, str) and resolution.strip():
        allowed = tuple(meta.get("resolutions") or ())
        value = resolution.strip()
        if allowed and value in allowed:
            payload["resolution"] = value
        elif allowed:
            notes.append(
                f"'resolution={value}' is not valid for this model "
                f"(accepts {', '.join(allowed)}); dropped"
            )
        else:
            notes.append("'resolution' is not supported by this model; dropped")

    for enum_name, explicit in (
        ("quality", kwargs.get("quality")),
        ("background", kwargs.get("background")),
        ("output_format", kwargs.get("output_format")),
    ):
        value = _image_api_enum(enum_name, explicit, meta, config_key, notes)
        if value:
            payload[enum_name] = value

    compression = _coerce_int(
        _image_api_setting("output_compression", kwargs.get("output_compression"), config_key)
    )
    if compression is not None:
        if meta.get("compression"):
            payload["output_compression"] = max(0, min(100, compression))
        else:
            notes.append("'output_compression' is not supported by this model; dropped")

    seed = _coerce_int(_image_api_setting("seed", kwargs.get("seed"), config_key))
    if seed is not None:
        if meta.get("seed"):
            payload["seed"] = seed
        else:
            notes.append("'seed' is not supported by this model; dropped")

    count = _coerce_int(_image_api_setting("n", kwargs.get("n"), config_key))
    if count is not None and count > 1:
        max_n = int(meta.get("max_n") or 1)
        if count > max_n:
            notes.append(f"'n={count}' exceeds this model's cap of {max_n}; clamped")
        payload["n"] = max(1, min(count, max_n))

    if references:
        max_refs = int(meta.get("max_refs") or 0)
        usable = references[:max_refs] if max_refs else []
        if len(references) > len(usable):
            notes.append(
                f"{len(references)} reference image(s) supplied but this model "
                f"accepts {max_refs}; extras dropped"
            )
        if usable:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}} for url in usable
            ]

    return payload, notes


def _extract_image_api_error(response: Any, fallback: str) -> str:
    """Flatten either error shape this endpoint produces into one line.

    ``{"error": {"message", "code"}}`` covers routing / auth / unknown model;
    ``{"success": false, "error": {"name": "ZodError", "message": "<json>"}}``
    covers request validation, where ``message`` is a JSON-encoded array of
    issues that is unreadable as-is.
    """
    if response is None:
        return fallback
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - non-JSON error body
        text = getattr(response, "text", "") or ""
        return text[:300] or fallback

    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, str) and error.strip():
        return error.strip()
    if not isinstance(error, dict):
        return json.dumps(body)[:300] if body else fallback

    message = error.get("message")
    if error.get("name") == "ZodError" and isinstance(message, str):
        try:
            issues = json.loads(message)
        except Exception:  # noqa: BLE001
            return message[:300]
        parts: List[str] = []
        for issue in issues if isinstance(issues, list) else []:
            if not isinstance(issue, dict):
                continue
            field = ".".join(str(p) for p in (issue.get("path") or [])) or "request"
            parts.append(f"{field}: {issue.get('message') or 'invalid'}")
        return "; ".join(parts)[:400] or message[:300]

    if isinstance(message, str) and message.strip():
        return message.strip()[:300]
    return json.dumps(error)[:300]


def _extension_for(media_type: Optional[str], fallback: str = "png") -> str:
    if isinstance(media_type, str):
        return _MEDIA_TYPE_EXTENSIONS.get(media_type.split(";", 1)[0].strip().lower(), fallback)
    return fallback


def _model_slug(model_id: str) -> str:
    """Filename-safe fragment for the cache prefix (``openai/gpt-image-2`` …)."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_id)


def _save_image_api_entry(entry: Dict[str, Any], prefix: str) -> Optional[str]:
    """Persist one ``data[]`` entry to the image cache; return its path.

    ``None`` when the entry carries neither base64 nor a URL. Raises on write
    failure so the caller can report ``io_error``.
    """
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return str(save_b64_image(b64, prefix=prefix, extension=_extension_for(entry.get("media_type"))))

    url = entry.get("url")
    if isinstance(url, str) and url.strip():
        return str(save_url_image(url.strip(), prefix=prefix))

    return None


class OpenRouterCompatImageProvider(ImageGenProvider):
    """Image generation over an OpenRouter-compatible chat-completions endpoint.

    Instantiated once per backend (OpenRouter, Nous Portal). The two differ only
    in which runtime provider supplies ``(base_url, api_key)`` and in the config
    namespace used for the model override.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        display_name: str,
        runtime_name: str,
        config_key: str,
        model_env_var: str,
        setup_schema: Dict[str, Any],
        supports_image_api: bool = False,
    ) -> None:
        self._name = provider_name
        self._display = display_name
        self._runtime_name = runtime_name
        self._config_key = config_key
        self._model_env_var = model_env_var
        self._setup_schema = setup_schema
        self._live_models_cache: Optional[tuple] = None
        self._image_api_models_cache: Optional[tuple] = None
        # OpenRouter only: Nous Portal proxies the chat-completions protocol
        # and has no /images/generations route, so routing a model there would
        # turn a working setup into a 404.
        self._supports_image_api = supports_image_api

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display

    def _resolve_runtime(self) -> Dict[str, Any]:
        """Resolve ``(base_url, api_key)`` via the shared runtime resolver."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        return resolve_runtime_provider(requested=self._runtime_name)

    def is_available(self) -> bool:
        try:
            runtime = self._resolve_runtime()
        except Exception as exc:  # noqa: BLE001 - treat resolution failure as unavailable
            logger.debug("%s runtime resolution failed: %s", self._name, exc)
            return False
        return bool(str(runtime.get("api_key") or "").strip())

    def capabilities(self) -> Dict[str, Any]:
        # Both text-to-image and image-to-image (reference grounding) — the
        # latter is what makes this backend usable for pet sprite rows.
        max_refs = _MAX_REFERENCE_IMAGES
        if self._supports_image_api:
            # Report the cap of the model that would actually service the next
            # call, so the tool schema advertises the right number. Image API
            # models take far more references than chat-completions does.
            chain = self._resolve_model_chain()
            resolved = chain[0] if chain else ""
            if resolved in _IMAGE_API_MODELS:
                max_refs = int(_image_api_model_meta(resolved).get("max_refs") or max_refs)
        return {
            "modalities": ["text", "image"],
            "max_reference_images": max_refs,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Picker catalog: the endpoint's full live image-model surface.

        For Image-API-capable backends (OpenRouter) this is the union of the
        dedicated ``GET /images/models`` catalog (40+ models: Seedream, Flux,
        Recraft, Qwen, MAI, Krea, ...) and the chat-completions image models,
        so every image model the endpoint serves — including ones released
        after this code shipped — is selectable in ``hermes tools``. Nous
        Portal (no ``/images`` route) lists the chat-completions catalog only.
        Offline fallback: the static default chain plus the curated Image API
        snapshot.
        """
        merged: Dict[str, Dict[str, Any]] = {}
        if self._supports_image_api:
            for entry in self._image_api_live_models():
                merged[entry["id"]] = entry
        for entry in self._live_models():
            merged.setdefault(entry["id"], entry)
        if merged:
            priority = {DEFAULT_MODEL: 0, _FALLBACK_MODEL: 1}
            return sorted(
                merged.values(), key=lambda m: (priority.get(m["id"], 2), m["id"])
            )

        models = [
            {
                "id": DEFAULT_MODEL,
                "display": "OpenAI GPT-5.4 Image 2",
                "strengths": "Highest fidelity; best prompt adherence; slower on OpenRouter",
            },
            {
                "id": _FALLBACK_MODEL,
                "display": "Gemini 3 Pro Image",
                "strengths": "Fast, reliable fallback with good layout adherence",
            },
        ]
        if self._supports_image_api:
            # Curated snapshot keeps the Image API models pickable offline.
            models.extend(
                {
                    "id": model_id,
                    "display": meta["display"],
                    "strengths": f"{meta['strengths']} (Image API)",
                }
                for model_id, meta in _IMAGE_API_MODELS.items()
                if model_id not in (DEFAULT_MODEL, _FALLBACK_MODEL)
            )
        return models

    def _image_api_live_models(self) -> List[Dict[str, Any]]:
        """Cached live ``GET /images/models`` entries (``[]`` when unreachable).

        Curated metadata from :data:`_IMAGE_API_MODELS` wins for known ids;
        unknown (newly released) models get the API-provided name and a
        generic strengths line so they are pickable the day they launch.
        """
        import time

        cached = self._image_api_models_cache
        if cached is not None and time.monotonic() - cached[1] < _LIVE_CACHE_TTL:
            return cached[0]
        models: List[Dict[str, Any]] = []
        try:
            import requests

            runtime = self._resolve_runtime()
            api_key = str(runtime.get("api_key") or "").strip()
            base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
            if base_url:
                response = requests.get(
                    f"{base_url}/images/models",
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                    timeout=_LIVE_TIMEOUT,
                )
                response.raise_for_status()
                for entry in response.json().get("data") or []:
                    if not isinstance(entry, dict):
                        continue
                    model_id = entry.get("id")
                    if not isinstance(model_id, str) or not model_id.strip():
                        continue
                    model_id = model_id.strip()
                    meta = _IMAGE_API_MODELS.get(model_id, {})
                    arch_raw = entry.get("architecture")
                    arch: Dict[str, Any] = arch_raw if isinstance(arch_raw, dict) else {}
                    models.append(
                        {
                            "id": model_id,
                            "display": meta.get("display", entry.get("name") or model_id),
                            "strengths": meta.get(
                                "strengths", "Image API model (from live OpenRouter catalog)"
                            ),
                            "input_modalities": arch.get("input_modalities") or [],
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - offline/unauth → fallback path
            logger.debug("%s live Image API catalog unavailable: %s", self._name, exc)
            models = []
        self._image_api_models_cache = (models, time.monotonic())
        return models

    def _live_models(self) -> List[Dict[str, Any]]:
        """Cached live catalog for this backend (``[]`` when unreachable)."""
        import time

        cached = self._live_models_cache
        if cached is not None and time.monotonic() - cached[1] < _LIVE_CACHE_TTL:
            return cached[0]
        models: List[Dict[str, Any]] = []
        try:
            runtime = self._resolve_runtime()
            api_key = str(runtime.get("api_key") or "").strip()
            base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
            if base_url:
                models = _fetch_live_image_models(base_url, api_key)
        except Exception as exc:  # noqa: BLE001 - offline/unauth → static fallback
            logger.debug("%s live image model catalog unavailable: %s", self._name, exc)
            models = []
        self._live_models_cache = (models, time.monotonic())
        return models

    def default_model(self) -> Optional[str]:
        # This is the catalog default, not the effective runtime model.
        # Runtime overrides are resolved separately by _resolve_model_chain().
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return dict(self._setup_schema)

    def _resolve_model(self, explicit: Optional[str] = None) -> str:
        """Pick the image model (first of :meth:`_resolve_model_chain`)."""
        return self._resolve_model_chain(explicit)[0]

    def _resolve_model_chain(self, explicit: Optional[str] = None) -> list[str]:
        """Ordered model attempts for this request.

        Precedence: explicit caller override (the ``model`` kwarg) → the
        provider's ``*_IMAGE_MODEL`` env override → scoped
        ``image_gen.<provider>.model`` → top-level ``image_gen.model`` (written
        by ``hermes tools``) → the quality-first default chain.

        Any explicit user/model selection means "use this exact model", so no
        fallback. Only the bare default chain carries a Gemini fallback.
        """
        if isinstance(explicit, str) and explicit.strip():
            return [explicit.strip()]
        env_override = os.environ.get(self._model_env_var, "").strip()
        if env_override:
            return [env_override]
        cfg = _load_image_gen_config()
        scoped = cfg.get(self._config_key) if isinstance(cfg.get(self._config_key), dict) else {}
        if isinstance(scoped, dict):
            value = scoped.get("model")
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        top = cfg.get("model")
        if isinstance(top, str) and top.strip():
            return [top.strip()]
        return _dedupe_models(list(_DEFAULT_MODEL_CHAIN))

    def _generate_via_image_api(
        self,
        *,
        model_id: str,
        prompt: str,
        semantic_aspect: str,
        references: List[str],
        base_url: str,
        headers: Dict[str, str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Serve one model through ``POST {base_url}/images/generations``.

        Returns the usual success/error dict. A failure that the caller may
        retry on the next model of the chain carries a private ``_retryable``
        flag, which :meth:`generate` strips before returning.
        """
        import requests

        def _fail(error: str, error_type: str, retryable: bool = False) -> Dict[str, Any]:
            response = error_response(
                error=error,
                error_type=error_type,
                provider=self._name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=semantic_aspect,
            )
            if retryable:
                response["_retryable"] = True
            return response

        # References become data URIs here rather than reusing the chat path's
        # `content`, because that one is clamped to 3 and these models take up
        # to 16.
        usable_refs: List[str] = []
        unreadable: List[str] = []
        try:
            for ref in references:
                part = _to_image_url_part(ref)
                if part:
                    usable_refs.append(part)
                else:
                    unreadable.append(str(ref))
        except Exception as exc:  # noqa: BLE001 - blocked by the file-safety guard
            return _fail(f"Could not load reference image: {exc}", "io_error")

        # An edit whose every source image failed to load must not quietly
        # become a text-to-image generation: that bills a new picture unrelated
        # to what the caller asked to edit, and success=True hides it.
        if unreadable and not usable_refs:
            return _fail(
                "Could not read the reference image(s) requested for editing: "
                + ", ".join(unreadable)
                + ". Refusing to silently fall back to text-to-image.",
                "io_error",
            )

        payload, notes = _build_image_api_payload(
            model_id=model_id,
            prompt=prompt,
            semantic_aspect=semantic_aspect,
            references=usable_refs,
            config_key=self._config_key,
            kwargs=kwargs,
        )
        if unreadable:
            notes.insert(0, f"dropped unreadable reference image(s): {', '.join(unreadable)}")

        timeout = _REQUEST_TIMEOUT
        configured = _image_api_setting("timeout", kwargs.get("timeout"), self._config_key)
        try:
            if configured is not None:
                timeout = max(1.0, float(configured))
        except (TypeError, ValueError):
            logger.debug("%s: ignoring non-numeric image API timeout %r", self._name, configured)

        endpoint = f"{base_url}/images/generations"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                # (connect, read): an unreachable endpoint fails in seconds
                # instead of burning the whole read budget on a connection that
                # will never be established.
                timeout=(min(_IMAGE_API_CONNECT_TIMEOUT, timeout), timeout),
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            message = _extract_image_api_error(resp, str(exc))
            logger.error(
                "%s image API generation failed (%s) on %s: %s",
                self._name, status, model_id, message,
            )
            # 401/403 are answered first so one account-level rejection does
            # not get two different error_types depending on where in the chain
            # it landed.
            if status in (401, 403):
                return _fail(f"{self._display} rejected the API key ({status}): {message}", "auth_error")
            if status == 404:
                return _fail(
                    f"Model '{model_id}' does not exist on the OpenRouter Image API "
                    f"(its catalog is separate from chat-completions — check "
                    f"GET {base_url}/images/models).",
                    "model_access",
                    retryable=True,
                )
            return _fail(
                f"{self._display} image generation failed ({status}): {message}",
                "api_error",
                retryable=status in _IMAGE_API_FALLBACK_STATUSES,
            )
        except requests.Timeout:
            return _fail(
                f"{self._display} image generation timed out ({int(timeout)}s)",
                "timeout",
                retryable=True,
            )
        except requests.ConnectionError as exc:
            return _fail(f"{self._display} connection error: {exc}", "connection_error")
        except requests.RequestException as exc:
            return _fail(f"{self._display} request failed: {exc}", "api_error")

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            return _fail(f"{self._display} returned invalid JSON: {exc}", "invalid_response")

        entries = body.get("data") if isinstance(body, dict) else None
        entries = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []
        if not entries:
            return _fail(
                f"{self._display} returned no image data for '{model_id}'.",
                "empty_response",
                retryable=True,
            )

        prefix = f"{self._name}_{_model_slug(model_id)}"
        try:
            saved = [p for p in (_save_image_api_entry(e, prefix) for e in entries) if p]
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Could not save generated image: {exc}", "io_error")

        if not saved:
            return _fail(
                f"{self._display} response carried neither b64_json nor url.",
                "empty_response",
            )

        extra: Dict[str, Any] = {
            "endpoint": "images/generations",
            "exact_aspect_ratio": payload.get("aspect_ratio"),
        }
        for key in ("resolution", "quality", "background", "output_format", "seed", "n"):
            if key in payload:
                extra[key] = payload[key]
        if len(saved) > 1:
            extra["additional_images"] = saved[1:]
        if usable_refs:
            extra["reference_images_used"] = len(payload.get("input_references") or [])
        if notes:
            extra["notes"] = notes

        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            if isinstance(usage.get("cost"), (int, float)):
                extra["cost_usd"] = usage["cost"]
            if isinstance(usage.get("total_tokens"), int):
                extra["total_tokens"] = usage["total_tokens"]

        return success_response(
            image=saved[0],
            model=model_id,
            prompt=prompt,
            aspect_ratio=semantic_aspect,
            provider=self._name,
            modality="image" if usable_refs else "text",
            extra=extra,
        )

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import requests

        try:
            runtime = self._resolve_runtime()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not resolve {self._display} credentials: {exc}",
                error_type="missing_api_key",
                provider=self._name,
                aspect_ratio=aspect_ratio,
            )
        api_key = str(runtime.get("api_key") or "").strip()
        base_url = str(runtime.get("base_url") or "").strip().rstrip("/")
        if not api_key or not base_url:
            return error_response(
                error=(
                    f"No {self._display} credentials found. "
                    f"Configure {self._display} in `hermes tools` → Image Generation."
                ),
                error_type="missing_api_key",
                provider=self._name,
                aspect_ratio=aspect_ratio,
            )

        model_chain = self._resolve_model_chain(kwargs.get("model"))
        aspect = resolve_aspect_ratio(aspect_ratio)
        or_aspect = _ASPECT_RATIOS.get(aspect, "1:1")

        # Collect every reference: the pet generator passes local paths via the
        # ``reference_images`` kwarg; the generic tool surface uses ``image_url``
        # / ``reference_image_urls``. Accept all three.
        references: List[str] = []
        for ref in kwargs.get("reference_images") or []:
            references.append(str(ref))
        if image_url:
            references.append(str(image_url))
        for ref in reference_image_urls or []:
            references.append(str(ref))

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref in references[:_MAX_REFERENCE_IMAGES]:
            part = _to_image_url_part(ref)
            if part:
                content.append({"type": "image_url", "image_url": {"url": part}})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (harmless against Nous Portal).
            "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
            "X-Title": "Hermes Agent",
        }
        last_error: Optional[Dict[str, Any]] = None
        for i, model_id in enumerate(model_chain):
            is_last = i == len(model_chain) - 1

            # Surface selection. An Image API model cannot be served by
            # /chat/completions (and vice versa), so this is a routing
            # decision, not a preference.
            if self._supports_image_api and _select_surface(
                model_id, base_url, api_key, self._config_key
            ) == "images":
                outcome = self._generate_via_image_api(
                    model_id=model_id,
                    prompt=prompt,
                    semantic_aspect=aspect,
                    references=references,
                    base_url=base_url,
                    headers=headers,
                    kwargs=kwargs,
                )
                if outcome.get("success"):
                    return outcome
                if is_last or not outcome.get("_retryable"):
                    outcome.pop("_retryable", None)
                    return outcome
                outcome.pop("_retryable", None)
                logger.info(
                    "%s model %s failed on the image API; retrying with fallback %s",
                    self._name,
                    model_id,
                    model_chain[i + 1],
                )
                last_error = outcome
                continue

            payload: Dict[str, Any] = {
                "model": model_id,
                "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": content}],
                "image_config": {"aspect_ratio": or_aspect},
            }
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
            except requests.HTTPError as exc:
                resp = exc.response
                status = resp.status_code if resp is not None else 0
                try:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:300])
                except Exception:  # noqa: BLE001
                    err_msg = resp.text[:300] if resp is not None else str(exc)
                logger.error("%s image gen failed (%d) on %s: %s", self._name, status, model_id, err_msg)
                hint = _access_error_hint(self._display, model_id, self._model_env_var, status, err_msg)
                if hint and not is_last:
                    logger.info(
                        "%s model %s unavailable; retrying with fallback %s",
                        self._name,
                        model_id,
                        model_chain[i + 1],
                    )
                    continue
                last_error = error_response(
                    error=hint or f"{self._display} image generation failed ({status}): {err_msg}",
                    error_type="model_access" if hint else "api_error",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
                return last_error
            except requests.Timeout:
                if not is_last:
                    logger.info(
                        "%s model %s timed out; retrying with fallback %s",
                        self._name,
                        model_id,
                        model_chain[i + 1],
                    )
                    continue
                return error_response(
                    error=f"{self._display} image generation timed out "
                    f"({int(_REQUEST_TIMEOUT)}s)",
                    error_type="timeout",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            except requests.ConnectionError as exc:
                return error_response(
                    error=f"{self._display} connection error: {exc}",
                    error_type="connection_error",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            try:
                result = response.json()
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"{self._display} returned invalid JSON: {exc}",
                    error_type="invalid_response",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            images = _extract_images(result)
            if not images:
                if not is_last:
                    logger.info(
                        "%s model %s returned no image; retrying with fallback %s",
                        self._name,
                        model_id,
                        model_chain[i + 1],
                    )
                    continue
                # A response with text but no image usually means the model didn't
                # honor image output (wrong model or modalities); surface that.
                return error_response(
                    error=(
                        f"{self._display} returned no image. Ensure the model "
                        f"'{model_id}' supports image output."
                    ),
                    error_type="empty_response",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            first = images[0]
            try:
                if first.startswith("data:"):
                    b64 = first.split(",", 1)[1] if "," in first else ""
                    saved_path = save_b64_image(b64, prefix=f"{self._name}_gen")
                else:
                    saved_path = save_url_image(first, prefix=f"{self._name}_gen")
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    error=f"Could not save generated image: {exc}",
                    error_type="io_error",
                    provider=self._name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            return success_response(
                image=str(saved_path),
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self._name,
            )

        return last_error or error_response(
            error=f"{self._display} image generation failed after trying all candidate models.",
            error_type="api_error",
            provider=self._name,
            model=model_chain[-1] if model_chain else "",
            prompt=prompt,
            aspect_ratio=aspect,
        )


def _build_providers() -> List[OpenRouterCompatImageProvider]:
    return [
        OpenRouterCompatImageProvider(
            provider_name="openrouter",
            display_name="OpenRouter",
            runtime_name="openrouter",
            config_key="openrouter",
            model_env_var="OPENROUTER_IMAGE_MODEL",
            supports_image_api=True,
            setup_schema={
                "name": "OpenRouter (image)",
                "badge": "paid",
                "tag": "Gemini Flash Image, gpt-image-2, Krea 2, Qwen Image 3 & more via OpenRouter; uses OPENROUTER_API_KEY",
                "env_vars": [
                    {
                        "key": "OPENROUTER_API_KEY",
                        "prompt": "OpenRouter API key",
                        "url": "https://openrouter.ai/keys",
                    }
                ],
            },
        ),
        OpenRouterCompatImageProvider(
            provider_name="nous",
            display_name="Nous Portal",
            runtime_name="nous",
            config_key="nous",
            model_env_var="NOUS_IMAGE_MODEL",
            setup_schema={
                "name": "Nous Portal (image)",
                "badge": "subscription",
                "tag": "Reference-grounded image generation via Nous Portal (OpenRouter-backed)",
                "env_vars": [],
                "requires_nous_auth": True,
            },
        ),
    ]


def register(ctx: Any) -> None:
    """Register the OpenRouter + Nous Portal image gen providers."""
    for provider in _build_providers():
        ctx.register_image_gen_provider(provider)
