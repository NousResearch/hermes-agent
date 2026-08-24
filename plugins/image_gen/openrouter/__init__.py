"""OpenRouter-compatible image generation backend (OpenRouter + Nous Portal).

Both speak the OpenAI-style ``/chat/completions`` image protocol (``modalities:
["image","text"]``, references as ``image_url`` parts, output in
``choices[0].message.images[].image_url.url``); only ``(base_url, api_key)`` differs,
via :func:`hermes_cli.runtime_provider.resolve_runtime_provider`. OpenRouter alone also
has the Dedicated Image API (``POST /images/generations``, own catalog, exact ratios,
up to 16 references) — see :func:`_select_surface`.
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
    DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response, resolve_aspect_ratio, save_b64_image,
    save_url_image, success_response)
from plugins.image_gen._common import error_factory, load_image_gen_config, post_json

logger = logging.getLogger(__name__)

# Quality-first default chain: OpenAI first, Gemini 3 Pro Image when it is gated /
# unavailable / times out. Any explicit override is exact — no auto fallback.
DEFAULT_MODEL = "openai/gpt-5.4-image-2"
_FALLBACK_MODEL = "google/gemini-3-pro-image"
_DEFAULT_MODEL_CHAIN = (DEFAULT_MODEL, _FALLBACK_MODEL)
_MODEL_PRIORITY = {DEFAULT_MODEL: 0, _FALLBACK_MODEL: 1}

# Semantic aspect ratio → OpenRouter ``image_config.aspect_ratio``.
_ASPECT_RATIOS = {"square": "1:1", "landscape": "16:9", "portrait": "9:16"}
_MAX_REFERENCE_IMAGES = 3  # Gemini Flash Image accepts up to 3 input images per prompt.
_REQUEST_TIMEOUT = 300.0  # per image call; a cold quality-first row can run past 3 minutes.

# Curated metadata for well-known chat-completions image models.
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
_EXCLUDED_MODEL_PREFIXES = ("openrouter/auto",)  # router pseudo-models advertise image output
_LIVE_CACHE_TTL = 300.0
_LIVE_TIMEOUT = 10.0
_load_image_gen_config = load_image_gen_config

# ---------------------------------------------------------------------------
# OpenRouter Dedicated Image API (POST /images/generations)
# ---------------------------------------------------------------------------

# Knob env prefix (``OPENROUTER_IMAGE_API_QUALITY`` …); ``OPENROUTER_IMAGE_MODEL`` picks the model.
_IMAGE_API_ENV_PREFIX = "OPENROUTER_IMAGE_API_"
# Separate connect budget: no TLS in 20s means the endpoint is down — don't wait out the read budget.
_IMAGE_API_CONNECT_TIMEOUT = 20.0
# ``/images/models`` probes keyed by base URL: ``(fetched_at, ids)``; empty sets cached too.
_CATALOG_TTL_SECONDS = 900.0
_CATALOG_CACHE: Dict[str, Tuple[float, frozenset]] = {}

_GEMINI_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9",
)
_MAI_RATIOS = ("1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "auto")
_KREA_RATIOS = ("1:1", "4:3", "3:2", "16:9", "4:5", "2:3", "9:16")
_OPENAI_QUALITY = ("auto", "low", "medium", "high")
_NO_KNOBS = {"quality": (), "background": (), "output_format": (), "compression": False, "seed": False}


def _image_api_model(display: str, strengths: str, **spec: Any) -> Dict[str, Any]:
    """Curated Image API model entry; unspecified knobs default to "not supported"."""
    return {"display": display, "strengths": strengths, **_NO_KNOBS, "resolutions": (), **spec}


# Curated Image API models + the parameters each declares in ``GET /images/models``. An
# id missing here still works (no per-model knob filtering; one cached catalog probe).
# Keys mirror the payload field they gate; an empty tuple means "no such knob".
_IMAGE_API_MODELS: Dict[str, Dict[str, Any]] = {
    "google/gemini-3.1-flash-lite-image": _image_api_model(
        "Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "Cheap and fast; 14 exact aspect ratios; 14 reference images",
        aspect_ratios=_GEMINI_RATIOS, resolutions=("1K",), max_n=1, max_refs=14),
    "google/gemini-3.1-flash-image": _image_api_model(
        "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "Same ratios as Lite plus resolution control (512/1K/2K/4K)",
        aspect_ratios=_GEMINI_RATIOS, resolutions=("512", "1K", "2K", "4K"), max_n=1, max_refs=14),
    "openai/gpt-image-2": _image_api_model(
        "OpenAI GPT Image 2",
        "Best editing fidelity; up to 16 references; strongest prompt adherence",
        aspect_ratios=("1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9", "auto"),
        quality=_OPENAI_QUALITY, background=("auto", "opaque"), compression=True, max_n=10, max_refs=16,
    ),
    "openai/gpt-image-1-mini": _image_api_model(
        "OpenAI GPT Image 1 Mini",
        "The only model here with background=transparent (cut-out PNG)",
        aspect_ratios=("1:1", "3:2", "2:3", "auto"),
        quality=_OPENAI_QUALITY, background=("auto", "transparent", "opaque"), compression=True,
        max_n=10, max_refs=16),
    "microsoft/mai-image-2.5": _image_api_model(
        "Microsoft MAI-Image-2.5", "Standard ratios; a good second opinion next to Gemini",
        aspect_ratios=_MAI_RATIOS, max_n=1, max_refs=1),
    "microsoft/mai-image-2.5-pro": _image_api_model(
        "Microsoft MAI-Image-2.5 Pro", "Reach for it when gpt-image-2 misses the brief",
        aspect_ratios=_MAI_RATIOS, max_n=1, max_refs=1),
    "x-ai/grok-imagine-image-quality": _image_api_model(
        "Grok Imagine (Image Quality)",
        "Photoreal; widest exotic-ratio set (9:19.5, 20:9, 2:1 …); 1K/2K",
        aspect_ratios=(
            "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1",
            "auto"),
        resolutions=("1K", "2K"), max_n=1, max_refs=3),
    "krea/krea-2-medium": _image_api_model(
        "Krea 2 Medium", "Realistic, expressive styles; deterministic via seed",
        aspect_ratios=_KREA_RATIOS, resolutions=("1K",), seed=True, max_n=1, max_refs=1),
    "krea/krea-2-medium-turbo": _image_api_model(
        "Krea 2 Medium Turbo", "Cheapest here — bulk content, cards, thumbnails; seed support",
        aspect_ratios=_KREA_RATIOS, resolutions=("1K",), seed=True, max_n=1, max_refs=1),
    "qwen/qwen-image-3-pro": _image_api_model(
        "Qwen Image 3 Pro", "Precise small text and detail rendering; n up to 6; 1K/2K; seed",
        aspect_ratios=("1:1", "1:2", "1:4", "2:1", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "9:16", "16:9"),
        resolutions=("1K", "2K"), seed=True, max_n=6, max_refs=4),
}

# For catalog models the table doesn't describe; empty ``aspect_ratios`` = unknown enum → omitted.
_UNKNOWN_IMAGE_API_MODEL = _image_api_model("", "", aspect_ratios=(), max_n=1, max_refs=16)

# Union of exact ratios across all models; sanity-checks an override for a model with unknown enum.
_ENDPOINT_ASPECT_RATIOS = frozenset({
    "1:1", "1:2", "1:4", "1:8", "2:1", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16",
    "16:9", "9:19.5", "19.5:9", "9:20", "20:9", "9:21", "21:9", "auto",
})

# Semantic ratio → exact ratios, best first (``landscape`` degrades to 3:2 where 16:9 is missing).
_ASPECT_PREFERENCES: Dict[str, Tuple[str, ...]] = {
    "landscape": ("16:9", "3:2", "4:3", "5:4", "21:9", "2:1", "19.5:9", "20:9", "4:1", "8:1"),
    "portrait": ("9:16", "2:3", "3:4", "4:5", "9:21", "1:2", "9:19.5", "9:20", "1:4", "1:8"),
    "square": ("1:1",),
}

_MEDIA_TYPE_EXTENSIONS = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif",
    "image/svg+xml": "svg",
}

# Statuses worth retrying on the next chain model. 400 is our own payload (would repeat);
# 401/403 are account-level and answered first; 502 is how the Image API reports an unbilled failure.
_IMAGE_API_FALLBACK_STATUSES = frozenset({402, 404, 408, 409, 425, 429, 500, 502, 503, 504})

# Stay on ``/chat/completions`` whatever the image catalog says — the tested defaults.
_CHAT_ONLY_MODELS = frozenset({DEFAULT_MODEL, _FALLBACK_MODEL})

_IMAGE_API_EXTRA_KEYS = ("resolution", "quality", "background", "output_format", "seed", "n")

# Enum knobs: payload field → catalog key holding the allowed values.
_IMAGE_API_ENUMS = (
    ("resolution", "resolutions"), ("quality", None), ("background", None), ("output_format", None))
# Integer knobs: payload field → (catalog gate flag, clamp).
_IMAGE_API_INTS = (
    ("output_compression", "compression", lambda v: max(0, min(100, v))), ("seed", "seed", lambda v: v),
)

_ATTRIBUTION_HEADERS = {
    "Content-Type": "application/json",
    # OpenRouter attribution headers (harmless against Nous Portal).
    "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
    "X-Title": "Hermes Agent",
}


def _to_image_url_part(ref: str) -> Optional[str]:
    """URLs pass through; local files inline as base64 data URIs (``None`` when unreadable)."""
    ref = str(ref or "").strip()
    if not ref:
        return None
    if ref.startswith(("http://", "https://", "data:")):
        return ref
    path = Path(ref)
    from agent.file_safety import raise_if_read_blocked  # credential-read guard before inlining

    raise_if_read_blocked(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.debug("could not read reference image %s: %s", ref, exc)
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _dict_at(node: Any, key: str) -> Dict[str, Any]:
    value = node.get(key) if isinstance(node, dict) else None
    return value if isinstance(value, dict) else {}


def _list_at(node: Any, key: str) -> List[Any]:
    value = node.get(key) if isinstance(node, dict) else None
    return value if isinstance(value, list) else []


def _extract_images(payload: Dict[str, Any]) -> List[str]:
    """Generated image URLs from ``choices[].message.images[].image_url.url``."""
    out: List[str] = []
    for choice in _list_at(payload, "choices"):
        for image in _list_at(_dict_at(choice, "message"), "images"):
            url = _dict_at(image, "image_url").get("url")
            if isinstance(url, str) and url.strip():
                out.append(url.strip())
    return out


def _access_error_hint(display: str, model_id: str, env_var: str, status: int, err_msg: str) -> Optional[str]:
    """Hint for an access-gated OpenAI image model (valid key, but the *model* needs account
    enablement / BYOK, so "check your key" would mislead). ``None`` otherwise."""
    if not model_id.startswith("openai/"):
        return None
    low = (err_msg or "").lower()
    gated = status in (402, 403, 404) or any(
        s in low for s in ("no endpoints", "no allowed", "not a valid model", "data policy"))
    if not gated:
        return None
    return (
        f"{display} can't reach image model '{model_id}' ({status}) — enable OpenAI "
        f"image access in your {display} account, or set {env_var}={_FALLBACK_MODEL}.")


def _get_catalog(base_url: str, path: str, api_key: str, timeout: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """``(model_id, entry)`` pairs from ``GET {base_url}{path}``'s ``data[]``; raises on HTTP failure."""
    import requests

    response = requests.get(
        f"{base_url}{path}", headers={"Authorization": f"Bearer {api_key}"} if api_key else {}, timeout=timeout,
    )
    response.raise_for_status()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for entry in _list_at(response.json(), "data"):
        model_id = entry.get("id") if isinstance(entry, dict) else None
        if isinstance(model_id, str) and model_id.strip():
            out.append((model_id.strip(), entry))
    return out


def _fetch_catalog(
    base_url: str, api_key: str, *, path: str, meta: Dict[str, Dict[str, Any]], generic: str,
    image_output_only: bool,
) -> List[Dict[str, Any]]:
    """Picker rows from ``GET {base_url}{path}``; raises on failure. ``image_output_only`` keeps
    image-output models minus router pseudo-models; curated ``meta`` wins for known ids."""
    out: List[Dict[str, Any]] = []
    for model_id, entry in _get_catalog(base_url, path, api_key, _LIVE_TIMEOUT):
        arch = _dict_at(entry, "architecture")
        if image_output_only and (
            model_id.startswith(_EXCLUDED_MODEL_PREFIXES) or "image" not in (arch.get("output_modalities") or [])
        ):
            continue
        known = meta.get(model_id, {})
        out.append({
            "id": model_id,
            "display": known.get("display", entry.get("name") or model_id),
            "strengths": known.get("strengths", generic),
            "input_modalities": arch.get("input_modalities") or [],
        })
    return out


def _fetch_image_api_catalog(base_url: str, api_key: str) -> frozenset:
    """Model ids from ``GET {base_url}/images/models``, cached per base URL. Any failure caches an
    empty set (→ chat-completions): guessing "images" would 404 a working chat setup."""
    cached = _CATALOG_CACHE.get(base_url)
    if cached and (time.monotonic() - cached[0]) < _CATALOG_TTL_SECONDS:
        return cached[1]
    ids: set = set()
    try:
        catalog = _get_catalog(base_url, "/images/models", api_key, (_IMAGE_API_CONNECT_TIMEOUT, 30.0))
        ids = {model_id for model_id, _entry in catalog}
    except Exception as exc:  # noqa: BLE001 - probe must never break generation
        logger.debug("image API catalog probe failed for %s: %s", base_url, exc)
    resolved = frozenset(ids)
    _CATALOG_CACHE[base_url] = (time.monotonic(), resolved)
    return resolved


def _image_api_model_meta(model_id: str) -> Dict[str, Any]:
    """Catalog metadata for *model_id*, or permissive defaults when unknown."""
    return _IMAGE_API_MODELS.get(model_id, _UNKNOWN_IMAGE_API_MODEL)


def _select_surface(model_id: str, base_url: str, api_key: str, config_key: str) -> str:
    """``"images"`` or ``"chat"``: config/env ``surface`` forces it; curated ids are offline and
    deterministic; unknown ids consult the cached live catalog (a positive probe must route there
    or a model the live picker offered would 404; offline they stay on chat)."""
    if not model_id:
        return "chat"
    forced = _image_api_setting("surface", None, config_key)
    if isinstance(forced, str) and forced.strip().lower() in {"images", "chat"}:
        return forced.strip().lower()
    if model_id in _CHAT_ONLY_MODELS:
        return "chat"
    if model_id in _IMAGE_API_MODELS or model_id in _fetch_image_api_catalog(base_url, api_key):
        return "images"
    return "chat"


def _image_api_setting(name: str, explicit: Any, config_key: str) -> Any:
    """Knob: call kwarg → ``OPENROUTER_IMAGE_API_<NAME>`` env → scoped config; blanks are unset."""
    if explicit is not None and not (isinstance(explicit, str) and not explicit.strip()):
        return explicit
    env_value = os.environ.get(f"{_IMAGE_API_ENV_PREFIX}{name.upper()}", "").strip()
    if env_value:
        return env_value
    value = _dict_at(_load_image_gen_config(), config_key).get(name)
    return (value.strip() or None) if isinstance(value, str) else value


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort int (env vars arrive as strings); ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value.strip()) if isinstance(value, str) else None
    except ValueError:
        return None


def _pick_exact_aspect_ratio(
    semantic: str, meta: Dict[str, Any], forced: Optional[str], notes: List[str]
) -> Optional[str]:
    """Exact ``aspect_ratio`` or ``None`` to omit. *forced* wins when supported; otherwise the
    downgrade is noted and the per-model mapping of *semantic* applies."""
    supported: Tuple[str, ...] = tuple(meta.get("aspect_ratios") or ())
    if isinstance(forced, str) and forced.strip():
        value = forced.strip()
        if (supported and value in supported) or (not supported and value in _ENDPOINT_ASPECT_RATIOS):
            return value
        notes.append(
            f"requested aspect_ratio '{value}' is unsupported by this model; "
            f"used the '{semantic}' mapping instead")
    if not supported:
        # Out-of-enum aspect_ratio is a hard 400 (unknown *parameters* are ignored) — omit it.
        notes.append(
            "model is not in this backend's catalog, so its aspect_ratio enum is "
            f"unknown; the field was omitted and '{semantic}' was not applied")
        return None
    for candidate in _ASPECT_PREFERENCES.get(semantic, ()):
        if candidate in supported:
            return candidate
    return "auto" if "auto" in supported else supported[0]


def _image_api_enum(
    name: str, explicit: Any, meta: Dict[str, Any], config_key: str, notes: List[str], *,
    meta_key: Optional[str] = None,
) -> Optional[str]:
    """Resolve an enum knob and drop it when this model doesn't accept it."""
    value = _image_api_setting(name, explicit, config_key)
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    allowed: Tuple[str, ...] = tuple(meta.get(meta_key or name) or ())
    if not allowed:
        notes.append(f"'{name}' is not supported by this model; dropped")
        return None
    if value not in allowed:
        notes.append(f"'{name}={value}' is not valid for this model (accepts {', '.join(allowed)}); dropped")
        return None
    return value


def _build_image_api_payload(
    *, model_id: str, prompt: str, semantic_aspect: str, references: List[str], config_key: str,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """``/images/generations`` body: ``(payload, notes)``. Every knob is filtered against what the
    model declares: the endpoint silently *ignores* unknown parameters, which would let a caller
    believe ``background=transparent`` took effect on a model without it."""
    meta = _image_api_model_meta(model_id)
    notes: List[str] = []
    payload: Dict[str, Any] = {"model": model_id, "prompt": prompt}

    forced_ratio = _image_api_setting("aspect_ratio", kwargs.get("aspect_ratio_exact"), config_key)
    ratio = _pick_exact_aspect_ratio(semantic_aspect, meta, forced_ratio, notes)
    if ratio:
        payload["aspect_ratio"] = ratio
    for enum_name, meta_key in _IMAGE_API_ENUMS:
        value = _image_api_enum(enum_name, kwargs.get(enum_name), meta, config_key, notes, meta_key=meta_key)
        if value:
            payload[enum_name] = value
    for name, gate, clamp in _IMAGE_API_INTS:
        value = _coerce_int(_image_api_setting(name, kwargs.get(name), config_key))
        if value is None:
            continue
        if meta.get(gate):
            payload[name] = clamp(value)
        else:
            notes.append(f"'{name}' is not supported by this model; dropped")
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
                f"accepts {max_refs}; extras dropped")
        if usable:
            payload["input_references"] = [{"type": "image_url", "image_url": {"url": url}} for url in usable]
    return payload, notes


def _extract_image_api_error(response: Any, fallback: str) -> str:
    """One-line error from ``{"error": {"message"}}`` (routing/auth) or the ZodError shape
    ``{"error": {"name": "ZodError", "message": "<json issue array>"}}`` (validation)."""
    if response is None:
        return fallback
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - non-JSON error body
        return (getattr(response, "text", "") or "")[:300] or fallback

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
        parts = [
            f"{'.'.join(str(p) for p in (issue.get('path') or [])) or 'request'}: "
            f"{issue.get('message') or 'invalid'}"
            for issue in (issues if isinstance(issues, list) else []) if isinstance(issue, dict)
        ]
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
    """Cache one ``data[]`` entry; ``None`` without b64/URL. Raises on write failure."""
    b64 = entry.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return str(save_b64_image(b64, prefix=prefix, extension=_extension_for(entry.get("media_type"))))
    url = entry.get("url")
    if isinstance(url, str) and url.strip():
        return str(save_url_image(url.strip(), prefix=prefix))
    return None


def _image_api_extra(
    payload: Dict[str, Any], saved: List[str], usable_refs: List[str], notes: List[str], body: Any
) -> Dict[str, Any]:
    """Success ``extra`` for an Image API result (knobs sent, extra images, usage)."""
    extra: Dict[str, Any] = {"endpoint": "images/generations", "exact_aspect_ratio": payload.get("aspect_ratio")}
    extra.update({key: payload[key] for key in _IMAGE_API_EXTRA_KEYS if key in payload})
    if len(saved) > 1:
        extra["additional_images"] = saved[1:]
    if usable_refs:
        extra["reference_images_used"] = len(payload.get("input_references") or [])
    if notes:
        extra["notes"] = notes
    usage = _dict_at(body, "usage")
    if isinstance(usage.get("cost"), (int, float)):
        extra["cost_usd"] = usage["cost"]
    if isinstance(usage.get("total_tokens"), int):
        extra["total_tokens"] = usage["total_tokens"]
    return extra


class OpenRouterCompatImageProvider(ImageGenProvider):
    """One instance per backend (OpenRouter, Nous Portal); they differ only in the runtime
    provider supplying ``(base_url, api_key)``, the config namespace, and Image API support."""

    def __init__(
        self, *, provider_name: str, display_name: str, runtime_name: str, config_key: str,
        model_env_var: str, setup_schema: Dict[str, Any], supports_image_api: bool = False,
    ) -> None:
        self._name = provider_name
        self._display = display_name
        self._runtime_name = runtime_name
        self._config_key = config_key
        self._model_env_var = model_env_var
        self._setup_schema = setup_schema
        self._live_models_cache: Optional[tuple] = None
        self._image_api_models_cache: Optional[tuple] = None
        self._supports_image_api = supports_image_api

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display

    def _credentials(self) -> Tuple[str, str]:
        """``(api_key, base_url)`` — either may be ``""``; raises on resolution failure."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested=self._runtime_name)
        return (
            str(runtime.get("api_key") or "").strip(), str(runtime.get("base_url") or "").strip().rstrip("/"),
        )

    def is_available(self) -> bool:
        try:
            return bool(self._credentials()[0])
        except Exception as exc:  # noqa: BLE001 - treat resolution failure as unavailable
            logger.debug("%s runtime resolution failed: %s", self._name, exc)
            return False

    def capabilities(self) -> Dict[str, Any]:
        # Report the reference cap of the model that would service the next call.
        max_refs = _MAX_REFERENCE_IMAGES
        if self._supports_image_api:
            resolved = self._resolve_model_chain()[0]
            if resolved in _IMAGE_API_MODELS:
                max_refs = int(_IMAGE_API_MODELS[resolved].get("max_refs") or max_refs)
        return {"modalities": ["text", "image"], "max_reference_images": max_refs}

    def list_models(self) -> List[Dict[str, Any]]:
        """Live catalog: OpenRouter = ``GET /images/models`` ∪ chat-completions image models (new
        releases selectable); Nous Portal = chat-completions only. Offline: default chain + snapshot."""
        merged: Dict[str, Dict[str, Any]] = {}
        if self._supports_image_api:
            merged = {entry["id"]: entry for entry in self._image_api_live_models()}
        for entry in self._live_models():
            merged.setdefault(entry["id"], entry)
        if merged:
            return sorted(merged.values(), key=lambda m: (_MODEL_PRIORITY.get(m["id"], 2), m["id"]))
        models = [{"id": model_id, **meta} for model_id, meta in _KNOWN_MODEL_META.items()]
        if self._supports_image_api:
            models.extend(
                {"id": model_id, "display": meta["display"], "strengths": f"{meta['strengths']} (Image API)"}
                for model_id, meta in _IMAGE_API_MODELS.items())
        return models

    def _cached_catalog(self, attr: str, label: str, **fetch: Any) -> List[Dict[str, Any]]:
        """Cached (per TTL) live catalog rows for this backend; ``[]`` when unreachable."""
        cached = getattr(self, attr)
        if cached is not None and time.monotonic() - cached[1] < _LIVE_CACHE_TTL:
            return cached[0]
        models: List[Dict[str, Any]] = []
        try:
            api_key, base_url = self._credentials()
            if base_url:
                models = _fetch_catalog(base_url, api_key, **fetch)
        except Exception as exc:  # noqa: BLE001 - offline/unauth → fallback path
            logger.debug("%s live %s unavailable: %s", self._name, label, exc)
            models = []
        setattr(self, attr, (models, time.monotonic()))
        return models

    def _image_api_live_models(self) -> List[Dict[str, Any]]:
        return self._cached_catalog(
            "_image_api_models_cache", "Image API catalog", path="/images/models", meta=_IMAGE_API_MODELS,
            generic="Image API model (from live OpenRouter catalog)", image_output_only=False)

    def _live_models(self) -> List[Dict[str, Any]]:
        return self._cached_catalog(
            "_live_models_cache", "image model catalog", path="/models", meta=_KNOWN_MODEL_META,
            generic="Image-output model (from live OpenRouter catalog)", image_output_only=True)

    def default_model(self) -> Optional[str]:
        # The catalog default, not the effective runtime model (_resolve_model_chain).
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return dict(self._setup_schema)

    def _resolve_model(self, explicit: Optional[str] = None) -> str:
        return self._resolve_model_chain(explicit)[0]

    def _resolve_model_chain(self, explicit: Optional[str] = None) -> list[str]:
        """``model`` kwarg → ``*_IMAGE_MODEL`` env → ``image_gen.<provider>.model`` →
        ``image_gen.model`` → default chain. Only the bare default chain carries a fallback."""
        for candidate in (explicit, os.environ.get(self._model_env_var)):
            if isinstance(candidate, str) and candidate.strip():
                return [candidate.strip()]
        cfg = _load_image_gen_config()
        for candidate in (_dict_at(cfg, self._config_key).get("model"), cfg.get("model")):
            if isinstance(candidate, str) and candidate.strip():
                return [candidate.strip()]
        return list(_DEFAULT_MODEL_CHAIN)

    def _generate_via_image_api(
        self, *, model_id: str, prompt: str, semantic_aspect: str, references: List[str],
        base_url: str, headers: Dict[str, str], kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """One ``POST {base_url}/images/generations`` attempt; a chain-retryable failure carries a
        private ``_retryable`` flag that :meth:`generate` strips."""
        base_fail = error_factory(self._name, semantic_aspect, model=model_id, prompt=prompt)

        def _fail(error: str, error_type: str, retryable: bool = False) -> Dict[str, Any]:
            response = base_fail(error, error_type)
            if retryable:
                response["_retryable"] = True
            return response

        # Not the chat path's `content`: that is clamped to 3 refs, these models take up to 16.
        usable_refs: List[str] = []
        unreadable: List[str] = []
        try:
            for ref in references:
                part = _to_image_url_part(ref)
                (usable_refs if part else unreadable).append(part or str(ref))
        except Exception as exc:  # noqa: BLE001 - blocked by the file-safety guard
            return _fail(f"Could not load reference image: {exc}", "io_error")
        # An edit with no readable source must not silently bill a text-to-image picture.
        if unreadable and not usable_refs:
            return _fail(
                "Could not read the reference image(s) requested for editing: "
                + ", ".join(unreadable) + ". Refusing to silently fall back to text-to-image.",
                "io_error")
        payload, notes = _build_image_api_payload(
            model_id=model_id, prompt=prompt, semantic_aspect=semantic_aspect,
            references=usable_refs, config_key=self._config_key, kwargs=kwargs)
        if unreadable:
            notes.insert(0, f"dropped unreadable reference image(s): {', '.join(unreadable)}")

        timeout = _REQUEST_TIMEOUT
        configured = _image_api_setting("timeout", kwargs.get("timeout"), self._config_key)
        try:
            if configured is not None:
                timeout = max(1.0, float(configured))
        except (TypeError, ValueError):
            logger.debug("%s: ignoring non-numeric image API timeout %r", self._name, configured)

        # (connect, read): an unreachable endpoint fails in seconds, not the whole read budget.
        body, failure = post_json(
            f"{base_url}/images/generations", headers=headers, payload=payload,
            timeout=(min(_IMAGE_API_CONNECT_TIMEOUT, timeout), timeout), label=self._display,
            error_message=lambda resp, exc: _extract_image_api_error(resp, str(exc)),
            catch_request_exception=True)
        if failure is not None:
            if failure.kind != "http":
                return _fail(failure.error, failure.error_type, retryable=failure.kind == "timeout")
            status, message = failure.status, failure.message
            logger.error("%s image API generation failed (%s) on %s: %s", self._name, status, model_id, message)
            # 401/403 first so an account-level rejection gets one error_type anywhere in the chain.
            if status in (401, 403):
                return _fail(f"{self._display} rejected the API key ({status}): {message}", "auth_error")
            if status == 404:
                return _fail(
                    f"Model '{model_id}' does not exist on the OpenRouter Image API "
                    f"(its catalog is separate from chat-completions — check "
                    f"GET {base_url}/images/models).",
                    "model_access", retryable=True)
            return _fail(failure.error, "api_error", retryable=status in _IMAGE_API_FALLBACK_STATUSES)

        entries = [e for e in _list_at(body, "data") if isinstance(e, dict)]
        if not entries:
            return _fail(
                f"{self._display} returned no image data for '{model_id}'.", "empty_response", retryable=True,
            )
        prefix = f"{self._name}_{_model_slug(model_id)}"
        try:
            saved = [p for p in (_save_image_api_entry(e, prefix) for e in entries) if p]
        except Exception as exc:  # noqa: BLE001
            return _fail(f"Could not save generated image: {exc}", "io_error")
        if not saved:
            return _fail(f"{self._display} response carried neither b64_json nor url.", "empty_response")
        return success_response(
            image=saved[0], model=model_id, prompt=prompt, aspect_ratio=semantic_aspect, provider=self._name,
            modality="image" if usable_refs else "text",
            extra=_image_api_extra(payload, saved, usable_refs, notes, body))

    def _generate_via_chat(
        self, *, model_id: str, prompt: str, aspect: str, content: List[Dict[str, Any]],
        base_url: str, headers: Dict[str, str],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """One ``/chat/completions`` attempt: ``(result, retry_reason)``; reason set when the
        chain may continue with the fallback model."""
        fail = error_factory(self._name, aspect, model=model_id, prompt=prompt)
        payload: Dict[str, Any] = {
            "model": model_id,
            "modalities": ["image", "text"],
            "messages": [{"role": "user", "content": content}],
            "image_config": {"aspect_ratio": _ASPECT_RATIOS.get(aspect, "1:1")},
        }
        result, failure = post_json(
            f"{base_url}/chat/completions", headers=headers, payload=payload, timeout=_REQUEST_TIMEOUT,
            label=self._display)
        if failure is not None:
            if failure.kind != "http":
                reason = "timed out" if failure.kind == "timeout" else None
                return fail(failure.error, failure.error_type), reason
            status, message = failure.status, failure.message
            logger.error("%s image gen failed (%d) on %s: %s", self._name, status, model_id, message)
            hint = _access_error_hint(self._display, model_id, self._model_env_var, status, message)
            if hint:
                return fail(hint, "model_access"), "unavailable"
            return fail(failure.error, "api_error"), None

        images = _extract_images(result)
        if not images:
            # Text but no image usually means the model didn't honor image output.
            return fail(
                f"{self._display} returned no image. Ensure the model '{model_id}' supports image output.",
                "empty_response",
            ), "returned no image"
        first = images[0]
        try:
            if first.startswith("data:"):
                b64 = first.split(",", 1)[1] if "," in first else ""
                saved_path = save_b64_image(b64, prefix=f"{self._name}_gen")
            else:
                saved_path = save_url_image(first, prefix=f"{self._name}_gen")
        except Exception as exc:  # noqa: BLE001
            return fail(f"Could not save generated image: {exc}", "io_error"), None
        return success_response(
            image=str(saved_path), model=model_id, prompt=prompt, aspect_ratio=aspect, provider=self._name,
        ), None

    def generate(
        self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *,
        image_url: Optional[str] = None, reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        fail = error_factory(self._name, aspect_ratio)
        try:
            api_key, base_url = self._credentials()
        except Exception as exc:  # noqa: BLE001
            return fail(f"Could not resolve {self._display} credentials: {exc}", "missing_api_key")
        if not api_key or not base_url:
            return fail(
                f"No {self._display} credentials found. "
                f"Configure {self._display} in `hermes tools` → Image Generation.",
                "missing_api_key")

        model_chain = self._resolve_model_chain(kwargs.get("model"))
        aspect = resolve_aspect_ratio(aspect_ratio)
        # ``reference_images`` (pet generator, local paths) + the generic ``image_url`` surface.
        references = [str(ref) for ref in kwargs.get("reference_images") or []]
        if image_url:
            references.append(str(image_url))
        references.extend(str(ref) for ref in reference_image_urls or [])
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for ref in references[:_MAX_REFERENCE_IMAGES]:
            part = _to_image_url_part(ref)
            if part:
                content.append({"type": "image_url", "image_url": {"url": part}})
        headers = {"Authorization": f"Bearer {api_key}", **_ATTRIBUTION_HEADERS}

        last_error: Optional[Dict[str, Any]] = None
        for i, model_id in enumerate(model_chain):
            # Image API and chat models are not interchangeable: this is routing, not preference.
            surface = "chat"
            if self._supports_image_api:
                surface = _select_surface(model_id, base_url, api_key, self._config_key)
            if surface == "images":
                outcome = self._generate_via_image_api(
                    model_id=model_id, prompt=prompt, semantic_aspect=aspect,
                    references=references, base_url=base_url, headers=headers, kwargs=kwargs)
                reason = "failed on the image API" if outcome.pop("_retryable", False) else None
            else:
                outcome, reason = self._generate_via_chat(
                    model_id=model_id, prompt=prompt, aspect=aspect, content=content,
                    base_url=base_url, headers=headers)
            if outcome.get("success") or reason is None or i == len(model_chain) - 1:
                return outcome
            logger.info(
                "%s model %s %s; retrying with fallback %s", self._name, model_id, reason, model_chain[i + 1],
            )
            last_error = outcome

        return last_error or error_response(
            error=f"{self._display} image generation failed after trying all candidate models.",
            error_type="api_error", provider=self._name,
            model=model_chain[-1] if model_chain else "", prompt=prompt, aspect_ratio=aspect)


def _build_providers() -> List[OpenRouterCompatImageProvider]:
    return [
        OpenRouterCompatImageProvider(
            provider_name="openrouter", display_name="OpenRouter", runtime_name="openrouter",
            config_key="openrouter", model_env_var="OPENROUTER_IMAGE_MODEL", supports_image_api=True,
            setup_schema={
                "name": "OpenRouter (image)",
                "badge": "paid",
                "tag": "Gemini Flash Image, gpt-image-2, Krea 2, Qwen Image 3 & more via OpenRouter; uses OPENROUTER_API_KEY",
                "env_vars": [{
                    "key": "OPENROUTER_API_KEY", "prompt": "OpenRouter API key", "url": "https://openrouter.ai/keys",
                }],
            }),
        OpenRouterCompatImageProvider(
            provider_name="nous", display_name="Nous Portal", runtime_name="nous", config_key="nous",
            model_env_var="NOUS_IMAGE_MODEL",
            setup_schema={
                "name": "Nous Portal (image)",
                "badge": "subscription",
                "tag": "Reference-grounded image generation via Nous Portal (OpenRouter-backed)",
                "env_vars": [],
                "requires_nous_auth": True,
            }),
    ]


def register(ctx: Any) -> None:
    """Register the OpenRouter + Nous Portal image gen providers."""
    for provider in _build_providers():
        ctx.register_image_gen_provider(provider)
