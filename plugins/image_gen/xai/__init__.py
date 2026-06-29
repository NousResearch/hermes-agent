"""xAI image generation backend.

Exposes xAI's Grok Imagine image models as an
:class:`ImageGenProvider` implementation, aligned with the current official
xAI Imagine API (https://docs.x.ai/developers/model-capabilities/imagine).

Features:
- Text-to-image generation (``/v1/images/generations``)
- Image editing (``/v1/images/edits``) — single image (JSON ``image`` object),
  xAI Files API ``file_id`` inputs, and multi-image edit (``images: [...]``,
  up to 3 sources)
- Official aspect ratios plus the Hermes ``landscape``/``square``/``portrait``
  aliases
- Optional multi-output (``n``) — first output stays in ``result["image"]``;
  all outputs are reported in ``result["images"]``
- Optional ``storage_options`` pass-through (default-off; never injects a
  public URL)
- Resolutions (1K, 2K)
- Base64 output saved to cache with a MIME-derived extension

Selection precedence (first hit wins):
1. ``XAI_IMAGE_MODEL`` env var
2. ``image_gen.xai.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`

Unknown or deprecated model ids soft-fall back to :data:`DEFAULT_MODEL`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    is_data_uri,
    is_http_url,
    local_image_to_data_uri,
    normalize_reference_images,
    save_b64_image,
    save_url_image,
    success_response,
)
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

_MODELS: Dict[str, Dict[str, Any]] = {
    # Quality is the current official recommendation (the former
    # ``grok-imagine-image-pro`` was deprecated by xAI on 2026-05-15). List it
    # first so it is the catalog default surfaced by ``default_model()``.
    "grok-imagine-image-quality": {
        "display": "Grok Imagine Image (Quality)",
        "speed": "~10-20s",
        "strengths": "Higher fidelity / detail; current official default.",
    },
    "grok-imagine-image": {
        "display": "Grok Imagine Image",
        "speed": "~5-10s",
        "strengths": "Faster / lower-cost standard model.",
    },
}

# Deprecated ids (e.g. ``grok-imagine-image-pro``) are intentionally absent so
# they can never resolve as the active model; ``_resolve_model()`` soft-falls
# back to this default for unknown/deprecated selections.
DEFAULT_MODEL = "grok-imagine-image-quality"

# ---------------------------------------------------------------------------
# Aspect ratio (xAI-specific). The shared three-value resolve_aspect_ratio()
# in agent.image_gen_provider clamps everything to landscape/square/portrait,
# which would make the official wire ratios below unreachable — so xAI payloads
# resolve their own ratios via _resolve_xai_aspect_ratio().
# ---------------------------------------------------------------------------

# Hermes user-facing aliases -> official xAI wire ratio.
_XAI_ASPECT_ALIASES = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# Official xAI aspect-ratio wire values
# (https://docs.x.ai/developers/model-capabilities/images/generation).
_XAI_OFFICIAL_ASPECT_RATIOS = {
    "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2",
    "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto",
}

# Landscape is the Hermes default; 16:9 is its official wire value.
_DEFAULT_XAI_WIRE_RATIO = "16:9"

# xAI resolutions
_XAI_RESOLUTIONS = {"1k", "2k"}

DEFAULT_RESOLUTION = "1k"

# Response MIME type -> local cache file extension (FR8). Unknown/missing
# falls back to "png".
_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
}

# xAI storage_options expiry bounds (seconds): 1 hour .. 30 days.
_STORAGE_EXPIRY_MIN = 3600
_STORAGE_EXPIRY_MAX = 2592000


def _resolve_xai_aspect_ratio(value: Any) -> str:
    """Resolve a requested aspect ratio to an official xAI wire value (FR2).

    Accepts Hermes aliases (``landscape``/``square``/``portrait``) and the
    official wire ratios (``16:9``, ``4:3``, ``20:9``, ``auto``, ...). Invalid
    or non-string input soft-falls back to the landscape default wire ratio
    rather than raising, so a bad value never crashes a generation call.
    """
    if not isinstance(value, str):
        return _DEFAULT_XAI_WIRE_RATIO
    v = value.strip().lower()
    if v in _XAI_ASPECT_ALIASES:
        return _XAI_ASPECT_ALIASES[v]
    if v in _XAI_OFFICIAL_ASPECT_RATIOS:
        return v
    return _DEFAULT_XAI_WIRE_RATIO


def _echo_aspect_ratio(value: Any) -> str:
    """Return the caller-facing ``aspect_ratio`` echo (FR10 backward-compat).

    Preserves the caller's recognized alias / official ratio in the result
    dict (e.g. ``square`` stays ``square`` even though the wire payload carried
    ``1:1``). Unrecognized values fall back to the landscape default.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _XAI_ASPECT_ALIASES or v in _XAI_OFFICIAL_ASPECT_RATIOS:
            return v
    return DEFAULT_ASPECT_RATIO


def _ext_from_mime(mime: Any) -> str:
    """Map a response ``mime_type`` to a local cache file extension (FR8)."""
    return _MIME_TO_EXT.get(str(mime or "").strip().lower(), "png")


def _normalize_n(value: Any) -> Optional[int]:
    """Return a positive int for the xAI ``n`` field, or None to omit it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _is_valid_storage_expiry(value: Any) -> bool:
    """True when *value* is an int within xAI's documented expiry window."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _STORAGE_EXPIRY_MIN <= value <= _STORAGE_EXPIRY_MAX
    )


def _validate_storage_options(
    opts: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate caller-supplied ``storage_options`` (FR7).

    Returns ``(opts, None)`` when safe to pass through unchanged, or
    ``(None, error_message)`` otherwise. Never injects keys — in particular it
    never adds ``public_url``, preserving the privacy default (no public URL
    for private user media unless the caller explicitly requested one).
    """
    if not isinstance(opts, dict):
        return None, "storage_options must be an object/dict"
    filename = opts.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        return None, "storage_options.filename is required when storage_options is provided"
    if "expires_after" in opts and not _is_valid_storage_expiry(opts["expires_after"]):
        return None, (
            f"storage_options.expires_after must be an integer between "
            f"{_STORAGE_EXPIRY_MIN} and {_STORAGE_EXPIRY_MAX} seconds"
        )
    if "public_url" in opts:
        public_url = opts["public_url"]
        if isinstance(public_url, bool):
            pass
        elif isinstance(public_url, dict):
            if "expires_after" in public_url and not _is_valid_storage_expiry(
                public_url["expires_after"]
            ):
                return None, (
                    f"storage_options.public_url.expires_after must be an integer "
                    f"between {_STORAGE_EXPIRY_MIN} and {_STORAGE_EXPIRY_MAX} seconds"
                )
        else:
            return None, "storage_options.public_url must be a boolean or an object"
    return opts, None


def _build_xai_image_ref(ref: Any) -> Dict[str, Any]:
    """Normalize one edit-input reference into an xAI image object (FR4/FR5).

    Accepts an http(s) URL or data URI (passed by reference), an xAI Files API
    id (``file_...`` -> ``{"file_id": ...}``), or a local filesystem path
    (validated and inlined as a base64 data URI). Raises :class:`ValueError`
    on any unusable input so callers can surface a clear ``invalid_input``.
    A ``file_...`` id is never probed on the local filesystem.
    """
    s = str(ref or "").strip()
    if not s:
        raise ValueError(
            "image reference is empty (expected a local path, http(s) URL, "
            "data URI, or xAI file id)"
        )
    if s.startswith("file_"):
        if len(s) <= len("file_"):
            raise ValueError("xAI file id is empty; expected a non-empty id like 'file_abc123'")
        return {"file_id": s}
    if is_http_url(s) or is_data_uri(s):
        return {"url": s, "type": "image_url"}
    return {"url": local_image_to_data_uri(s), "type": "image_url"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_xai_config() -> Dict[str, Any]:
    """Read ``image_gen.xai`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        xai_section = section.get("xai") if isinstance(section, dict) else None
        return xai_section if isinstance(xai_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.xai config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which model to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("XAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_xai_config()
    candidate = cfg.get("model") if isinstance(cfg.get("model"), str) else None
    if candidate and candidate in _MODELS:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _resolve_resolution() -> str:
    """Get configured resolution."""
    cfg = _load_xai_config()
    res = cfg.get("resolution") if isinstance(cfg.get("resolution"), str) else None
    if res and res in _XAI_RESOLUTIONS:
        return res
    return DEFAULT_RESOLUTION


# ---------------------------------------------------------------------------
# Shared HTTP helpers (used by both generate and edit)
# ---------------------------------------------------------------------------


def _xai_headers(api_key: str) -> Dict[str, str]:
    """Build the JSON request headers for an xAI image endpoint."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": hermes_xai_user_agent(),
    }


def _post_xai_image(
    endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    *,
    op_label: str,
    provider_name: str,
    model_id: str,
    prompt: str,
    aspect: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """POST to an xAI image endpoint and return ``(result_json, error_dict)``.

    Exactly one slot is non-None. Centralizes the HTTP error / timeout /
    connection / invalid-JSON handling shared by image generation and image
    editing. ``op_label`` ("image generation" / "image edit") is woven into
    error messages so each surface reads naturally.
    """
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
    except requests.HTTPError as exc:
        response = exc.response
        status = response.status_code if response is not None else 0
        try:
            err_msg = response.json().get("error", {}).get("message", response.text[:300])
        except Exception:
            err_msg = response.text[:300] if response is not None else str(exc)
        logger.error("xAI %s failed (%d): %s", op_label, status, err_msg)
        return None, error_response(
            error=f"xAI {op_label} failed ({status}): {err_msg}",
            error_type="api_error",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )
    except requests.Timeout:
        return None, error_response(
            error=f"xAI {op_label} timed out (120s)",
            error_type="timeout",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )
    except requests.ConnectionError as exc:
        return None, error_response(
            error=f"xAI connection error: {exc}",
            error_type="connection_error",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    try:
        return response.json(), None
    except Exception as exc:
        return None, error_response(
            error=f"xAI returned invalid JSON: {exc}",
            error_type="invalid_response",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )


def _extract_and_cache_images(
    result: Dict[str, Any],
    *,
    provider_name: str,
    model_id: str,
    prompt: str,
    aspect: str,
    prefix: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Process every ``data[i]`` entry from an xAI image response (FR3/FR8).

    Returns ``(outputs, error_dict)`` with exactly one slot non-None.
    ``outputs`` is a list of per-image dicts, each carrying at minimum an
    ``image`` ref (local cache path or bare URL) plus ``mime_type`` /
    ``file_output`` when the response provides them. ``b64_json`` is decoded
    and saved with a MIME-derived extension; a ``url`` is downloaded — falling
    back to the bare URL when caching fails, mirroring the generation path
    (xAI's ``imgen.x.ai`` URLs expire fast, so a cache miss must not become a
    hard tool error). A response with no usable entries is an
    ``empty_response``.
    """
    data = result.get("data")
    if not isinstance(data, list) or not data:
        return None, error_response(
            error="xAI returned no image data",
            error_type="empty_response",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    outputs: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        url = item.get("url")
        mime = item.get("mime_type")
        file_output = item.get("file_output")

        if b64:
            try:
                ref = str(save_b64_image(b64, prefix=prefix, extension=_ext_from_mime(mime)))
            except Exception as exc:
                return None, error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider=provider_name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
        elif url:
            try:
                ref = str(save_url_image(url, prefix=prefix))
            except Exception as exc:
                logger.warning(
                    "xAI image URL %s could not be cached (%s); falling back to bare URL.",
                    url,
                    exc,
                )
                ref = url
        else:
            continue

        entry: Dict[str, Any] = {"image": ref}
        if mime:
            entry["mime_type"] = mime
        if file_output is not None:
            entry["file_output"] = file_output
        outputs.append(entry)

    if not outputs:
        return None, error_response(
            error="xAI response contained neither b64_json nor URL",
            error_type="empty_response",
            provider=provider_name,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
        )

    return outputs, None


def _build_response_extra(
    result: Dict[str, Any],
    outputs: List[Dict[str, Any]],
    *,
    base_extra: Optional[Dict[str, Any]] = None,
    n: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble additive success metadata for an xAI image response (FR3/FR8).

    Preserves the legacy contract — ``result["image"]`` (the first output)
    stays the primary ref — while surfacing first-image ``mime_type`` /
    ``file_output``, top-level ``storage_error`` / ``public_url_error`` /
    ``usage`` when present, and the full ``images`` list when there is more
    than one output (or ``n`` explicitly requested multiple).
    """
    extra: Dict[str, Any] = dict(base_extra or {})
    first = outputs[0]
    if "mime_type" in first:
        extra["mime_type"] = first["mime_type"]
    if "file_output" in first:
        extra["file_output"] = first["file_output"]
    for key in ("storage_error", "public_url_error", "usage"):
        if key in result and result[key] is not None:
            extra[key] = result[key]
    if len(outputs) > 1 or (isinstance(n, int) and n > 1):
        extra["images"] = outputs
    return extra


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class XAIImageGenProvider(ImageGenProvider):
    """xAI ``grok-imagine-image`` backend."""

    @property
    def name(self) -> str:
        return "xai"

    @property
    def display_name(self) -> str:
        return "xAI (Grok)"

    def is_available(self) -> bool:
        creds = resolve_xai_http_credentials()
        return bool(creds.get("api_key"))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta.get("display", model_id),
                "speed": meta.get("speed", ""),
                "strengths": meta.get("strengths", ""),
            }
            for model_id, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        # Auth resolution is delegated to the shared ``xai_grok`` post_setup
        # hook (``hermes_cli/tools_config.py``); identical to the TTS / video
        # gen entries so users see the same OAuth-or-API-key choice for every
        # xAI service.
        return {
            "name": "xAI Grok Imagine (image)",
            "badge": "paid",
            "tag": "grok-imagine-image — text-to-image; uses xAI Grok OAuth or XAI_API_KEY",
            "env_vars": [],
            "post_setup": "xai_grok",
        }

    def capabilities(self) -> Dict[str, Any]:
        """Return xAI image input support for the shared image_generate tool.

        The general ``image_generate`` surface can pass ``image_url`` and
        ``reference_image_urls`` into :meth:`generate`; this provider routes
        those inputs to the same xAI edit implementation used by the dedicated
        ``image_edit`` tool. xAI documents up to three edit input images.
        """
        return {"modalities": ["text", "image"], "max_reference_images": 3}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate images, or route image inputs to xAI's edit endpoint.

        ``aspect_ratio`` accepts the Hermes aliases or any official xAI wire
        ratio (FR2). Optional keyword args (direct/provider callers only — the
        agent tool schema stays minimal): ``n`` for multi-output (FR3) and
        ``storage_options`` for Files API persistence (FR7, default-off, never
        injects a public URL). The first output stays in ``result["image"]``;
        all outputs (when more than one) are reported in ``result["images"]``.

        Backward compatibility with the shared image generation tool is
        preserved: when ``image_url`` or ``reference_image_urls`` are supplied,
        route through :meth:`edit` instead of silently ignoring image inputs.
        """
        source_images: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            source_images.append(image_url.strip())
        refs = normalize_reference_images(reference_image_urls)
        if refs:
            source_images.extend(refs)
        if source_images:
            edit_kwargs: Dict[str, Any] = {}
            for key in ("n", "storage_options"):
                if key in kwargs:
                    edit_kwargs[key] = kwargs[key]
            if len(source_images) == 1:
                return self.edit(
                    prompt=prompt,
                    image=source_images[0],
                    aspect_ratio=aspect_ratio,
                    **edit_kwargs,
                )
            return self.edit(
                prompt=prompt,
                image=None,
                aspect_ratio=aspect_ratio,
                images=source_images,
                **edit_kwargs,
            )

        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        provider_name = str(creds.get("provider") or "xai").strip() or "xai"
        aspect_echo = _echo_aspect_ratio(aspect_ratio)
        if not api_key:
            return error_response(
                error="No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.",
                error_type="missing_api_key",
                provider=provider_name,
                aspect_ratio=aspect_echo,
            )

        model_id, _meta = _resolve_model()
        xai_ar = _resolve_xai_aspect_ratio(aspect_ratio)
        resolution = _resolve_resolution()
        xai_res = resolution if resolution in _XAI_RESOLUTIONS else DEFAULT_RESOLUTION

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "aspect_ratio": xai_ar,
            "resolution": xai_res,
            # Prefer inline bytes over xAI's temporary imgen.x.ai URL: the URL
            # path can be blocked by CDN/IP policy before Hermes can cache it.
            "response_format": "b64_json",
        }

        # Optional multi-output (FR3). Only emitted when a positive int ``n`` is
        # supplied; the agent schema never exposes it.
        n = _normalize_n(kwargs.get("n"))
        if n is not None:
            payload["n"] = n

        # Optional storage_options pass-through (FR7). Validated for safe shape;
        # default-off and never adds public_url on the caller's behalf.
        storage_options = kwargs.get("storage_options")
        if storage_options is not None:
            validated, opt_err = _validate_storage_options(storage_options)
            if opt_err is not None:
                return error_response(
                    error=opt_err,
                    error_type="invalid_input",
                    provider=provider_name,
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect_echo,
                )
            payload["storage_options"] = validated

        headers = _xai_headers(api_key)
        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")

        result, error = _post_xai_image(
            f"{base_url}/images/generations",
            payload,
            headers,
            op_label="image generation",
            provider_name=provider_name,
            model_id=model_id,
            prompt=prompt,
            aspect=aspect_echo,
        )
        if error is not None:
            return error

        # xAI's Grok Imagine returns ephemeral ``imgen.x.ai/xai-tmp-*`` URLs
        # that 404 within minutes (#26942); ``_extract_and_cache_images``
        # materialises the bytes locally so the gateway has a stable path.
        outputs, error = _extract_and_cache_images(
            result,
            provider_name=provider_name,
            model_id=model_id,
            prompt=prompt,
            aspect=aspect_echo,
            prefix=f"xai_{model_id}",
        )
        if error is not None:
            return error

        return success_response(
            image=outputs[0]["image"],
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect_echo,
            provider="xai",
            extra=_build_response_extra(result, outputs, base_extra={"resolution": xai_res}, n=n),
        )

    # ------------------------------------------------------------------
    # Image edit / image-to-image
    # ------------------------------------------------------------------

    def supports_edit(self) -> bool:
        return True

    def edit(
        self,
        prompt: str,
        image: Optional[str] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Edit an image with xAI's ``/v1/images/edits`` endpoint.

        Single-image input (``image``) may be a local filesystem path (inlined
        as a base64 ``data:`` URI after validation), an ``http(s)`` URL, a
        ``data:`` URI, or an xAI Files API id (``file_...``). Per the official
        xAI docs the source is wrapped in an object —
        ``"image": {"url": "<url-or-data-uri>", "type": "image_url"}`` for URLs
        / data URIs, or ``"image": {"file_id": "file_..."}`` for stored files
        (https://docs.x.ai/developers/model-capabilities/images/editing).

        Multi-image edit (``images=[...]``, keyword-only, FR6) accepts up to 3
        sources of the same kinds and sends ``"images": [...]``. ``image`` and
        ``images`` are mutually exclusive. Note: for single-image edit the xAI
        output respects the *input* image ratio — ``aspect_ratio`` does not
        reliably control single-image edit output; for multi-image edit it
        overrides the first-input default.

        Optional keyword args mirror :meth:`generate`: ``n`` (multi-output,
        FR3) and ``storage_options`` (FR7). Reuses the same credential
        resolution, user-agent, error handling, and output caching.
        """
        creds = resolve_xai_http_credentials()
        api_key = str(creds.get("api_key") or "").strip()
        provider_name = str(creds.get("provider") or "xai").strip() or "xai"
        aspect_echo = _echo_aspect_ratio(aspect_ratio)

        def _invalid(message: str, *, model: str = "") -> Dict[str, Any]:
            return error_response(
                error=message,
                error_type="invalid_input",
                provider=provider_name,
                model=model,
                prompt=prompt if isinstance(prompt, str) else "",
                aspect_ratio=aspect_echo,
            )

        if not api_key:
            return error_response(
                error="No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.",
                error_type="missing_api_key",
                provider=provider_name,
                aspect_ratio=aspect_echo,
            )

        if not prompt or not str(prompt).strip():
            return _invalid("prompt is required and must be a non-empty string")

        images = kwargs.get("images")
        image_ref = str(image or "").strip()

        # ``image`` and ``images`` are mutually exclusive (FR6).
        if image_ref and images is not None:
            return _invalid("Provide either 'image' or 'images' for an edit, not both")

        model_id, _meta = _resolve_model()
        xai_ar = _resolve_xai_aspect_ratio(aspect_ratio)
        resolution = _resolve_resolution()
        xai_res = resolution if resolution in _XAI_RESOLUTIONS else DEFAULT_RESOLUTION

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            # Mirrors the generation surface; xAI uses this to override the
            # first-input default for multi-image edits and ignores it where
            # single-image output must follow the input ratio.
            "aspect_ratio": xai_ar,
            # xAI Imagine edits accept the same literal resolution values as
            # generations ("1k" / "2k"). Honor image_gen.xai.resolution so
            # reference-image edits do not silently downshift to service default.
            "resolution": xai_res,
            # Prefer inline bytes over xAI's ephemeral imgen.x.ai URLs.
            "response_format": "b64_json",
        }

        # Build the source-image payload. Local files are validated and inlined
        # as data URIs; http(s)/data URIs pass by reference; ``file_...`` ids
        # ride the Files API object and are never probed on disk.
        if images is not None:
            if not isinstance(images, list) or len(images) == 0:
                return _invalid(
                    "images must be a non-empty list of up to 3 image references",
                    model=model_id,
                )
            if len(images) > 3:
                return _invalid(
                    "xAI multi-image edit accepts at most 3 images",
                    model=model_id,
                )
            try:
                payload["images"] = [_build_xai_image_ref(ref) for ref in images]
            except ValueError as exc:
                return _invalid(str(exc), model=model_id)
        else:
            if not image_ref:
                return _invalid(
                    "image is required (local path, http(s) URL, data URI, or xAI "
                    "file id) — or provide 'images' for a multi-image edit",
                    model=model_id,
                )
            try:
                payload["image"] = _build_xai_image_ref(image_ref)
            except ValueError as exc:
                return _invalid(str(exc), model=model_id)

        # Optional multi-output (FR3) and storage_options (FR7), as in generate.
        n = _normalize_n(kwargs.get("n"))
        if n is not None:
            payload["n"] = n
        storage_options = kwargs.get("storage_options")
        if storage_options is not None:
            validated, opt_err = _validate_storage_options(storage_options)
            if opt_err is not None:
                return _invalid(opt_err, model=model_id)
            payload["storage_options"] = validated

        headers = _xai_headers(api_key)
        base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")

        result, error = _post_xai_image(
            f"{base_url}/images/edits",
            payload,
            headers,
            op_label="image edit",
            provider_name=provider_name,
            model_id=model_id,
            prompt=prompt,
            aspect=aspect_echo,
        )
        if error is not None:
            return error

        outputs, error = _extract_and_cache_images(
            result,
            provider_name=provider_name,
            model_id=model_id,
            prompt=prompt,
            aspect=aspect_echo,
            prefix=f"xai_edit_{model_id}",
        )
        if error is not None:
            return error

        return success_response(
            image=outputs[0]["image"],
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect_echo,
            provider="xai",
            modality="image",
            extra=_build_response_extra(result, outputs, base_extra={"resolution": xai_res}, n=n),
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(XAIImageGenProvider())
