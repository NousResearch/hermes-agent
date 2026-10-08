"""Image generation provider ABC.

Providers register via ``PluginContext.register_image_gen_provider()`` (from
``<repo>/plugins/image_gen/<name>/`` or ``~/.hermes/plugins/image_gen/<name>/``);
the one selected by ``image_gen.provider`` services every ``image_generate`` call.
One tool covers text-to-image and editing: ``image_url`` / ``reference_image_urls``
route to the provider's edit endpoint, otherwise text-to-image. Mirrors
``agent/video_gen_provider.py``. Response dicts come from :func:`success_response`
/ :func:`error_response` (``success, image, model, prompt, aspect_ratio, modality,
provider`` + ``error, error_type`` on failure).
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent import provider_media
from agent.provider_base import CatalogProviderBase

logger = logging.getLogger(__name__)


VALID_ASPECT_RATIOS: Tuple[str, ...] = ("landscape", "square", "portrait")
DEFAULT_ASPECT_RATIO = "landscape"


class ImageGenProvider(CatalogProviderBase):
    """Abstract base class for an image generation backend. Subclasses implement
    :attr:`name` and :meth:`generate`; ``list_models`` entries may add
    ``speed`` / ``strengths`` / ``price`` for the picker."""

    def capabilities(self) -> Dict[str, Any]:
        """``modalities`` (``"text"`` and/or ``"image"``) and ``max_reference_images``; optionally
        ``supports_upscale`` (bool) and ``creative_controls`` (names from the tool's creative-control
        vocabulary: ``creativity``, ``intensity``, ``complexity``, ``movement``). Surfaced in the
        dynamic tool schema so the model knows when ``image_url`` / ``upscale`` / each control is
        honored, and only declared controls are passed to :meth:`generate`; the text-only default
        keeps non-overriding providers backward compatible."""
        return {"modalities": ["text"], "max_reference_images": 0}

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate an image, or edit ``image_url`` (``reference_image_urls`` are extra
        style/composition refs, clamped to ``max_reference_images``); any source image
        routes to the edit endpoint. Return :func:`success_response` / :func:`error_response`.
        Unknown ``kwargs`` MUST be ignored (forward compat); ``upscale`` (bool) is a
        post-generation high-res pass, reported as ``upscaled: True`` in ``extra``."""


def resolve_aspect_ratio(value: Optional[str]) -> str:
    """Clamp to :data:`VALID_ASPECT_RATIOS`; invalid values coerce to landscape so
    the tool surface forgives agent mistakes instead of rejecting them."""
    v = value.strip().lower() if isinstance(value, str) else ""
    return v if v in VALID_ASPECT_RATIOS else DEFAULT_ASPECT_RATIO


def normalize_reference_images(value: Any) -> Optional[List[str]]:
    """Coerce a str or list into a clean list of non-blank strings; ``None`` when
    nothing usable remains so providers treat "no refs" as one sentinel."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    return [item.strip() for item in value if isinstance(item, str) and item.strip()] or None


_GENERATED_IMAGE_KIND = f"{provider_media.GENERATED_SUBDIR}/images"


def save_b64_image(b64_data: str, *, prefix: str = "image", extension: str = "png") -> Path:
    """Decode base64 image data into ``$HERMES_HOME/cache/generated/images/``; return the path."""
    return provider_media.save_b64(_GENERATED_IMAGE_KIND, b64_data, prefix=prefix, extension=extension)


_MAX_NATIVE_IMAGE_BYTES = 25 * 1024 * 1024
_MAX_NATIVE_IMAGE_PIXELS = 40_000_000
_NATIVE_IMAGE_FORMATS = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}
_NATIVE_IMAGE_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


def save_native_b64_image(b64_data: str, *, output_format: Optional[str] = None) -> Path:
    """Strict, bounded validation for paid server-side results, then atomic local delivery.

    Keep the older plugin helper's permissive contract unchanged. Provider IDs/format text
    never become paths, and diagnostics never include the encoded image.
    """
    import base64
    import binascii
    import io
    import warnings
    from PIL import Image

    if not isinstance(b64_data, str) or not b64_data:
        raise ValueError("Missing image base64 result.")
    if len(b64_data) > ((_MAX_NATIVE_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("Image encoded size exceeds the limit.")
    expected = None
    if output_format is not None:
        expected = _NATIVE_IMAGE_FORMATS.get(output_format.lower()) if isinstance(output_format, str) else None
        if expected is None:
            raise ValueError("Unsupported image output format.")
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid image base64.") from exc
    if not raw or len(raw) > _MAX_NATIVE_IMAGE_BYTES:
        raise ValueError("Image decoded size is empty or exceeds the limit.")
    signature = (
        "PNG" if raw.startswith(b"\x89PNG\r\n\x1a\n") else
        "JPEG" if raw.startswith(b"\xff\xd8\xff") else
        "WEBP" if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP" else None
    )
    if signature is None or (expected and signature != expected):
        raise ValueError("Image format/signature mismatch or unsupported image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != signature or image.width * image.height > _MAX_NATIVE_IMAGE_PIXELS:
                    raise ValueError("Invalid image format or pixel size.")
                image.verify()
            # JPEG's verify() is a no-op; a bounded decode also rejects truncated scan data.
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or oversized image data.") from exc
    return provider_media.save_bytes_atomic(
        _GENERATED_IMAGE_KIND, raw, prefix="responses_image", extension=_NATIVE_IMAGE_EXTENSIONS[signature],
    )


_URL_IMAGE_CONTENT_TYPES = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif",
}


def save_url_image(
    url: str, *, prefix: str = "image", timeout: float = 60.0, max_bytes: int = 25 * 1024 * 1024,
) -> Path:
    """Download an (often ephemeral) image URL into ``$HERMES_HOME/cache/generated/images/``. Raises on
    network / HTTP / oversize / empty errors so callers can fall back to the bare URL."""
    return provider_media.save_url(
        _GENERATED_IMAGE_KIND, url, prefix=prefix, timeout=timeout, max_bytes=max_bytes,
        chunk_size=64 * 1024, content_types=_URL_IMAGE_CONTENT_TYPES,
        url_extensions=("png", "jpg", "jpeg", "webp", "gif"), default_extension="png",
        label="Image", empty_error="Image at {url} returned 0 bytes; refusing to cache.",
    )


def success_response(
    *,
    image: str,
    model: str,
    prompt: str,
    aspect_ratio: str,
    provider: str,
    modality: str = "text",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Uniform success dict; ``extra`` keys are added without overriding standard ones."""
    payload: Dict[str, Any] = {
        "success": True, "image": image, "model": model, "prompt": prompt,
        "aspect_ratio": aspect_ratio, "modality": modality, "provider": provider,
    }
    for k, v in (extra or {}).items():
        payload.setdefault(k, v)
    return payload


def error_response(
    *,
    error: str,
    error_type: str = "provider_error",
    provider: str = "",
    model: str = "",
    prompt: str = "",
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
) -> Dict[str, Any]:
    """Build a uniform error response dict."""
    return {
        "success": False, "image": None, "error": error, "error_type": error_type,
        "model": model, "prompt": prompt, "aspect_ratio": aspect_ratio, "provider": provider,
    }
