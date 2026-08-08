"""Size constraints for images embedded in native model payloads.

This module intentionally operates on in-memory bytes.  The cached source file
is never modified.  Small images pass through byte-for-byte; oversized images
are decoded with Pillow, EXIF-oriented, and re-encoded as JPEG or PNG until the
complete base64 data-URL payload fits the configured limit.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The ChatGPT Codex transport's effective image ceiling varies with the full
# request envelope.  Keep enough headroom for prompts and tool schemas while
# preserving useful phone-photo resolution.  Both values remain configurable
# for providers with different proven limits.
DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES = 900_000
DEFAULT_NATIVE_IMAGE_MAX_DIMENSION = 2560


def _positive_int(raw: Any, default: int) -> int:
    """Return a positive integer config value, otherwise *default*."""
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw if raw > 0 else default
    if isinstance(raw, str):
        try:
            parsed = int(raw.strip())
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
    return default


def agent_config_int(key: str, default: int) -> int:
    """Positive-int value of ``agent.<key>`` from the cached configuration.

    Malformed, missing, zero, or negative values fall back to ``default``.
    The config loader caches on file mtime, so this is safe on hot paths.
    """
    agent_cfg: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        raw_agent_cfg = cfg.get("agent") if isinstance(cfg, dict) else None
        if isinstance(raw_agent_cfg, dict):
            agent_cfg = raw_agent_cfg
    except Exception as exc:  # pragma: no cover - defensive config fallback
        logger.debug("image_payloads: could not load config; using defaults: %s", exc)

    return _positive_int(agent_cfg.get(key), default)


def get_native_image_limits() -> tuple[int, int]:
    """Load native-image limits from the cached Hermes configuration.

    ``native_image_max_payload_bytes`` limits the complete ASCII data URL
    (``data:<mime>;base64,<encoded bytes>``), not the raw file size.
    Malformed, missing, zero, or negative values fall back to safe defaults.
    """
    return (
        agent_config_int(
            "native_image_max_payload_bytes",
            DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES,
        ),
        agent_config_int(
            "native_image_max_dimension",
            DEFAULT_NATIVE_IMAGE_MAX_DIMENSION,
        ),
    )


def data_url_payload_size(raw_size: int, mime: str) -> int:
    """Return the exact ASCII byte length of a base64 data URL."""
    encoded_size = 4 * ((max(0, raw_size) + 2) // 3)
    return len(f"data:{mime};base64,") + encoded_size


def _has_transparency(image: Any) -> bool:
    """Conservatively detect images that must retain an alpha channel."""
    try:
        if "A" in image.getbands():
            return True
    except Exception:
        pass
    return "transparency" in getattr(image, "info", {})


def _encode_candidate(image: Any, output_format: str, quality: Optional[int]) -> bytes:
    buf = BytesIO()
    if output_format == "JPEG":
        image.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling="4:2:0",
        )
    else:
        image.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def constrain_image_payload(
    raw: bytes,
    mime: str,
    *,
    max_payload_bytes: int,
    max_dimension: int,
) -> Optional[tuple[bytes, str]]:
    """Return provider-safe image bytes/MIME within native payload limits.

    If the complete data URL is within ``max_payload_bytes`` and the longest
    image side is within ``max_dimension``, the original bytes and MIME are
    returned unchanged.  Otherwise Pillow processes the image entirely in
    memory: the first frame is selected for animated inputs, EXIF orientation
    is applied, and the image is encoded as JPEG (opaque) or PNG (transparent).
    JPEG quality is reduced before dimensions are reduced.  Images are never
    upscaled.

    ``None`` is a safe failure result for oversized images Pillow cannot decode
    or images that cannot fit even at minimum dimensions.  Callers should skip
    that attachment rather than sending the original oversized payload.
    """
    max_payload_bytes = _positive_int(
        max_payload_bytes, DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES
    )
    max_dimension = _positive_int(
        max_dimension, DEFAULT_NATIVE_IMAGE_MAX_DIMENSION
    )

    over_payload = data_url_payload_size(len(raw), mime) > max_payload_bytes
    requires_processing = over_payload

    try:
        from PIL import Image, ImageOps
    except ImportError:
        if over_payload:
            logger.warning(
                "image_payloads: Pillow unavailable; skipping oversized native image"
            )
            return None
        # Pillow is a core dependency, but preserve a byte-safe attachment if a
        # broken/minimal installation lacks it. Dimension validation is the only
        # unavailable check in this fallback.
        return raw, mime

    try:
        with Image.open(BytesIO(raw)) as opened:
            over_dimension = max(opened.size) > max_dimension
            requires_processing = over_payload or over_dimension
            if not requires_processing:
                return raw, mime

            frame_count = int(getattr(opened, "n_frames", 1) or 1)
            if frame_count > 1:
                logger.info(
                    "image_payloads: oversized animated image has %d frames; "
                    "using the first frame",
                    frame_count,
                )
                opened.seek(0)

            image = ImageOps.exif_transpose(opened)
            image.load()
            # Detach from the source context before it closes.
            image = image.copy()
    except Exception as exc:
        if requires_processing:
            logger.warning(
                "image_payloads: Pillow could not process oversized image; "
                "skipping attachment: %s",
                exc,
            )
            return None
        # A byte-safe corrupt/mislabelled image follows the historical
        # best-effort behavior; provider validation remains the final arbiter.
        logger.debug(
            "image_payloads: could not inspect byte-safe image dimensions; "
            "preserving original bytes: %s",
            exc,
        )
        return raw, mime

    original_dimensions = image.size
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        constrained_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(constrained_size, Image.Resampling.LANCZOS)

    transparent = _has_transparency(image)
    if transparent:
        output_format = "PNG"
        output_mime = "image/png"
        if image.mode not in {"RGBA", "LA", "P"}:
            image = image.convert("RGBA")
        quality_steps: tuple[Optional[int], ...] = (None,)
    else:
        output_format = "JPEG"
        output_mime = "image/jpeg"
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        # Search from high to low quality so we use the available transport
        # budget instead of jumping directly from 85 to 75.  The finer upper
        # range matters for phone photos, where a small quality-step difference
        # can leave substantial transport budget unused.
        quality_steps = (
            95,
            92,
            90,
            89,
            88,
            87,
            85,
            82,
            80,
            78,
            75,
            72,
            70,
            68,
            65,
            62,
            60,
            58,
            55,
            50,
            45,
            40,
            35,
        )

    best_size: Optional[int] = None
    for _attempt in range(12):
        for quality in quality_steps:
            try:
                candidate = _encode_candidate(image, output_format, quality)
            except Exception as exc:
                logger.warning(
                    "image_payloads: failed to encode constrained image as %s: %s",
                    output_format,
                    exc,
                )
                return None
            payload_size = data_url_payload_size(len(candidate), output_mime)
            best_size = payload_size if best_size is None else min(best_size, payload_size)
            if payload_size <= max_payload_bytes:
                if image.size != original_dimensions:
                    logger.info(
                        "image_payloads: constrained native image dimensions "
                        "%dx%d -> %dx%d",
                        original_dimensions[0],
                        original_dimensions[1],
                        image.width,
                        image.height,
                    )
                return candidate, output_mime

        if image.size == (1, 1):
            break

        # Estimate the linear scale needed from the smallest candidate at this
        # size, with headroom for codec non-linearity.  Cap at 0.9 so every
        # unsuccessful round makes measurable progress.
        assert best_size is not None
        scale = min(0.9, (max_payload_bytes / best_size) ** 0.5 * 0.95)
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        if new_size == image.size:
            new_size = (max(1, image.width - 1), max(1, image.height - 1))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        best_size = None

    logger.warning(
        "image_payloads: could not fit native image under %d-byte data-URL "
        "payload limit; skipping attachment",
        max_payload_bytes,
    )
    return None


__all__ = [
    "DEFAULT_NATIVE_IMAGE_MAX_DIMENSION",
    "DEFAULT_NATIVE_IMAGE_MAX_PAYLOAD_BYTES",
    "constrain_image_payload",
    "data_url_payload_size",
    "get_native_image_limits",
]
