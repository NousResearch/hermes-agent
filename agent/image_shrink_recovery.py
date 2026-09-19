"""Reactive image-shrink recovery: re-encode oversized native image parts.

Byte-verbatim extraction from ``agent/conversation_compression`` (Part of #79926, #78647);
``agent/conversation_compression`` re-exports every name below, same object, so patch
targets and ``module.attr`` callers keep working. Recovers from a provider rejecting an
image part for size (Anthropic's 5 MB / per-side pixel cap, Codex's tile-patch budget):
data-URL parts over the byte budget or over the pixel cap are re-encoded through
``tools.vision_tools._resize_image_for_vision``; http(s) image URLs are left untouched.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.conversation_compression")


# 4 MB leaves headroom under Anthropic's 5 MB; shrinking loses quality but only
# runs after a confirmed provider rejection, so the alternative is failure.
_IMAGE_SHRINK_TARGET_BYTES = 4 * 1024 * 1024
_IMAGE_SUFFIX_BY_MIME = {
    "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/bmp": ".bmp",
}


def _data_url_mime(header: str, default: str = "image/jpeg") -> str:
    """``image/*`` mime from a ``data:`` URL header, else ``default``."""
    if header.startswith("data:"):
        candidate = header[len("data:") :].split(";", 1)[0].strip()
        if candidate.startswith("image/"):
            return candidate
    return default


def _decode_pixels(data_url: str) -> Optional[tuple]:
    """``(width, height)`` of a base64 data URL; None when Pillow is missing or the payload is corrupt."""
    try:
        import base64, io
        _, _, data_d = data_url.partition(",")
        if not data_d or not data_url.startswith("data:"):
            return None
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(data_d))) as _img:
            return _img.size
    except Exception:
        return None


def _shrink_data_url(url: str, *, max_dimension: int, resize_fn: Any) -> tuple:
    """Return ``(resized_url, unshrinkable)`` for a data URL.
    ``resized_url`` is None when no rewrite applied. ``unshrinkable`` is True only when the image violated a
    constraint and resizing failed to satisfy that same constraint, so the caller knows a retry is pointless.
    The accept gate MUST use the axis that triggered the shrink: a pixel downscale can re-encode to MORE bytes
    (PNG non-monotonic); a byte-only reject wedges."""
    target_bytes = _IMAGE_SHRINK_TARGET_BYTES
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, False
    triggered_by = "bytes" if len(url) > target_bytes else None  # over byte budget
    if triggered_by is None:
        # Bytes fine; check pixels against the provider cap (tiny bytes, huge pixels).
        dims = _decode_pixels(url)
        if dims is None or max(dims) <= max_dimension:
            return None, False
        triggered_by = "dimension"
    try:
        header, _, data = url.partition(",")
        mime = _data_url_mime(header)
        import base64 as _b64
        raw = _b64.b64decode(data)
        tmp = tempfile.NamedTemporaryFile(
            prefix="hermes_shrink_", suffix=_IMAGE_SUFFIX_BY_MIME.get(mime, ".jpg"), delete=False
        )
        try:
            tmp.write(raw)
            tmp.close()
            resized = resize_fn(
                Path(tmp.name), mime_type=mime, max_base64_bytes=target_bytes, max_dimension=max_dimension
            )
        finally:
            with contextlib.suppress(Exception):
                Path(tmp.name).unlink(missing_ok=True)
        if not resized:
            return None, True  # Pillow couldn't help
        new_dims = _decode_pixels(resized)
        if triggered_by == "bytes":
            # Byte budget is binding — bytes must shrink; and the resizer may return an
            # over-cap blob (long side freezes at the 64px short-side floor) → still 400.
            ok = len(resized) < len(url) and (new_dims is None or max(new_dims) <= max_dimension)
        elif new_dims is not None:
            # Dimension cap is binding: accept a byte-larger re-encode if now within cap.
            ok = max(new_dims) <= max_dimension
        else:
            # Can't verify dimensions: fall back to the bytes-must-shrink gate so we never
            # accept an unverifiable byte-larger blob.
            ok = len(resized) < len(url)
        return (resized, False) if ok else (None, True)
    except Exception as exc:
        logger.warning("image-shrink recovery: re-encode failed — %s", exc)
        return None, triggered_by is not None


def _source_to_data_url(source: Any) -> Optional[str]:
    """Anthropic ``{"type": "base64", ...}`` image source → data URL, else None."""
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None
    media_type = str(source.get("media_type") or "image/jpeg").strip()
    return f"data:{media_type if media_type.startswith('image/') else 'image/jpeg'};base64,{data}"


def _write_data_url_to_source(source: dict, data_url: str) -> dict:
    """Return a NEW source dict carrying the re-encoded payload.
    Copy-on-write: parts may be shared with the persistent history, so mutating in place would store the
    degraded image; the caller replaces the part."""
    header, _, data = data_url.partition(",")
    return {**source, "type": "base64", "media_type": _data_url_mime(header), "data": data}


def try_shrink_image_parts_in_messages(api_messages: list, *, max_dimension: int = 8000) -> bool:
    """Re-encode oversized native image parts to recover from image-too-large errors.
    Mutates ``api_messages`` in place. Returns True if any part was replaced, False if nothing to shrink or
    Pillow could not help. Targets data-URL parts over 4 MB or ``max_dimension`` (Anthropic's per-side pixel
    cap, parsed from the rejection by the caller); http(s) image URLs are left untouched."""
    if not api_messages:
        return False
    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False
    changed_count = 0
    # Track over-target parts that could not be shrunk: if any remain, a retry
    # re-sends the same payload and wastes the single retry budget.
    unshrinkable_oversized = 0

    def _shrink(url: Any) -> tuple:
        return _shrink_data_url(url, max_dimension=max_dimension, resize_fn=_resize_image_for_vision)

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Copy-on-write: part/source dicts can alias stored history, so build a new
        # content list and reassign msg["content"] on the per-call copy.
        new_content: list | None = None
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            replacement = None
            if ptype == "image":
                source = part.get("source")
                resized, unshrinkable = _shrink(_source_to_data_url(source) or "")
                if resized and isinstance(source, dict):
                    replacement = {**part, "source": _write_data_url_to_source(source, resized)}
            elif ptype in {"image_url", "input_image"}:
                image_value = part.get("image_url")
                # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
                # OpenAI Responses: {"image_url": "data:..."}
                if isinstance(image_value, dict):
                    resized, unshrinkable = _shrink(image_value.get("url", ""))
                    if resized:
                        replacement = {**part, "image_url": {**image_value, "url": resized}}
                elif isinstance(image_value, str):
                    resized, unshrinkable = _shrink(image_value)
                    if resized:
                        replacement = {**part, "image_url": resized}
                else:
                    continue
            else:
                continue
            if replacement is not None:
                if new_content is None:
                    new_content = list(content)
                new_content[part_idx] = replacement
                changed_count += 1
            elif unshrinkable:
                unshrinkable_oversized += 1
        if new_content is not None:
            msg["content"] = new_content
    target_mb = _IMAGE_SHRINK_TARGET_BYTES / (1024 * 1024)
    if changed_count:
        logger.info("image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB", changed_count, target_mb)
    if unshrinkable_oversized:
        # An unshrinkable oversized image makes retry pointless; signal no progress even
        # if others shrank so the caller surfaces the original error.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)", unshrinkable_oversized,
            target_mb,
        )
        return False
    return changed_count > 0
