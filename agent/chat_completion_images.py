"""Materialize Chat Completions image sidecars before Relay/history serialization."""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import warnings
from typing import Any

from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)
_IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
_MAX_IMAGE_BYTES = 25 * 1024 * 1024


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    result = getattr(value, name, None)
    if result is None:
        extra = getattr(value, "model_extra", None)
        if isinstance(extra, dict):
            result = extra.get(name)
    return result


def _validate_image(raw: bytes, extension: str | None = None) -> None:
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as decoded:
            formats = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP", "gif": "GIF"}
            if decoded.format not in formats.values() or (extension and decoded.format != formats[extension]):
                raise ValueError("unsupported or mismatched image type")
            decoded.verify()


def materialize_chat_images(message: Any, seen: set[str] | None = None, *, save: bool = True) -> list[str]:
    """Strip inline sidecars before serialization; return new delivery/error markers.

    ``seen`` belongs to the logical request, not its streaming attempts.
    ``save=False`` discards a cancelled response without filesystem/network work.
    Ordinary text and tool calls are left intact.
    """
    images = _field(message, "images")
    if images is None:
        return []
    if not save:
        clear_chat_images(message)
        return []
    seen = seen if seen is not None else set()
    markers = [] if isinstance(images, (list, tuple)) else ["[Provider image could not be saved: unsupported image payload.]"]
    images = images if isinstance(images, (list, tuple)) else ()
    for image in images:
        url = _field(_field(image, "image_url"), "url")
        if not isinstance(url, str) or not url:
            markers.append("[Provider image could not be saved: unsupported image payload.]")
            continue
        # Reject oversized sidecars before allocating their encoded copy/hash.
        if len(url) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4 + 64:
            markers.append("[Provider image could not be saved: image too large.]")
            continue
        key = hashlib.sha256(url.encode()).hexdigest()
        if key in seen:
            continue
        try:
            if url.startswith("data:"):
                header, encoded = url.split(",", 1)
                mime = header[5:].removesuffix(";base64")
                extension = _IMAGE_TYPES.get(mime)
                if extension is None or not header.endswith(";base64"):
                    raise ValueError("unsupported image encoding")
                if len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4:
                    raise ValueError("image too large")
                raw = base64.b64decode(encoded, validate=True)
                if not raw or len(raw) > _MAX_IMAGE_BYTES:
                    raise ValueError("empty or oversized image")
                _validate_image(raw, extension)
                from agent.provider_media import save_bytes
                path = save_bytes("generated/images", raw, prefix="chat_image", extension=extension)
            elif url.startswith(("https://", "http://")):
                from agent.image_gen_provider import save_url_image
                path = save_url_image(url, prefix="chat_image", max_bytes=_MAX_IMAGE_BYTES)
                try:
                    _validate_image(path.read_bytes())
                except Exception:
                    path.unlink(missing_ok=True)
                    raise
            else:
                raise ValueError("unsupported image URL")
            marker = f"MEDIA:{path}"
        except Exception as exc:
            # Provider URLs/data and download exceptions can contain secrets. Log only the class.
            logger.warning("Chat image materialization failed (%s)", type(exc).__name__)
            marker = "[Provider image could not be saved.]"
        seen.add(key)
        markers.append(marker)
    clear_chat_images(message)
    if markers:
        content = flatten_message_text(_field(message, "content"), sep="") or ""
        content = content + ("\n" if content else "") + "\n".join(markers)
        if isinstance(message, dict):
            message["content"] = content
        else:
            message.content = content
    return markers


def clear_chat_images(message: Any) -> None:
    # Clear the sidecar even on failure: no base64 in Relay records or session history.
    if isinstance(message, dict):
        message.pop("images", None)
    else:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            extra.pop("images", None)
        if hasattr(message, "images"):
            message.images = None


def materialize_chat_response(response: Any, *, save: bool = True, seen: set[str] | None = None,
                              markers: list[str] | None = None) -> Any:
    for choice in _field(response, "choices") or ():
        message = _field(choice, "message")
        if message is not None:
            new_markers = materialize_chat_images(message, seen, save=save)
            if markers is not None:
                markers.extend(new_markers)
    return response
