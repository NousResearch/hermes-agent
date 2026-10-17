"""Deliverable paths in an ``image_generate`` tool result, which carries a JSON payload, not a ``MEDIA:`` tag."""

from __future__ import annotations

import json
from typing import Callable

# The primary image's path fields in preference order; they name one file, so only the first usable one counts.
_PRIMARY_IMAGE_FIELDS = ("host_image", "image", "agent_visible_image")


def image_generate_media_paths(content: str, accept: Callable[[str], bool] = bool) -> list[str]:
    """The primary image, then every ``additional_images`` entry, of a successful payload; ``[]`` otherwise.

    A multi-image backend (OpenRouter ``n``) bills every image it returns, so delivering the primary
    alone silently drops the rest. ``accept`` filters each path (deliverable shape, already sent)."""
    try:
        payload = json.loads(content)
    except ValueError:
        return []
    if not isinstance(payload, dict) or not payload.get("success"):
        return []
    primary = next((path for field in _PRIMARY_IMAGE_FIELDS
                    if isinstance(path := payload.get(field), str) and accept(path)), None)
    extras = payload.get("additional_images")
    extras = [path for path in extras if isinstance(path, str) and accept(path)] if isinstance(extras, list) else []
    return ([primary] if primary else []) + extras
