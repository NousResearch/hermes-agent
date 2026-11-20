"""Image tags a reply turns into attachments (``BasePlatformAdapter.extract_images``).

``||![alt](url)||`` marks an image a spoiler: its alt comes back as a ``SpoilerAlt``, which the
adapters that can hide an image read with ``is_spoiler``.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable

MD_IMAGE_TAG_PATTERN = r'!\[(?P<alt>[^\]]*)\]\((?P<url>https?://[^\s\)]+)\)'
HTML_IMAGE_TAG_PATTERN = r'<img\s+src=["\']?(?P<url>https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
# The markdown tag inside optional spoiler bars. Used only on text with "||" in it: the optional
# lead costs the plain pattern its fast literal scan.
_SPOILERABLE_MD_IMAGE_TAG_PATTERN = r'(?P<bars>\|\|[ \t]*)?' + MD_IMAGE_TAG_PATTERN + r'(?(bars)[ \t]*\|\|)'
_IMAGE_URL_MARKERS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery')


class SpoilerAlt(str):
    """The alt text of a ``||![alt](url)||`` image. Images travel as ``(url, alt)`` pairs that
    every ``send_multiple_images`` override unpacks, so the mark rides on the alt."""

    spoiler = True


def is_spoiler(alt: Any) -> bool:
    """Whether an image's alt text carries the ``||![alt](url)||`` spoiler mark."""
    return getattr(alt, "spoiler", False) is True


def spoiler_kwargs(sender: Callable, alt: Any) -> dict[str, bool]:
    """``{"spoiler": True}`` for a marked image when ``sender`` (``send_image`` or
    ``send_animation``) declares ``spoiler``; otherwise ``{}``, so an adapter that cannot hide
    images is called exactly as before."""
    if is_spoiler(alt) and "spoiler" in inspect.signature(sender).parameters:
        return {"spoiler": True}
    return {}


def hidden_image_link(url: str, alt: str = "") -> str:
    """An image that goes as text, hidden in spoiler bars; the ``<>`` stop Discord's preview."""
    url = url.replace("<", "%3C").replace(">", "%3E")
    return f"||[{alt}](<{url}>)||" if alt else f"||<{url}>||"


def _md_pattern(text: str) -> str:
    return _SPOILERABLE_MD_IMAGE_TAG_PATTERN if "||" in text else MD_IMAGE_TAG_PATTERN


def find_images(content: str) -> list[tuple[str, str]]:
    """``[(url, alt), ...]`` for the ``![alt](url)`` tags whose URL looks like an image (a
    ``SpoilerAlt`` inside spoiler bars), then the ``<img src=...>`` tags."""
    images: list[tuple[str, str]] = [
        (m["url"], SpoilerAlt(m["alt"]) if m.groupdict().get("bars") else m["alt"])
        for m in re.finditer(_md_pattern(content), content)
        if any(marker in m["url"].lower() for marker in _IMAGE_URL_MARKERS)]
    images.extend((m["url"], "") for m in re.finditer(HTML_IMAGE_TAG_PATTERN, content))
    return images


def remove_image_tags(text: str, urls: set[str], trailing: str = "") -> str:
    """``text`` without the image tags whose URL is in ``urls``, each with what the ``trailing``
    regex matches after it."""
    def replace(m: re.Match) -> str:
        return "" if m["url"] in urls else m[0]

    for pattern in (_md_pattern(text), HTML_IMAGE_TAG_PATTERN):
        text = re.sub(f"(?:{pattern})(?P<trailing>{trailing})", replace, text)
    return text


def extract_images(content: str) -> tuple[list[tuple[str, str]], str]:
    """``(find_images(content), content with those tags removed)``."""
    images = find_images(content)
    if not images:
        return images, content
    cleaned = remove_image_tags(content, {url for url, _ in images})
    return images, re.sub(r'\n{3,}', '\n\n', cleaned).strip()  # leftover blank lines
