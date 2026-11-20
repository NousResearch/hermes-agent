"""Image tags a reply turns into attachments (``BasePlatformAdapter.extract_images``)."""

from __future__ import annotations

import re

MD_IMAGE_TAG_PATTERN = r'!\[(?P<alt>[^\]]*)\]\((?P<url>https?://[^\s\)]+)\)'
HTML_IMAGE_TAG_PATTERN = r'<img\s+src=["\']?(?P<url>https?://[^\s"\'<>]+)["\']?\s*/?>\s*(?:</img>)?'
_IMAGE_URL_MARKERS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'fal-cdn', 'replicate.delivery')


def find_images(content: str) -> list[tuple[str, str]]:
    """``[(url, alt), ...]`` for the ``![alt](url)`` tags whose URL looks like an image, then the
    ``<img src=...>`` tags."""
    images: list[tuple[str, str]] = [
        (m["url"], m["alt"]) for m in re.finditer(MD_IMAGE_TAG_PATTERN, content)
        if any(marker in m["url"].lower() for marker in _IMAGE_URL_MARKERS)]
    images.extend((m["url"], "") for m in re.finditer(HTML_IMAGE_TAG_PATTERN, content))
    return images


def remove_image_tags(text: str, urls: set[str], trailing: str = "") -> str:
    """``text`` without the image tags whose URL is in ``urls``, each with what the ``trailing``
    regex matches after it."""
    def replace(m: re.Match) -> str:
        return "" if m["url"] in urls else m[0]

    for pattern in (MD_IMAGE_TAG_PATTERN, HTML_IMAGE_TAG_PATTERN):
        text = re.sub(f"(?:{pattern})(?P<trailing>{trailing})", replace, text)
    return text


def extract_images(content: str) -> tuple[list[tuple[str, str]], str]:
    """``(find_images(content), content with those tags removed)``."""
    images = find_images(content)
    if not images:
        return images, content
    cleaned = remove_image_tags(content, {url for url, _ in images})
    return images, re.sub(r'\n{3,}', '\n\n', cleaned).strip()  # leftover blank lines
