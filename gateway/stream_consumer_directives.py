"""Hold-back check for attachments still being streamed: ``MEDIA:<path>``, ``[[…]]`` flags and
markdown/HTML image tags (the display cleaner hides each one only once it is complete)."""

from __future__ import annotations

import re

from gateway.platforms.base import HTML_IMAGE_TAG_PATTERN, MD_IMAGE_TAG_PATTERN, BasePlatformAdapter

# A delivery directive still being typed on the open (last) line: "MEDIA:" the cleaner has not
# stripped yet whose remainder can still become a path (MEDIA_TAG_CLEANUP_RE's openers: a quote,
# ~/, /, a drive letter), or a prefix of "MEDIA" not glued to a preceding letter or digit. A
# complete tag has no left boundary, so a prefix after punctuation ("图片：MED") is held too.
# Prose such as "social media: Twitter" is not a path, so it never holds the line back.
_PENDING_MEDIA_RE = re.compile(
    r"""MEDIA:\s*(?:$|[`"'~/\\]|[A-Za-z](?::|$))"""
    r"""|(?<![A-Za-z0-9])[`"'*_]{0,3}M(?:E(?:D(?:I(?:A)?)?)?)?$""",
    re.IGNORECASE)
_PENDING_FLAG_RE = re.compile(r"\[\[?[A-Za-z_]*\]?$")
_STREAM_FLAGS = ("[[audio_as_voice]]", "[[as_document]]")
# An image tag still being typed (``extract_images`` hides the complete one): ``![alt``,
# ``![alt](https://ho`` or ``<img src="…``. A bare "!" or "<" counts only at the start of a word,
# so prose exclamations ("Great!") keep streaming. Whitespace in the URL ends the hold: that tag is
# never extracted, so it stays visible either way.
_PENDING_IMAGE_RE = re.compile(
    r"""!\[[^\]\n]*(?:\](?:\([^)\s]*)?)?$|(?<!\S)!$"""
    r"""|<img\s[^>\n]*$|(?<!\S)<(?:i(?:m(?:g)?)?)?$""",
    re.IGNORECASE)


def strip_image_tags_for_display(text: str) -> str:
    """Hide the image tags ``extract_images`` delivers as attachments post-stream.

    Each tag goes with the spaces and up to two line breaks after it, and nothing else is trimmed
    or collapsed: an interim frame must stay a prefix of the next one, so text in front of the tag
    (already shown) is never rewritten.
    """
    if "![" not in text and "<img" not in text:
        return text
    images, _ = BasePlatformAdapter.extract_images(text)
    if not images:
        return text
    urls = {url for url, _ in images}
    for pattern, url_group in ((MD_IMAGE_TAG_PATTERN, 2), (HTML_IMAGE_TAG_PATTERN, 1)):
        text = re.sub(f"(?:{pattern})[ \\t]*(?:\\r?\\n){{0,2}}",
                      lambda m, g=url_group: "" if m.group(g) in urls else m.group(0), text)
    return text


def has_pending_media_directive(cleaned: str) -> bool:
    """True while the open line of cleaned streamed text may still become a hidden attachment.

    Complete deliverable tags are already gone from *cleaned*, so a ``MEDIA:`` or image tag left
    there is unfinished or undeliverable; the final (non-interim) flush still shows the latter.
    Showing the unfinished part and hiding it later would break prefix-stable draft frames.
    """
    line = cleaned.rsplit("\n", 1)[-1]
    if _PENDING_MEDIA_RE.search(line) or _PENDING_IMAGE_RE.search(line):
        return True
    flag = _PENDING_FLAG_RE.search(line)
    return bool(flag) and any(f.startswith(flag.group(0)) for f in _STREAM_FLAGS)
