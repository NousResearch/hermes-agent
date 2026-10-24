"""Hold-back check for delivery directives still being streamed (``MEDIA:<path>``, ``[[…]]`` flags)."""

from __future__ import annotations

import re

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


def has_pending_media_directive(cleaned: str) -> bool:
    """True while the open line of cleaned streamed text may still become a delivery directive.

    Complete deliverable tags are already gone from *cleaned*, so a ``MEDIA:`` left there is
    unfinished or undeliverable; the final (non-interim) flush still shows the latter.
    """
    line = cleaned.rsplit("\n", 1)[-1]
    if _PENDING_MEDIA_RE.search(line):
        return True
    flag = _PENDING_FLAG_RE.search(line)
    return bool(flag) and any(f.startswith(flag.group(0)) for f in _STREAM_FLAGS)
