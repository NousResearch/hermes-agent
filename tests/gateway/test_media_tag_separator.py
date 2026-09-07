"""Regression tests for #68773 — MEDIA tags without a separator merge paths.

Before the fix, ``MEDIA_EXTENSIONLESS_TAG_RE`` used a greedy character class
``[^\s\n`\"']+`` that would silently absorb the next ``MEDIA:`` keyword when
two tags were emitted back-to-back (``MEDIA:/a.pngMEDIA:/b.png``), producing
an invalid merged path that was then rejected by
``validate_media_delivery_path`` and dropped silently.

The same pattern also failed for ``MEDIA:/path/file.pngSome text`` — the
fallback would treat the trailing text as part of the path.
"""

from gateway.platforms.base import (
    MEDIA_EXTENSIONLESS_TAG_RE,
    MEDIA_TAG_CLEANUP_RE,
    _strip_media_tag_directives,
)


def test_known_extension_regex_splits_glued_tags():
    """``MEDIA_TAG_CLEANUP_RE`` must stop at the next ``MEDIA:`` keyword (#68773).

    Previously the primary regex used greedy ``\S+`` in the path class,
    so two tags glued together (``MEDIA:/a.pngMEDIA:/b.png``) merged into
    one invalid path (``/a.pngMEDIA:/b.png``) and were silently dropped by
    ``validate_media_delivery_path``. The fix uses non-greedy quantifiers
    and accepts ``MEDIA:`` in the trailing lookahead.
    """
    text = "MEDIA:/tmp/file.pngMEDIA:/tmp/file2.png"
    matches = list(MEDIA_TAG_CLEANUP_RE.finditer(text))
    paths = [m.group("path") for m in matches]
    assert paths == ["/tmp/file.png", "/tmp/file2.png"], paths


def test_strip_media_directives_handles_glued_known_extension_tags(tmp_path):
    """Two known-extension tags glued together must each be delivered (#68773)."""
    png1 = tmp_path / "a.png"
    png1.write_bytes(b"\x89PNG\r\n\x1a\n")
    png2 = tmp_path / "b.png"
    png2.write_bytes(b"\x89PNG\r\n\x1a\n")

    text = f"MEDIA:{png1}MEDIA:{png2}"
    cleaned = _strip_media_tag_directives(text)
    # Both MEDIA: tokens consumed; the leading MEDIA: prefix is gone.
    assert "MEDIA:" not in cleaned, f"Greedy merge leaked: {cleaned!r}"


