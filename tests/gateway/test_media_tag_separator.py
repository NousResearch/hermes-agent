"""Regression tests for #68773 — MEDIA tags without a separator merge paths.

Before the fix, ``MEDIA_EXTENSIONLESS_TAG_RE`` used a greedy character class
``[^\\s\\n`\"']+`` that would silently absorb the next ``MEDIA:`` keyword when
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


def test_extensionless_regex_does_not_absorb_next_media_keyword():
    """Two extensionless tags glued together must each match independently."""
    text = "MEDIA:/tmp/CaddyfileMEDIA:/tmp/Dockerfile"
    matches = list(MEDIA_EXTENSIONLESS_TAG_RE.finditer(text))
    paths = [m.group("path") for m in matches]
    assert paths == ["/tmp/Caddyfile", "/tmp/Dockerfile"], paths


def test_extensionless_regex_does_not_absorb_following_text():
    """An extensionless tag glued to text containing `MEDIA:` must stop at the next tag.

    The known-extension case is covered separately by
    ``test_strip_media_directives_does_not_drop_known_ext_tag_followed_by_text``
    — there the primary regex's separator requirement leaves the text visible.

    For the fallback regex, the realistic threat is a *second* ``MEDIA:`` tag
    glued to the first path; this test pins that behavior.
    """
    text = "MEDIA:/tmp/CaddyfileMEDIA:/tmp/Dockerfile and text"
    match = MEDIA_EXTENSIONLESS_TAG_RE.search(text)
    assert match is not None
    assert match.group("path") == "/tmp/Caddyfile"


def test_extensionless_regex_still_matches_normal_cases():
    """The fix must not regress the well-formed extensionless paths."""
    text = "see MEDIA:/tmp/Caddyfile for details"
    match = MEDIA_EXTENSIONLESS_TAG_RE.search(text)
    assert match is not None
    assert match.group("path") == "/tmp/Caddyfile"


def test_known_extension_regex_splits_glued_tags():
    """``MEDIA_TAG_CLEANUP_RE`` must stop at the next ``MEDIA:`` keyword (#68773).

    Previously the primary regex used greedy ``\\S+`` in the path class,
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


def test_strip_media_directives_handles_glued_extensionless_tags(tmp_path):
    """``_strip_media_tag_directives`` must not produce a merged invalid path.

    With two real files glued together, ``validate_media_delivery_path``
    accepts the first valid path and skips the second because the merged
    string is not a real file. After the fix, the second tag should be
    matched independently and also accepted.
    """
    caddy = tmp_path / "Caddyfile"
    caddy.write_text("example.com")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch")

    text = f"MEDIA:{caddy}MEDIA:{dockerfile}"
    cleaned = _strip_media_tag_directives(text)
    # Both tags stripped; no leftover MEDIA: token from greedy merge.
    assert "MEDIA:" not in cleaned, f"Greedy merge leaked: {cleaned!r}"


def test_strip_media_directives_does_not_drop_known_ext_tag_followed_by_text(tmp_path):
    """A known-extension tag glued to text must leave the text visible.

    The primary regex requires a separator after the extension, so
    ``MEDIA:/file.pngSome text`` does not match. After the fallback runs,
    ``_path_lacks_deliverable_extension`` sees ``.pngSome`` as a non-known
    extension and the strip function returns ``match.group(0)`` unchanged —
    so the original text stays visible (no silent drop, no merged invalid
    path).
    """
    png = tmp_path / "real.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")

    text = f"MEDIA:{png}Some text"
    cleaned = _strip_media_tag_directives(text)
    # The full original is preserved — no silent truncation of the file or text.
    assert cleaned == text