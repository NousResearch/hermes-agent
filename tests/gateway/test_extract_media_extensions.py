"""Regression tests for issue #37318: MEDIA:<path> tag must recognize
common document / code / data extensions.

Historically the ``MEDIA:`` tag regex in ``gateway/platforms/base.py``
(``extract_media``) and the inline ``_TOOL_MEDIA_RE`` regexes in
``gateway/run.py`` (used for history deduplication of media tags produced
by tools) only whitelisted a narrow set of extensions — ``png``, ``jpg``,
``pdf``, ``txt``, ``csv``, ``apk``, ``ipa`` and similar. Common document,
data, and code file types were silently dropped: a response that ended
with ``MEDIA:/home/user/report.md`` was stripped from the body by the
loose cleanup regex, the file was never delivered, and the agent
received no signal that the tag had been lost.

Fix: add ``md``, ``markdown``, ``json``, ``yaml``, ``yml``, ``toml``,
``py``, ``js``, ``sh``, ``tar.gz``, ``tgz`` to the whitelist at both
call sites, plus ``.markdown``, ``.toml``, ``.py``, ``.js``, ``.sh`` to
the shared ``MEDIA_DELIVERY_EXTS`` tuple (which is the single source
of truth that drives both ``MEDIA_TAG_CLEANUP_RE`` and
``extract_local_files``).

These tests pin the public behavior: a ``MEDIA:<path>`` tag ending in
any of the newly-supported extensions must survive the extraction
pipeline (path captured, tag stripped from cleaned text).
"""

import os
import sys

import pytest

# Make sure the project root is on sys.path so ``import gateway.*`` works
# when the test is invoked from any working directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gateway.platforms.base import (  # noqa: E402
    MEDIA_DELIVERY_EXTS,
    BasePlatformAdapter,
)


# All the new extensions added in the #37318 fix.  Used to parametrize
# the positive tests so a regression on any single extension is loud.
NEW_EXTS = (
    "md",
    "markdown",
    "json",
    "yaml",
    "yml",
    "toml",
    "py",
    "js",
    "sh",
    "tar.gz",
    "tgz",
)


# Extensions that have been supported for a while — pinned here so we
# notice if a refactor accidentally drops them.
EXISTING_EXTS = (
    "png",
    "jpg",
    "pdf",
    "txt",
    "csv",
    "apk",
    "ipa",
)


class TestMediaDeliveryExtsTuple:
    """The shared MEDIA_DELIVERY_EXTS tuple lists the new extensions.

    The tuple is the single source of truth that drives
    ``MEDIA_TAG_CLEANUP_RE`` (used by ``extract_media``) and
    ``extract_local_files``.  If a new extension is not in the tuple it
    cannot be delivered, no matter what else is fixed elsewhere.
    """

    @pytest.mark.parametrize("ext", NEW_EXTS)
    def test_new_extension_in_tuple(self, ext):
        assert f".{ext}" in MEDIA_DELIVERY_EXTS, (
            f"Expected .{ext} in MEDIA_DELIVERY_EXTS for issue #37318, "
            f"but it was missing. Current tuple: {MEDIA_DELIVERY_EXTS!r}"
        )

    @pytest.mark.parametrize("ext", EXISTING_EXTS)
    def test_existing_extension_still_in_tuple(self, ext):
        """Pre-existing extensions must not be dropped by the refactor."""
        assert f".{ext}" in MEDIA_DELIVERY_EXTS, (
            f"Pre-existing extension .{ext} was dropped from MEDIA_DELIVERY_EXTS"
        )


class TestExtractMediaRecognizesNewExtensions:
    """``extract_media`` (BasePlatformAdapter) must capture MEDIA:<path>
    tags ending in any of the newly-supported extensions and strip them
    from the cleaned text."""

    @pytest.mark.parametrize("ext", NEW_EXTS)
    def test_new_extension_is_extracted(self, ext):
        path = f"/home/user/output.{ext}"
        content = f"Here is the file you asked for.\nMEDIA:{path}\nDone."

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert len(media) == 1, (
            f"Expected exactly one MEDIA tag for .{ext} path, got: {media!r}"
        )
        captured_path, is_voice = media[0]
        # ``os.path.expanduser`` should pass the path through unchanged
        # when it is already absolute and not starting with ``~``.
        assert captured_path == path
        assert is_voice is False
        # The tag itself must be removed from the user-visible text.
        assert "MEDIA:" not in cleaned
        assert f".{ext}" not in cleaned or path not in cleaned

    def test_md_file_path_is_not_filtered_out(self):
        """The exact failure case from issue #37318: a .md path must
        survive ``extract_media`` (regression: it was silently dropped)."""
        content = (
            "I have written the report to disk.\n"
            "MEDIA:/home/user/projects/notes.md\n"
            "Let me know what you think."
        )

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert len(media) == 1, (
            f".md MEDIA tag was filtered out — got media={media!r}, "
            f"cleaned={cleaned!r}"
        )
        captured, _ = media[0]
        assert captured == "/home/user/projects/notes.md"
        assert "MEDIA:" not in cleaned
        assert "I have written the report" in cleaned

    def test_json_file_path_is_not_filtered_out(self):
        """Second regression case from the issue: .json paths must also
        survive extraction (agents commonly emit structured artifacts)."""
        content = (
            "Here are the analysis results.\n"
            "MEDIA:/tmp/analysis.json\n"
            "Summary above, raw data attached."
        )

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert len(media) == 1, (
            f".json MEDIA tag was filtered out — got media={media!r}, "
            f"cleaned={cleaned!r}"
        )
        captured, _ = media[0]
        assert captured == "/tmp/analysis.json"
        assert "MEDIA:" not in cleaned

    def test_png_still_works(self):
        """Bonus: ensure existing supported extensions (.png) still pass
        after the whitelist extension — guards against accidentally
        breaking the original behavior while adding the new entries."""
        content = "Image attached.\nMEDIA:/tmp/screenshot.png\n"

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert len(media) == 1
        assert media[0][0] == "/tmp/screenshot.png"
        assert "MEDIA:" not in cleaned

    def test_unknown_extension_still_filtered(self):
        """An extension that is NOT in the whitelist must NOT be captured
        by ``extract_media``.  This is the intentional behaviour: a
        ``MEDIA:`` tag with an unknown extension is left in the body so
        the downstream ``extract_local_files`` pass can still see the
        bare path (see issue #34517 referenced in base.py)."""
        content = "MEDIA:/home/user/mystery.weirdext\n"

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert media == [], (
            f"Unknown-extension MEDIA tag should NOT be captured by "
            f"extract_media, but got: {media!r}"
        )

    def test_multiple_new_extension_tags_in_one_response(self):
        """Multiple MEDIA tags of different new-extension types in the
        same response are all captured."""
        content = (
            "Several artifacts:\n"
            "MEDIA:/home/user/report.md\n"
            "MEDIA:/home/user/data.json\n"
            "MEDIA:/home/user/script.py\n"
            "End of reply."
        )

        media, cleaned = BasePlatformAdapter.extract_media(content)

        assert len(media) == 3
        captured_paths = sorted(p for p, _ in media)
        assert captured_paths == [
            "/home/user/data.json",
            "/home/user/report.md",
            "/home/user/script.py",
        ]
        assert "MEDIA:" not in cleaned


class TestBasePlatformAdapterExtractMediaRegression:
    """``BasePlatformAdapter.extract_media`` (the static method) is the
    real public surface — there is no module-level helper.  This test
    exercises the same paths via a minimal subclass to confirm the
    fix is wired through correctly end-to-end (the regex used internally
    is the shared ``MEDIA_TAG_CLEANUP_RE`` which is built from
    ``MEDIA_DELIVERY_EXTS``)."""

    def test_md_path_is_extracted(self):
        content = "MEDIA:/tmp/notes.md\n"

        # ``extract_media`` is a ``@staticmethod`` — call it on a
        # throwaway subclass to avoid any ``__init__`` surprises.
        class _Stub(BasePlatformAdapter):
            platform_name = "test"

            def __init__(self):  # pragma: no cover - never used
                pass

        media, cleaned = _Stub.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/tmp/notes.md"
        assert "MEDIA:" not in cleaned

    def test_json_path_is_extracted(self):
        content = "MEDIA:/tmp/data.json\n"

        class _Stub(BasePlatformAdapter):
            platform_name = "test"

            def __init__(self):  # pragma: no cover - never used
                pass

        media, cleaned = _Stub.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/tmp/data.json"
        assert "MEDIA:" not in cleaned
