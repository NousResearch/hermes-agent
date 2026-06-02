"""Tests for issue #37318: _TOOL_MEDIA_RE in gateway/run.py must accept the
same set of file extensions as MEDIA_DELIVERY_EXTS / MEDIA_TAG_CLEANUP_RE in
gateway/platforms/base.py.

The bug: the agent emits ``MEDIA:/tmp/report.md`` (or any of the extensions
in MEDIA_DELIVERY_EXTS that weren't in the hardcoded run.py whitelist —
``.md``, ``.json``, ``.yaml``, ``.html``, ``.svg``, etc.). The dispatch-site
regex in ``base.py`` correctly recognises these because it is built from the
shared ``MEDIA_DELIVERY_EXTS`` tuple. The narrower regex baked into ``run.py``
silently drops them, so:

  * ``_collect_auto_append_media_tags`` (run.py L692) skips them — they are
    never appended to outgoing messages.
  * The history-dedup scan (run.py L17765) doesn't see them — stale tags can
    leak through.

Fix: build the run.py regex from the same ``MEDIA_DELIVERY_EXTS`` alternation
exposed by ``gateway.platforms.base``, so the two patterns stay in lock-step
by construction.
"""

import re

import pytest

from gateway.platforms.base import (
    MEDIA_DELIVERY_EXTS,
    MEDIA_EXT_ALTERNATION,
    MEDIA_TAG_CLEANUP_RE,
)
from gateway.run import _TOOL_MEDIA_RE


# Extensions that issue #37318 specifically calls out as silently dropped.
# Source-file extensions (.py / .js / .sh) from the issue are deliberately
# NOT added to MEDIA_DELIVERY_EXTS — see the NOTE in gateway/platforms/base.py.
# Auto-shipping source files based on a bare-path mention would surprise users.
MISSING_EXTS_FROM_ISSUE = [
    "md", "markdown", "json", "yaml", "yml", "toml",
    # Plus extensions that ALREADY lived in MEDIA_DELIVERY_EXTS but were
    # missing from the run.py whitelist (silent drift, the real bug):
    "html", "htm", "svg", "bmp", "tiff", "tsv", "xml", "odt", "ods", "odp",
    "key", "tar", "bz2", "xz",
]


class TestRunPyMediaExtParity:
    """run.py's _TOOL_MEDIA_RE must accept every ext in MEDIA_DELIVERY_EXTS."""

    @pytest.mark.parametrize("ext", [e.lstrip(".") for e in MEDIA_DELIVERY_EXTS])
    def test_every_delivery_ext_matched_by_run_pattern(self, ext):
        """Every extension MEDIA_DELIVERY_EXTS advertises must match.

        Without this, ``base.py.extract_media`` happily strips and ships the
        tag while ``run.py._collect_auto_append_media_tags`` silently
        discards it from the auto-append path — the exact contract drift
        that produced #37318.
        """
        tag = f"MEDIA:/tmp/example.{ext}"
        assert _TOOL_MEDIA_RE.search(tag) is not None, (
            f"run.py _TOOL_MEDIA_RE rejected MEDIA:/tmp/example.{ext} "
            f"even though .{ext} is in MEDIA_DELIVERY_EXTS"
        )

    @pytest.mark.parametrize("ext", MISSING_EXTS_FROM_ISSUE)
    def test_issue_37318_named_extensions_match(self, ext):
        """The specific extensions called out in issue #37318 now match."""
        tag = f"MEDIA:/path/to/file.{ext}"
        match = _TOOL_MEDIA_RE.search(tag)
        assert match is not None, f"Issue #37318 ext .{ext} should match: {tag}"
        assert match.group(1) == f"/path/to/file.{ext}"

    def test_run_pattern_uses_shared_alternation(self):
        """run.py must source its alternation from base.py to prevent drift.

        We don't dictate HOW the run.py regex is constructed — only that it
        accepts every extension in MEDIA_EXT_ALTERNATION. This guard ensures
        a future contributor who adds a new extension to MEDIA_DELIVERY_EXTS
        doesn't have to remember to also hand-edit the run.py whitelist.
        """
        # Pick an extension out of the shared alternation and try it.
        for raw in MEDIA_EXT_ALTERNATION.split("|"):
            # Strip optional groups like (?:...) — bare ext only
            ext = raw.replace("\\.", ".")
            if "?" in ext or "(" in ext or ")" in ext:
                # e.g. 'jpe?g', 'docx?'. Pick a concrete form.
                if ext == "jpe?g":
                    ext = "jpg"
                elif ext == "docx?":
                    ext = "doc"
                elif ext == "xlsx?":
                    ext = "xls"
                elif ext == "pptx?":
                    ext = "ppt"
                else:
                    continue  # skip exotic patterns
            tag = f"MEDIA:/tmp/x.{ext}"
            assert _TOOL_MEDIA_RE.search(tag), (
                f"run.py _TOOL_MEDIA_RE missed .{ext} from shared "
                f"MEDIA_EXT_ALTERNATION — patterns have drifted"
            )

    # ── Regression: pre-existing behaviour preserved ───────────────

    @pytest.mark.parametrize("tag,expected", [
        ("MEDIA:/tmp/output.png", "/tmp/output.png"),
        ("MEDIA:/var/log/r.pdf", "/var/log/r.pdf"),
        ("MEDIA:~/Downloads/a.jpg", "~/Downloads/a.jpg"),
        ("MEDIA:C:\\Users\\t\\image.png", "C:\\Users\\t\\image.png"),
        ("MEDIA:D:/data/report.pdf", "D:/data/report.pdf"),
    ])
    def test_existing_paths_still_match(self, tag, expected):
        """Unix, home-relative, and Windows paths from existing tests still match."""
        m = _TOOL_MEDIA_RE.search(tag)
        assert m is not None, f"regression: {tag} no longer matches"
        assert m.group(1) == expected

    @pytest.mark.parametrize("text", [
        "No MEDIA tag here",
        "MEDIA:relative/path/file.png",   # no anchor
        "MEDIA:file.md",                   # no directory
        "MEDIA:/path/to/file.unknown",     # unsupported ext
        "MEDIA:/path/to/file",             # no extension
    ])
    def test_invalid_inputs_still_rejected(self, text):
        """The fix must not loosen anchoring or accept arbitrary extensions."""
        assert _TOOL_MEDIA_RE.search(text) is None, (
            f"should still reject: {text}"
        )


class TestExtractMediaIssue37318:
    """End-to-end: base.py extract_media accepts the same ``.md`` tag that
    triggered the user-visible bug in #37318."""

    def test_md_tag_extracted(self):
        from gateway.platforms.base import BasePlatformAdapter
        content = "Here is the doc MEDIA:/tmp/report.md"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/tmp/report.md"
        assert "MEDIA:" not in cleaned
