"""Regression tests: emphasis-wrapped MEDIA tags for extension-less and
unknown-extension files deliver instead of being silently dropped.

Sibling of the known-extension emphasis fix in MEDIA_TAG_CLEANUP_RE: models
routinely wrap a MEDIA tag in Markdown emphasis when presenting a file to the
user (``**MEDIA:script.lua**``). For extension-less (Caddyfile) and
unknown-extension (.py, .lua, .log, .weirdext, ...) tags — the
``MEDIA_EXTENSIONLESS_TAG_RE`` validated-delivery pass — the bare path class
greedily absorbed the trailing ``**`` into the candidate path, so disk
validation failed and the file was silently never delivered, with the tag
left visible in the delivered text. ``*`` terminates the bare path instead;
``_`` stays a path character (``my_script.lua``).
"""

import os

import pytest

from gateway.platforms.base import BasePlatformAdapter


class TestEmphasisWrappedUnknownExtensionTags:
    def test_bold_lua_tag_delivers(self, tmp_path):
        p = tmp_path / "script.lua"
        p.write_text("-- guidance\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"Here: **MEDIA:{p}**")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned

    def test_bold_extensionless_tag_delivers(self, tmp_path):
        p = tmp_path / "Caddyfile"
        p.write_text("localhost\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"Config: **MEDIA:{p}**")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned

    def test_italic_py_tag_delivers(self, tmp_path):
        p = tmp_path / "app.py"
        p.write_text("import sys\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"*MEDIA:{p}*")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned

    def test_bold_italic_log_tag_delivers(self, tmp_path):
        p = tmp_path / "server.log"
        p.write_text("x\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"***MEDIA:{p}***")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned

    def test_bold_tag_followed_by_prose_keeps_prose(self, tmp_path):
        p = tmp_path / "script.lua"
        p.write_text("--\n")
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Here **MEDIA:{p}** is the script you wanted"
        )
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "is the script you wanted" in cleaned
        assert "MEDIA:" not in cleaned

    def test_leading_and_trailing_underscore_emphasis(self, tmp_path):
        # ``_`` is a legal path character (my_script.lua), so a trailing ``_``
        # after the tag is indistinguishable from a possible filename
        # continuation — it cannot be treated as a closing emphasis marker.
        # The correct contract here is fail-safe: the tag stays visible (no
        # silent drop), because disk validation can't resolve the candidate.
        p = tmp_path / "my_script.lua"
        p.write_text("--\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"See _MEDIA:{p}_ here")
        assert media == []
        assert "MEDIA:" in cleaned  # visible, never a silent black hole

    def test_underscore_inside_filename_delivers(self, tmp_path):
        # ``_`` stays a path character within the path itself.
        p = tmp_path / "my_script.lua"
        p.write_text("--\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"See MEDIA:{p} here")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned

    def test_streamed_display_strips_bold_unknown_extension(self, tmp_path):
        """Display cleanup must agree with attachment delivery for a bold .lua tag."""
        p = tmp_path / "script.lua"
        p.write_text("-- delivered\n")
        text = f"See **MEDIA:{p}**; done"
        assert BasePlatformAdapter.strip_media_directives_for_display(text) == "See ; done"



class TestEmphasisWrappedKnownExtensionTagsStillDeliver:
    """The extensionless-pass change must not regress the known-ext path."""

    def test_bold_png_tag_delivers(self, tmp_path):
        p = tmp_path / "chart.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"**MEDIA:{p}**")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "MEDIA:" not in cleaned


class TestUndeliverableTagsStayVisible:
    """No emphasis-wrapped silent black hole: a nonexistent path stays visible."""

    @pytest.mark.parametrize("wrap", ["**", "*", "`", ""])
    def test_nonexistent_emphasis_path_stays_visible(self, tmp_path, wrap):
        p = tmp_path / "nope.lua"
        # file intentionally not created
        text = f"See {wrap}MEDIA:{p}{wrap} (should not attach)"
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert media == []
        assert "MEDIA:" in cleaned  # visible, not silently dropped


class TestSpacedPathsStillWork:
    """#24032 progressive space-extension must survive the charset change."""

    def test_spaced_unknown_extension_path_delivers(self, tmp_path):
        p = tmp_path / "map data.log"
        p.write_text("x\n")
        media, cleaned = BasePlatformAdapter.extract_media(
            f"MEDIA:{p} is the log you asked for"
        )
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
        assert "is the log you asked for" in cleaned

    def test_glued_tags_do_not_merge(self, tmp_path):
        a = tmp_path / "Caddyfile"
        b = tmp_path / "Dockerfile"
        a.write_text("localhost\n")
        b.write_text("FROM alpine\n")
        media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:{a}MEDIA:{b}")
        got = sorted(os.path.realpath(x) for x, _ in media)
        assert got == sorted([os.path.realpath(str(a)), os.path.realpath(str(b))])
        assert "MEDIA:" not in cleaned
