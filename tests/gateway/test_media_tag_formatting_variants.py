"""Regression tests: MEDIA tag formatting variants that previously broke delivery.

Covers the two follow-up fixes layered on top of the salvaged contributor PRs:

1. Sentence-final punctuation — ``MEDIA:/x/data.csv.`` at the end of a
   sentence must extract ``data.csv`` (the trailing ``.`` is a boundary, not
   part of the path), while multi-part extensions (``archive.tar.gz``)
   remain intact.

2. Inline-code-wrapped tags — a whole ``MEDIA:`` tag inside inline backticks
   (`` `MEDIA:/path.csv` ``) is a real delivery directive when the path
   validates on disk; prose examples with non-existent paths stay masked
   (#35695), and fenced code blocks are always masked.
"""

import os

import pytest

from gateway.platforms.base import BasePlatformAdapter


@pytest.fixture()
def real_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("x,y\n1,2\n")
    return str(p)


@pytest.fixture()
def real_targz(tmp_path):
    p = tmp_path / "archive.tar.gz"
    p.write_bytes(b"\x1f\x8b")
    return str(p)


class TestTrailingPunctuation:


    def test_multipart_extension_not_truncated(self, real_targz):
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{real_targz}")
        assert [p for p, _ in media] == [real_targz]


class TestInlineCodeWrappedTags:
    def test_real_path_in_inline_code_delivers(self, real_file):
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Here is your file `MEDIA:{real_file}`"
        )
        assert [p for p, _ in media] == [real_file]
        assert "MEDIA:" not in cleaned


class TestEmphasisAndDedupeIntegration:
    """End-to-end matrix over the salvaged contributor fixes."""

    def test_bold_wrapped_tag_delivers(self, real_file):
        media, cleaned = BasePlatformAdapter.extract_media(f"**MEDIA:{real_file}**")
        assert [p for p, _ in media] == [real_file]
        assert "MEDIA:" not in cleaned

    def test_two_tags_one_line_both_deliver(self, tmp_path):
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("1")
        b.write_text("2")
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{a} MEDIA:{b}")
        assert [p for p, _ in media] == [str(a), str(b)]


