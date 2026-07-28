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
    def test_sentence_final_period_extracts_path(self, real_file):
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Saved your data. MEDIA:{real_file}."
        )
        assert [p for p, _ in media] == [real_file]
        assert "MEDIA:" not in cleaned

    def test_period_then_more_prose(self, real_file):
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Done: MEDIA:{real_file}. Enjoy!"
        )
        assert [p for p, _ in media] == [real_file]
        assert "Enjoy!" in cleaned

    def test_multipart_extension_not_truncated(self, real_targz):
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{real_targz}")
        assert [p for p, _ in media] == [real_targz]

    def test_multipart_extension_with_trailing_period(self, real_targz):
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{real_targz}.")
        assert [p for p, _ in media] == [real_targz]


class TestInlineCodeWrappedTags:
    def test_real_path_in_inline_code_delivers(self, real_file):
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Here is your file `MEDIA:{real_file}`"
        )
        assert [p for p, _ in media] == [real_file]
        assert "MEDIA:" not in cleaned

    def test_nonexistent_path_in_inline_code_stays_masked(self):
        text = "Use the format `MEDIA:/nonexistent/example.csv` to attach files."
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert media == []
        assert "`MEDIA:/nonexistent/example.csv`" in cleaned

    def test_fenced_code_block_always_masked(self, real_file):
        text = f"```\nMEDIA:{real_file}\n```"
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert media == []
        assert real_file in cleaned

    def test_inline_code_non_media_untouched(self, real_file):
        text = f"Run `ls -la` then see MEDIA:{real_file}"
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert [p for p, _ in media] == [real_file]
        assert "`ls -la`" in cleaned


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

    def test_duplicate_tags_deliver_once(self, real_file):
        media, _ = BasePlatformAdapter.extract_media(
            f"MEDIA:{real_file} and again MEDIA:{real_file}"
        )
        assert [p for p, _ in media] == [real_file]

    def test_glued_as_document_delivers(self, real_file):
        media, _ = BasePlatformAdapter.extract_media(
            f"MEDIA:{real_file}[[as_document]]"
        )
        assert [p for p, _ in media] == [real_file]

    def test_unknown_extension_real_file_delivers(self, tmp_path):
        p = tmp_path / "script.py"
        p.write_text("print('hi')\n")
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{p}")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]

    def test_extensionless_real_file_delivers(self, tmp_path):
        p = tmp_path / "Caddyfile"
        p.write_text("localhost\n")
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{p}")
        assert [os.path.realpath(x) for x, _ in media] == [os.path.realpath(str(p))]
