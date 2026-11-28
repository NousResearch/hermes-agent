"""Regression tests: an extension the model detached from its basename.

A reply can carry ``MEDIA:/tmp/work/propunere. pdf`` (space between the dot and the
extension) or ``MEDIA:/tmp/work/propunere .pdf``. No file on disk is named that way, so
the tag matched no pass, was never delivered, and the raw ``MEDIA:`` text reached the chat
surface — the recipient saw a control tag and no attachment.

Two behaviours are pinned here:

1. Recovery — whitespace around the final dot is not part of any filename, so the candidate
   is retried with the extension rejoined, and only a candidate that validates on disk is
   used (a genuinely absent file is still not delivered).
2. Observability — a tag that survives because it could not be delivered keeps its PATH
   (visibility contract, #34517) and the gateway log names it.
"""

import os

from gateway.platforms.base import (
    BasePlatformAdapter,
    _strip_media_tag_directives,
)


class TestDetachedExtensionRecovery:
    def test_space_after_dot_delivers(self, tmp_path):
        target = tmp_path / "propunere.pdf"
        target.write_bytes(b"%PDF-1.4")
        media, cleaned = BasePlatformAdapter.extract_media(
            f"Gata: MEDIA:{tmp_path / 'propunere'}. pdf"
        )
        assert [os.path.realpath(p) for p, _ in media] == [os.path.realpath(str(target))]
        assert "MEDIA:" not in cleaned
        assert "propunere. pdf" not in cleaned

    def test_space_before_dot_delivers(self, tmp_path):
        target = tmp_path / "raport.docx"
        target.write_bytes(b"PK\x03\x04")
        media, _ = BasePlatformAdapter.extract_media(
            f"MEDIA:{tmp_path / 'raport'} .docx"
        )
        assert [os.path.realpath(p) for p, _ in media] == [os.path.realpath(str(target))]

    def test_four_detached_tags_one_line_all_deliver(self, tmp_path):
        names = ["doc.docx", "doc.pdf", "doc_complet.pdf", "doc_rezumat.pdf"]
        for name in names:
            (tmp_path / name).write_bytes(b"x")
        line = " ".join(
            f"MEDIA:{tmp_path / name.rsplit('.', 1)[0]}. {name.rsplit('.', 1)[1]}"
            for name in names
        )
        media, cleaned = BasePlatformAdapter.extract_media(line)
        assert sorted(os.path.basename(p) for p, _ in media) == sorted(names)
        assert "MEDIA:" not in cleaned

    def test_spaced_filename_without_detached_extension_still_delivers(self, tmp_path):
        target = tmp_path / "my server.log"
        target.write_text("log line\n")
        media, _ = BasePlatformAdapter.extract_media(f"MEDIA:{target} is the log")
        assert [os.path.realpath(p) for p, _ in media] == [os.path.realpath(str(target))]

    def test_absent_file_is_not_delivered(self, tmp_path):
        media, _ = BasePlatformAdapter.extract_media(
            f"MEDIA:{tmp_path / 'missing'}. pdf"
        )
        assert media == []


class TestUndeliveredTagIsNamed:
    def test_unresolved_tag_keeps_path_and_is_logged(self, caplog):
        text = "Saved to MEDIA:/nonexistent/data.weirdext done"
        with caplog.at_level("INFO", logger="gateway.platforms.base"):
            media, cleaned = BasePlatformAdapter.extract_media(text)
        assert media == []
        assert "/nonexistent/data.weirdext" in cleaned
        assert "left undelivered" in caplog.text
        assert "/nonexistent/data.weirdext" in caplog.text

    def test_code_block_example_untouched(self):
        text = "Example:\n```\nMEDIA:/example/path.weirdext\n```\ndone"
        out = _strip_media_tag_directives(text)
        assert "MEDIA:/example/path.weirdext" in out
