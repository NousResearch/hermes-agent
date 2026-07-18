"""[[artifact:path|title]] tag normalization for messaging-platform delivery.

Browser surfaces (hermes-webui) render the tag themselves; messaging platforms
have no stable-URL concept, so the gateway rewrites the tag to its bare file
path and the existing bare-path deliverable pipeline ships the file natively.
Fake-first: no gateway process, no real platform — pure text transformation
plus one integration check against extract_local_files with a real tmp file.
"""

import os
import tempfile

from gateway.platforms.base import BasePlatformAdapter


norm = BasePlatformAdapter.normalize_artifact_tags


class TestNormalizeArtifactTags:
    def test_plain_tag_becomes_bare_path(self):
        assert norm("Done! [[artifact:/tmp/report.html]]") == "Done! /tmp/report.html"

    def test_tag_with_title_keeps_only_path(self):
        assert norm("[[artifact:/tmp/report.html|Q3 Report]]") == "/tmp/report.html"

    def test_tilde_path_preserved_for_downstream_expansion(self):
        assert norm("[[artifact:~/ws/chart.png|Chart]]") == "~/ws/chart.png"

    def test_multiple_tags(self):
        out = norm("[[artifact:/tmp/a.html|A]] und [[artifact:/tmp/b.pdf]]")
        assert out == "/tmp/a.html und /tmp/b.pdf"

    def test_no_tag_passthrough_is_identity(self):
        s = "normal reply with [[as_document]] and MEDIA:/tmp/x.png"
        assert norm(s) is s

    def test_fenced_code_block_untouched(self):
        s = "Nutze das so:\n```\n[[artifact:/pfad/seite.html|Titel]]\n```\nfertig"
        assert norm(s) == s

    def test_inline_code_untouched(self):
        s = "Das Tag `[[artifact:/x.html|T]]` bleibt stehen"
        assert norm(s) == s

    def test_malformed_tags_left_alone(self):
        assert norm("[[artifact:]]") == "[[artifact:]]"
        assert norm("[[artifact:/a\nb]]") == "[[artifact:/a\nb]]"

    def test_empty_and_none_safe(self):
        assert norm("") == ""


class TestDeliverablePipelineIntegration:
    def test_normalized_tag_path_is_picked_up_by_extract_local_files(self):
        fd, path = tempfile.mkstemp(suffix=".pdf", prefix="artifact-tag-", dir="/tmp")
        os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(b"%PDF-1.4 test")
            text = norm(f"Hier dein Report: [[artifact:{path}|Report]]")
            assert path in text and "[[artifact:" not in text
            files, cleaned = BasePlatformAdapter.extract_local_files(text)
            assert files == [path]
            assert path not in cleaned
        finally:
            os.unlink(path)

    def test_missing_file_tag_degrades_to_visible_path_text(self):
        text = norm("Siehe [[artifact:/tmp/never-written-xyz.html|Report]]")
        assert text == "Siehe /tmp/never-written-xyz.html"
        files, cleaned = BasePlatformAdapter.extract_local_files(text)
        assert files == []
        assert "/tmp/never-written-xyz.html" in cleaned
