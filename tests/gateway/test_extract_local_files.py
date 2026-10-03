"""
Tests for extract_local_files() — auto-detection of bare local file paths
in model response text for native media delivery.

Covers: path matching, code-block exclusion, URL rejection, tilde expansion,
deduplication, text cleanup, and extension routing.

Based on PR #1636 by sudoingX (salvaged + hardened).
"""

from unittest.mock import patch

import pytest

from gateway.platforms.base import BasePlatformAdapter

from gateway.platforms.base import (
    BasePlatformAdapter,
    _is_sensitive_bare_local_file_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(content: str, existing_files: set[str] | None = None):
    """
    Run extract_local_files with os.path.isfile mocked to return True
    for any path in *existing_files* (expanded form).  If *existing_files*
    is None every path passes.
    """
    existing = existing_files

    def fake_isfile(p):
        if existing is None:
            return True
        return p in existing

    def fake_expanduser(p):
        if p.startswith("~/"):
            return "/home/user" + p[1:]
        return p

    with patch("os.path.isfile", side_effect=fake_isfile), \
         patch("os.path.expanduser", side_effect=fake_expanduser):
        return BasePlatformAdapter.extract_local_files(content)


# ---------------------------------------------------------------------------
# Basic detection
# ---------------------------------------------------------------------------

class TestBasicDetection:

    def test_absolute_path_image(self):
        paths, cleaned = _extract("Here is the screenshot /root/screenshots/game.png enjoy")
        assert paths == ["/root/screenshots/game.png"]
        assert "/root/screenshots/game.png" not in cleaned
        assert "Here is the screenshot" in cleaned

    def test_tilde_path_image(self):
        paths, cleaned = _extract("Check out ~/photos/cat.jpg for the cat")
        assert paths == ["/home/user/photos/cat.jpg"]
        assert "~/photos/cat.jpg" not in cleaned

    def test_video_extensions(self):
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            text = f"Video at /tmp/clip{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/clip{ext}"

    def test_image_extensions(self):
        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            text = f"Image at /tmp/pic{ext} here"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/pic{ext}"

    def test_document_extensions(self):
        """Documents (PDF, Word, plain text, etc.) ship as file uploads."""
        for ext in (".pdf", ".docx", ".doc", ".odt", ".rtf", ".txt", ".md"):
            text = f"Report at /tmp/report{ext} attached"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/report{ext}"


    def test_path_at_line_start(self):
        paths, _ = _extract("/var/data/image.png")
        assert paths == ["/var/data/image.png"]

    def test_spreadsheet_and_data_extensions(self):
        """Spreadsheets and non-sensitive structured data ship as file uploads."""
        for ext in (".xlsx", ".xls", ".csv", ".tsv", ".json", ".xml"):
            text = f"Data at /tmp/data{ext} ready"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed for {ext}"
            assert paths[0] == f"/tmp/data{ext}"

    def test_bare_yaml_paths_are_not_auto_attached(self):
        """YAML is usually config; require explicit MEDIA: for upload."""
        for ext in (".yaml", ".yml"):
            text = f"Data at /tmp/data{ext} ready"
            paths, cleaned = _extract(text)
            assert paths == []
            assert cleaned == text


# ---------------------------------------------------------------------------
# Non-existent files are skipped
# ---------------------------------------------------------------------------

class TestIsfileGuard:

    def test_nonexistent_path_skipped(self):
        """Paths that don't exist on disk are not extracted."""
        paths, cleaned = _extract(
            "See /tmp/nope.png here",
            existing_files=set(),  # nothing exists
        )
        assert paths == []
        assert "/tmp/nope.png" in cleaned  # not stripped


# ---------------------------------------------------------------------------
# URL false-positive prevention
# ---------------------------------------------------------------------------

class TestURLRejection:

    def test_https_url_not_matched(self):
        """Paths embedded in HTTP URLs must not be extracted."""
        paths, cleaned = _extract("Visit https://example.com/images/photo.png for details")
        # The regex lookbehind should prevent matching the URL's path segment
        # Even if it did match, isfile would be False for /images/photo.png
        # (we mock isfile to True-for-all here, so the lookbehind is the guard)
        assert paths == []
        assert "https://example.com/images/photo.png" in cleaned


# ---------------------------------------------------------------------------
# Code block exclusion
# ---------------------------------------------------------------------------

class TestCodeBlockExclusion:

    def test_fenced_code_block_skipped(self):
        text = "Here's how:\n```python\nimg = open('/tmp/image.png')\n```\nDone."
        paths, cleaned = _extract(text)
        assert paths == []
        assert "/tmp/image.png" in cleaned  # not stripped

    def test_inline_code_skipped(self):
        text = "Use the path `/tmp/image.png` in your config"
        paths, cleaned = _extract(text)
        assert paths == []
        assert "`/tmp/image.png`" in cleaned


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_paths_deduplicated(self):
        text = "See /tmp/img.png and also /tmp/img.png again"
        paths, _ = _extract(text)
        assert paths == ["/tmp/img.png"]


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

class TestTextCleanup:

    def test_path_removed_from_text(self):
        paths, cleaned = _extract("Before /tmp/x.png after")
        assert "Before" in cleaned
        assert "after" in cleaned
        assert "/tmp/x.png" not in cleaned

    def test_excessive_blank_lines_collapsed(self):
        text = "Before\n\n\n/tmp/x.png\n\n\nAfter"
        _, cleaned = _extract(text)
        assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self):
        paths, cleaned = _extract("")
        assert paths == []
        assert cleaned == ""


    @pytest.mark.parametrize(
        "content,expected",
        [
            # Backslash separators (native Windows style)
            ("See C:\\Users\\test\\image.png here", "C:\\Users\\test\\image.png"),
            # Forward slashes with drive letter (common in cross-platform code)
            ("See C:/Users/test/image.png here", "C:/Users/test/image.png"),
            # Non-C: drive
            ("Video at D:/data/clip.mp4 ready", "D:/data/clip.mp4"),
            # Lowercase drive letter
            ("Path e:/audio/track.mp3 done", "e:/audio/track.mp3"),
        ],
    )
    def test_windows_drive_letter_paths_matched(self, content, expected):
        """Windows drive-letter paths (C:/..., C:\\...) must be detected (#34632).

        Prior behavior anchored on (?:~/|/) only, which silently dropped
        Windows absolute paths so the agent's bare-path references were
        sent as text instead of native uploads.
        """
        paths, cleaned = _extract(content)
        assert paths == [expected]
        assert expected not in cleaned

    def test_relative_windows_path_not_matched(self):
        """A bare Windows-style filename without a drive letter must still
        not match (e.g. ``foo\\bar.png`` is treated as relative, like its
        Unix sibling ``foo/bar.png``)."""
        paths, _ = _extract("File at foo\\bar.png here")
        assert paths == []


    def test_path_followed_by_punctuation(self):
        """Path followed by comma, period, paren should still match."""
        for suffix in [",", ".", ")", ":", ";"]:
            text = f"See /tmp/img.png{suffix} details"
            paths, _ = _extract(text)
            assert len(paths) == 1, f"Failed with suffix '{suffix}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestExplicitMediaTagStillWorks:

    def test_explicit_media_tag_can_attach_yaml(self):
        media, cleaned = BasePlatformAdapter.extract_media("Ship MEDIA:/tmp/generated.yaml now")
        assert media == [("/tmp/generated.yaml", False)]
        assert cleaned == "Ship  now"


class TestSensitiveGuardDirect:
    """Exercise ``_is_sensitive_bare_local_file_path`` directly.

    The extraction suite above can only cover forms that survive the extension
    regex. These cases pin the guard's own contract, including the
    component-bounded word matching that replaced a raw substring test.
    """

    @pytest.mark.parametrize("path", [
        "/Users/alice/.ssh/id_rsa",
        "/Users/alice/.ssh/id_rsa.bak",
        "/Users/alice/.ssh/id_ed25519",
        "/Users/alice/certs/server.pem",
        "/Users/alice/certs/server.key",
        "/Users/alice/.hermes/config.yaml",
        "/Users/alice/.hermes/plots/chart.png",
        "/Users/alice/project/.env.local",
        "/Users/alice/data/api-token.json",
        "/Users/alice/data/service-secret.json",
        "/Users/alice/data/passwords.txt",
        "/Users/alice/data/credentials.json",
        # Archive containers are opaque and can hold anything, so the word
        # heuristic must apply to them exactly as it does to .json/.txt.
        # These previously bypassed it because archive extensions were not
        # in the word-scoped set, and every one of them is in
        # BARE_LOCAL_FILE_EXTS -- i.e. auto-attachable.
        "/Users/alice/out/secrets.zip",
        "/Users/alice/out/credentials.tar",
        "/Users/alice/out/passwords.gz",
        "/Users/alice/out/secrets.7z",
        "/Users/alice/out/api-tokens.zip",
        "/Users/alice/out/token.tgz",
    ])
    def test_guard_blocks_sensitive_paths(self, path):
        assert _is_sensitive_bare_local_file_path(path) is True

    @pytest.mark.parametrize("path", [
        # Component-bounded matching: these contain a sensitive WORD as a
        # substring but not as a whole component.
        "/Users/alice/out/tokenization.json",
        "/Users/alice/out/tokenizer.json",
        # Extension-scoped matching: the word IS a component, but the
        # extension is a binary render/artifact format that cannot be a
        # credential store.
        "/Users/alice/plots/token_counts.png",
        "/Users/alice/docs/secret-santa.pdf",
        "/Users/alice/docs/password-policy-guide.pdf",
        "/Users/alice/clips/tokens-per-second.mp4",
        # Plain safe artifacts.
        "/Users/alice/report.pdf",
        "/Users/alice/chart.png",
    ])
    def test_guard_allows_legitimate_artifacts(self, path):
        assert _is_sensitive_bare_local_file_path(path) is False

    def test_sensitive_json_candidate_is_rejected_without_a_filesystem_probe(self):
        """A matched sensitive candidate must never hit ``os.path.isfile()``."""
        text = "See /Users/alice/data/api-token.json for details"

        def exploding_isfile(p):
            raise AssertionError(f"isfile() probed a sensitive candidate: {p}")

        def fake_expanduser(p):
            return "/home/user" + p[1:] if p.startswith("~/") else p

        with patch("os.path.isfile", side_effect=exploding_isfile), \
                patch("os.path.expanduser", side_effect=fake_expanduser):
            paths, cleaned = BasePlatformAdapter.extract_local_files(text)

        assert paths == []
        assert cleaned == text

    def test_safe_artifact_still_attaches_next_to_sensitive_path(self):
        text = "Safe /tmp/report.pdf but do not send /Users/alice/.hermes/config.yaml"
        paths, cleaned = _extract(text, existing_files={
            "/tmp/report.pdf",
            "/Users/alice/.hermes/config.yaml",
        })
        assert paths == ["/tmp/report.pdf"]
        assert "/tmp/report.pdf" not in cleaned
        assert "/Users/alice/.hermes/config.yaml" in cleaned


class TestSensitivePathRejection:

    @pytest.mark.parametrize("path", [
        "/Users/alice/.hermes/config.yaml",
        "/Users/alice/.hermes/profiles/aegis/config.yaml",
        "/Users/alice/.hermes/.env",
        "/Users/alice/.hermes/auth.json",
        "/Users/alice/project/config.yaml",
        "/Users/alice/project/config.yml",
        "/Users/alice/project/.env",
        "/Users/alice/project/.env.local",
        # NOTE: id_rsa / id_ed25519 / *.pem are NOT extracted at all -- they
        # carry no BARE_LOCAL_FILE_EXTS extension, so extract_local_files()
        # never produces a candidate for the guard to reject. They are covered
        # directly against the guard in TestSensitiveGuardDirect below; keeping
        # them here would be a vacuous pass.
        "/Users/alice/certs/server.key",
        "/Users/alice/data/api-token.json",
        "/Users/alice/data/passwords.txt",
        "/Users/alice/data/service-secret.json",
        "/Users/alice/data/credentials.json",
    ])
    def test_sensitive_bare_paths_are_not_attached_or_stripped(self, path):
        text = f"The file is {path}"
        paths, cleaned = _extract(text)
        assert paths == []
        assert cleaned == text

    @pytest.mark.parametrize("path", [
        "/Users/alice/.ssh/id_rsa",
        "/Users/alice/.ssh/id_ed25519",
        "/Users/alice/certs/server.pem",
    ])
    def test_extensionless_and_pem_paths_produce_no_candidate(self, path):
        """Guard-independent proof: these forms never reach the guard.

        ``extract_local_files()`` only matches BARE_LOCAL_FILE_EXTS, which has
        no ``.pem`` and no extensionless form. Asserting "not attached" for
        them in the guard suite above would pass even with the guard deleted,
        so the real contract -- no candidate is ever produced -- is asserted
        here instead.
        """
        text = f"The file is {path}"
        with patch(
            "gateway.platforms.base._is_sensitive_bare_local_file_path",
            side_effect=AssertionError("guard must not be consulted"),
        ):
            paths, cleaned = _extract(text)
        assert paths == []
        assert cleaned == text
