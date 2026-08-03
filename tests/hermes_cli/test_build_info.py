"""Tests for hermes_cli.build_info — baked-in build SHA resolution.

The build SHA is written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg
into ``<project_root>/.hermes_build_sha``.  These tests cover the read-side
helper: missing file, malformed file, truncation, and error tolerance.
"""

from pathlib import Path
from unittest.mock import patch


def test_get_build_sha_returns_none_when_file_absent(tmp_path):
    """Source installs: no file present → None, callers fall back to git."""
    from hermes_cli import build_info

    missing = tmp_path / ".hermes_build_sha"  # never created

    with patch.object(build_info, "_BUILD_SHA_FILE", missing):
        assert build_info.get_build_sha() is None


def test_get_build_sha_reads_baked_file(tmp_path):
    """Docker image case: file exists with full 40-char SHA → truncated to 8."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("abcdef1234567890abcdef1234567890abcdef12\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() == "abcdef12"


def test_get_build_sha_respects_short_argument(tmp_path):
    """``short=N`` truncates to N chars; ``short<=0`` returns full SHA."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"
    sha_file.write_text(full_sha + "\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha(short=12) == "abcdef123456"
        assert build_info.get_build_sha(short=0) == full_sha
        assert build_info.get_build_sha(short=-1) == full_sha


def test_get_build_sha_strips_whitespace(tmp_path):
    """The Dockerfile uses ``printf '%s\\n'`` — strip the trailing newline."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("  abcdef1234567890\n\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() == "abcdef12"


def test_get_build_sha_returns_none_for_empty_file(tmp_path):
    """A whitespace-only file is treated as absent."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("   \n\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() is None


def test_get_build_sha_swallows_read_errors(tmp_path):
    """Any IO exception from the read returns None — never raises."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("abcdef1234567890\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file), \
         patch.object(Path, "read_text", side_effect=OSError("boom")):
        assert build_info.get_build_sha() is None
