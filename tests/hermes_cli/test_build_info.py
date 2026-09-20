"""Tests for hermes_cli.build_info — baked-in build SHA resolution.

The build SHA is written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg
into ``<project_root>/.hermes_build_sha``.  These tests cover the read-side
helper: missing file, malformed file, truncation, and error tolerance.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_get_build_sha_returns_none_when_file_absent(tmp_path):
    """Source installs: no file present → None, callers fall back to git."""
    from hermes_cli import build_info

    missing = tmp_path / ".hermes_build_sha"  # never created

    with patch.object(build_info, "_BUILD_SHA_FILE", missing):
        assert build_info.get_build_sha() is None


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


@pytest.mark.parametrize("sha_length", [40, 64], ids=["sha1", "sha256"])
def test_resolve_git_head_accepts_supported_object_formats(tmp_path, sha_length):
    """Runtime identity must match Git's full SHA in SHA-1 and SHA-256 repositories."""
    from hermes_cli import build_info

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    sha = "a" * sha_length
    (git_dir / "HEAD").write_text(sha + "\n", encoding="utf-8")

    assert build_info._resolve_git_head_sha(tmp_path) == sha


