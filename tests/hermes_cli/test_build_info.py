"""Tests for exact Hermes source-revision resolution.

Release artifacts carry ``hermes_cli/_build_metadata.json``; source checkouts
fall back to a clean full ``HEAD``. The legacy root-level Docker stamp remains
covered at the bottom for compatibility with older artifacts.
"""

from pathlib import Path
import json
import subprocess
from unittest.mock import patch

import pytest


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _clean_source_checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "build-info@example.test")
    _git(repo, "config", "user.name", "Build Info Test")
    (repo / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-qm", "seed")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_clean_source_checkout_reports_full_head(tmp_path):
    """A clean source artifact is identified by its exact committed HEAD."""
    from hermes_cli import build_info

    repo, revision = _clean_source_checkout(tmp_path)

    assert build_info.get_source_revision(repo) == revision


def test_dirty_source_checkout_does_not_claim_head(tmp_path):
    """Tracked source changes make HEAD an inexact artifact identity."""
    from hermes_cli import build_info

    repo, _ = _clean_source_checkout(tmp_path)
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_untracked_source_checkout_does_not_claim_head(tmp_path):
    """Untracked source can affect runtime behavior and is therefore dirty."""
    from hermes_cli import build_info

    repo, _ = _clean_source_checkout(tmp_path)
    (repo / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_source_without_metadata_or_git_reports_none(tmp_path):
    """An unpackaged tree cannot manufacture an exact source revision."""
    from hermes_cli import build_info

    assert build_info.get_source_revision(tmp_path) is None


def test_embedded_revision_precedes_dirty_source_checkout(tmp_path):
    """Immutable artifact metadata remains authoritative without clean Git."""
    from hermes_cli import build_info

    repo, _ = _clean_source_checkout(tmp_path)
    revision = "0123456789abcdef0123456789abcdef01234567"
    metadata = repo / "hermes_cli" / "_build_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps({"source_revision": revision}) + "\n",
        encoding="utf-8",
    )
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.get_source_revision(repo) == revision


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{",
        "{}",
        json.dumps({"source_revision": None}),
        json.dumps({"source_revision": "a" * 39}),
        json.dumps({"source_revision": "A" * 40}),
        json.dumps({"source_revision": "g" * 40}),
    ],
)
def test_malformed_embedded_revision_reports_none(tmp_path, payload):
    """A malformed artifact stamp can never be promoted to exact identity."""
    from hermes_cli import build_info

    repo, _ = _clean_source_checkout(tmp_path)
    metadata = repo / "hermes_cli" / "_build_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text(payload, encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_build_assertion_cannot_override_embedded_revision(tmp_path):
    """Rebuilding a packaged source preserves its immutable provenance."""
    from hermes_cli import build_info

    embedded = "0123456789abcdef0123456789abcdef01234567"
    asserted = "89abcdef0123456789abcdef0123456789abcdef"
    build_info.write_build_metadata(tmp_path, embedded)

    assert build_info.resolve_build_source_revision(tmp_path, asserted) == embedded


def test_build_assertion_must_match_clean_checkout(tmp_path):
    """A builder variable cannot relabel a Git checkout as another commit."""
    from hermes_cli import build_info

    repo, revision = _clean_source_checkout(tmp_path)
    other = "89abcdef0123456789abcdef0123456789abcdef"

    assert build_info.resolve_build_source_revision(repo, revision) == revision
    assert build_info.resolve_build_source_revision(repo, other) is None


def test_build_assertion_cannot_bless_dirty_checkout(tmp_path):
    """A dirty checkout remains inexact even with a valid asserted SHA."""
    from hermes_cli import build_info

    repo, revision = _clean_source_checkout(tmp_path)
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.resolve_build_source_revision(repo, revision) is None


def test_source_archive_may_use_validated_build_assertion(tmp_path):
    """Git-free builders can attest the revision used to create an artifact."""
    from hermes_cli import build_info

    revision = "0123456789abcdef0123456789abcdef01234567"

    assert build_info.resolve_build_source_revision(tmp_path, revision) == revision
    assert build_info.resolve_build_source_revision(tmp_path, "a" * 39) is None


def test_short_build_sha_comes_from_canonical_metadata(tmp_path):
    """Legacy short displays derive from the same full artifact identity."""
    from hermes_cli import build_info

    revision = "0123456789abcdef0123456789abcdef01234567"
    build_info.write_build_metadata(tmp_path, revision)

    assert build_info.get_build_sha(project_root=tmp_path) == revision[:8]


def test_explicit_null_metadata_blocks_stale_legacy_fallback(tmp_path):
    """A new artifact that declines identity cannot inherit an older stamp."""
    from hermes_cli import build_info

    build_info.write_build_metadata(tmp_path, None)
    legacy = tmp_path / ".hermes_build_sha"
    legacy.write_text("a" * 40 + "\n", encoding="utf-8")

    with patch.object(build_info, "_PROJECT_ROOT", tmp_path), \
         patch.object(build_info, "_BUILD_SHA_FILE", legacy):
        assert build_info.get_build_sha() is None


def test_get_build_sha_returns_none_when_file_absent(tmp_path):
    """Source installs: no file present → None, callers fall back to git."""
    from hermes_cli import build_info

    missing = tmp_path / ".hermes_build_sha"  # never created

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", missing):
        assert build_info.get_build_sha() is None


def test_get_build_sha_reads_baked_file(tmp_path):
    """Docker image case: file exists with full 40-char SHA → truncated to 8."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("abcdef1234567890abcdef1234567890abcdef12\n")

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() == "abcdef12"


def test_get_build_sha_respects_short_argument(tmp_path):
    """``short=N`` truncates to N chars; ``short<=0`` returns full SHA."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"
    sha_file.write_text(full_sha + "\n")

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha(short=12) == "abcdef123456"
        assert build_info.get_build_sha(short=0) == full_sha
        assert build_info.get_build_sha(short=-1) == full_sha


def test_get_build_sha_strips_whitespace(tmp_path):
    """The Dockerfile uses ``printf '%s\\n'`` — strip the trailing newline."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("  abcdef1234567890\n\n")

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() == "abcdef12"


def test_get_build_sha_returns_none_for_empty_file(tmp_path):
    """A whitespace-only file is treated as absent."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("   \n\n")

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha() is None


def test_get_build_sha_swallows_read_errors(tmp_path):
    """Any IO exception from the read returns None — never raises."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    sha_file.write_text("abcdef1234567890\n")

    with patch.object(build_info, "get_source_revision", return_value=None), \
         patch.object(build_info, "_BUILD_SHA_FILE", sha_file), \
         patch.object(Path, "read_text", side_effect=OSError("boom")):
        assert build_info.get_build_sha() is None
