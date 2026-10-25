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
    from hermes_cli import source_revision as build_info

    repo, revision = _clean_source_checkout(tmp_path)

    assert build_info.get_source_revision(repo) == revision


def test_dirty_source_checkout_does_not_claim_head(tmp_path):
    """Tracked source changes make HEAD an inexact artifact identity."""
    from hermes_cli import source_revision as build_info

    repo, _ = _clean_source_checkout(tmp_path)
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_untracked_source_checkout_does_not_claim_head(tmp_path):
    """Untracked source can affect runtime behavior and is therefore dirty."""
    from hermes_cli import source_revision as build_info

    repo, _ = _clean_source_checkout(tmp_path)
    (repo / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_source_without_metadata_or_git_reports_none(tmp_path):
    """An unpackaged tree cannot manufacture an exact source revision."""
    from hermes_cli import source_revision as build_info

    assert build_info.get_source_revision(tmp_path) is None


def test_embedded_revision_precedes_dirty_source_checkout(tmp_path):
    """Immutable artifact metadata remains authoritative without clean Git."""
    from hermes_cli import source_revision as build_info

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
    from hermes_cli import source_revision as build_info

    repo, _ = _clean_source_checkout(tmp_path)
    metadata = repo / "hermes_cli" / "_build_metadata.json"
    metadata.parent.mkdir()
    metadata.write_text(payload, encoding="utf-8")

    assert build_info.get_source_revision(repo) is None


def test_build_assertion_cannot_override_embedded_revision(tmp_path):
    """Rebuilding a packaged source preserves its immutable provenance."""
    from hermes_cli import source_revision as build_info

    embedded = "0123456789abcdef0123456789abcdef01234567"
    asserted = "89abcdef0123456789abcdef0123456789abcdef"
    build_info.write_build_metadata(tmp_path, embedded)

    assert build_info.resolve_build_source_revision(tmp_path, asserted) == embedded


def test_build_assertion_must_match_clean_checkout(tmp_path):
    """A builder variable cannot relabel a Git checkout as another commit."""
    from hermes_cli import source_revision as build_info

    repo, revision = _clean_source_checkout(tmp_path)
    other = "89abcdef0123456789abcdef0123456789abcdef"

    assert build_info.resolve_build_source_revision(repo, revision) == revision
    assert build_info.resolve_build_source_revision(repo, other) is None


def test_build_assertion_cannot_bless_dirty_checkout(tmp_path):
    """A dirty checkout remains inexact even with a valid asserted SHA."""
    from hermes_cli import source_revision as build_info

    repo, revision = _clean_source_checkout(tmp_path)
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert build_info.resolve_build_source_revision(repo, revision) is None


def test_source_archive_may_use_validated_build_assertion(tmp_path):
    """Git-free builders can attest the revision used to create an artifact."""
    from hermes_cli import source_revision as build_info

    revision = "0123456789abcdef0123456789abcdef01234567"

    assert build_info.resolve_build_source_revision(tmp_path, revision) == revision
    assert build_info.resolve_build_source_revision(tmp_path, "a" * 39) is None



@pytest.mark.parametrize("commit,dirty,expected", [("a" * 40, False, "a" * 40), ("a" * 40, True, None), ("A" * 40, False, None), ("0" * 40, False, None), ("abc", False, None), (None, False, None)])
def test_release_install_stamp_is_exact_or_null(tmp_path, commit, dirty, expected):
    from hermes_cli.source_revision import get_source_revision

    (tmp_path / "install-stamp.json").write_text(json.dumps({"commit": commit, "dirty": dirty, "source": "ci", "updateMechanism": "external"}))
    assert get_source_revision(tmp_path) == expected


def test_source_install_stamp_cannot_bless_dirty_checkout(tmp_path):
    from hermes_cli.source_revision import get_source_revision

    repo, revision = _clean_source_checkout(tmp_path)
    (repo / ".gitignore").write_text("install-stamp.json\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore stamp")
    revision = _git(repo, "rev-parse", "HEAD")
    (repo / "install-stamp.json").write_text(json.dumps({"commit": revision, "source": "git", "updateMechanism": "self"}))
    (repo / "tracked.py").write_text("VALUE = 3\n")
    assert get_source_revision(repo) is None

@pytest.mark.parametrize("payload", ["{", "[]", '{"commit": "abc"}', '{"commit": 7}'])
def test_malformed_install_stamp_does_not_fall_through_to_git(tmp_path, payload):
    from hermes_cli.source_revision import get_source_revision

    repo, _ = _clean_source_checkout(tmp_path)
    (repo / "install-stamp.json").write_text(payload)
    assert get_source_revision(repo) is None


def test_nested_package_does_not_inherit_unrelated_git_head(tmp_path):
    from hermes_cli.source_revision import get_source_revision

    repo, _ = _clean_source_checkout(tmp_path)
    nested = repo / "unrelated-package"
    nested.mkdir()
    assert get_source_revision(nested) is None


def test_failed_status_cannot_claim_clean_head(tmp_path):
    from hermes_cli import source_revision

    repo, _ = _clean_source_checkout(tmp_path)
    with patch.object(source_revision.subprocess, "run", side_effect=OSError("no git")):
        assert source_revision.get_source_revision(repo) is None


def test_nix_wrapper_install_root_precedes_package_metadata(tmp_path, monkeypatch):
    from hermes_cli.source_revision import get_source_revision

    revision = "b" * 40
    (tmp_path / "install-stamp.json").write_text(json.dumps({
        "commit": revision, "source": "nix", "distribution": "nix", "updateMechanism": "external",
    }))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    assert get_source_revision() == revision
