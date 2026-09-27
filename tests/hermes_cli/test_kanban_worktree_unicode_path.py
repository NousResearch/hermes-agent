"""Regression: a composed (NFC) workspace path must resolve against its repo root.

macOS hands back DECOMPOSED path strings (NFD: ``o`` + U+0308) for names the user
typed in composed form (NFC: ``ö``). A OneDrive/FileProvider root such as
``OneDrive-Persönlich`` therefore arrives from ``git rev-parse --show-toplevel``
as NFD while the kanban DB row holds NFC. The raw ``Path`` comparisons in
``_resolve_worktree_workspace`` / ``_ensure_git_worktree`` /
``_cleanup_worktree_workspace`` then reported a real repo root as "not inside a
git repo" and dispatch died with a ``ValueError`` for every task on that board.

The identity comparison goes through ``_path_key`` (NFC-normalized), so Unicode
form can never decide a path identity question again.
"""

from __future__ import annotations

import subprocess
import unicodedata
from pathlib import Path

import pytest

from hermes_cli import kanban_db_workspace as kbw


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_path_key_ignores_unicode_form():
    """The identity key is form-insensitive but still distinguishes paths."""
    composed = "/tmp/repo-Persönlich"
    decomposed = _nfd(composed)

    assert composed != decomposed  # the two forms really are different strings
    assert kbw._path_key(composed) == kbw._path_key(decomposed)
    assert kbw._path_key(None) == ""
    assert kbw._path_key("/tmp/repo-Personlich") != kbw._path_key(composed)


@pytest.mark.platforms("macos")
def test_nfc_workspace_path_resolves_against_nfd_repo_root(tmp_path):
    """A task row in NFC form resolves on macOS, where git reports NFD."""
    repo = tmp_path / _nfd("repo-Persönlich")
    repo.mkdir()
    assert _git(repo.parent, "init", "-b", "main", str(repo)).returncode == 0
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    assert _git(repo, "add", "README.md").returncode == 0
    assert _git(repo, "commit", "-m", "init").returncode == 0

    composed = Path(_nfc(str(repo)))
    if str(composed) == str(repo):
        pytest.skip("filesystem does not round-trip the name in NFD here")

    class _Task:
        id = "t_unicode_probe"
        workspace_kind = "worktree"
        workspace_path = str(composed)
        branch_name = None

    resolved, _branch = kbw._resolve_worktree_workspace(_Task())

    expected = repo / ".worktrees" / "t_unicode_probe"
    assert expected.is_dir()
    assert _nfc(str(Path(resolved).resolve(strict=False))) == _nfc(
        str(expected.resolve(strict=False))
    )


@pytest.mark.platforms("macos")
def test_repo_root_treated_as_repo_root_in_other_unicode_form(tmp_path):
    """A root passed in NFC form must take the anchored-worktree path, not fail."""
    repo = tmp_path / _nfd("root-Persönlich")
    repo.mkdir()
    assert _git(repo.parent, "init", "-b", "main", str(repo)).returncode == 0
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    assert _git(repo, "add", "README.md").returncode == 0
    assert _git(repo, "commit", "-m", "init").returncode == 0

    composed_root = _nfc(str(repo))
    if composed_root == str(repo):
        pytest.skip("filesystem does not round-trip the name in NFD here")

    assert kbw._path_key(kbw._git_toplevel(Path(composed_root))) == kbw._path_key(repo)
