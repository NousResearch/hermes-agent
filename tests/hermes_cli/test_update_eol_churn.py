"""Regression: ``hermes update`` should take a managed checkout off autocrlf=true.

Git for Windows ships ``core.autocrlf=true`` in its system config, which
renormalizes this repo's LF text files to CRLF in the working tree and breaks
``git checkout`` on update. ``install.ps1`` pins ``core.autocrlf=false`` on new
installs, but checkouts created before that landed cannot receive the fix, so
``hermes update`` has to repair them.

The pin and the cleanup are one operation: under ``autocrlf=true`` git compares
normalized content, so the CRLF tree reads clean and pinning alone would expose
the whole tree as modified. These tests pin down that coupling.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.update_cmd import _normalize_managed_eol

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

GIT_CMD = ["git"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _managed_repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """A checkout in the broken state: LF in the index, CRLF in the worktree,
    ``core.autocrlf=true`` still set — i.e. what Git for Windows leaves behind.

    The CRLF is written by git's own checkout rather than by hand, which is how a
    real managed install gets there. It matters: git's stat cache then records the
    CRLF size, so the churn reads *clean* under ``autocrlf=true`` and is only
    visible once the pin is evaluated.
    """
    repo = tmp_path / "managed"
    repo.mkdir()
    _git(repo, "init")
    for name, body in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _git(repo, "-c", "core.autocrlf=false", "add", "-A")
    _git(repo, "commit", "-m", "init")
    _git(repo, "config", "core.autocrlf", "true")
    for name in files:
        (repo / name).unlink()
    _git(repo, "checkout", "--", ".")
    # Deterministic dirtiness: whether `git diff` content-checks an entry (and
    # so sees the CRLF churn) or trusts the stat cache depends on racy-git
    # detection — entries whose mtime equals the index timestamp get content-
    # compared, later ones read clean. On a fast runner a large checkout
    # straddles that boundary nondeterministically (CI flake: 92/661 of 1200
    # dirty). Bump every worktree mtime past the index write so ALL entries
    # are stat-stale and git must content-compare each one.
    import os as _os
    import time as _time

    bumped = _time.time() + 5
    for name in files:
        _os.utime(repo / name, (bumped, bumped))
    return repo


def _dirty(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "diff", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in out.stdout.splitlines() if line}


def _autocrlf(repo: Path) -> str:
    out = subprocess.run(
        ["git", "config", "--local", "--get", "core.autocrlf"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def test_churn_invisible_under_autocrlf_true_is_still_found(tmp_path: Path) -> None:
    """The reason the pin and the cleanup are one operation.

    A managed install's CRLF churn reads clean while ``autocrlf=true`` is set, so
    an implementation that only looked at ``git status`` would pin the config and
    hand the update a whole-tree autostash. Evaluating the tree as it would look
    pinned is what makes the churn visible in time to clear it.
    """
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\ny = 2\n"})
    as_git_sees_it = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert as_git_sees_it.stdout.strip() == ""
    assert b"\r\n" in (repo / "a.py").read_bytes()

    _normalize_managed_eol(GIT_CMD, repo)

    assert _dirty(repo) == set()
    assert b"\r\n" not in (repo / "a.py").read_bytes()
    assert _autocrlf(repo) == "false"


def test_churn_is_cleared_and_the_pin_is_persisted(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n", "b.md": b"# Title\n\nbody\n"})

    _normalize_managed_eol(GIT_CMD, repo)

    assert _dirty(repo) == set()
    assert b"\r\n" not in (repo / "a.py").read_bytes()
    assert _autocrlf(repo) == "false"


def test_real_edits_survive_even_when_line_endings_also_flipped(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"churn.py": b"y = 2\n", "both.py": b"z = 3\n"})
    # A genuine edit that ALSO got renormalized must not be discarded.
    (repo / "both.py").write_bytes(b"z = 3\r\nz += 1\r\n")

    _normalize_managed_eol(GIT_CMD, repo)

    assert _dirty(repo) == {"both.py"}
    assert (repo / "both.py").read_bytes() == b"z = 3\r\nz += 1\r\n"
    assert _autocrlf(repo) == "false"


def test_pin_alone_is_written_when_there_is_no_churn(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    (repo / "a.py").write_bytes(b"x = 1\n")

    _normalize_managed_eol(GIT_CMD, repo)

    assert _dirty(repo) == set()
    assert _autocrlf(repo) == "false"


@pytest.mark.skipif(sys.platform == "win32", reason="shim needs a POSIX shell")
def test_pin_is_withheld_when_the_churn_cannot_be_cleared(tmp_path: Path) -> None:
    """If normalization can't finish, the checkout is left as found — pinning
    anyway would surface churn we failed to clear."""
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    # Real git everywhere except the restore, which fails. Stands in for any
    # reason it cannot finish: a git too old for --pathspec-from-file, an
    # unwritable working tree, a file locked by a running process.
    shim = tmp_path / "git-no-checkout"
    shim.write_text(
        '#!/bin/sh\nfor a in "$@"; do [ "$a" = checkout ] && exit 1; done\nexec git "$@"\n'
    )
    shim.chmod(0o755)

    _normalize_managed_eol([str(shim)], repo)

    assert _autocrlf(repo) == "true"
    assert b"\r\n" in (repo / "a.py").read_bytes()


def test_already_pinned_checkout_is_untouched(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    _git(repo, "config", "core.autocrlf", "false")

    _normalize_managed_eol(GIT_CMD, repo)

    # Nothing to repair from this function's perspective: autocrlf is not true,
    # so it must not start restoring files behind the user's back.
    assert _dirty(repo) == {"a.py"}


def test_autocrlf_input_is_left_alone(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    _git(repo, "config", "core.autocrlf", "input")

    _normalize_managed_eol(GIT_CMD, repo)

    assert _autocrlf(repo) == "input"


def test_churn_across_more_files_than_fit_in_one_argv(tmp_path: Path) -> None:
    """The pathspec goes over stdin, so a fully renormalized tree fits.

    Passing thousands of paths as arguments overflows the Windows command-line
    limit; this asserts the batch is not argv-bound.
    """
    files = {f"pkg/mod_{i:04d}.py": f"VALUE = {i}\n".encode() for i in range(1200)}
    repo = _managed_repo(tmp_path, files)
    assert len(_dirty(repo)) == len(files)

    _normalize_managed_eol(GIT_CMD, repo)

    assert _dirty(repo) == set()
    assert _autocrlf(repo) == "false"


def test_non_repo_is_ignored(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _normalize_managed_eol(GIT_CMD, tmp_path)

    assert capsys.readouterr().out == ""
