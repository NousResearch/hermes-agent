"""Tests for scripts/ci/check_dco.py.

Two things the DCO check must get right (both raised in review of the first
draft):

* A sign-off is accepted when it matches the **author or the committer** —
  ``git commit -s`` signs off as the committer, so ``--author=... -s`` must pass.
* A ``Signed-off-by:`` line only counts when it is a real trailer, not prose
  sitting in the commit body.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_dco.py"
_spec = importlib.util.spec_from_file_location("check_dco", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load check_dco.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


# --- pure helpers ---------------------------------------------------------


def test_parse_signoff_trailers_extracts_only_signoff_keys():
    parsed = "Signed-off-by: Real Person <real@example.com>\nReviewed-by: Someone <s@example.com>"
    assert _mod.parse_signoff_trailers(parsed) == ["Real Person <real@example.com>"]


def test_parse_signoff_trailers_empty_when_no_trailers():
    assert _mod.parse_signoff_trailers("") == []


def test_is_signed_off_matches_author():
    author = "Ann Author <ann@example.com>"
    committer = "Ann Author <ann@example.com>"
    assert _mod.is_signed_off(author, committer, [author])


def test_is_signed_off_matches_committer_when_author_differs():
    # git commit --author="Ann" -s  →  author is Ann, committer + sign-off are Cody.
    author = "Ann Author <ann@example.com>"
    committer = "Cody Committer <cody@example.com>"
    assert _mod.is_signed_off(author, committer, [committer])


def test_is_signed_off_case_insensitive():
    author = "Ann Author <Ann@Example.com>"
    committer = author
    assert _mod.is_signed_off(author, committer, ["ann author <ann@example.com>"])


def test_is_signed_off_false_when_signoff_matches_nobody():
    author = "Ann Author <ann@example.com>"
    committer = "Cody Committer <cody@example.com>"
    assert not _mod.is_signed_off(author, committer, ["Ghost <ghost@example.com>"])


def test_is_signed_off_false_when_no_signoffs():
    assert not _mod.is_signed_off("A <a@x>", "A <a@x>", [])


@pytest.mark.parametrize(
    "email",
    [
        "49699333+dependabot[bot]@users.noreply.github.com",
        "github-actions[bot]@users.noreply.github.com",
        "12345+bot@noreply@github.com",
    ],
)
def test_is_bot_email_true(email):
    assert _mod.is_bot_email(email)


@pytest.mark.parametrize(
    "email",
    [
        "real@example.com",
        # A human using GitHub's noreply address is not a bot and must sign off.
        "octocat@users.noreply.github.com",
    ],
)
def test_is_bot_email_false_for_human(email):
    assert not _mod.is_bot_email(email)


# --- end-to-end against a real git repo -----------------------------------


def _run_git(repo: Path, *args: str, env: dict | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _commit(repo, message, *, author, committer=None, signoff=False) -> None:
    committer = committer or author
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": author[0],
        "GIT_AUTHOR_EMAIL": author[1],
        "GIT_COMMITTER_NAME": committer[0],
        "GIT_COMMITTER_EMAIL": committer[1],
    }
    args = ["commit", "--allow-empty", "-m", message]
    if signoff:
        args.append("-s")  # signs off as the committer, per git docs
    _run_git(repo, *args, env=env)


@pytest.fixture
def repo(tmp_path):
    """A git repo with one base commit; yields (path, base_sha).

    ``base_sha`` is captured before any test commit so the checked range
    (``base_sha..HEAD``) is exactly the commits the test adds.
    """
    _run_git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, "chore: base", author=("Base", "base@example.com"), signoff=True)
    base_sha = _run_git(tmp_path, "rev-parse", "HEAD").strip()
    return tmp_path, base_sha


def _check(repo, monkeypatch) -> int:
    path, base_sha = repo
    monkeypatch.chdir(path)
    return _mod.main(["check_dco.py", base_sha, "HEAD"])


def test_signed_off_commit_passes(repo, monkeypatch):
    _commit(repo[0], "feat: thing", author=("Ann", "ann@example.com"), signoff=True)
    assert _check(repo, monkeypatch) == 0


def test_author_committer_divergence_passes_on_committer_signoff(repo, monkeypatch):
    # The exact workflow the review flagged: author != committer, `-s` signs
    # off as the committer. This must pass.
    _commit(
        repo[0],
        "feat: applied patch",
        author=("Ann Author", "ann@example.com"),
        committer=("Cody Committer", "cody@example.com"),
        signoff=True,
    )
    assert _check(repo, monkeypatch) == 0


def test_unsigned_commit_fails(repo, monkeypatch):
    _commit(repo[0], "feat: unsigned", author=("Ann", "ann@example.com"), signoff=False)
    assert _check(repo, monkeypatch) == 1


def test_prose_decoy_is_not_a_trailer(repo, monkeypatch):
    # "Signed-off-by:" in the body, but followed by another paragraph, so it is
    # not in the trailer block. It must NOT satisfy the check.
    message = (
        "feat: sneaky\n\n"
        "Signed-off-by: Ann <ann@example.com>\n\n"
        "This trailing paragraph makes the line above prose, not a trailer."
    )
    _commit(repo[0], message, author=("Ann", "ann@example.com"), signoff=False)
    assert _check(repo, monkeypatch) == 1


def test_bot_commit_exempt_without_signoff(repo, monkeypatch):
    _commit(
        repo[0],
        "chore(deps): bump thing",
        author=("dependabot[bot]", "49699333+dependabot[bot]@users.noreply.github.com"),
        signoff=False,
    )
    assert _check(repo, monkeypatch) == 0


def test_empty_range_passes(repo, monkeypatch):
    assert _check(repo, monkeypatch) == 0
