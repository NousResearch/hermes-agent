"""Review-pane per-change actions act on exactly one hunk, and refuse a hunk the file no longer holds."""

import subprocess

import pytest

from hermes_cli import web_git


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          check=True, timeout=10).stdout


def read(path):
    return path.read_text(encoding="utf-8-sig")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    git(root, "init", "-q")
    target = root / "sub" / "f.txt"
    target.write_text("".join(f"{n}\n" for n in range(1, 31)), encoding="utf-8")
    git(root, "add", ".")
    git(root, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "init")
    target.write_text(read(target).replace("3\n", "three\n", 1).replace("25\n", "twentyfive\n"), encoding="utf-8")
    return root


def _hunks(cwd):
    diff = web_git.review_diff(cwd, "f.txt", "uncommitted", None, False)
    return ["@@" + h.rstrip("\n") for h in diff.split("\n@@")[1:]]


def test_one_hunk_stages_and_reverts_alone_from_a_subdirectory(repo):
    sub = str(repo / "sub")
    first, second = _hunks(sub)

    web_git.review_hunk(sub, "f.txt", second, "stage")
    assert "+twentyfive" in git(repo, "diff", "--cached") and "three" not in git(repo, "diff", "--cached")

    web_git.review_hunk(sub, "f.txt", first, "revert")
    text = read(repo / "sub" / "f.txt")
    assert "three" not in text and "twentyfive" in text
    assert not list((repo / ".git").glob("hermes-review-hunk-*"))


def test_a_hunk_the_file_no_longer_holds_is_refused_untouched(repo):
    sub = str(repo / "sub")
    first, _ = _hunks(sub)
    target = repo / "sub" / "f.txt"
    target.write_text(read(target).replace("three", "THREE"), encoding="utf-8")
    before = read(target)

    with pytest.raises(RuntimeError, match="no longer matches"):
        web_git.review_hunk(sub, "f.txt", first, "revert")
    assert read(target) == before
