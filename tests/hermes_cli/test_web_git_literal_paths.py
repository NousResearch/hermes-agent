"""Review rows must preserve literal Git filenames, including rename destinations."""
import subprocess

import pytest

from hermes_cli import web_git


@pytest.mark.platforms("linux", "macos")
def test_review_preserves_literal_paths_and_counts(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True, encoding="utf-8")

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    names = ["old => actual.txt", " café.txt ", "line\nbreak.txt"]
    for name in names:
        (tmp_path / name).write_text("before\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")
    for name in names:
        (tmp_path / name).write_text("after\nextra\n", encoding="utf-8")
    rows = web_git.review_list(str(tmp_path), "uncommitted", None)["files"]
    assert {row["path"]: (row["added"], row["removed"]) for row in rows} == {
        name: (2, 1) for name in names
    }


@pytest.mark.platforms("linux", "macos")
def test_review_counts_rename_with_literal_arrow(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (tmp_path / "old.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "baseline")
    destination = "new => real.txt"
    git("mv", "old.txt", destination)
    (tmp_path / destination).write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    git("add", ".")
    rows = web_git.review_list(str(tmp_path), "uncommitted", None)["files"]
    assert [(r["path"], r["added"], r["removed"]) for r in rows] == [(destination, 1, 0)]
