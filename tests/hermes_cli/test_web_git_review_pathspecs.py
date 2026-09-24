"""Review-pane git ops: untracked-directory expansion and literal pathspecs.

`web_git` backs the `/api/git/review/*` routes — the same Review pane the
desktop app drives over IPC, served for remote/web sessions. It had two bugs
that the Electron module also had:

1. `git status` collapses an untracked directory into one `dir/` row, but
   `git diff --no-index -- <devnull> dir/` cannot diff that (it pairs the
   operands as trees and fails looking for `dir/nul`), so clicking the row
   rendered "No diff to show" under a populated header.
2. Paths were passed as pathspecs, which git treats as globs. A real filename
   containing wildcards (`weird[1].txt`) also matched its neighbours
   (`weird1.txt`): reads showed the wrong file, and `add` / `reset` /
   `checkout` / `clean` acted on files the user never selected.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli.web_git import (
    file_diff_vs_head,
    review_diff,
    review_list,
    review_revert,
    review_stage,
    review_unstage,
)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A repo with one committed file."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "hermes-test@example.com")
    _git(tmp_path, "config", "user.name", "Hermes Test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "tracked.txt").write_text("tracked\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "base")
    return tmp_path


@pytest.fixture
def glob_repo(repo):
    """The nastiest shape for the pathspec bug: the decoy is TRACKED and
    MODIFIED, so it has both a real diff to leak and real edits to destroy."""
    (repo / "weird1.txt").write_text("neighbour original\n")
    _git(repo, "add", "weird1.txt")
    _git(repo, "commit", "-qm", "add neighbour")
    (repo / "weird1.txt").write_text("neighbour MODIFIED\n")
    (repo / "weird[1].txt").write_text("clicked file\n")
    return repo


def _uncommitted(repo, path, staged=False):
    return review_diff(str(repo), path, "uncommitted", None, staged)


# ── untracked directory rows ────────────────────────────────────────────────


def test_untracked_directory_is_listed_as_one_row(repo):
    (repo / "newdir").mkdir()
    (repo / "newdir" / "one.txt").write_text("first\n")

    paths = [f["path"] for f in review_list(str(repo), "uncommitted", None)["files"]]

    # The compact listing is deliberate; the diff path has to cope with it.
    assert "newdir/" in paths


def test_untracked_directory_expands_to_its_files(repo):
    (repo / "newdir" / "sub").mkdir(parents=True)
    (repo / "newdir" / "one.txt").write_text("first\n")
    (repo / "newdir" / "sub" / "two.txt").write_text("second\n")

    diff = _uncommitted(repo, "newdir/")

    assert "+first" in diff
    assert "+second" in diff
    # Every file is named so the renderer can label a multi-file payload.
    assert "newdir/one.txt" in diff
    assert "newdir/sub/two.txt" in diff


def test_untracked_directory_with_spaces_expands(repo):
    (repo / "Fallout Vault" / "20 Projects").mkdir(parents=True)
    (repo / "Fallout Vault" / "20 Projects" / "note.md").write_text("vault note\n")

    assert "+vault note" in _uncommitted(repo, "Fallout Vault/")


def test_untracked_directory_expansion_honors_gitignore(repo):
    (repo / ".gitignore").write_text("*.log\n")
    (repo / "logs").mkdir()
    (repo / "logs" / "keep.txt").write_text("kept\n")
    (repo / "logs" / "noisy.log").write_text("ignored\n")

    diff = _uncommitted(repo, "logs/")

    assert "+kept" in diff
    assert "+ignored" not in diff


def test_untracked_directory_expansion_is_capped_and_says_so(repo):
    (repo / "generated").mkdir()
    for i in range(60):
        (repo / "generated" / f"f-{i:03d}.txt").write_text(f"line {i}\n")

    diff = _uncommitted(repo, "generated/")

    assert "+line 0" in diff
    assert "10 more file(s) omitted" in diff


def test_nested_git_repo_is_opaque_not_a_crash(repo):
    nested = repo / "nested_repo"
    nested.mkdir()
    _git(nested, "init", "-q")
    (nested / "inner.txt").write_text("inner\n")

    # Opaque to the outer repo — the pane shows its folder empty-state for this.
    assert _uncommitted(repo, "nested_repo/") == ""


def test_untracked_single_file_still_synthesizes_an_all_add_diff(repo):
    (repo / "fresh.txt").write_text("alpha\nbeta\n")

    diff = _uncommitted(repo, "fresh.txt")

    assert "+alpha" in diff
    assert "+beta" in diff


def test_file_diff_vs_head_expands_an_untracked_directory(repo):
    (repo / "preview-dir").mkdir()
    (repo / "preview-dir" / "a.txt").write_text("preview line\n")

    assert "+preview line" in file_diff_vs_head(str(repo), "preview-dir/")


def test_file_diff_vs_head_still_empty_for_a_clean_tracked_file(repo):
    assert file_diff_vs_head(str(repo), "tracked.txt") == ""


# ── literal pathspecs: reads ────────────────────────────────────────────────


def test_review_diff_does_not_leak_a_glob_neighbour(glob_repo):
    # The worktree probe runs first, so without --literal-pathspecs this
    # returns the TRACKED neighbour's diff and the pane renders a file the
    # user never clicked.
    diff = _uncommitted(glob_repo, "weird[1].txt")

    assert "+clicked file" in diff
    assert "neighbour MODIFIED" not in diff


def test_file_diff_vs_head_does_not_leak_a_glob_neighbour(glob_repo):
    diff = file_diff_vs_head(str(glob_repo), "weird[1].txt")

    assert "+clicked file" in diff
    assert "neighbour MODIFIED" not in diff


# ── literal pathspecs: mutations (wrong-tree writes) ────────────────────────


def _staged_names(repo):
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def test_review_stage_stages_only_the_selected_file(glob_repo):
    review_stage(str(glob_repo), "weird[1].txt")

    staged = _staged_names(glob_repo)

    assert "weird[1].txt" in staged
    assert "weird1.txt" not in staged


def test_review_unstage_unstages_only_the_selected_file(glob_repo):
    _git(glob_repo, "add", "-A")
    review_unstage(str(glob_repo), "weird[1].txt")

    staged = _staged_names(glob_repo)

    # The neighbour must stay staged; only the clicked file comes back out.
    assert "weird1.txt" in staged
    assert "weird[1].txt" not in staged


def test_review_revert_does_not_discard_a_glob_neighbours_edits(glob_repo):
    # The destructive one: `git checkout HEAD -- 'weird[1].txt'` also restored
    # weird1.txt, silently throwing away the user's uncommitted work.
    review_revert(str(glob_repo), "weird[1].txt")

    assert (glob_repo / "weird1.txt").read_text() == "neighbour MODIFIED\n"
    assert not (glob_repo / "weird[1].txt").exists()


# ── staged rows show the whole story ────────────────────────────────────────


def test_partially_staged_file_shows_staged_and_unstaged(repo):
    (repo / "tracked.txt").write_text("tracked\nstaged line\n")
    _git(repo, "add", "tracked.txt")
    (repo / "tracked.txt").write_text("tracked\nstaged line\nunstaged line\n")

    # The row's +/- counts sum both sides, so the diff has to as well.
    diff = _uncommitted(repo, "tracked.txt", staged=True)

    assert "+staged line" in diff
    assert "+unstaged line" in diff


def test_staged_file_in_a_repo_with_no_commits_falls_back_to_the_index(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "hermes-test@example.com")
    _git(tmp_path, "config", "user.name", "Hermes Test")
    (tmp_path / "first.txt").write_text("first commit pending\n")
    _git(tmp_path, "add", "first.txt")

    # No HEAD to diff against — the --cached fallback keeps the panel populated.
    assert "+first commit pending" in review_diff(
        str(tmp_path), "first.txt", "uncommitted", None, True
    )
