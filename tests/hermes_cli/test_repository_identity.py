from __future__ import annotations

import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.repository_identity import (
    RepositoryIdentityError,
    select_repository_candidate,
    validate_repository_identity,
)


MARKERS = ["hermes_cli/kanban_db.py", "tests/hermes_cli/test_kanban_db.py"]


def _git_repo(root: Path, *, name: str = "hermes-agent") -> Path:
    (root / "hermes_cli").mkdir(parents=True)
    (root / "tests/hermes_cli").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f"[project]\nname = {name!r}\nversion = '0'\n", encoding="utf-8"
    )
    for marker in MARKERS:
        (root / marker).write_text("# fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_correct_checkout_returns_canonical_identity(tmp_path):
    root = _git_repo(tmp_path / "hermes")
    evidence = validate_repository_identity(
        root,
        expected_manifest_name="hermes-agent",
        required_markers=MARKERS,
    )
    assert Path(evidence["git_root"]) == root.resolve()
    assert evidence["missing"] == []


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda p: _git_repo(p, name="old-plugin"), "pyproject.toml"),
        (lambda p: (p.mkdir() or p), "canonical Git root"),
    ],
)
def test_wrong_or_non_git_candidate_fails_closed(tmp_path, factory, expected):
    candidate = factory(tmp_path / "candidate")
    with pytest.raises(RepositoryIdentityError) as exc:
        validate_repository_identity(
            candidate,
            expected_manifest_name="hermes-agent",
            required_markers=MARKERS,
        )
    message = str(exc.value)
    assert "BLOCKED/needs-input" in message
    assert str(candidate.resolve()) in message
    assert expected in message


def test_arbitrary_python_project_is_not_accepted(tmp_path):
    root = tmp_path / "python"
    root.mkdir()
    (root / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
    with pytest.raises(RepositoryIdentityError, match="canonical Git root"):
        validate_repository_identity(
            root,
            expected_manifest_name="hermes-agent",
            required_markers=MARKERS,
        )


def test_symlink_and_linked_worktree_are_checked_by_identity(tmp_path):
    root = _git_repo(tmp_path / "hermes")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    evidence = validate_repository_identity(
        alias,
        expected_manifest_name="hermes-agent",
        required_markers=MARKERS,
    )
    assert Path(evidence["candidate"]) == root.resolve()

    worktree = _committed_linked_worktree(root, "fixture", tmp_path / "worktree")
    assert validate_repository_identity(
        worktree,
        expected_manifest_name="hermes-agent",
        required_markers=MARKERS,
    )["git_root"] == str(worktree.resolve())


def test_missing_marker_is_audited(tmp_path):
    root = _git_repo(tmp_path / "hermes")
    (root / MARKERS[1]).unlink()
    with pytest.raises(RepositoryIdentityError, match="required source marker") as exc:
        validate_repository_identity(
            root,
            expected_manifest_name="hermes-agent",
            required_markers=MARKERS,
        )
    assert MARKERS[1] in str(exc.value)


@pytest.mark.parametrize("marker", ["../outside", "/tmp/outside"])
def test_marker_path_escape_fails_closed(tmp_path, marker):
    root = _git_repo(tmp_path / "hermes")
    with pytest.raises(RepositoryIdentityError, match="relative paths"):
        validate_repository_identity(
            root,
            expected_manifest_name="hermes-agent",
            required_markers=[marker],
        )


def test_candidate_selection_rejects_ambiguity_and_lists_evidence(tmp_path):
    first = _git_repo(tmp_path / "one")
    second = _git_repo(tmp_path / "two")
    with pytest.raises(RepositoryIdentityError, match="ambiguous validated candidates") as exc:
        select_repository_candidate(
            [first, second],
            expected_manifest_name="hermes-agent",
            required_markers=MARKERS,
        )
    assert str(first.resolve()) in str(exc.value)
    assert str(second.resolve()) in str(exc.value)


def test_kanban_worktree_resolution_honors_explicit_board_identity(tmp_path, monkeypatch):
    unrelated = _git_repo(tmp_path / "plugin", name="old-plugin")
    monkeypatch.setattr(
        kb,
        "read_board_metadata",
        lambda _board: {
            "repository_identity": {
                "expected_manifest_name": "hermes-agent",
                "required_markers": MARKERS,
            }
        },
    )
    with pytest.raises(ValueError, match="BLOCKED/needs-input"):
        kb._validate_workspace_repository(unrelated, board="default")


def _committed_linked_worktree(root: Path, branch: str, target: Path) -> Path:
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "init"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "worktree", "add", "-q", "-b", branch, str(target)],
        check=True,
    )
    return target


@pytest.mark.parametrize(
    "requested_branch",
    ["occupied", "other"],
    ids=["matching-branch-reuse", "different-branch-fallback"],
)
def test_worktree_resolution_validates_existing_linked_checkout_before_return(
    tmp_path, monkeypatch, requested_branch
):
    root = _git_repo(tmp_path / "wrong", name="old-plugin")
    target = _committed_linked_worktree(root, requested_branch, tmp_path / "linked")
    monkeypatch.setattr(
        kb,
        "read_board_metadata",
        lambda _board: {
            "repository_identity": {
                "expected_manifest_name": "hermes-agent",
                "required_markers": MARKERS,
            }
        },
    )
    task = SimpleNamespace(
        id="t_identity",
        branch_name="occupied" if requested_branch == "occupied" else "wanted",
        workspace_path=str(target),
    )
    with pytest.raises(ValueError, match="BLOCKED/needs-input"):
        kb._resolve_worktree_workspace(task, board="default")
