from __future__ import annotations

import dataclasses
import os
import stat
import subprocess
from pathlib import Path

import pytest

from agent.project_local import (
    canonical_project_identity,
    clear_project_local_cache,
    load_project_trust,
    project_has_recognized_files,
    project_trust_path,
    record_project_skill_consent,
    resolve_project_local_state,
)


@pytest.fixture(autouse=True)
def _clear_project_cache():
    clear_project_local_cache()
    yield
    clear_project_local_cache()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "README.md").write_text("repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Test User",
        "commit",
        "-m",
        "init",
    )
    return repo


def _write_project_skill(repo: Path, name: str, body: str = "Do useful work.") -> Path:
    skill = repo / ".hermes" / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(
        f"---\nname: {name}\ndescription: test\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill


def test_canonical_identity_collapses_subdirs_and_git_worktrees(tmp_path: Path):
    repo = _git_repo(tmp_path)
    subdir = repo / "src" / "pkg"
    subdir.mkdir(parents=True)
    worktree = tmp_path / "linked-worktree"
    _git(repo, "worktree", "add", str(worktree), "-b", "test-worktree")

    root_id = canonical_project_identity(repo)
    subdir_id = canonical_project_identity(subdir)
    worktree_id = canonical_project_identity(worktree)

    assert root_id is not None
    assert subdir_id is not None
    assert worktree_id is not None
    assert root_id.canonical_id == subdir_id.canonical_id
    assert root_id.canonical_id == worktree_id.canonical_id
    assert Path(root_id.git_common_dir).resolve() == Path(worktree_id.git_common_dir).resolve()


def test_recognized_files_ignore_bare_hermes_directory(tmp_path: Path):
    repo = _git_repo(tmp_path)
    (repo / ".hermes" / "plans").mkdir(parents=True)
    identity = canonical_project_identity(repo)
    assert identity is not None

    recognized, rejected = project_has_recognized_files(identity)

    assert recognized is False
    assert rejected == ""


def test_recognized_files_detect_project_skill(tmp_path: Path):
    repo = _git_repo(tmp_path)
    _write_project_skill(repo, "local-skill")
    identity = canonical_project_identity(repo)
    assert identity is not None

    recognized, rejected = project_has_recognized_files(identity)

    assert recognized is True
    assert rejected == ""


def test_recognized_files_detect_mcp_config(tmp_path: Path):
    repo = _git_repo(tmp_path)
    config = repo / ".hermes" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("mcp_servers:\n  local: {}\n", encoding="utf-8")
    identity = canonical_project_identity(repo)
    assert identity is not None

    recognized, rejected = project_has_recognized_files(identity)
    state = resolve_project_local_state(repo)

    assert recognized is True
    assert rejected == ""
    assert state.recognized is True
    assert state.mcp_manifest
    assert state.mcp_manifest_hash
    assert state.cache_signature() == {
        "project.canonical_id": state.canonical_id,
        "project.mcp_manifest_hash": state.mcp_manifest_hash,
    }

    before = state.mcp_manifest_hash
    config.write_text("mcp_servers:\n  local:\n    command: changed\n", encoding="utf-8")
    assert resolve_project_local_state(repo).mcp_manifest_hash != before


def test_recognized_file_cache_tracks_new_skill_under_existing_dir(tmp_path: Path):
    repo = _git_repo(tmp_path)
    skill_dir = repo / ".hermes" / "skills" / "late"
    skill_dir.mkdir(parents=True)
    identity = canonical_project_identity(repo)
    assert identity is not None

    recognized, rejected = project_has_recognized_files(identity)
    assert recognized is False
    assert rejected == ""

    (skill_dir / "SKILL.md").write_text(
        "---\nname: late\ndescription: test\n---\n\nNow visible.\n",
        encoding="utf-8",
    )

    recognized, rejected = project_has_recognized_files(identity)
    assert recognized is True
    assert rejected == ""


def test_symlinked_hermes_directory_is_rejected(tmp_path: Path):
    repo = _git_repo(tmp_path)
    target = tmp_path / "elsewhere"
    target.mkdir()
    try:
        (repo / ".hermes").symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    identity = canonical_project_identity(repo)
    assert identity is not None

    state = resolve_project_local_state(repo)

    assert state.recognized is False
    assert "symlinked .hermes" in state.rejected_reason
    assert state.cache_signature() == {
        "project.canonical_id": state.canonical_id,
        "project.rejected_reason": state.rejected_reason,
    }


def test_project_local_state_is_immutable_and_hashes_skills(tmp_path: Path):
    repo = _git_repo(tmp_path)
    _write_project_skill(repo, "alpha")

    state = resolve_project_local_state(repo)

    assert state.recognized is True
    assert state.skill_roots
    assert state.skills_manifest_hash
    assert state.cache_signature() == {
        "project.canonical_id": state.canonical_id,
        "project.skills_manifest_hash": state.skills_manifest_hash,
    }
    assert state.skills_manifest[0].name == "alpha"
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.canonical_id = "changed"  # type: ignore[misc]


def test_unrecognized_repo_keeps_identity_but_has_no_cache_signature(tmp_path: Path):
    repo = _git_repo(tmp_path)
    (repo / ".hermes" / "plans").mkdir(parents=True)

    state = resolve_project_local_state(repo)

    assert state.canonical_id
    assert state.recognized is False
    assert state.cache_signature() == {}


def test_default_resolution_uses_runtime_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo = _git_repo(tmp_path)
    _write_project_skill(repo, "alpha")
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    monkeypatch.setenv("TERMINAL_CWD", str(repo))

    state = resolve_project_local_state()

    assert state.recognized is True
    assert state.worktree_root == str(repo.resolve())


def test_trust_sidecar_round_trips_atomically_and_freezes_matching_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    repo = _git_repo(tmp_path)
    _write_project_skill(repo, "alpha")
    state = resolve_project_local_state(repo)
    assert state.skills_manifest_hash

    record_project_skill_consent(
        state.canonical_id,
        skills_manifest_hash=state.skills_manifest_hash,
        decision="allow",
    )

    trust = load_project_trust()
    assert trust["projects"][state.canonical_id]["skills"]["decision"] == "allow"
    mode = stat.S_IMODE(project_trust_path().stat().st_mode)
    assert mode == 0o600
    assert resolve_project_local_state(repo).consent_decision == "allow"

    _write_project_skill(repo, "alpha", body="Changed bytes.")
    assert resolve_project_local_state(repo).consent_decision == ""
