"""Tests for hermes_cli.kanban_completion_gates — v6.7 Tranche 1.

Closes hermes-jarvis#62 (workspace-diff verification), #28 (repo hygiene
gate), and #64 (per-role runtime floor). See hermes-jarvis#61 for the
bootstrap-paradox case study where a v6.7 swarm build chain rubber-stamped
9 tasks done in ~10 minutes with zero real deliverables.

Each gate is a pure function and gets a focused test that pins the exact
failure modes the 2026-06-09 chain demonstrated.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli.kanban_completion_gates import (
    RuntimeFloorViolation,
    StrayArtifactViolation,
    WorkspaceDiffViolation,
    verify_no_stray_artifacts,
    verify_runtime_floor,
    verify_workspace_diff,
)


# =====================================================================
# verify_runtime_floor — #64
# =====================================================================


class TestRuntimeFloor:
    def test_tony_20s_review_is_below_floor(self) -> None:
        """The exact case from 2026-06-09: Tony approved Wave A in 20s."""
        v = verify_runtime_floor("tony", started_at=1000, completed_at=1020)
        assert isinstance(v, RuntimeFloorViolation)
        assert v.actual_seconds == 20
        assert v.floor_seconds == 90
        assert "tony" in v.message().lower()
        assert "below" in v.message().lower()

    def test_friday_59s_implementation_is_below_floor(self) -> None:
        """Friday claimed 7 dispatcher gates implemented in 59s."""
        v = verify_runtime_floor("friday", started_at=1000, completed_at=1059)
        assert isinstance(v, RuntimeFloorViolation)
        assert v.floor_seconds == 300

    def test_tony_91s_review_passes(self) -> None:
        """One second above the floor is a pass — the floor is the floor."""
        assert verify_runtime_floor("tony", 1000, 1091) is None

    def test_jarvis_orchestration_has_no_floor(self) -> None:
        """Orchestration roles routinely complete in seconds and that's fine."""
        assert verify_runtime_floor("jarvis", 1000, 1001) is None

    def test_unknown_assignee_skips(self) -> None:
        """Don't invent floors for roles we haven't categorized."""
        assert verify_runtime_floor("rando-profile", 1000, 1001) is None

    def test_missing_assignee_skips(self) -> None:
        assert verify_runtime_floor(None, 1000, 1001) is None

    def test_missing_started_at_skips(self) -> None:
        """If the dispatcher never recorded started_at the gate can't fire."""
        assert verify_runtime_floor("tony", None, 1100) is None

    def test_allow_below_floor_opt_out(self) -> None:
        """Workers can justify fast completions via metadata."""
        assert (
            verify_runtime_floor("tony", 1000, 1020, allow_below_floor=True)
            is None
        )

    def test_completed_before_started_is_zero(self) -> None:
        """Clock skew / wrong order doesn't crash — actual=0, still below floor."""
        v = verify_runtime_floor("tony", 1100, 1000)
        assert v is not None
        assert v.actual_seconds == 0

    def test_case_insensitive_role_match(self) -> None:
        """Profile names sometimes capitalize differently — match insensitively."""
        v = verify_runtime_floor("Tony", 1000, 1020)
        assert v is not None


# =====================================================================
# verify_workspace_diff — #62
# =====================================================================


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    """A real git repo with one committed file on main, no other changes."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=tmp_path, check=True,
    )
    (tmp_path / "src.py").write_text("print('hi')\n")
    subprocess.run(["git", "add", "src.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True,
    )
    return tmp_path


class TestWorkspaceDiff:
    def test_friday_empty_diff_with_implementation_claim_rejects(
        self, git_workspace: Path,
    ) -> None:
        """The exact case from 2026-06-09: Friday's branch had no new commits
        but his summary claimed "Wave A dispatcher discipline gates
        implemented; tests cover #28-#34".
        """
        v = verify_workspace_diff(
            assignee="friday",
            workspace_kind="dir",
            workspace_path=str(git_workspace),
            summary="Wave A dispatcher discipline gates implemented; tests cover #28-#34",
        )
        assert isinstance(v, WorkspaceDiffViolation)
        assert "friday" in v.message().lower()
        assert "implementation" in v.message().lower() or "implement" in v.message().lower()

    def test_real_diff_with_implementation_claim_passes(
        self, git_workspace: Path,
    ) -> None:
        """A worker who actually did work and committed it gets through."""
        # Make a second commit so HEAD differs from main's first commit
        # but we still test against HEAD's diff against base. Setup: detach,
        # add a new commit, then diff stat will be non-empty against `main`
        # if HEAD has more.
        new_file = git_workspace / "feature.py"
        new_file.write_text("def real(): pass\n")
        subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=git_workspace, check=True)
        subprocess.run(["git", "add", "feature.py"], cwd=git_workspace, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "feat"], cwd=git_workspace, check=True,
        )
        v = verify_workspace_diff(
            assignee="friday",
            workspace_kind="dir",
            workspace_path=str(git_workspace),
            summary="Implemented feature module per spec",
        )
        assert v is None

    def test_review_role_skipped(self, git_workspace: Path) -> None:
        """Tony's deliverable is a verdict, not code — skip the diff gate."""
        assert (
            verify_workspace_diff(
                assignee="tony",
                workspace_kind="dir",
                workspace_path=str(git_workspace),
                summary="approve - implementation matches spec",
            )
            is None
        )

    def test_orchestration_role_skipped(self, git_workspace: Path) -> None:
        """JARVIS umbrella spawn doesn't ship code, even when body says
        'implemented chain'."""
        assert (
            verify_workspace_diff(
                assignee="jarvis",
                workspace_kind="dir",
                workspace_path=str(git_workspace),
                summary="Spawned and implemented the build chain",
            )
            is None
        )

    def test_scratch_workspace_skipped(self) -> None:
        """scratch workspaces have no diff target."""
        assert (
            verify_workspace_diff(
                assignee="friday",
                workspace_kind="scratch",
                workspace_path=None,
                summary="implemented thing",
            )
            is None
        )

    def test_no_implementation_claim_skipped(self, git_workspace: Path) -> None:
        """Summary that doesn't claim code work doesn't trip the gate."""
        assert (
            verify_workspace_diff(
                assignee="friday",
                workspace_kind="dir",
                workspace_path=str(git_workspace),
                summary="Investigated the issue; recommendations in comment.",
            )
            is None
        )

    def test_x_no_code_opt_out(self, git_workspace: Path) -> None:
        assert (
            verify_workspace_diff(
                assignee="friday",
                workspace_kind="dir",
                workspace_path=str(git_workspace),
                summary="Implemented the docs reshuffle",
                allow_no_code=True,
            )
            is None
        )

    def test_nonexistent_workspace_path_skipped(self) -> None:
        """We don't crash when workspace_path is wrong; we just skip."""
        assert (
            verify_workspace_diff(
                assignee="friday",
                workspace_kind="dir",
                workspace_path="/tmp/this-does-not-exist-v67",
                summary="implemented thing",
            )
            is None
        )


# =====================================================================
# verify_no_stray_artifacts — #28
# =====================================================================


def _git_init(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=tmp_path, check=True,
    )


class TestStrayArtifacts:
    def test_pr1_all_prior_block_evidence_files(self, tmp_path: Path) -> None:
        """The literal failure mode from agent-dashboard PR #1: a file
        named 'all prior block evidence files' (no extension) committed
        because the evidence-path gate took a descriptive phrase
        literally.
        """
        _git_init(tmp_path)
        stray = tmp_path / "all prior block evidence files"
        stray.write_text("nothing\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(
            ["git", "add", "all prior block evidence files", "src.py"],
            cwd=tmp_path, check=True,
        )
        v = verify_no_stray_artifacts("dir", str(tmp_path))
        assert isinstance(v, StrayArtifactViolation)
        assert "all prior block evidence files" in v.stray_paths

    def test_commit_hash_txt_stray(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "commit-hash.txt").write_text("abc123\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        v = verify_no_stray_artifacts("dir", str(tmp_path))
        assert v is not None
        assert "commit-hash.txt" in v.stray_paths

    def test_triage_dir_stray(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        td = tmp_path / "triage"
        td.mkdir()
        (td / "v6.4-report.md").write_text("notes\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        v = verify_no_stray_artifacts("dir", str(tmp_path))
        assert v is not None
        assert any("triage/" in p for p in v.stray_paths)

    def test_evidence_subdir_stray(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        ed = tmp_path / "changes" / "fix-14" / "evidence"
        ed.mkdir(parents=True)
        (ed / "out.json").write_text("{}\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        v = verify_no_stray_artifacts("dir", str(tmp_path))
        assert v is not None
        assert any("evidence" in p for p in v.stray_paths)

    def test_clean_repo_passes(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "src.py").write_text("print('hi')\n")
        (tmp_path / "README.md").write_text("# hi\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        assert verify_no_stray_artifacts("dir", str(tmp_path)) is None

    def test_shebang_file_without_extension_is_ok(self, tmp_path: Path) -> None:
        """Real scripts have shebangs — those aren't stray."""
        _git_init(tmp_path)
        (tmp_path / "bin").mkdir()
        script = tmp_path / "bin" / "deploy"
        script.write_text("#!/bin/bash\necho hi\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        assert verify_no_stray_artifacts("dir", str(tmp_path)) is None

    def test_scratch_workspace_skipped(self) -> None:
        assert verify_no_stray_artifacts("scratch", None) is None

    def test_x_stray_ok_opt_out(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "all prior block evidence files").write_text("x\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        assert (
            verify_no_stray_artifacts("dir", str(tmp_path), allow_stray=True)
            is None
        )

    def test_nonexistent_workspace_path_skipped(self) -> None:
        assert (
            verify_no_stray_artifacts("dir", "/tmp/does-not-exist-v67") is None
        )

    def test_tmp_prefixed_files_stray(self, tmp_path: Path) -> None:
        _git_init(tmp_path)
        (tmp_path / "tmp-scratch").write_text("x\n")
        (tmp_path / "src.py").write_text("print('hi')\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        v = verify_no_stray_artifacts("dir", str(tmp_path))
        assert v is not None
        assert "tmp-scratch" in v.stray_paths
