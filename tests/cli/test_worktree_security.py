"""Security-focused integration tests for CLI worktree setup.

The snapshot/quarantine cases assert the recovery-first local-safety candidate.
They are not evidence that the separate upstream aggressive-reclamation policy
has been selected for integration.
"""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing real cli._setup_worktree behavior."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
    return repo


def _force_remove_worktree(info: dict | None) -> None:
    if not info:
        return
    subprocess.run(
        ["git", "worktree", "remove", info["path"], "--force"],
        cwd=info["repo_root"],
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "branch", "-D", info["branch"]],
        cwd=info["repo_root"],
        capture_output=True,
        check=False,
    )


class TestWorktreeSafety:
    def test_rejects_parent_directory_file_traversal(self, git_repo):
        import cli as cli_mod

        outside_file = git_repo.parent / "sensitive.txt"
        outside_file.write_text("SENSITIVE DATA")
        (git_repo / ".worktreeinclude").write_text("../sensitive.txt\n")

        info = None
        try:
            info = cli_mod._setup_worktree(str(git_repo))
            assert info is not None

            wt_path = Path(info["path"])
            assert not (wt_path.parent / "sensitive.txt").exists()
            assert not (wt_path / "../sensitive.txt").resolve().exists()
        finally:
            _force_remove_worktree(info)
    def test_dirty_worktree_is_preserved_with_untracked_payload(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        try:
            marker = Path(info["path"]) / "recoverable-untracked.txt"
            marker.write_text("only local copy\n")

            cli_mod._cleanup_worktree(info)

            assert Path(info["path"]).exists()
            assert marker.read_text() == "only local copy\n"
        finally:
            _force_remove_worktree(info)

    def test_ignored_payload_is_preserved(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        try:
            worktree = Path(info["path"])
            (worktree / ".gitignore").write_text("recoverable-output/\n")
            payload = worktree / "recoverable-output" / "report.txt"
            payload.parent.mkdir()
            payload.write_text("ignored but recoverable\n")

            cli_mod._cleanup_worktree(info)

            assert worktree.exists()
            assert payload.read_text() == "ignored but recoverable\n"
        finally:
            _force_remove_worktree(info)

    def test_clean_worktree_is_quarantined_with_branch_and_archive_ref(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        archive_path = git_repo / ".worktrees" / ".archive" / Path(info["path"]).name
        try:
            oid = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=info["path"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            cli_mod._cleanup_worktree(info)

            assert not Path(info["path"]).exists()
            assert archive_path.exists()
            assert subprocess.run(
                ["git", "show-ref", "--verify", f"refs/heads/{info['branch']}"],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
            assert subprocess.run(
                ["git", "show-ref", "--verify", f"refs/hermes/archive/{info['branch']}/{oid}"],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
        finally:
            subprocess.run(
                ["git", "worktree", "remove", str(archive_path), "--force"],
                cwd=git_repo,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "branch", "-D", info["branch"]],
                cwd=git_repo,
                capture_output=True,
                check=False,
            )

    def test_preexisting_archive_ref_allows_quarantine(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        source_path = Path(info["path"])
        archive_path = git_repo / ".worktrees" / ".archive" / source_path.name
        try:
            oid = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source_path,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            archive_ref = f"refs/hermes/archive/{info['branch']}/{oid}"
            subprocess.run(["git", "update-ref", archive_ref, oid], cwd=git_repo, check=True)

            cli_mod._cleanup_worktree(info)

            assert archive_path.exists()
            assert subprocess.run(
                ["git", "show-ref", "--verify", archive_ref],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
        finally:
            subprocess.run(
                ["git", "worktree", "remove", str(archive_path), "--force"],
                cwd=git_repo,
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "branch", "-D", info["branch"]],
                cwd=git_repo,
                capture_output=True,
                check=False,
            )

    def test_quarantine_move_failure_keeps_worktree_and_branch(self, git_repo, monkeypatch):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        real_run = subprocess.run
        oid = real_run(
            ["git", "rev-parse", "HEAD"],
            cwd=info["path"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        def fail_move(args, **kwargs):
            if args[1:3] == ["worktree", "move"]:
                return subprocess.CompletedProcess(args, 1, "", "simulated move failure")
            return real_run(args, **kwargs)

        monkeypatch.setattr(subprocess, "run", fail_move)
        try:
            cli_mod._cleanup_worktree(info)

            assert Path(info["path"]).exists()
            assert real_run(
                ["git", "show-ref", "--verify", f"refs/heads/{info['branch']}"],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
            assert real_run(
                [
                    "git",
                    "show-ref",
                    "--verify",
                    f"refs/hermes/archive/{info['branch']}/{oid}",
                ],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
        finally:
            monkeypatch.setattr(subprocess, "run", real_run)
            _force_remove_worktree(info)

    def test_no_remote_unique_commit_is_preserved(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        try:
            marker = Path(info["path"]) / "unique-commit.txt"
            marker.write_text("only reachable from worktree branch\n")
            subprocess.run(["git", "add", marker.name], cwd=info["path"], check=True)
            subprocess.run(
                ["git", "commit", "-m", "unique worktree commit"],
                cwd=info["path"],
                capture_output=True,
                check=True,
            )

            cli_mod._cleanup_worktree(info)

            assert Path(info["path"]).exists()
            assert subprocess.run(
                ["git", "show-ref", "--verify", f"refs/heads/{info['branch']}"],
                cwd=git_repo,
                capture_output=True,
            ).returncode == 0
        finally:
            _force_remove_worktree(info)

    def test_foreign_lock_is_preserved_during_exit_cleanup(self, git_repo):
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo), sync_base=False)
        assert info is not None
        try:
            subprocess.run(
                ["git", "worktree", "unlock", info["path"]],
                cwd=git_repo,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "worktree", "lock", "--reason", "another tool", info["path"]],
                cwd=git_repo,
                capture_output=True,
                check=True,
            )

            cli_mod._cleanup_worktree(info)

            assert Path(info["path"]).exists()
            assert cli_mod._worktree_lock_is_live(str(git_repo), info["path"]) == "unknown"
        finally:
            subprocess.run(
                ["git", "worktree", "unlock", info["path"]],
                cwd=git_repo,
                capture_output=True,
                check=False,
            )
            _force_remove_worktree(info)

    def test_rejects_parent_directory_directory_traversal(self, git_repo):
        import cli as cli_mod

        outside_dir = git_repo.parent / "outside-dir"
        outside_dir.mkdir()
        (outside_dir / "secret.txt").write_text("SENSITIVE DIR DATA")
        (git_repo / ".worktreeinclude").write_text("../outside-dir\n")

        info = None
        try:
            info = cli_mod._setup_worktree(str(git_repo))
            assert info is not None

            wt_path = Path(info["path"])
            escaped_dir = wt_path.parent / "outside-dir"
            assert not escaped_dir.exists()
            assert not escaped_dir.is_symlink()
        finally:
            _force_remove_worktree(info)

    def test_rejects_symlink_that_resolves_outside_repo(self, git_repo):
        import cli as cli_mod

        outside_file = git_repo.parent / "linked-secret.txt"
        outside_file.write_text("LINKED SECRET")
        try:
            (git_repo / "leak.txt").symlink_to(outside_file)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlink privilege is unavailable")
            raise
        (git_repo / ".worktreeinclude").write_text("leak.txt\n")

        info = None
        try:
            info = cli_mod._setup_worktree(str(git_repo))
            assert info is not None

            assert not (Path(info["path"]) / "leak.txt").exists()
        finally:
            _force_remove_worktree(info)

    def test_allows_valid_file_include(self, git_repo):
        import cli as cli_mod

        (git_repo / ".env").write_text("SECRET=***\n")
        (git_repo / ".worktreeinclude").write_text(".env\n")

        info = None
        try:
            info = cli_mod._setup_worktree(str(git_repo))
            assert info is not None

            copied = Path(info["path"]) / ".env"
            assert copied.exists()
            assert copied.read_text() == "SECRET=***\n"
        finally:
            _force_remove_worktree(info)

    def test_allows_valid_directory_include(self, git_repo):
        import cli as cli_mod

        assets_dir = git_repo / ".venv" / "lib"
        assets_dir.mkdir(parents=True)
        (assets_dir / "marker.txt").write_text("venv marker")
        (git_repo / ".worktreeinclude").write_text(".venv\n")

        info = None
        try:
            info = cli_mod._setup_worktree(str(git_repo))
            assert info is not None

            linked_dir = Path(info["path"]) / ".venv"
            if os.name == "nt" and not linked_dir.is_symlink():
                # Production deliberately falls back to copytree when Windows
                # Developer Mode / SeCreateSymbolicLinkPrivilege is unavailable.
                assert linked_dir.is_dir()
            else:
                assert linked_dir.is_symlink()
            assert (linked_dir / "lib" / "marker.txt").read_text() == "venv marker"
        finally:
            _force_remove_worktree(info)
