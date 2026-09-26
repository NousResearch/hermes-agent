"""Git trampoline recovery; branch updates use the real target-identity suite."""

import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _isolate_venv_holders(monkeypatch):
    """The update flow's venv-holder guard sees the live gateway processes on
    a dev machine and aborts with SystemExit 2 before reaching the branch
    logic under test.  Isolate it so the test exercises the intended path."""
    monkeypatch.setattr("hermes_cli.update_cmd_windows._detect_venv_python_processes", lambda: [])


class TestGitTrampolineSelfHeal:
    """Proactive Git-for-Windows trampoline self-heal (#87876).

    A broken bin\\git.exe / cmd\\git.exe shim (~46KB) refuses every git call
    with a "BUG (fork bomb)" guard instead of re-execing the real git-core
    binary. _ensure_non_trampoline_git detects this up front and swaps in a
    real git binary when one can be located, so the normal git update path
    survives instead of degrading to the ZIP fallback.
    """

    @staticmethod
    def _fake_run_healthy(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="git version 2.50.0.windows.1\n", stderr=""
        )

    @staticmethod
    def _fake_run_trampoline(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="BUG (fork bomb): tried to spawn itself, check your PATH\n",
        )

    @pytest.mark.platforms("windows")
    def test_healthy_git_command_unchanged(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_healthy,
            ),
            patch("hermes_cli.update_cmd._locate_real_git") as locate,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        locate.assert_not_called()

    @pytest.mark.platforms("windows")
    def test_trampoline_swaps_to_real_git(self, capsys):
        from pathlib import Path

        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        real = Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch(
                "hermes_cli.update_cmd._locate_real_git", return_value=real
            ),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == [str(real), "-c", "windows.appendAtomically=false"]
        out = capsys.readouterr().out
        assert "switching to real git" in out

    @pytest.mark.platforms("windows")
    def test_trampoline_no_real_git_keeps_command(self, capsys):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch("hermes_cli.update_cmd._locate_real_git", return_value=None),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        out = capsys.readouterr().out
        assert "ZIP path" in out

    @pytest.mark.platforms("not windows")
    def test_off_windows_noop(self):
        from hermes_cli import update_cmd

        git_cmd = ["git"]
        with patch("hermes_cli.update_cmd.subprocess.run") as run:
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        run.assert_not_called()

    def test_portable_git_candidates_check_shared_root_first(self, tmp_path, monkeypatch):
        # Profile-scoped layout: HERMES_HOME = <root>/profiles/foo, but the
        # PortableGit tree lives under the SHARED root (monerostar review on
        # #88136). The candidate list must check get_default_hermes_root()
        # before the profile home.
        import hermes_constants
        from hermes_cli.update_cmd_git import _portable_git_candidates

        root = tmp_path / "root"
        profile_home = root / "profiles" / "foo"

        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile_home)

        candidates = _portable_git_candidates()
        assert candidates[0] == (
            root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )
        assert candidates[1] == (
            profile_home / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        )


class TestShallowFetchDoesNotFakeOrphanDivergence:
    """Regression for #123346.

    ``git fetch --depth 1`` (the documented workaround for the slow-fetch hang) marks the
    repository shallow and truncates the remote-tracking ref. When the truncation cuts below
    local HEAD, ``merge-base`` reports no common ancestor for a checkout whose history is
    intact — and the updater would read that as orphan divergence and force-reset a branch
    that never diverged. These tests drive real git repositories, because the whole failure
    lives in git's shallow-boundary bookkeeping.
    """

    @staticmethod
    def _git(repo, *args):
        return subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True,
        )

    @staticmethod
    def _shallow_file(repo):
        return (repo / ".git" / "shallow").exists()

    @pytest.fixture()
    def checkout(self, tmp_path):
        """A 'remote' with 6 commits and a full clone sitting 3 commits behind it."""
        remote = tmp_path / "remote.git"
        seed = tmp_path / "seed"
        seed.mkdir()
        self._git(seed, "init", "-q", "-b", "main")
        self._git(seed, "config", "user.email", "t@example.com")
        self._git(seed, "config", "user.name", "t")
        for i in range(6):
            (seed / "f.txt").write_text(f"c{i}\n", encoding="utf-8")
            self._git(seed, "add", "-A")
            self._git(seed, "commit", "-qm", f"c{i}")
        self._git(seed, "clone", "-q", "--bare", str(seed), str(remote))

        work = tmp_path / "work"
        self._git(tmp_path, "clone", "-q", str(remote), str(work))
        self._git(work, "config", "user.email", "t@example.com")
        self._git(work, "config", "user.name", "t")
        self._git(work, "reset", "-q", "--hard", "HEAD~3")
        return work

    def _shallow_prefetch(self, repo):
        """The user-side workaround from #93759: a depth-1 fetch before updating."""
        self._git(repo, "fetch", "-q", "--depth", "1", "origin", "main")
        assert self._shallow_file(repo)

    def test_depth_prefetch_makes_merge_base_report_no_ancestor(self, checkout):
        """Pins the premise: this is what the depth fetch does to the ancestry test."""
        self._shallow_prefetch(checkout)
        assert subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=str(checkout), capture_output=True,
        ).returncode != 0, "premise broken: a depth fetch no longer hides the common ancestor"

    def test_ancestry_probe_deepens_rather_than_reporting_orphans(self, checkout, monkeypatch):
        """_has_common_ancestor must recover the truth instead of answering False."""
        from hermes_cli.update_cmd import _has_common_ancestor
        self._shallow_prefetch(checkout)
        monkeypatch.setattr(update_cmd, "_m", lambda: type(
            "M", (), {"PROJECT_ROOT": str(checkout)})())
        assert _has_common_ancestor(["git"], "origin/main") is True
        assert not self._shallow_file(checkout), "the checkout stayed shallow after a re-test"

    def test_deepened_ancestry_allows_fast_forward_instead_of_reset(self, checkout, monkeypatch, capsys):
        """The user-visible contract: no force-reset, no 'orphan divergence' message."""
        self._shallow_prefetch(checkout)
        head_before = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(checkout),
            capture_output=True, text=True, check=True).stdout.strip()
        monkeypatch.setattr(update_cmd, "_m", lambda: type(
            "M", (), {"PROJECT_ROOT": str(checkout)})())
        update_cmd._reconcile_diverged_checkout(["git"], "main", head_before)
        out = capsys.readouterr().out
        assert "orphan divergence" not in out
        head_after = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(checkout),
            capture_output=True, text=True, check=True).stdout.strip()
        # Landed on the remote tip, and the branch was moved, not reset-and-forked.
        assert head_after == subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=str(checkout),
            capture_output=True, text=True, check=True).stdout.strip()

    def test_genuine_orphans_in_a_complete_clone_still_reset(self, checkout, monkeypatch, capsys):
        """A complete clone with genuinely unrelated history keeps the old reset behaviour.

        Built by replacing every ref in the clone with an unrelated root commit, so
        ``main`` itself shares nothing with ``origin/main`` — the re-init / corrupted-HEAD
        case the rescue ref exists for. The shallow-deepening must not paper over this.
        """
        orphan = checkout.parent / "orphan"
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(orphan)], check=True,
        )
        self._git(orphan, "config", "user.email", "t@example.com")
        self._git(orphan, "config", "user.name", "t")
        (orphan / "unrelated.txt").write_text("u\n", encoding="utf-8")
        self._git(orphan, "add", "-A")
        self._git(orphan, "commit", "-qm", "unrelated-root")
        # Move the clone's main (and the ref the updater compares against) onto that root.
        # The object has to exist in the clone first, so bring it across.
        self._git(checkout, "fetch", "-q", str(orphan), "main")
        unrelated = self._git(orphan, "rev-parse", "HEAD").stdout.strip()
        self._git(checkout, "update-ref", "refs/heads/main", unrelated)
        self._git(checkout, "update-ref", "refs/remotes/origin/main", unrelated)
        self._git(checkout, "reset", "-q", "--hard", "main")
        # Now make origin/main a DIFFERENT unrelated history, so no ancestor exists at all.
        other = checkout.parent / "other"
        subprocess.run(["git", "init", "-q", "-b", "main", str(other)], check=True)
        self._git(other, "config", "user.email", "t@example.com")
        self._git(other, "config", "user.name", "t")
        (other / "remote.txt").write_text("r\n", encoding="utf-8")
        self._git(other, "add", "-A")
        self._git(other, "commit", "-qm", "remote-root")
        self._git(checkout, "fetch", "-q", str(other), "main")
        self._git(checkout, "update-ref", "refs/remotes/origin/main",
                   self._git(other, "rev-parse", "HEAD").stdout.strip())

        head_before = self._git(checkout, "rev-parse", "HEAD").stdout.strip()
        monkeypatch.setattr(update_cmd, "_m", lambda: type(
            "M", (), {"PROJECT_ROOT": str(checkout)})())
        update_cmd._reconcile_diverged_checkout(["git"], "main", head_before)
        out = capsys.readouterr().out
        assert "orphan divergence" in out
        assert not self._shallow_file(checkout)
        # And it really did reset onto the remote ref.
        assert self._git(checkout, "rev-parse", "HEAD").stdout.strip() == self._git(
            checkout, "rev-parse", "origin/main").stdout.strip()
