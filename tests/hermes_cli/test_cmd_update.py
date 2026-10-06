"""Git trampoline recovery; branch updates use the real target-identity suite."""

import subprocess
from types import SimpleNamespace
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


def test_pm_workspace_uses_managed_checkout_for_updates(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    install_root = tmp_path / "install"
    workspace.mkdir()
    (install_root / ".git").mkdir(parents=True)
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", workspace)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(install_root))

    assert update_cmd._update_project_root() == install_root


def test_plain_update_command_uses_managed_root_for_git_execution(tmp_path, monkeypatch):
    """The apply path must not select Git from one root and run it in another."""
    workspace = tmp_path / "workspace"
    install_root = tmp_path / "install"
    workspace.mkdir()
    (install_root / ".git").mkdir(parents=True)
    main = update_cmd._m()
    monkeypatch.setattr(main, "PROJECT_ROOT", workspace)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(install_root))

    monkeypatch.setattr(update_cmd, "git_operation_in_progress", lambda root: None)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: SimpleNamespace(
        gw_input_fn=None, assume_yes=True, keep_stash=False, switch_branch=False,
        discard_local_changes=False, no_gateway_restart=False))
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_record_pre_update_backup_outcome", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_record_snapshot_stage", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(main, "_desktop_packaged_executable", lambda *_: None)
    monkeypatch.setattr(main, "_desktop_dist_exists", lambda *_: False)
    monkeypatch.setattr(main, "_installed_desktop_apps", lambda: [])
    monkeypatch.setattr(update_cmd, "_map_ssl_cert_file_for_git", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_ensure_non_trampoline_git", lambda command: command)
    monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_normalize_managed_eol", lambda *_: None)
    monkeypatch.setattr(update_cmd, "_get_origin_url", lambda *_: "https://github.com/NousResearch/hermes-agent.git")
    monkeypatch.setattr(update_cmd, "_source_completion_request", lambda *args: {})
    monkeypatch.setattr(update_cmd, "_source_update_channel", lambda *_: "main")

    class ReachedGitFetch(Exception):
        pass

    def fetch(_runner, _git_cmd, _args, root):
        assert root == install_root
        raise ReachedGitFetch

    monkeypatch.setattr("hermes_cli.gitlock.fetch_with_partial_clone_recovery", fetch)
    monkeypatch.setattr("hermes_cli.gitlock.is_partial_clone_pack_objects_crash", lambda *_: False)

    args = SimpleNamespace(
        yes=True, keep_stash=False, switch_branch=False, no_gateway_restart=False,
        pre_update_version=None, branch="main", channel=None)
    with pytest.raises(ReachedGitFetch):
        update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert main.PROJECT_ROOT == install_root
