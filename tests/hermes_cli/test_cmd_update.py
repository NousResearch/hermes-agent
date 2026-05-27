"""Tests for cmd_update — branch fallback when remote branch doesn't exist."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update, PROJECT_ROOT


def _make_run_side_effect(branch="main", verify_ok=True, commit_count="0"):
    """Build a side_effect function for subprocess.run that simulates git commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD  (get current branch)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        # git rev-parse --verify origin/{branch}  (check remote branch exists)
        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        # git rev-list HEAD..origin/{branch} --count
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        # Fallback: return a successful CompletedProcess with empty stdout
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


class TestCmdUpdateBranchFallback:
    """cmd_update falls back to main when current branch has no remote counterpart."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_falls_back_to_main_when_branch_not_on_remote(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="fix/stoicneko", verify_ok=False, commit_count="3"
        )

        cmd_update(mock_args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        # rev-list should use origin/main, not origin/fix/stoicneko
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert len(rev_list_cmds) == 1
        assert "origin/main" in rev_list_cmds[0]
        assert "origin/fix/stoicneko" not in rev_list_cmds[0]

        # pull should use main, not fix/stoicneko
        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 1
        assert "main" in pull_cmds[0]

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_uses_current_branch_when_on_remote(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="2"
        )

        cmd_update(mock_args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert len(rev_list_cmds) == 1
        assert "origin/main" in rev_list_cmds[0]

        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 1
        assert "main" in pull_cmds[0]

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_already_up_to_date(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        cmd_update(mock_args)

        captured = capsys.readouterr()
        assert "Already up to date!" in captured.out

        # Should NOT have called pull
        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 0

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_update_refreshes_repo_and_tui_node_dependencies(
        self, mock_run, mock_which, mock_args
    ):
        from hermes_cli import main as hm

        mock_which.side_effect = {"uv": "/usr/bin/uv", "npm": "/usr/bin/npm"}.get
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        with patch.object(hm, "_is_termux_env", return_value=False):
            cmd_update(mock_args)

        npm_calls = [
            (call.args[0], call.kwargs.get("cwd"))
            for call in mock_run.call_args_list
            if call.args and call.args[0][0] == "/usr/bin/npm"
        ]

        # cmd_update runs npm commands in three locations:
        #   1. repo root  — slash-command / TUI bridge deps
        #   2. ui-tui/    — Ink TUI deps
        #   3. web/       — install + "npm run build" for the web frontend
        #
        # Repo-root and ui-tui installs intentionally omit `--silent` and run
        # without `capture_output` so optional postinstall scripts (e.g.
        # `@askjo/camofox-browser`'s browser-binary fetch) print progress —
        # otherwise long downloads look like a hang (#18840).  The web/ install
        # keeps `--silent` because its build step is short and noisy.
        update_flags = [
            "/usr/bin/npm",
            "ci",
            "--no-fund",
            "--no-audit",
            "--progress=false",
        ]
        assert npm_calls[:2] == [
            (update_flags, PROJECT_ROOT),
            (update_flags, PROJECT_ROOT / "ui-tui"),
        ]
        if len(npm_calls) > 2:
            assert npm_calls[2:] == [
                (["/usr/bin/npm", "ci", "--silent"], PROJECT_ROOT / "web"),
                (["/usr/bin/npm", "run", "build"], PROJECT_ROOT / "web"),
            ]

        # Regression for #18840: repo root + ui-tui installs must stream
        # output (capture_output=False) so postinstall progress is visible
        # to the user.
        repo_and_tui_calls = [
            call
            for call in mock_run.call_args_list
            if call.args
            and call.args[0][0] == "/usr/bin/npm"
            and call.args[0][1] == "ci"
            and call.kwargs.get("cwd") in {PROJECT_ROOT, PROJECT_ROOT / "ui-tui"}
        ]
        assert len(repo_and_tui_calls) == 2
        for call in repo_and_tui_calls:
            assert call.kwargs.get("capture_output") is False, (
                "repo-root / ui-tui npm install must stream output "
                "(no capture_output) so postinstall progress is visible"
            )

    def test_update_non_interactive_runs_safe_config_migrations(self, mock_args, capsys):
        """Dashboard/web updates apply non-interactive migrations before restart."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=["MISSING_KEY"]
        ), patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[{"key": "new.option", "default": True}],
        ), patch("hermes_cli.config.check_config_version", return_value=(1, 2)), patch(
            "hermes_cli.config.migrate_config",
            return_value={"env_added": [], "config_added": ["new.option"]},
        ), patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = False
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            from hermes_cli.config import migrate_config

            migrate_config.assert_called_once_with(interactive=False, quiet=False)
            captured = capsys.readouterr()
            assert "applying safe config migrations" in captured.out
            assert "API keys require manual entry" in captured.out


class TestCmdUpdateProfileSkillSync:
    """cmd_update syncs bundled skills to all profiles, including the active one.

    Regression guard for #16176: previously the active profile was excluded
    from the seed_profile_skills loop, leaving it on stale skill content.
    """

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_active_profile_included_in_skill_sync(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        from pathlib import Path

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )

        default_p = SimpleNamespace(name="default", path=Path("/fake/.hermes"))
        active_p = SimpleNamespace(name="bit", path=Path("/fake/.hermes/profiles/bit"))
        other_p = SimpleNamespace(name="work", path=Path("/fake/.hermes/profiles/work"))
        all_profiles = [default_p, active_p, other_p]

        synced_paths = []

        def fake_seed(path, quiet=False):
            synced_paths.append(path)
            return {"copied": [], "updated": [], "user_modified": []}

        empty_sync = {"copied": [], "updated": [], "user_modified": [], "cleaned": []}

        with (
            patch("hermes_cli.profiles.list_profiles", return_value=all_profiles),
            patch("hermes_cli.profiles.seed_profile_skills", side_effect=fake_seed),
            patch("tools.skills_sync.sync_skills", return_value=empty_sync),
        ):
            cmd_update(mock_args)

        assert active_p.path in synced_paths, (
            f"Active profile 'bit' must be included in skill sync; got: {synced_paths}"
        )
        assert set(synced_paths) == {p.path for p in all_profiles}, (
            f"All profiles must be synced; got: {synced_paths}"
        )

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_single_profile_default_is_synced(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        from pathlib import Path

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )

        default_p = SimpleNamespace(name="default", path=Path("/fake/.hermes"))
        synced_paths = []

        def fake_seed(path, quiet=False):
            synced_paths.append(path)
            return {"copied": [], "updated": [], "user_modified": []}

        empty_sync = {"copied": [], "updated": [], "user_modified": [], "cleaned": []}

        with (
            patch("hermes_cli.profiles.list_profiles", return_value=[default_p]),
            patch("hermes_cli.profiles.seed_profile_skills", side_effect=fake_seed),
            patch("tools.skills_sync.sync_skills", return_value=empty_sync),
        ):
            cmd_update(mock_args)

        assert default_p.path in synced_paths


class TestCmdUpdateBranchFlag:
    """``hermes update --branch <name>`` targets the requested branch.

    The CLI default stays 'main'; --branch lets callers pick a different
    target without monkey-patching the implementation.
    """

    def _branch_side_effect(self, current_branch, target_branch, *, checkout_fails=False, track_fails=False, commit_count="0"):
        """Mock side-effect that knows about checkout/track behavior.

        - ``current_branch``  what ``git rev-parse --abbrev-ref HEAD`` returns
        - ``target_branch``   passed via --branch; what we expect the code to switch to
        - ``checkout_fails``  if True, ``git checkout <target>`` returns non-zero
                              (simulates branch absent locally; code should retry with -B)
        - ``track_fails``     if True, ``git checkout -B <target> origin/<target>`` ALSO fails
                              (simulates branch absent on origin too)
        - ``commit_count``    rev-list count returned (0 = up-to-date, >0 = behind)
        """

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)

            if "rev-parse" in joined and "--abbrev-ref" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{current_branch}\n", stderr="")

            if "checkout" in joined and "-B" in joined:
                rc = 128 if track_fails else 0
                err = f"fatal: '{target_branch}' did not match any file(s) known to git\n" if track_fails else ""
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "checkout" in joined and "-B" not in joined and "rev-parse" not in joined:
                rc = 128 if checkout_fails else 0
                err = f"error: pathspec '{target_branch}' did not match\n" if checkout_fails else ""
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "rev-list" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return side_effect

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_pulls_against_named_branch(self, mock_run, _mock_which, capsys):
        """--branch bb/gui makes rev-list and pull target origin/bb/gui."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="bb/gui", target_branch="bb/gui", commit_count="3"
        )
        args = SimpleNamespace(branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        # rev-list must compare against origin/bb/gui, not origin/main
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("origin/bb/gui" in c for c in rev_list_cmds), rev_list_cmds
        assert not any("origin/main" in c for c in rev_list_cmds), rev_list_cmds

        # pull must target bb/gui
        pull_cmds = [c for c in commands if "pull" in c and "ff-only" in c]
        assert any("bb/gui" in c and "main" not in c.split() for c in pull_cmds), pull_cmds

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_defaults_to_main_when_none(self, mock_run, _mock_which, capsys):
        """No --branch (or --branch=None) preserves the historical 'main' default."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="main", target_branch="main", commit_count="0"
        )
        args = SimpleNamespace(branch=None)

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert all("origin/main" in c for c in rev_list_cmds), rev_list_cmds

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_switches_from_different_branch(self, mock_run, _mock_which, capsys):
        """When HEAD is on main and --branch=bb/gui, switch to bb/gui first."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="main", target_branch="bb/gui", commit_count="2"
        )
        args = SimpleNamespace(branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # First checkout call should switch us to bb/gui (not -B; happy-path branch exists locally)
        checkout_cmds = [c for c in commands if "checkout" in c and "rev-parse" not in c]
        assert len(checkout_cmds) >= 1
        assert "bb/gui" in checkout_cmds[0]

        out = capsys.readouterr().out
        assert "switching to bb/gui" in out

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_tracks_remote_when_branch_absent_locally(self, mock_run, _mock_which, capsys):
        """If local lacks the branch but origin has it, fall back to ``checkout -B``."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="main",
            target_branch="bb/gui",
            checkout_fails=True,  # plain checkout fails
            track_fails=False,    # -B from origin/bb/gui succeeds
            commit_count="2",
        )
        args = SimpleNamespace(branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # Should have BOTH a failed `checkout bb/gui` AND a successful `checkout -B bb/gui origin/bb/gui`
        track_cmds = [c for c in commands if "checkout" in c and "-B" in c]
        assert len(track_cmds) == 1
        assert "bb/gui" in track_cmds[0]
        assert "origin/bb/gui" in track_cmds[0]

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_branch_flag_fails_when_branch_missing_everywhere(self, mock_run, _mock_which, capsys):
        """If branch doesn't exist locally OR on origin, exit non-zero with clear error."""
        mock_run.side_effect = self._branch_side_effect(
            current_branch="main",
            target_branch="nonexistent",
            checkout_fails=True,
            track_fails=True,
            commit_count="0",
        )
        args = SimpleNamespace(branch="nonexistent")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "does not exist locally or on origin" in out
        assert "nonexistent" in out


class TestCmdUpdateCheckBranchFlag:
    """``hermes update --check --branch <name>`` honors the branch override.

    The check path used to call ``git rev-list HEAD..origin/<branch> --count``
    with ``check=True``. When the branch didn't exist on origin, the fetch
    silently succeeded (no refspec) but rev-list exited 128 and a raw
    ``CalledProcessError`` propagated to the user. These tests pin the
    friendlier behavior: detect-the-missing-ref before rev-list, exit 1
    with a clear message.
    """

    def _check_side_effect(
        self,
        target_branch: str,
        *,
        verify_ok: bool = True,
        commit_count: str = "0",
        upstream_fetch_ok: bool = True,
    ):
        """Mock side-effect for the _cmd_update_check git pipeline.

        - ``target_branch``      what we expect compare ref to point at
        - ``verify_ok``          if False, ``git rev-parse --verify --quiet
                                 origin/<branch>`` fails (branch missing
                                 on origin)
        - ``commit_count``       rev-list count (0 = up-to-date)
        - ``upstream_fetch_ok``  if False, ``git fetch upstream`` fails
                                 (forces fallback to origin on branch==main)
        """

        def side_effect(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)

            if "fetch" in joined and "upstream" in joined:
                rc = 0 if upstream_fetch_ok else 128
                err = "" if upstream_fetch_ok else "fatal: 'upstream' does not appear to be a git repository\n"
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

            if "fetch" in joined and "origin" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            if "rev-parse" in joined and "--verify" in joined:
                rc = 0 if verify_ok else 1
                return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

            if "rev-list" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        return side_effect

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_branch_compares_against_named_origin_branch(
        self, mock_run, _mock_method, capsys
    ):
        """--check --branch bb/gui compares against origin/bb/gui, never origin/main."""
        mock_run.side_effect = self._check_side_effect(
            target_branch="bb/gui", verify_ok=True, commit_count="2"
        )
        args = SimpleNamespace(check=True, branch="bb/gui")

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # Non-main branch skips upstream probe entirely.
        assert not any("fetch" in c and "upstream" in c for c in commands), commands
        # Verify and rev-list both target origin/bb/gui.
        verify_cmds = [c for c in commands if "rev-parse" in c and "--verify" in c]
        assert any("origin/bb/gui" in c for c in verify_cmds), verify_cmds
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("origin/bb/gui" in c for c in rev_list_cmds), rev_list_cmds
        assert not any("origin/main" in c for c in rev_list_cmds), rev_list_cmds

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_branch_missing_on_origin_exits_cleanly(
        self, mock_run, _mock_method, capsys
    ):
        """If origin/<branch> doesn't exist, surface a friendly error and exit 1.

        Pre-fix this case raised CalledProcessError from rev-list's check=True
        and dumped a Python traceback to stdout.
        """
        mock_run.side_effect = self._check_side_effect(
            target_branch="ghost", verify_ok=False
        )
        args = SimpleNamespace(check=True, branch="ghost")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        # No raw Python traceback.
        assert "Traceback" not in out
        assert "CalledProcessError" not in out
        # Friendly message naming the branch.
        assert "ghost" in out
        assert "not found" in out

        # rev-list must never have been called once verify failed.
        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        assert not any("rev-list" in c for c in commands), commands

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_check_default_main_still_prefers_upstream(
        self, mock_run, _mock_method, capsys
    ):
        """No --branch (or --branch=None) preserves the upstream-then-origin probe."""
        mock_run.side_effect = self._check_side_effect(
            target_branch="main", verify_ok=True, commit_count="0"
        )
        args = SimpleNamespace(check=True, branch=None)

        cmd_update(args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        # Should have tried upstream first.
        assert any("fetch" in c and "upstream" in c for c in commands), commands
        # Compare ref is upstream/main (upstream fetch succeeded).
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert any("upstream/main" in c for c in rev_list_cmds), rev_list_cmds

    @patch("hermes_cli.config.detect_install_method", return_value="pip")
    @patch("hermes_cli.banner.check_via_pypi", return_value=0)
    @patch("subprocess.run")
    def test_check_branch_warns_on_pypi_install(
        self, mock_run, _mock_pypi, _mock_method, capsys
    ):
        """PyPI install + --branch=<non-main> surfaces a warning instead of silent drop."""
        args = SimpleNamespace(check=True, branch="bb/gui")

        cmd_update(args)

        out = capsys.readouterr().out
        assert "--branch is ignored for PyPI installs" in out
        assert "bb/gui" in out


class TestCmdUpdateZipBranchRefusal:
    """``hermes update --branch=<non-main>`` must refuse on the ZIP fallback path.

    The ZIP fallback hard-codes a GitHub archive URL for main.zip; honoring
    --branch arbitrarily would require remote-branch existence checks the
    fallback can't easily do. Refusing is the right move — silently lying
    about which branch got installed is the bug --branch was meant to prevent.
    """

    def test_zip_fallback_refuses_non_main_branch(self, capsys):
        from hermes_cli.main import _update_via_zip

        args = SimpleNamespace(branch="bb/gui")
        with pytest.raises(SystemExit) as exc_info:
            _update_via_zip(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "bb/gui" in out
        assert "not supported" in out
        # No actual download attempted.
        assert "Downloading latest version" not in out


def test_is_termux_env_true_for_termux_prefix():
    from hermes_cli import main as hm

    assert hm._is_termux_env({"PREFIX": "/data/data/com.termux/files/usr"}) is True


def test_is_termux_env_false_for_non_termux_prefix():
    from hermes_cli import main as hm

    assert hm._is_termux_env({"PREFIX": "/usr/local"}) is False


def test_load_installable_optional_extras_supports_termux_group(tmp_path, monkeypatch):
    from hermes_cli import main as hm

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
name = "x"
version = "0.0.0"

[project.optional-dependencies]
all = ["x[mcp]"]
termux-all = ["x[termux]", "x[mcp]"]
mcp = ["mcp>=1"]
termux = ["rich>=14"]
""".strip()
    )
    monkeypatch.setattr(hm, "PROJECT_ROOT", tmp_path)

    assert hm._load_installable_optional_extras(group="all") == ["mcp"]
    assert hm._load_installable_optional_extras(group="termux-all") == ["termux", "mcp"]
