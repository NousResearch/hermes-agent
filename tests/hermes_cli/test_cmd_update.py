"""Tests for cmd_update — branch fallback when remote branch doesn't exist."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update
from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _isolate_venv_holders(monkeypatch):
    """The update flow's venv-holder guard sees the live gateway processes on
    a dev machine and aborts with SystemExit 2 before reaching the branch
    logic under test.  Isolate it so the test exercises the intended path."""
    monkeypatch.setattr("hermes_cli.update_cmd_windows._detect_venv_python_processes", lambda: [])


@pytest.fixture(autouse=True)
def _isolate_product_preparation(monkeypatch):
    """These tests exercise update orchestration, not PM installs or npm builds."""
    monkeypatch.setattr(update_cmd, "_prepare_updated_checkout", lambda *a, **k: None)


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


pytestmark = pytest.mark.usefixtures(
    "isolated_update_processes", "isolated_update_checkout",
)


class TestCmdUpdateBranchFallback:
    """cmd_update falls back to main when current branch has no remote counterpart."""




    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_on_fork_checks_upstream_when_origin_up_to_date(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        """Regression for issue #26172: forks whose local HEAD already matches
        origin/main must still consult upstream/main before printing
        "Already up to date!" — otherwise a fork that's caught up to its own
        origin but behind NousResearch/hermes-agent silently misses updates.
        """
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(hm, "_sync_with_upstream_if_needed") as sync_mock, patch.object(
            update_cmd, "_check_and_apply_config_migration"
        ):
            cmd_update(mock_args)

        expected_git_cmd = (
            ["git", "-c", "windows.appendAtomically=false"] if hm._is_windows() else ["git"]
        )
        sync_mock.assert_called_once_with(
            expected_git_cmd,
            # Resolved live: the module autouse fixture pins PROJECT_ROOT to
            # tmp_path, so the imported constant would be stale here.
            hm.PROJECT_ROOT,
            assume_yes=False,
            input_fn=None,
        )
        captured = capsys.readouterr()
        assert "Already up to date!" in captured.out

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_yes_on_fork_without_upstream_does_not_claim_up_to_date(
        self, mock_run, _mock_which, capsys
    ):
        """#97052 review: genuine fork, no upstream remote, HEAD == origin/main,
        --yes. The prompt is skipped without mutating remotes, and because the
        official repo was never consulted the completion line must not claim
        plain "Already up to date!"."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(
            update_cmd, "_has_upstream_remote", return_value=False
        ), patch.object(
            update_cmd, "_should_skip_upstream_prompt", return_value=False
        ), patch.object(
            update_cmd, "_add_upstream_remote"
        ) as add_remote, patch.object(
            update_cmd, "_mark_skip_upstream_prompt"
        ) as mark_skip, patch("builtins.input") as stdin_input:
            cmd_update(SimpleNamespace(yes=True))

        stdin_input.assert_not_called()
        add_remote.assert_not_called()
        mark_skip.assert_not_called()
        captured = capsys.readouterr()
        assert "Skipping upstream setup (non-interactive run)." in captured.out
        assert "official repo not checked" in captured.out
        assert "Already up to date!" not in captured.out

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_current_checkout_runtime_verification_failure_is_durable(
        self,
        mock_run,
        _mock_which,
        mock_args,
    ):
        """Prepared products must still pass runtime and durable outcome checks."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_args.gateway = True
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(hm, "_sync_with_upstream_if_needed"), patch.object(
            update_cmd,
            "_post_update_sqlite_runtime_status",
            return_value=(False, SimpleNamespace(sqlite_version_string="3.46.1")),
        ) as runtime_check, patch.object(
            update_cmd, "_write_gateway_update_exit_code"
        ) as write_gateway_exit, patch(
            "hermes_cli.update_receipt.finalize_update_receipt"
        ) as finalize_receipt, patch(
            "hermes_cli.update_receipt.finalize_pending_update_receipt"
        ):
            with pytest.raises(SystemExit) as exit_info:
                cmd_update(mock_args)

        assert exit_info.value.code == 1
        runtime_check.assert_called_once_with()
        write_gateway_exit.assert_called_once_with(False)
        finalize_receipt.assert_called_once_with("partial")

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_fork_upstream_sync_that_moves_head_runs_post_update_steps(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        """A fork sync that pulls code must continue through post-update work."""
        from hermes_cli import main as hm
        from hermes_cli import update_cmd

        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        # The first two reads bracket the upstream sync (aaaaaaa -> bbbbbbb:
        # the sync moved HEAD). The NEXT two bracket the pull inside the
        # normal update path (bbbbbbb -> ccccccc) — the head-moved no-op
        # guard added after this PR exits 1 when that pair is equal, so the
        # mock must show the pull advancing HEAD too.
        shas = iter(["aaaaaaa", "bbbbbbb", "bbbbbbb", "ccccccc"])

        with patch.object(
            hm,
            "_get_origin_url",
            return_value="https://github.com/example/hermes-agent.git",
        ), patch.object(
            update_cmd,
            "_capture_head_sha",
            side_effect=lambda *_args, **_kwargs: next(shas, "ccccccc"),
        ), patch.object(
            hm, "_sync_with_upstream_if_needed"
        ), patch.object(
            update_cmd,
            "_run_post_update_maintenance",
            # Unlike product preparation, this phase only runs after a pull.
            # Stop before skills sync and fleet restart; the regression took
            # the current-checkout path instead and never reached this phase.
            side_effect=SystemExit(0),
        ) as post_update_step:
            with pytest.raises(SystemExit) as exit_info:
                cmd_update(mock_args)

        assert exit_info.value.code == 0
        post_update_step.assert_called_once()
        captured = capsys.readouterr()
        assert "Already up to date!" not in captured.out

    def test_update_non_interactive_runs_safe_config_migrations(self, mock_args, capsys):
        """Dashboard/web updates apply non-interactive migrations before restart."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=["MISSING_KEY"]
        ), patch(
            "hermes_cli.config.get_missing_config_fields",
            return_value=[{"key": "new.option", "default": True}],
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 2)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": ["new.option"]},
        ) as migrate_config, patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = False
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            migrate_config.assert_called_once_with(interactive=False, quiet=False)
            captured = capsys.readouterr()
            assert "applying safe config migrations" in captured.out
            assert "API keys require manual entry" in captured.out


class TestCmdUpdateMigrationPrompt:
    """The config-migration prompt names what changed and skips the prompt
    entirely when only the config format version moved.

    Regression guard for the contentless-prompt report (ScottFive / Tt2021):
    previously the prompt printed only counts ("1 new config option") and
    asked "configure them now?" even for pure version bumps, where saying
    yes looked like a no-op.
    """

    def test_version_bump_only_applies_silently_without_prompt(
        self, mock_args, capsys
    ):
        """Only the version moved → apply non-interactively, never prompt."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=[]
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=[]
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(5, 24)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": [], "warnings": []},
        ) as mock_migrate:
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            mock_migrate.assert_called_once_with(interactive=False, quiet=True)
            out = capsys.readouterr().out
            assert "Updating config format (v5 → v24)" in out
            assert "no new settings to configure" in out
            # The misleading question must NOT appear for a pure version bump.
            assert "configure them now" not in out.lower()

    def test_version_bump_only_surfaces_migration_resets(
        self, mock_args, capsys
    ):
        """A quiet version-bump migration that RESETS a user setting must say so.

        Regression for #86656: the v33→v34 personality reset ran with
        quiet=True and its results dict was discarded, so the update printed
        "no new settings to configure" while silently wiping
        display.personality. Migration-step mutations (config_added) and
        warnings must be re-surfaced even in the silent branch.
        """
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=[]
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=[]
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(33, 34)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={
                "env_added": [],
                "config_added": ["display.personality=none (one-time reset)"],
                "warnings": ["Disabled suspicious MCP server 'evil'"],
            },
        ):
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            out = capsys.readouterr().out
            assert "Updating config format (v33 → v34)" in out
            assert "no new settings to configure" in out
            # The migration's mutation note and warning must NOT be swallowed.
            assert "display.personality=none (one-time reset)" in out
            assert "Disabled suspicious MCP server 'evil'" in out

    def test_new_options_are_listed_by_name_before_prompt(
        self, mock_args, capsys
    ):
        """New env/config keys are printed by name so the user can decide."""
        env_items = [
            {"name": "FOO_API_KEY", "description": "Foo service API key"},
        ]
        cfg_items = [
            {"key": "display.new_widget", "description": "New config option: display.new_widget"},
        ]
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input", return_value="n"), patch(
            "hermes_cli.config.get_missing_env_vars", return_value=env_items
        ), patch(
            "hermes_cli.config.get_missing_config_fields", return_value=cfg_items
        ), patch(
            "hermes_cli.update_cmd._reload_config_modules"
        ), patch(
            "hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 24)
        ), patch(
            "hermes_cli.update_cmd._run_migrate_config_fresh",
            return_value={"env_added": [], "config_added": [], "warnings": []},
        ), patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            out = capsys.readouterr().out
            # Names, not just counts.
            assert "FOO_API_KEY" in out
            assert "Foo service API key" in out
            assert "display.new_widget" in out


class TestConfigVersionCheckUsesFreshModules:
    """Regression: config migration must use freshly-reloaded modules, not the
    sys.modules cache from before git pull.

    Before the fix, ``hermes update`` ran in the PRE-pull Python process.
    After ``git pull`` updated the source on disk, function-level imports
    returned the OLD cached ``hermes_cli.config`` module — so
    ``DEFAULT_CONFIG["_config_version"]`` was stale and
    ``check_config_version()`` reported ``(33, 33)`` "up to date" even though
    the freshly-pulled code had v34 with a migration to run. The personality
    reset migration (#81946) was silently skipped this way.
    """

    def test_run_config_check_fresh_reloads_modules(self):
        """_run_config_check_fresh must call _reload_config_modules which
        force-reloads the config modules from disk.

        Regression: config migration was silently skipped because
        sys.modules held the OLD hermes_cli.config with the OLD
        DEFAULT_CONFIG["_config_version"] after git pull.
        """
        from unittest.mock import patch

        import hermes_cli.update_cmd as update_cmd

        with patch.object(update_cmd, "_reload_config_modules") as mock_reload:
            update_cmd._run_config_check_fresh()

        mock_reload.assert_called_once()


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

        # the ff-only merge must target origin/bb/gui
        merge_cmds = [c for c in commands if "merge --ff-only" in c]
        assert any("origin/bb/gui" in c and "origin/main" not in c for c in merge_cmds), merge_cmds


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


class TestCmdUpdateZipBranchRefusal:
    """``hermes update --branch=<non-main>`` must refuse on the ZIP fallback path.

    The ZIP fallback hard-codes a GitHub archive URL for main.zip; honoring
    --branch arbitrarily would require remote-branch existence checks the
    fallback can't easily do. Refusing is the right move — silently lying
    about which branch got installed is the bug --branch was meant to prevent.
    """

    def test_zip_fallback_refuses_non_main_branch(self, capsys):
        from hermes_cli.update_cmd_zip import _update_via_zip

        args = SimpleNamespace(branch="bb/gui")
        with pytest.raises(SystemExit) as exc_info:
            _update_via_zip(args)
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert "bb/gui" in out
        assert "not supported" in out
        # No actual download attempted.
        assert "Downloading latest version" not in out


class TestZipDesktopPreservation:
    def test_git_failure_zip_fallback_preserves_desktop(self, tmp_path, monkeypatch):
        """The Windows ZIP fallback keeps Desktop intact when replacing ``apps/``.

        The built app survives the source swap and preparation retains the
        pre-update desktop selection (#70337/#87331).
        """
        import zipfile

        from hermes_cli import main as hm
        from hermes_cli import update_cmd
        from hermes_cli import update_cmd_maint, update_cmd_zip

        project_root = tmp_path / "hermes-agent"
        (project_root / ".git").mkdir(parents=True)
        desktop_dir = project_root / "apps" / "desktop"
        packaged_exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
        packaged_exe.parent.mkdir(parents=True)
        packaged_exe.write_bytes(b"desktop")

        def write_source_zip(_url, destination):
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("hermes-agent-main/apps/desktop/package.json", "{}")

        def fail_git_fetch(command, **_kwargs):
            if "fetch" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        preparations = []

        def prepare_checkout(root, *, desktop):
            preparations.append((root, desktop, packaged_exe.read_bytes()))

        monkeypatch.setattr(hm, "PROJECT_ROOT", project_root)
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(hm, "_run_pre_update_backup", lambda _args: None)
        monkeypatch.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
        monkeypatch.setattr(hm, "_get_origin_url", lambda *_args: "")
        monkeypatch.setattr(
            hm,
            "_desktop_packaged_executable",
            lambda _desktop_dir: packaged_exe if packaged_exe.exists() else None,
        )
        monkeypatch.setattr(hm, "_desktop_dist_exists", lambda _desktop_dir: False)
        monkeypatch.setattr(update_cmd_maint, "_prepare_updated_checkout", prepare_checkout)
        monkeypatch.setattr(hm, "_clear_bytecode_cache", lambda *_args: 0)
        monkeypatch.setattr(hm, "_record_bytecode_fingerprint", lambda: None)
        monkeypatch.setattr(hm, "_refresh_bootstrap_cache_scripts", lambda _branch: None)

        monkeypatch.setattr(update_cmd, "_discard_lockfile_churn", lambda *_args: None)
        monkeypatch.setattr(update_cmd, "_normalize_managed_eol", lambda *_args: None)
        monkeypatch.setattr(
            update_cmd,
            "_validate_critical_modules_import",
            lambda *_args: (True, None, None),
        )

        monkeypatch.setattr(update_cmd, "_print_curator_first_run_notice", lambda: None)
        monkeypatch.setattr(update_cmd, "_print_curator_recent_run_notice", lambda: None)
        monkeypatch.setattr("hermes_cli.update_cmd_maint._refresh_dashboard_after_update", lambda **kwargs: None)
        monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path / "hermes-home")

        with (
            patch("hermes_cli.config.load_config", return_value={}),
            # This test legitimately exercises the real ZIP fallback; undo the
            # module autouse fixture's fail-fast tripwire for exactly this run.
            patch(
                "hermes_cli.update_cmd._update_via_zip",
                update_cmd_zip._update_via_zip,
            ),
            patch("subprocess.run", side_effect=fail_git_fetch),
            patch("urllib.request.urlretrieve", side_effect=write_source_zip),
            patch(
                "tools.skills_sync.sync_skills",
                return_value={
                    "copied": [],
                    "updated": [],
                    "user_modified": [],
                    "cleaned": [],
                    "relocated": [],
                },
            ),
            patch("hermes_cli.model_catalog.seed_cache_from_checkout", return_value=False),
        ):
            update_cmd._cmd_update_impl(
                SimpleNamespace(yes=True, force=True, force_venv=True, branch=None),
                gateway_mode=False,
            )

        assert preparations == [(project_root, True, b"desktop")]
        assert packaged_exe.exists()
        assert packaged_exe.read_bytes() == b"desktop"


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

    def test_healthy_git_command_unchanged(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch("sys.platform", "win32"),
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_healthy,
            ),
            patch("hermes_cli.update_cmd._locate_real_git") as locate,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        locate.assert_not_called()

    def test_trampoline_swaps_to_real_git(self, capsys):
        from pathlib import Path

        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        real = Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
        with (
            patch("sys.platform", "win32"),
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

    def test_trampoline_no_real_git_keeps_command(self, capsys):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch("sys.platform", "win32"),
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

    def test_off_windows_noop(self):
        from hermes_cli import update_cmd

        git_cmd = ["git"]
        with (
            patch("sys.platform", "linux"),
            patch("hermes_cli.update_cmd.subprocess.run") as run,
        ):
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
