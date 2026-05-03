"""Tests for cmd_update — branch fallback when remote branch doesn't exist."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import UpdateReplayResult, cmd_update, PROJECT_ROOT


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

        # The update-count check should use origin/main, not origin/fix/stoicneko.
        rev_list_cmds = [c for c in commands if "HEAD..origin/main --count" in c]
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

        rev_list_cmds = [c for c in commands if "HEAD..origin/main --count" in c]
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
        mock_which.side_effect = {"uv": "/usr/bin/uv", "npm": "/usr/bin/npm"}.get
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )

        cmd_update(mock_args)

        npm_calls = [
            (call.args[0], call.kwargs.get("cwd"))
            for call in mock_run.call_args_list
            if call.args and call.args[0][0] == "/usr/bin/npm"
        ]

        # cmd_update runs npm commands in three locations:
        #   1. repo root  — slash-command / TUI bridge deps
        #   2. ui-tui/    — Ink TUI deps
        #   3. web/       — web UI frontend deps + build
        full_flags = [
            "/usr/bin/npm",
            "ci",
            "--silent",
            "--no-fund",
            "--no-audit",
            "--progress=false",
        ]
        assert npm_calls == [
            (full_flags, PROJECT_ROOT),
            (full_flags, PROJECT_ROOT / "ui-tui"),
            (["/usr/bin/npm", "ci", "--silent"], PROJECT_ROOT / "web"),
            (["/usr/bin/npm", "run", "build"], PROJECT_ROOT / "web"),
        ]

    def test_update_non_interactive_skips_migration_prompt(self, mock_args, capsys):
        """When stdin/stdout aren't TTYs, config migration prompt is skipped."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=["MISSING_KEY"]
        ), patch("hermes_cli.config.get_missing_config_fields", return_value=[]), patch(
            "hermes_cli.config.check_config_version", return_value=(1, 2)
        ), patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = False
            mock_run.side_effect = _make_run_side_effect(
                branch="main", verify_ok=True, commit_count="1"
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            captured = capsys.readouterr()
            assert "Non-interactive session" in captured.out


@patch("hermes_cli.main.replay_missing_update_commits")
@patch("hermes_cli.main.create_update_rescue_ref", return_value="refs/hermes/update-rescue/test")
@patch("hermes_cli.main.collect_update_snapshot")
@patch("hermes_cli.main._stash_local_changes_if_needed", return_value=None)
@patch("hermes_cli.main._invalidate_update_cache")
@patch("hermes_cli.main._clear_bytecode_cache", return_value=0)
@patch("hermes_cli.main._install_python_dependencies_with_optional_fallback")
@patch("hermes_cli.main._update_node_dependencies")
@patch("hermes_cli.main._build_web_ui")
@patch("hermes_cli.main._sync_with_upstream_if_needed")
@patch("hermes_cli.main._finalize_update_output")
@patch("hermes_cli.main._print_update_final_report")
@patch("hermes_cli.main._install_hangup_protection", return_value={})
@patch("hermes_cli.main._get_origin_url", return_value="git@github.com:NousResearch/hermes-agent.git")
@patch("hermes_cli.main._is_fork", return_value=False)
@patch("shutil.which", return_value=None)
@patch("subprocess.run")
def test_update_replay_failure_stops_before_post_update_side_effects(
    mock_run,
    mock_which,
    mock_is_fork,
    mock_get_origin_url,
    mock_install_hangup,
    mock_print_report,
    mock_finalize_output,
    mock_sync_upstream,
    mock_build_web,
    mock_update_node_deps,
    mock_install_python_deps,
    mock_clear_bytecode,
    mock_invalidate_cache,
    mock_stash,
    mock_collect_snapshot,
    mock_create_rescue_ref,
    mock_replay_commits,
    mock_args,
):
    mock_collect_snapshot.return_value = SimpleNamespace(
        head_before="abc123deadbeef",
        branch_before="main",
        upstream_ref="origin/main",
        upstream_head_before="def456deadbeef",
        status_short="",
        ahead_commits=["c1"],
        ahead_count=1,
        dirty_tree=False,
    )
    mock_replay_commits.return_value = UpdateReplayResult(
        skipped_commits=[],
        replayed_commits=[],
        conflicted_commit="c1",
        succeeded=False,
    )

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "fetch", "origin"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="def456deadbeef\n", stderr="")
        if cmd == ["git", "status", "--short", "--branch"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="## main...origin/main [ahead 1]\n", stderr=""
            )
        if cmd == ["git", "rev-list", "HEAD..origin/main", "--count"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1\n", stderr="")
        if cmd == ["git", "pull", "--ff-only", "origin", "main"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    mock_run.side_effect = fake_run

    with pytest.raises(SystemExit) as excinfo:
        cmd_update(mock_args)

    assert excinfo.value.code == 1
    mock_replay_commits.assert_called_once()
    mock_invalidate_cache.assert_not_called()
    mock_clear_bytecode.assert_not_called()
    mock_sync_upstream.assert_not_called()
    mock_install_python_deps.assert_not_called()
    mock_update_node_deps.assert_not_called()
    mock_build_web.assert_not_called()
    mock_print_report.assert_called_once()
    report = mock_print_report.call_args.args[0]
    assert report["gateway_service_health"] == "not reached"
    assert report["replay_conflict_commit"] == "c1"
    mock_finalize_output.assert_called_once()
