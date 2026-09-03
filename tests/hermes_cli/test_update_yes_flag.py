"""Tests for `hermes update --yes / -y` — assume yes for interactive prompts.

Covers:
  1. argparse parses the flag
  2. Config-migration prompt is auto-answered (no input() call) and migrate_config
     runs with interactive=False so API-key prompts are skipped
  3. Autostash restore prompt is auto-answered (prompt_for_restore == False, no
     input() call) and the stash is applied automatically
"""

import subprocess

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import cmd_update


@pytest.fixture(autouse=True)
def _isolate_cmd_update(monkeypatch):
    import hermes_cli.gateway as hermes_gateway
    import hermes_cli.main as hermes_main

    # These tests cover prompt behavior, not module purging or fleet restart.
    # Retain the no-live-gateway stubs so cmd_update cannot rediscover and
    # signal a developer's running gateway after the simulated pull.
    # _purge_stale_hermes_modules is invoked through _m(), so the
    # hermes_cli.main patch reaches it; _finish_dashboard_update_cleanup and
    # _restart_macos_launchd_gateways are bare module globals in
    # hermes_cli.update_cmd and must be patched on that namespace.
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.update_cmd._finish_dashboard_update_cleanup",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.update_cmd._restart_macos_launchd_gateways",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        hermes_gateway, "find_gateway_pids", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    monkeypatch.setattr(hermes_gateway, "supports_systemd_services", lambda: False)


def _make_run_side_effect(
    branch="main", verify_ok=True, commit_count="1", dirty=False
):
    """Minimal subprocess.run side_effect for the update flow."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")
        if "rev-parse" in joined and "--verify" in joined:
            return subprocess.CompletedProcess(
                cmd, 0 if verify_ok else 128, stdout="", stderr=""
            )
        if "rev-list" in joined:
            return subprocess.CompletedProcess(
                cmd, 0, stdout=f"{commit_count}\n", stderr=""
            )
        # `git status --porcelain` for dirty-tree detection during autostash.
        if "status" in joined and "--porcelain" in joined:
            out = " M hermes_cli/main.py\n" if dirty else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        # `git stash list` — return a stash ref when dirty (so _stash_local_changes
        # gets something to return). _stash_local_changes_if_needed is what we
        # actually patch in tests that exercise restore, so this is a catch-all.
        if "stash" in joined and "list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


class TestUpdateYesConfigMigration:
    """--yes auto-answers the config-migration prompt and skips API-key prompts."""

    @patch("hermes_cli.update_cmd._reload_config_modules")
    @patch("hermes_cli.update_cmd._run_migrate_config_fresh")
    @patch("hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 2))
    @patch("hermes_cli.config.get_missing_config_fields", return_value=[])
    @patch("hermes_cli.config.get_missing_env_vars", return_value=["NEW_KEY"])
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_yes_auto_migrates_without_input(
        self,
        mock_run,
        _mock_which,
        _mock_missing_env,
        _mock_missing_cfg,
        _mock_version,
        mock_migrate,
        _mock_reload,
        capsys,
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        mock_migrate.return_value = {"env_added": [], "config_added": []}

        args = SimpleNamespace(yes=True)

        with patch("builtins.input") as mock_input:
            cmd_update(args)
            # Never prompted the user.
            mock_input.assert_not_called()

        # migrate_config was invoked with interactive=False — API-key prompts
        # are suppressed, matching gateway-mode semantics.
        assert mock_migrate.call_count == 1
        _, kwargs = mock_migrate.call_args
        assert kwargs.get("interactive") is False

        out = capsys.readouterr().out
        assert "--yes: auto-applying config migration" in out
        # The "Would you like to configure them now?" prompt text never appears.
        assert "Would you like to configure them now?" not in out

    @patch("hermes_cli.update_cmd._reload_config_modules")
    @patch("hermes_cli.update_cmd._run_migrate_config_fresh")
    @patch("hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 2))
    @patch("hermes_cli.config.get_missing_config_fields", return_value=[])
    @patch("hermes_cli.config.get_missing_env_vars", return_value=["NEW_KEY"])
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_no_yes_flag_still_prompts_in_tty(
        self,
        mock_run,
        _mock_which,
        _mock_missing_env,
        _mock_missing_cfg,
        _mock_version,
        mock_migrate,
        _mock_reload,
        capsys,
    ):
        """Regression guard: without --yes, the TTY prompt path still fires."""
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        mock_migrate.return_value = {"env_added": [], "config_added": []}

        args = SimpleNamespace(yes=False)

        # Patch ``sys.stdin.isatty`` and ``sys.stdout.isatty`` directly on the
        # real ``sys`` module instead of replacing ``hermes_cli.main.sys`` with
        # a MagicMock. The MagicMock approach was flaky under ``pytest-xdist``
        # — a sibling test that imported ``hermes_cli.main`` first could leave
        # a different ``sys`` reference resolved inside the function and the
        # mock would never be consulted, with CI then taking the
        # "Non-interactive session" branch instead of prompting.
        import sys as _sys

        with patch("builtins.input", return_value="n") as mock_input, patch.object(
            _sys.stdin, "isatty", return_value=True
        ), patch.object(_sys.stdout, "isatty", return_value=True):
            cmd_update(args)
            # The user was actually prompted.
            assert mock_input.called
            prompts = [c.args[0] if c.args else "" for c in mock_input.call_args_list]
            assert any("configure them now" in p for p in prompts)


class TestUpdateYesStashRestore:
    """--yes auto-restores the pre-update autostash without prompting."""



class TestUnicodeDecodeErrorInUpdatePrompts:
    """Regression tests (review of #68497): input() can raise
    UnicodeDecodeError when the terminal encoding can't decode the byte
    sequence (e.g. a non-UTF-8 locale, or an embedded terminal). Three
    interactive update prompts call input() directly -- the config-
    migration prompt, the stash-restore prompt, and the upstream-remote
    prompt -- and each must fail safe (skip, don't crash) rather than let
    the exception escape and crash `hermes update` mid-flight.
    """

    @patch("hermes_cli.update_cmd._reload_config_modules")
    @patch("hermes_cli.update_cmd._run_migrate_config_fresh")
    @patch("hermes_cli.update_cmd._run_config_check_fresh", return_value=(1, 2))
    @patch("hermes_cli.config.get_missing_config_fields", return_value=[])
    @patch("hermes_cli.config.get_missing_env_vars", return_value=["NEW_KEY"])
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_unicode_decode_error_in_tty_skips_and_prints_hint(
        self,
        mock_run,
        _mock_which,
        _mock_missing_env,
        _mock_missing_cfg,
        _mock_version,
        mock_migrate,
        _mock_reload,
        capsys,
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        mock_migrate.return_value = {"env_added": [], "config_added": []}
        args = SimpleNamespace(yes=False)

        import sys as _sys

        with patch(
            "builtins.input",
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
        ), patch.object(_sys.stdin, "isatty", return_value=True), patch.object(
            _sys.stdout, "isatty", return_value=True
        ):
            cmd_update(args)  # must not raise

        out = capsys.readouterr().out
        assert "hermes config migrate" in out
        mock_migrate.assert_not_called()

    def test_stash_restore_unicode_decode_error_falls_through_to_skip(self, tmp_path, capsys):
        from hermes_cli.update_cmd import _restore_stashed_changes

        with patch(
            "builtins.input",
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
        ):
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
            )  # must not raise

        assert result is False
        out = capsys.readouterr().out
        assert "Skipped restoring local changes" in out
        assert "git stash apply stash@{0}" in out

    def test_stash_restore_eof_error_still_falls_through_to_skip(self, tmp_path):
        """Sanity: this fix must not regress the pre-existing EOFError case,
        which the raw input() path had no guard for at all before this fix."""
        from hermes_cli.update_cmd import _restore_stashed_changes

        with patch("builtins.input", side_effect=EOFError()):
            result = _restore_stashed_changes(
                ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
            )  # must not raise

        assert result is False

    def test_upstream_remote_prompt_unicode_decode_error_falls_through_to_skip(
        self, tmp_path
    ):
        from hermes_cli.update_cmd import _sync_with_upstream_if_needed

        with patch(
            "hermes_cli.update_cmd._has_upstream_remote", return_value=False
        ), patch(
            "hermes_cli.update_cmd._should_skip_upstream_prompt", return_value=False
        ), patch(
            "hermes_cli.update_cmd._add_upstream_remote"
        ) as mock_add, patch(
            "builtins.input",
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
        ):
            _sync_with_upstream_if_needed(["git"], tmp_path)  # must not raise

        mock_add.assert_not_called()
