"""Tests for `hermes update --yes / -y` — assume yes for interactive prompts.

Covers:
  1. argparse parses the flag
  2. Config-migration prompt is auto-answered (no input() call) and migrate_config
     runs with interactive=False so API-key prompts are skipped
  3. Autostash restore prompt is auto-answered (prompt_for_restore == False, no
     input() call) and the stash is applied automatically
"""

import importlib
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# Other tests in the suite (notably ``test_env_loader.py`` and
# ``test_skills_subparser.py``) evict ``hermes_cli.main`` from
# ``sys.modules`` mid-run.  When that happens on the same xdist worker
# before we run, a module-top ``from hermes_cli.main import cmd_update``
# binds against the *old* module object — and ``cmd_update.__globals__``
# stays pointed at the old ``hermes_cli.main.__dict__``.
# ``@patch("hermes_cli.main.X")`` then patches the *new* module's
# attribute, but ``cmd_update`` still resolves every internal call
# (``_install_hangup_protection``, ``_cmd_update_impl``,
# ``_stash_local_changes_if_needed``, ``_restore_stashed_changes``, ``sys``,
# …) via the stale ``__globals__``, so every patch is a no-op and the
# real production code runs through.  This is the same xdist pollution
# that ``test_update_stale_dashboard.py`` works around — see the
# autouse fixture there for the original write-up.
#
# Get ``cmd_update`` from the *live* module inside each test instead.
def _live_cmd_update():
    live = sys.modules.get("hermes_cli.main")
    if live is None:
        live = importlib.import_module("hermes_cli.main")
    return live.cmd_update


@pytest.fixture(autouse=True)
def _ensure_hermes_cli_main_loaded():
    """Make sure ``hermes_cli.main`` is loaded before tests in this file run.

    If a prior test on the worker evicted it, re-import here so
    ``@patch("hermes_cli.main.X")`` targets the same module that
    ``_live_cmd_update()`` will resolve against.
    """
    if "hermes_cli.main" not in sys.modules:
        importlib.import_module("hermes_cli.main")
    yield


# `cmd_update` wraps `_cmd_update_impl` in `_install_hangup_protection`, which
# replaces ``sys.stdout`` / ``sys.stderr`` with ``_UpdateOutputStream`` mirrors
# during the update.  In tests that patch ``hermes_cli.main.sys`` to make
# isatty() return True, that wrapping subverts the patch on stdout: the
# wrapper stores the original (still-mocked) stream as ``_original`` but
# replaces ``sys.stdout`` with itself, so subsequent ``sys.stdout.isatty()``
# calls go through the wrapper's ``isatty()``, whose return value depends on
# whether earlier writes flagged ``_original_broken``.  Bypass the wrapping
# with a no-op state so isatty() answers the mock directly.
_HANGUP_PROTECTION_NOOP_STATE = {
    "prev_stdout": None,
    "prev_stderr": None,
    "log_file": None,
    "installed": False,
}


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

    @patch("hermes_cli.main._finalize_update_output")
    @patch(
        "hermes_cli.main._install_hangup_protection",
        return_value=dict(_HANGUP_PROTECTION_NOOP_STATE),
    )
    @patch("hermes_cli.config.migrate_config")
    @patch("hermes_cli.config.check_config_version", return_value=(1, 2))
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
        _mock_install_hangup,
        _mock_finalize,
        capsys,
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        mock_migrate.return_value = {"env_added": [], "config_added": []}

        args = SimpleNamespace(yes=True)

        with patch("builtins.input") as mock_input:
            _live_cmd_update()(args)
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

    @patch("hermes_cli.main._finalize_update_output")
    @patch(
        "hermes_cli.main._install_hangup_protection",
        return_value=dict(_HANGUP_PROTECTION_NOOP_STATE),
    )
    @patch("hermes_cli.config.migrate_config")
    @patch("hermes_cli.config.check_config_version", return_value=(1, 2))
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
        _mock_install_hangup,
        _mock_finalize,
        capsys,
    ):
        """Regression guard: without --yes, the TTY prompt path still fires."""
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="1"
        )
        mock_migrate.return_value = {"env_added": [], "config_added": []}

        args = SimpleNamespace(yes=False)

        with patch("builtins.input", return_value="n") as mock_input, patch(
            "hermes_cli.main.sys"
        ) as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            mock_sys.stdout.isatty.return_value = True
            _live_cmd_update()(args)
            # The user was actually prompted.
            assert mock_input.called
            prompts = [c.args[0] if c.args else "" for c in mock_input.call_args_list]
            assert any("configure them now" in p for p in prompts)


class TestUpdateYesStashRestore:
    """--yes auto-restores the pre-update autostash without prompting."""

    @patch("hermes_cli.main._finalize_update_output")
    @patch(
        "hermes_cli.main._install_hangup_protection",
        return_value=dict(_HANGUP_PROTECTION_NOOP_STATE),
    )
    @patch("hermes_cli.main._restore_stashed_changes")
    @patch(
        "hermes_cli.main._stash_local_changes_if_needed",
        return_value="stash@{0}",
    )
    @patch("hermes_cli.config.check_config_version", return_value=(1, 1))
    @patch("hermes_cli.config.get_missing_config_fields", return_value=[])
    @patch("hermes_cli.config.get_missing_env_vars", return_value=[])
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_yes_restores_stash_without_prompting(
        self,
        mock_run,
        _mock_which,
        _mock_missing_env,
        _mock_missing_cfg,
        _mock_version,
        _mock_stash,
        mock_restore,
        _mock_install_hangup,
        _mock_finalize,
        capsys,
    ):
        # Not on main → cmd_update switches to main → autostash fires.
        mock_run.side_effect = _make_run_side_effect(
            branch="feature-branch", verify_ok=True, commit_count="1", dirty=True
        )

        args = SimpleNamespace(yes=True)

        _live_cmd_update()(args)

        # _restore_stashed_changes was called, and called with prompt_user=False
        # every time (so the user never sees "Restore local changes now?").
        assert mock_restore.called
        for call in mock_restore.call_args_list:
            assert call.kwargs.get("prompt_user") is False, (
                f"Expected prompt_user=False under --yes, got {call.kwargs}"
            )
