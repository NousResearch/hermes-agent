"""An unattended Windows console must gate the update prompts off (#92303).

A hidden console — a Scheduled Task with ``-WindowStyle Hidden``, or a
``CREATE_NO_WINDOW`` child that inherits an open stdin handle nobody writes —
reports a TTY pair, so the ``isatty()``-only gates classified the run as
interactive and a bare ``input()`` blocked the update forever: it neither
completed nor failed. #97052 closed the fork-upstream leg with a TTY gate and
no reader threads; the residual case (a TTY pair, no ``--yes``, nobody
present) stayed open. These tests pin the shared ``_interactive_console``
predicate and every prompt site that routes through it: a console window that
exists but is not visible is unattended, no window at all is ConPTY (Windows
Terminal) and stays interactive, and a failed probe fails open.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_cmd_config
from hermes_cli import update_cmd_git
from tests.hermes_cli.test_update_target_identity import git


def _tty(stdin: bool, stdout: bool):
    return (
        patch.object(update_cmd.sys.stdin, "isatty", return_value=stdin),
        patch.object(update_cmd.sys.stdout, "isatty", return_value=stdout),
    )


def _windows_console(monkeypatch, *, hwnd: int, visible: bool):
    """Stand in for the win32 console probes on any host (the #92303 reports are Windows-only)."""
    import ctypes

    monkeypatch.setattr(update_cmd.sys, "platform", "win32")
    monkeypatch.setattr(
        ctypes,
        "windll",
        SimpleNamespace(
            kernel32=SimpleNamespace(GetConsoleWindow=lambda: hwnd),
            user32=SimpleNamespace(IsWindowVisible=lambda _hwnd: visible),
        ),
        raising=False,
    )


class TestInteractiveConsole:
    def test_hidden_windows_console_is_unattended(self, monkeypatch):
        _windows_console(monkeypatch, hwnd=0x1234, visible=False)
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is False

    def test_visible_windows_console_still_prompts(self, monkeypatch):
        _windows_console(monkeypatch, hwnd=0x1234, visible=True)
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is True

    def test_conpty_without_a_console_window_still_prompts(self, monkeypatch):
        # Windows Terminal runs children on ConPTY: GetConsoleWindow() is NULL and a human IS there.
        _windows_console(monkeypatch, hwnd=0, visible=False)
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is True

    def test_probe_failure_fails_open(self, monkeypatch):
        import ctypes

        class _Exploding:
            @property
            def kernel32(self):
                raise OSError("no windll on this host")

        monkeypatch.setattr(update_cmd.sys, "platform", "win32")
        monkeypatch.setattr(ctypes, "windll", _Exploding(), raising=False)
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is True

    def test_posix_tty_still_prompts(self):
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is True

    def test_non_tty_stdio_is_unattended(self):
        p_in, p_out = _tty(False, True)
        with p_in, p_out:
            assert update_cmd._interactive_console() is False


@pytest.fixture
def checkout_with_local_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "git-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    (root / "cli.py").write_text("value = 1\n", encoding="utf8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    head = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", head)
    (root / "cli.py").write_text(
        "value = 2\n", encoding="utf8"
    )  # local edit -> autostash
    return root


class TestStashRestoreGate:
    """``prompt_for_restore`` is the gate the #92303 hang sat behind."""

    def _prepare(self):
        return update_cmd._prepare_checkout_for_update(
            ["git"],
            "main",
            "main",
            is_fork=False,
            assume_yes=False,
            gateway_mode=False,
            gw_input_fn=None,
            switch_branch=False,
            _windows_gateway_resume=None,
        )

    def test_hidden_console_does_not_prompt_for_restore(self, checkout_with_local_edit):
        with patch.object(update_cmd, "_interactive_console", return_value=False):
            plan = self._prepare()
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is False

    def test_attended_console_still_prompts(self, checkout_with_local_edit):
        with patch.object(update_cmd, "_interactive_console", return_value=True):
            plan = self._prepare()
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is True


class TestNonInteractiveLocalChangesConsultation:
    def test_hidden_console_consults_updates_non_interactive_local_changes(
        self, monkeypatch
    ):
        monkeypatch.setattr(update_cmd, "_checkout_version", lambda: "0.0.0")
        monkeypatch.setattr(
            update_cmd,
            "_updates_config",
            lambda: {"non_interactive_local_changes": "discard"},
        )
        args = SimpleNamespace(
            yes=False, keep_stash=False, switch_branch=False, no_gateway_restart=False
        )
        with patch.object(update_cmd, "_interactive_console", return_value=False):
            opts = update_cmd._resolve_update_options(args, gateway_mode=False)
        assert opts.discard_local_changes is True


class TestConfigMigrationPrompt:
    def test_hidden_console_takes_the_safe_auto_path(self):
        with (
            patch.object(update_cmd, "_interactive_console", return_value=False),
            patch("builtins.input") as stdin_input,
        ):
            answer = update_cmd_config._ask_configure_new_options(
                assume_yes=False, gateway_mode=False
            )
        assert answer == "auto"
        stdin_input.assert_not_called()


class TestForkUpstreamPrompt:
    def test_hidden_console_skips_prompt_without_touching_remotes(self, tmp_path):
        with (
            patch.object(update_cmd, "_has_upstream_remote", return_value=False),
            patch.object(
                update_cmd, "_should_skip_upstream_prompt", return_value=False
            ),
            patch.object(
                update_cmd, "_add_upstream_remote", return_value=True
            ) as add_remote,
            patch.object(update_cmd, "_mark_skip_upstream_prompt") as mark_skip,
            patch.object(update_cmd, "_interactive_console", return_value=False),
            patch("builtins.input") as stdin_input,
        ):
            added = update_cmd_git._offer_upstream_remote(
                ["git"], tmp_path, assume_yes=False, input_fn=None
            )
        assert added is False
        stdin_input.assert_not_called()
        add_remote.assert_not_called()
        mark_skip.assert_not_called()
