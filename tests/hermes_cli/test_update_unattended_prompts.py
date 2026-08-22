"""Unattended-update prompt guards (#92303).

Scheduled Task / CI / hidden-window runs keep stdin OPEN but unattended:
``input()`` blocks forever instead of raising EOFError, hanging
``hermes update`` at 'Restore local changes now? [Y/n]'. Interactive
prompts in the update path must detect this and take a safe default.
"""

from io import StringIO
from unittest.mock import patch

from hermes_cli.update_cmd import (
    _prompt_answerable,
    _restore_stashed_changes,
    _sync_with_upstream_if_needed,
)


class _FakeTTY(StringIO):
    def isatty(self):
        return True


class _FakePipe(StringIO):
    def isatty(self):
        return False


def test_prompt_answerable_requires_tty_stdin_and_stdout():
    with patch("sys.stdin", _FakeTTY()), patch("sys.stdout", _FakeTTY()):
        assert _prompt_answerable() is True
    with patch("sys.stdin", _FakePipe()), patch("sys.stdout", _FakeTTY()):
        assert _prompt_answerable() is False
    with patch("sys.stdin", _FakeTTY()), patch("sys.stdout", _FakePipe()):
        assert _prompt_answerable() is False
    with patch("sys.stdin", None):
        assert _prompt_answerable() is False


def test_stash_restore_parks_stash_when_unattended(tmp_path, capsys):
    with patch("builtins.input") as mock_input, patch(
        "hermes_cli.update_cmd._prompt_answerable", return_value=False
    ):
        result = _restore_stashed_changes(
            ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
        )

    assert result is False
    mock_input.assert_not_called()
    out = capsys.readouterr().out
    assert "unattended run" in out
    assert "Skipped restoring local changes" in out
    assert "git stash apply stash@{0}" in out


def test_upstream_prompt_takes_default_when_unattended(tmp_path):
    with patch("hermes_cli.update_cmd._has_upstream_remote", return_value=False), patch(
        "hermes_cli.update_cmd._should_skip_upstream_prompt", return_value=False
    ), patch("hermes_cli.update_cmd._add_upstream_remote") as mock_add, patch(
        "builtins.input"
    ) as mock_input, patch(
        "hermes_cli.update_cmd._prompt_answerable", return_value=False
    ):
        _sync_with_upstream_if_needed(["git"], tmp_path)

    mock_input.assert_not_called()
    mock_add.assert_not_called()


def test_restore_prompt_still_reads_input_when_interactive(tmp_path):
    with patch("builtins.input", return_value="n") as mock_input, patch(
        "hermes_cli.update_cmd._prompt_answerable", return_value=True
    ):
        result = _restore_stashed_changes(
            ["git"], tmp_path, "stash@{0}", prompt_user=True, input_fn=None,
        )

    assert result is False
    mock_input.assert_called_once()
