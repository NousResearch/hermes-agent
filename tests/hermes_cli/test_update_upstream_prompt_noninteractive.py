"""The fork-upstream prompt must never block a non-interactive update (#60240).

`_sync_with_upstream_if_needed` asks "Add official repo as 'upstream' remote?"
on fork checkouts with no upstream remote. In unattended contexts (CI, cron,
the desktop updater hand-off) stdin is open but nobody answers, so a bare
``input()`` blocks forever. These tests pin the gate: under ``assume_yes`` or
a non-TTY stdio pair the prompt is skipped as a decline WITHOUT persisting the
skip marker or touching git remotes, and both update call sites forward the
interaction state.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd


@pytest.fixture
def fork_without_upstream(tmp_path):
    with patch.object(
        update_cmd, "_has_upstream_remote", return_value=False
    ), patch.object(
        update_cmd, "_should_skip_upstream_prompt", return_value=False
    ), patch.object(
        update_cmd, "_add_upstream_remote", return_value=True
    ) as add_remote, patch.object(
        update_cmd, "_mark_skip_upstream_prompt"
    ) as mark_skip, patch("builtins.input") as stdin_input:
        yield SimpleNamespace(
            cwd=tmp_path,
            add_remote=add_remote,
            mark_skip=mark_skip,
            stdin_input=stdin_input,
        )


def _tty(stdin: bool, stdout: bool):
    return (
        patch.object(update_cmd.sys.stdin, "isatty", return_value=stdin),
        patch.object(update_cmd.sys.stdout, "isatty", return_value=stdout),
    )


class TestUpstreamPromptNonInteractive:
    def test_assume_yes_skips_prompt_without_touching_remotes(
        self, fork_without_upstream, capsys
    ):
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            update_cmd._sync_with_upstream_if_needed(
                ["git"], fork_without_upstream.cwd, assume_yes=True
            )

        fork_without_upstream.stdin_input.assert_not_called()
        fork_without_upstream.add_remote.assert_not_called()
        fork_without_upstream.mark_skip.assert_not_called()
        assert "Skipping upstream setup" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "stdin_tty,stdout_tty", [(False, False), (False, True), (True, False)]
    )
    def test_non_tty_skips_prompt_without_persisting_decline(
        self, fork_without_upstream, capsys, stdin_tty, stdout_tty
    ):
        p_in, p_out = _tty(stdin_tty, stdout_tty)
        with p_in, p_out:
            update_cmd._sync_with_upstream_if_needed(
                ["git"], fork_without_upstream.cwd
            )

        fork_without_upstream.stdin_input.assert_not_called()
        fork_without_upstream.add_remote.assert_not_called()
        fork_without_upstream.mark_skip.assert_not_called()
        assert "Skipping upstream setup" in capsys.readouterr().out

    def test_gateway_prompt_routes_through_input_fn(self, fork_without_upstream):
        prompts = []

        def gw_input(prompt, default=""):
            prompts.append((prompt, default))
            return "n"

        p_in, p_out = _tty(False, False)
        with p_in, p_out:
            update_cmd._sync_with_upstream_if_needed(
                ["git"], fork_without_upstream.cwd, input_fn=gw_input
            )

        assert prompts == [("Add official repo as 'upstream' remote? [y/N]", "n")]
        fork_without_upstream.stdin_input.assert_not_called()
        fork_without_upstream.add_remote.assert_not_called()
        fork_without_upstream.mark_skip.assert_called_once_with()

    def test_interactive_decline_still_persists_marker(self, fork_without_upstream):
        fork_without_upstream.stdin_input.return_value = "n"
        p_in, p_out = _tty(True, True)
        with p_in, p_out:
            update_cmd._sync_with_upstream_if_needed(
                ["git"], fork_without_upstream.cwd
            )

        fork_without_upstream.stdin_input.assert_called_once()
        fork_without_upstream.add_remote.assert_not_called()
        fork_without_upstream.mark_skip.assert_called_once_with()

    def test_interactive_accept_adds_upstream(self, fork_without_upstream):
        fork_without_upstream.stdin_input.return_value = "y"
        with patch.object(
            update_cmd, "_count_commits_between", return_value=-1
        ), patch.object(update_cmd.subprocess, "run"):
            p_in, p_out = _tty(True, True)
            with p_in, p_out:
                update_cmd._sync_with_upstream_if_needed(
                    ["git"], fork_without_upstream.cwd
                )

        fork_without_upstream.add_remote.assert_called_once_with(
            ["git"], fork_without_upstream.cwd
        )
        fork_without_upstream.mark_skip.assert_not_called()
