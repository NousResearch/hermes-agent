"""The autostash restore question must follow the updater's own interactivity gate.

Docs define non-interactive updates (gateway, desktop/chat app, ``--yes``) as never
prompting: ``updates.non_interactive_local_changes`` decides the stash's fate
(``website/docs/reference/cli-commands.md`` "Local source changes",
``website/docs/user-guide/configuration.md``, ``website/docs/getting-started/updating.md``).
``_resolve_update_options`` already implements that definition (``gateway_mode or
assume_yes or not TTY`` → consult the config), but ``_prepare_checkout_for_update``
forced the restore prompt ON whenever ``gateway_mode`` was set — so a gateway-driven
update asked a question no terminal could answer and the forwarder's default "n"
parked the user's local patches. These tests pin ``plan.prompt_for_restore`` against
the same gate: a prompt only when a real terminal is present and nothing already
declared the run non-interactive.
"""
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd
from tests.hermes_cli.test_update_target_identity import git, update_tree


@pytest.fixture
def dirty_checkout(tmp_path, monkeypatch):
    """A git checkout one commit behind origin/main with a dirty tracked file."""
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    (root / "cli.py").write_text("value = 1\n", encoding="utf8")
    git(root, "add", ".")
    git(root, "commit", "-qm", "old")
    old = git(root, "rev-parse", "HEAD")
    (root / "cli.py").write_text("value = 2\n", encoding="utf8")
    git(root, "commit", "-qam", "new")
    tip = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", tip)
    git(root, "checkout", "-q", "--detach", old)
    # Dirty tracked file: the update will stash it, so a restore decision exists.
    (root / "cli.py").write_text("value = 1\n# local patch\n", encoding="utf8")
    return root


def prepare(root, *, assume_yes=False, gateway_mode=False):
    return update_cmd._prepare_checkout_for_update(
        ["git"], "main", update_cmd._current_branch_name(["git"], check=True),
        is_fork=False, assume_yes=assume_yes, gateway_mode=gateway_mode, gw_input_fn=None,
        switch_branch=False, _windows_gateway_resume=None,
    )


def _tty(stdin: bool, stdout: bool):
    return (
        patch.object(update_cmd.sys.stdin, "isatty", return_value=stdin),
        patch.object(update_cmd.sys.stdout, "isatty", return_value=stdout),
    )


class TestRestorePromptInteractivityGate:
    def test_gateway_update_never_prompts_to_restore_the_autostash(self, dirty_checkout):
        """Gateway-driven update: the question must not be asked at all."""
        root = dirty_checkout
        with ExitStack() as stack:
            for ctx in _tty(True, True):  # even a TTY cannot outrank gateway mode
                stack.enter_context(ctx)
            plan = prepare(root, gateway_mode=True)
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is False

    def test_gateway_update_without_a_tty_restores_instead_of_prompting(self, dirty_checkout):
        """The real-world gateway case: no terminal exists to answer the question."""
        root = dirty_checkout
        plan = prepare(root, gateway_mode=True)  # pytest stdio is not a TTY
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is False

    def test_interactive_terminal_update_still_prompts(self, dirty_checkout):
        """A real terminal keeps the ask-before-restore behaviour unchanged."""
        root = dirty_checkout
        with ExitStack() as stack:
            for ctx in _tty(True, True):
                stack.enter_context(ctx)
            plan = prepare(root, gateway_mode=False)
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is True

    def test_non_tty_cli_run_does_not_prompt(self, dirty_checkout):
        """Desktop-app/agent hand-offs run without a TTY and must not be asked."""
        root = dirty_checkout
        plan = prepare(root, gateway_mode=False)
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is False

    def test_assume_yes_still_skips_the_prompt(self, dirty_checkout):
        root = dirty_checkout
        with ExitStack() as stack:
            for ctx in _tty(True, True):
                stack.enter_context(ctx)
            plan = prepare(root, assume_yes=True, gateway_mode=True)
        assert plan.auto_stash_ref is not None
        assert plan.prompt_for_restore is False

    def test_clean_tree_never_prompts(self, dirty_checkout):
        root = dirty_checkout
        git(root, "checkout", "--", "cli.py")
        with ExitStack() as stack:
            for ctx in _tty(True, True):
                stack.enter_context(ctx)
            plan = prepare(root, gateway_mode=False)
        assert plan.auto_stash_ref is None
        assert plan.prompt_for_restore is False