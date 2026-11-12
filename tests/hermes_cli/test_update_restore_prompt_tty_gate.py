"""Gateway updates must restore their autostash without an interactive question."""

import sys

import pytest

import hermes_cli.main as cli_main
from hermes_cli import update_cmd
from tests.hermes_cli.test_update_scoped_autostash import _behind_clone, _git
from tests.hermes_cli.test_update_target_identity import git, update_tree


@pytest.mark.parametrize("gateway,stdin_tty,stdout_tty,yes", [
    (True, True, True, False),
    (True, False, False, False),
    (False, False, True, False),
    (False, True, False, False),
    (False, True, True, True),
])
def test_noninteractive_update_restores_without_question(tmp_path, monkeypatch, gateway, stdin_tty, stdout_tty, yes):
    work = _behind_clone(tmp_path)
    (work / "pkg" / "new.py").unlink()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", work)
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)

    def no_question(*args, **kwargs):
        raise AssertionError("non-interactive update asked a question")

    monkeypatch.setattr("builtins.input", no_question)
    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=yes, gateway_mode=gateway,
        gw_input_fn=no_question, switch_branch=False, _windows_gateway_resume=None)
    assert plan.auto_stash_ref is not None
    cli_main._restore_stashed_changes(
        ["git"], work, plan.auto_stash_ref, prompt_user=plan.prompt_for_restore, input_fn=no_question)
    assert (work / "a.txt").read_text() == "a\nmine\n"
    assert _git(work, "stash", "list") == ""


def test_interactive_up_to_date_update_still_asks_to_restore(tmp_path, monkeypatch):
    work = _behind_clone(tmp_path)
    (work / "pkg" / "new.py").unlink()
    _git(work, "stash", "push", "-q", "--include-untracked")
    _git(work, "merge", "--ff-only", "-q", "origin/main")
    (work / "a.txt").write_text("a\nmine\n")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", work)
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    plan = update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=False, assume_yes=False, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, _windows_gateway_resume=None)
    assert plan.auto_stash_ref is not None
    assert plan.prompt_for_restore is True


@pytest.mark.parametrize("policy", ["stash", "discard"])
def test_gateway_update_applies_configured_local_change_policy(update_tree, monkeypatch, policy):
    tree = update_tree
    git(tree.clone, "checkout", "-q", "main")
    original = (tree.clone / ".gitignore").read_text()
    (tree.clone / ".gitignore").write_text(original + "# local patch\n")
    tree.args.channel = "main"
    tree.args.gateway = True
    tree.args.yes = False
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {"non_interactive_local_changes": policy})
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    def gateway_answer(prompt, default=""):
        assert "Restore local changes" not in prompt
        return default

    monkeypatch.setattr(update_cmd, "_gateway_prompt", gateway_answer)
    monkeypatch.setattr(cli_main, "_sync_with_upstream_if_needed", update_cmd._sync_with_upstream_if_needed)
    cli_main.cmd_update(tree.args)
    assert git(tree.clone, "rev-parse", "HEAD") == tree.newer
    assert (tree.clone / ".gitignore").read_text() == original + ("# local patch\n" if policy == "stash" else "")
    assert git(tree.clone, "stash", "list") == ""
