"""Regression tests for ``hermes update --stay-on-branch`` (#80041).

Desktop product updates run from a checkout that may be parked on a feature
branch. ``hermes update`` switches to the tracked branch/channel target and the
desktop rebuild then packages THAT checkout — switching back to the feature
branch (or auto-restoring its stashed WIP onto the target) is what left
desktop users on an old version while the CLI printed "already up to date".

With the flag the update must end on the target branch and the feature-branch
autostash stays parked for manual recovery. These tests pin that contract
against the decomposed update flow (``_resolve_update_options`` /
``_finish_already_up_to_date`` / ``_pull_updates``).

RED proof: before #80041 the park path did not exist —
``_restore_stashed_changes`` ran unconditionally whenever a stash existed, so
``test_parks_stash_when_stay_on_branch`` (``assert_not_called`` + the manual
recovery notice) could not have passed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import update_cmd


def _plan(**overrides):
    base = dict(
        auto_stash_ref="refs/stash@{0}",
        prompt_for_restore=False,
        parked_branch_switched=False,
        switch_block_reason=None,
        upstream_checked=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestStayOnBranchOptions:
    """The flag resolves into _UpdateOptions (getattr default keeps old callers whole)."""

    def test_flag_parsed_into_options(self, monkeypatch):
        monkeypatch.setattr(update_cmd, "_checkout_version", lambda: "v0.22.2")
        monkeypatch.setattr(update_cmd, "_updates_config", dict)
        args = SimpleNamespace(
            yes=True, keep_stash=False, switch_branch=False,
            stay_on_branch=True, no_gateway_restart=False,
        )
        opts = update_cmd._resolve_update_options(args, gateway_mode=False)
        assert opts.stay_on_branch is True

    def test_flag_defaults_false_when_absent(self, monkeypatch):
        monkeypatch.setattr(update_cmd, "_checkout_version", lambda: "v0.22.2")
        monkeypatch.setattr(update_cmd, "_updates_config", dict)
        args = SimpleNamespace(
            yes=True, keep_stash=False, switch_branch=False, no_gateway_restart=False,
        )
        opts = update_cmd._resolve_update_options(args, gateway_mode=False)
        assert opts.stay_on_branch is False


class TestFinishAlreadyUpToDateStashSettle:
    """"Already up to date" path: the flag parks the stash instead of restoring
    feature-branch WIP onto the tracked branch."""

    def test_parks_stash_when_stay_on_branch(self, capsys):
        m = MagicMock()
        completion = {"expected_sha": "deadbeef"}
        with (
            patch.object(update_cmd, "_m", return_value=m),
            patch.object(update_cmd, "_capture_head_sha", return_value="abc1234"),
            patch.object(update_cmd, "_complete_source_update") as complete,
        ):
            update_cmd._finish_already_up_to_date(
                ["git"], "main", "feat/while-i-work", _plan(),
                gw_input_fn=None, completion_request=completion,
                stay_on_branch=True,
            )
        m._restore_stashed_changes.assert_not_called()
        m._discard_stashed_changes.assert_not_called()
        out = capsys.readouterr().out
        assert "preserved in stash" in out
        assert "git stash apply refs/stash@{0}" in out
        # The completion still lands — parking the stash must not skip the tail.
        assert complete.called

    def test_restores_stash_without_flag(self):
        m = MagicMock()
        with (
            patch.object(update_cmd, "_m", return_value=m),
            patch.object(update_cmd, "_capture_head_sha", return_value="abc1234"),
            patch.object(update_cmd, "_complete_source_update"),
        ):
            update_cmd._finish_already_up_to_date(
                ["git"], "main", "feat/while-i-work", _plan(),
                gw_input_fn=None, completion_request=None,
                stay_on_branch=False,
            )
        m._restore_stashed_changes.assert_called_once()

    def test_pulled_path_wires_the_flag(self):
        """_pull_updates accepts stay_on_branch (its finally chain parks the
        stash on success) and both call sites pass opts.stay_on_branch — a
        dropped kwarg during a refactor silently re-enables auto-restore."""
        import inspect

        sig = inspect.signature(update_cmd._pull_updates)
        assert "stay_on_branch" in sig.parameters
        assert sig.parameters["stay_on_branch"].default is False
        src = inspect.getsource(update_cmd._pull_updates)
        assert "elif stay_on_branch:" in src
        impl = inspect.getsource(update_cmd)
        assert impl.count("stay_on_branch=opts.stay_on_branch") == 2


class TestParserAcceptsFlag:
    """`hermes update --stay-on-branch` must survive argparse (dest stay_on_branch)."""

    @pytest.fixture()
    def update_parser(self):
        import argparse

        from hermes_cli.subcommands import update as update_sub

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        update_sub.build_update_parser(sub, cmd_update=lambda _args: None)
        return parser

    def test_flag_parses_to_true(self, update_parser):
        ns = update_parser.parse_args(["update", "--stay-on-branch"])
        assert ns.stay_on_branch is True

    def test_flag_defaults_false(self, update_parser):
        ns = update_parser.parse_args(["update"])
        assert ns.stay_on_branch is False
