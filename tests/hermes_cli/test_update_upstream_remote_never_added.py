"""`hermes update` never creates an `upstream` remote.

The fork-upstream sync used to answer its own prompt with `git remote add upstream`, so an
interactive update on a fork gained a remote pointing at the official repository. The sync now
only compares against an `upstream` remote the user configured: without one, the update runs
against origin alone (no prompt, no remote mutation, no network call).
"""

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd


@pytest.fixture
def fork_without_upstream(tmp_path):
    with patch.object(update_cmd, "_has_upstream_remote", return_value=False), \
            patch.object(update_cmd, "_add_upstream_remote") as add_remote, \
            patch.object(update_cmd.subprocess, "run") as run, \
            patch("builtins.input") as stdin_input:
        yield SimpleNamespace(cwd=tmp_path, add_remote=add_remote, run=run, stdin_input=stdin_input)


class TestNoUpstreamRemoteIsCreated:
    def test_interactive_update_neither_prompts_nor_adds(self, fork_without_upstream):
        assert update_cmd._sync_with_upstream_if_needed(["git"], fork_without_upstream.cwd) is False
        fork_without_upstream.stdin_input.assert_not_called()
        fork_without_upstream.add_remote.assert_not_called()
        fork_without_upstream.run.assert_not_called()

    def test_assume_yes_leaves_remotes_alone(self, fork_without_upstream):
        assert update_cmd._sync_with_upstream_if_needed(
            ["git"], fork_without_upstream.cwd, assume_yes=True) is False
        fork_without_upstream.stdin_input.assert_not_called()
        fork_without_upstream.add_remote.assert_not_called()

    def test_the_gateway_prompt_path_never_asks(self, fork_without_upstream):
        prompts = []

        def gateway_input(prompt, default=""):
            prompts.append(prompt)
            return "y"

        assert update_cmd._sync_with_upstream_if_needed(
            ["git"], fork_without_upstream.cwd, input_fn=gateway_input) is False
        assert prompts == []
        fork_without_upstream.add_remote.assert_not_called()


def test_a_real_checkout_without_upstream_gains_no_remote(tmp_path, monkeypatch):
    """The gate itself, against git rather than a patched predicate."""
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path)}
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True,
                   capture_output=True, env=env)

    assert update_cmd._sync_with_upstream_if_needed(["git"], tmp_path) is False
    remotes = subprocess.run(["git", "remote"], cwd=tmp_path, check=True,
                             capture_output=True, text=True, env=env).stdout
    assert remotes.split() == []
