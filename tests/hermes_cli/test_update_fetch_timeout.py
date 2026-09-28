"""A dead-stalled network fetch ends `hermes update` with an error, never a hang (#93759, #95777)
— and a merely *slow* fetch is allowed to finish.

`_git_run(network=True)` bounds the wait. A transfer (fetch/clone/pull/push) delegates the stall
bound to git itself — `http.lowSpeedLimit`/`lowSpeedTime` — because git can tell "the remote
stopped sending" from "the remote is slow"; a wall-clock cap cannot, and as one it killed healthy
transfers on slow uplinks. Since installs are shallow, every
update re-fetches a whole snapshot and each retry restarts from zero, so that misfire left such an
install permanently unable to update. subprocess.run's timeout stays as an absolute ceiling.
Request/response network git (ls-remote) keeps the elapsed bound; local git stays unbounded.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.update_cmd as update_cmd


def _timeout(cmd, **kwargs):
    if "timeout" in kwargs:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return MagicMock(returncode=0, stdout="ok", stderr="")


def _ok(cmd, **kwargs):
    return MagicMock(returncode=0, stdout="ok", stderr="")


@pytest.fixture
def repo(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))


class TestTransferIsBoundedBySilenceNotElapsedTime:
    def test_fetch_delegates_the_stall_bound_to_git(self, repo):
        with patch.object(update_cmd.subprocess, "run", side_effect=_ok) as run:
            update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

        argv = run.call_args.args[0]
        assert f"http.lowSpeedLimit={update_cmd.NETWORK_GIT_MIN_BYTES_PER_SEC}" in argv
        assert f"http.lowSpeedTime={update_cmd.NETWORK_GIT_TIMEOUT_SECONDS}" in argv
        # The config must precede the subcommand, or git rejects it.
        assert argv.index("-c") < argv.index("fetch")

    def test_a_slow_transfer_is_not_killed_by_the_elapsed_bound(self, repo):
        """The regression: the wall clock must no longer be the transfer's real limit."""
        with patch.object(update_cmd.subprocess, "run", side_effect=_ok) as run:
            update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

        assert run.call_args.kwargs["timeout"] == update_cmd.NETWORK_GIT_MAX_SECONDS
        assert update_cmd.NETWORK_GIT_MAX_SECONDS > update_cmd.NETWORK_GIT_TIMEOUT_SECONDS

    def test_ssh_transfers_keep_only_the_ceiling_and_never_touch_the_user_ssh_transport(self, repo):
        """http.* is inert over ssh; overriding the user's ssh transport to bound it costs more."""
        with patch.object(update_cmd.subprocess, "run", side_effect=_ok) as run:
            update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

        assert "GIT_SSH_COMMAND" not in run.call_args.kwargs["env"]
        assert "core.sshCommand" not in " ".join(run.call_args.args[0])

    def test_the_absolute_ceiling_still_reports_a_failed_run(self, repo):
        """A wedged child is still killed — the guard this bound exists for is intact."""
        with patch.object(update_cmd.subprocess, "run", side_effect=_timeout):
            result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

        assert result.returncode == 124
        assert "timed out" in result.stderr and "fetch" in result.stderr

    def test_check_true_raises_on_the_ceiling(self, repo):
        with patch.object(update_cmd.subprocess, "run", side_effect=_timeout):
            with pytest.raises(subprocess.CalledProcessError) as exc:
                update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True, check=True)
        assert exc.value.returncode == 124


class TestNonTransferGitIsUnchanged:
    def test_ls_remote_keeps_the_elapsed_bound_and_no_transfer_config(self, repo):
        with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
            result = update_cmd._git_run(["git"], ["ls-remote", "origin", "main"], network=True)

        assert result.returncode != 0
        assert "timed out" in result.stderr and "ls-remote" in result.stderr
        assert run.call_args.kwargs["timeout"] == update_cmd.NETWORK_GIT_TIMEOUT_SECONDS
        assert "http.lowSpeedLimit=" not in " ".join(run.call_args.args[0])
        # The no-prompt guard still rides along with the bound.
        assert run.call_args.kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"

    def test_local_git_stays_unbounded_and_check_true_raises(self, repo):
        with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
            assert update_cmd._git_run(["git"], ["rev-parse", "HEAD"]).returncode == 0
            assert "timeout" not in run.call_args.kwargs

            try:
                update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True, check=True)
            except subprocess.CalledProcessError as exc:
                assert exc.returncode == 124
            else:
                raise AssertionError("check=True must raise on a timed-out fetch")
