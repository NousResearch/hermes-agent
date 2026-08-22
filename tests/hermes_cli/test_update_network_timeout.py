"""Every git network call in ``hermes update`` must be bounded.

``hermes update`` and ``hermes update --check`` shell out to git for seven
remote operations (fetch / pull / push).  None of them passed ``timeout=``,
so a remote that completes the TCP handshake and then stops responding —
captive-portal Wi-Fi, a half-up VPN, a filtering corporate proxy, a throttled
forge — left the CLI printing ``→ Fetching updates...`` forever, with Ctrl-C
the only exit.

Two invariants are pinned here:

1. every remote-talking invocation carries a positive ``timeout`` kwarg;
2. a ``subprocess.TimeoutExpired`` is handled, never propagated as a
   traceback — the user-facing entry points exit non-zero with a diagnostic,
   while the optional fork/upstream sync degrades and lets the update
   continue.

The asymmetry in (2) is deliberate: the upstream sync is a convenience layered
on top of the user's actual update, so a stalled *upstream* must not block it,
whereas a stalled *origin* means there is nothing to update against.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update
from hermes_cli.update_cmd import (
    _GIT_NETWORK_TIMEOUT_SECONDS,
    _sync_fork_with_upstream,
    _sync_with_upstream_if_needed,
)

# Verbs that actually talk to a remote.  ``git stash push`` and
# ``git merge --ff-only`` are purely local and are intentionally excluded.
_NETWORK_VERBS = ("fetch", "pull", "push")


def _joined(call) -> str:
    return " ".join(str(part) for part in call.args[0])


def _network_calls(mock_run) -> list:
    """Return the ``subprocess.run`` calls that reach a remote."""
    found = []
    for call in mock_run.call_args_list:
        if not call.args:
            continue
        joined = _joined(call)
        if "stash" in joined:  # `git stash push` never leaves the machine
            continue
        words = joined.split()
        if any(verb in words for verb in _NETWORK_VERBS):
            found.append(call)
    return found


def _assert_all_bounded(mock_run, *, expected_at_least: int) -> None:
    calls = _network_calls(mock_run)
    assert len(calls) >= expected_at_least, [_joined(c) for c in calls]
    for call in calls:
        timeout = call.kwargs.get("timeout")
        assert timeout is not None, f"unbounded git network call: {_joined(call)}"
        assert timeout > 0, f"non-positive timeout on: {_joined(call)}"


def _drain(capsys) -> tuple:
    """Return ``(stdout, stdout + stderr)`` from ``capsys``.

    The diagnostics under test are printed to stdout, but a leaked traceback
    or an unhandled exception surfaces on *stderr* — so "no traceback" has to
    be asserted against both streams, or the assertion is false confidence.
    """
    captured = capsys.readouterr()
    return captured.out, captured.out + captured.err


def _check_side_effect(
    *,
    upstream_fetch_ok: bool = True,
    timeout_on: str | None = None,
    commit_count: str = "0",
):
    """Drive ``_cmd_update_check``'s git pipeline.

    ``timeout_on`` is a substring; the matching invocation raises
    ``subprocess.TimeoutExpired`` instead of returning.
    """

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(part) for part in cmd)

        if timeout_on and timeout_on in joined:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

        if "fetch" in joined and "upstream" in joined:
            rc = 0 if upstream_fetch_ok else 128
            err = "" if upstream_fetch_ok else "fatal: 'upstream' does not appear to be a git repository\n"
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=err)

        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


class TestUpdateCheckFetchesAreBounded:
    """``hermes update --check`` — the three ``_cmd_update_check`` fetches."""

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_upstream_fetch_is_bounded(self, mock_run, _method):
        mock_run.side_effect = _check_side_effect()

        cmd_update(SimpleNamespace(check=True, branch=None))

        _assert_all_bounded(mock_run, expected_at_least=1)
        assert any("upstream" in _joined(c) for c in _network_calls(mock_run))

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_origin_fallback_fetch_is_bounded(self, mock_run, _method):
        """Upstream missing → origin fallback must be bounded too."""
        mock_run.side_effect = _check_side_effect(upstream_fetch_ok=False)

        cmd_update(SimpleNamespace(check=True, branch=None))

        calls = _network_calls(mock_run)
        _assert_all_bounded(mock_run, expected_at_least=2)
        assert any(
            "fetch" in _joined(c) and "origin" in _joined(c) for c in calls
        ), [_joined(c) for c in calls]

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_non_default_branch_fetch_is_bounded(self, mock_run, _method):
        """``--check --branch <name>`` skips upstream and fetches origin directly."""
        mock_run.side_effect = _check_side_effect()

        cmd_update(SimpleNamespace(check=True, branch="bb/gui"))

        calls = _network_calls(mock_run)
        _assert_all_bounded(mock_run, expected_at_least=1)
        assert not any("upstream" in _joined(c) for c in calls), [_joined(c) for c in calls]


class TestUpdateCheckTimeoutHandling:
    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_upstream_timeout_falls_back_to_origin(self, mock_run, _method, capsys):
        """A stalled upstream is treated like a failed upstream: try origin."""
        mock_run.side_effect = _check_side_effect(timeout_on="fetch upstream")

        cmd_update(SimpleNamespace(check=True, branch=None))

        out, both = _drain(capsys)
        assert "Traceback" not in both
        assert "TimeoutExpired" not in both
        assert "timed out" in out
        assert "falling back to origin" in out
        assert any(
            "fetch" in _joined(c) and "origin" in _joined(c)
            for c in _network_calls(mock_run)
        )

    @patch("hermes_cli.config.detect_install_method", return_value="git")
    @patch("subprocess.run")
    def test_origin_timeout_exits_cleanly(self, mock_run, _method, capsys):
        """A stalled origin has no fallback: diagnose and exit 1, no traceback."""
        mock_run.side_effect = _check_side_effect(timeout_on="fetch origin")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(SimpleNamespace(check=True, branch="bb/gui"))
        assert exc_info.value.code == 1

        out, both = _drain(capsys)
        assert "Traceback" not in both
        assert "TimeoutExpired" not in both
        assert "Timed out" in out
        assert str(_GIT_NETWORK_TIMEOUT_SECONDS) in out


def _apply_side_effect(*, timeout_on: str | None = None, commit_count: str = "0"):
    """Drive ``_cmd_update_impl`` far enough to exercise the apply-path fetch."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(part) for part in cmd)

        if timeout_on and timeout_on in joined:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="main\n", stderr="")

        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


class TestUpdateApplyFetch:
    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_apply_fetch_is_bounded(self, mock_run, _which):
        mock_run.side_effect = _apply_side_effect()

        cmd_update(SimpleNamespace(branch="main"))

        _assert_all_bounded(mock_run, expected_at_least=1)

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_apply_fetch_timeout_exits_cleanly(self, mock_run, _which, capsys):
        mock_run.side_effect = _apply_side_effect(timeout_on="fetch origin")

        with pytest.raises(SystemExit) as exc_info:
            cmd_update(SimpleNamespace(branch="main"))
        assert exc_info.value.code == 1

        out, both = _drain(capsys)
        assert "Traceback" not in both
        assert "TimeoutExpired" not in both
        assert "Timed out" in out


def _upstream_sync_side_effect(
    *,
    timeout_on: str | None = None,
    origin_ahead: str = "0",
    upstream_ahead: str = "3",
):
    """Drive ``_sync_with_upstream_if_needed`` down to the fetch/pull/push."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(part) for part in cmd)

        if timeout_on and timeout_on in joined:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 1))

        # `_count_commits_between(base, head)` → `rev-list --count base..head`
        if "rev-list" in joined and "upstream/main..origin/main" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{origin_ahead}\n", stderr="")
        if "rev-list" in joined and "origin/main..upstream/main" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{upstream_ahead}\n", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


class TestUpstreamSyncIsBounded:
    """The fork-sync path: ``fetch upstream``, ``pull upstream``, ``push origin``."""

    @patch("subprocess.run")
    def test_fetch_pull_and_push_are_all_bounded(self, mock_run, tmp_path):
        mock_run.side_effect = _upstream_sync_side_effect()

        _sync_with_upstream_if_needed(["git"], tmp_path)

        calls = _network_calls(mock_run)
        joined = [_joined(c) for c in calls]
        assert any("fetch upstream main" in j for j in joined), joined
        assert any("pull --ff-only upstream main" in j for j in joined), joined
        assert any("push origin main" in j for j in joined), joined
        _assert_all_bounded(mock_run, expected_at_least=3)

    @patch("subprocess.run")
    def test_fork_push_is_bounded(self, mock_run, tmp_path):
        mock_run.side_effect = _upstream_sync_side_effect()

        assert _sync_fork_with_upstream(["git"], tmp_path) is True

        _assert_all_bounded(mock_run, expected_at_least=1)


class TestUpstreamSyncDegradesOnTimeout:
    """A stalled upstream must be skipped, not fatal — the update goes on."""

    @patch("subprocess.run")
    def test_upstream_fetch_timeout_skips_sync(self, mock_run, tmp_path, capsys):
        mock_run.side_effect = _upstream_sync_side_effect(timeout_on="fetch upstream")

        # Must return normally — no SystemExit, no TimeoutExpired escaping.
        _sync_with_upstream_if_needed(["git"], tmp_path)

        out, both = _drain(capsys)
        assert "Traceback" not in both
        assert "TimeoutExpired" not in both
        assert "timed out" in out
        assert "Skipping upstream sync" in out
        # The pull must not have been attempted after the fetch stalled.
        assert not any("pull" in _joined(c) for c in _network_calls(mock_run))

    @patch("subprocess.run")
    def test_upstream_pull_timeout_skips_sync(self, mock_run, tmp_path, capsys):
        mock_run.side_effect = _upstream_sync_side_effect(timeout_on="pull --ff-only")

        _sync_with_upstream_if_needed(["git"], tmp_path)

        out, both = _drain(capsys)
        assert "Traceback" not in both
        assert "TimeoutExpired" not in both
        assert "timed out" in out
        assert "Skipping upstream sync" in out
        # Bailed before the fork push.
        assert not any("push" in _joined(c) for c in _network_calls(mock_run))

    @patch("subprocess.run")
    def test_fork_push_timeout_degrades_to_false(self, mock_run, tmp_path):
        """Contract test, not a regression test.

        ``_sync_fork_with_upstream`` already had a catch-all ``except``, so
        this passes with or without the fix — it pins the degrade contract
        that makes bounding the push safe.  Its companion,
        ``test_fork_push_is_bounded``, is the one that is red without the fix.
        """
        mock_run.side_effect = _upstream_sync_side_effect(timeout_on="push origin")

        assert _sync_fork_with_upstream(["git"], tmp_path) is False
