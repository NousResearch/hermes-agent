"""Regression for #74973 — `hermes update` must not leave the gateway down.

On macOS the update's launchd branch guarded the restart behind
``launchctl list <label>`` exiting 0. A job that has been *booted out* of
launchd exits non-zero there, so the whole restart branch was skipped — with
no ``else`` and no message. The update printed ``✓ Update complete!`` and
exited 0 while the gateway was stopped *and* deregistered, which ``KeepAlive``
cannot recover because the job definition is gone. Messaging adapters and
cron stayed dark until someone manually ran ``hermes gateway restart``.

``launchctl list`` is also not a reliable loaded/unloaded classifier: it is
session-scoped and can exit non-zero while the job is alive in its gui/user
domain (PR #75021 review). The fix therefore does not classify at all — when
the plist exists it always calls ``launchd_restart()``, which drains a live
PID, kickstarts with ``-k``, and falls back to bootout/bootstrap/kickstart
when the job is genuinely unloaded.
"""

from __future__ import annotations

import subprocess

import pytest

from hermes_cli import update_cmd


class _FakePlist:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


@pytest.fixture
def launchd(monkeypatch):
    """Stub hermes_cli.gateway so no real launchctl call is made."""
    calls: list[str] = []
    state = {"plist": _FakePlist(True), "restart_exc": None}
    subprocess_calls: list[list] = []

    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway", raising=False)
    monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: state["plist"], raising=False)

    def fake_restart():
        if state["restart_exc"] is not None:
            raise state["restart_exc"]
        calls.append("restart")

    monkeypatch.setattr(gateway_mod, "launchd_restart", fake_restart, raising=False)

    def fake_run(*args, **kwargs):
        subprocess_calls.append(args[0] if args else [])
        return subprocess.CompletedProcess(args=args[0] if args else [], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    return calls, state, subprocess_calls


class TestLaunchdRestartAfterUpdate:
    def test_plist_present_always_restarts_without_classifying(self, launchd, capsys):
        """The restart must not be gated on `launchctl list`.

        `list` can exit non-zero while the job is alive in its domain
        (`launchctl print gui/<uid>/<label>` reports state=running with a
        PID). Routing that state to a plain start would leave the old-code
        process running, because `kickstart` without `-k` does not terminate
        a running service. The helper therefore performs no list-based
        classification at all — `launchd_restart()` handles every
        plist-present state.
        """
        calls, state, subprocess_calls = launchd

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == (["ai.hermes.gateway"], [])
        assert calls == ["restart"]
        # No `launchctl list` classification happens in this helper.
        assert subprocess_calls == []
        assert "NOT running" not in capsys.readouterr().out

    def test_restart_failure_warns_that_gateway_is_down(self, launchd, capsys):
        calls, state, _ = launchd
        state["restart_exc"] = subprocess.CalledProcessError(
            returncode=1, cmd=["launchctl", "kickstart"], stderr="kickstart refused"
        )

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], ["ai.hermes.gateway"])
        out = capsys.readouterr().out
        assert "Gateway restart failed" in out
        assert "kickstart refused" in out
        assert "hermes gateway restart" in out

    @pytest.mark.parametrize(
        "exc",
        [
            FileNotFoundError("launchctl"),
            subprocess.TimeoutExpired(cmd=["launchctl", "kickstart"], timeout=90),
        ],
    )
    def test_launchctl_unusable_is_not_swallowed(self, launchd, capsys, exc):
        """A missing binary or a timeout used to `pass` silently."""
        calls, state, _ = launchd
        state["restart_exc"] = exc

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], ["ai.hermes.gateway"])
        assert calls == []
        out = capsys.readouterr().out
        assert "Could not restart the gateway" in out
        assert "hermes gateway restart" in out

    def test_no_plist_is_not_a_launchd_install(self, launchd, capsys):
        """No service definition → nothing to restart, and nothing to warn about."""
        calls, state, _ = launchd
        state["plist"] = _FakePlist(False)

        assert update_cmd._restart_launchd_gateway_after_update(supervision_verify=False) == ([], [])
        assert calls == []
        assert capsys.readouterr().out == ""


# `launchctl print gui/<uid>/<label>` excerpt for a running service, matching
# the real output shape (tab-indented, lowercase `pid = <N>`).
_PRINT_OUTPUT_RUNNING = """\
ai.hermes.gateway = {
\tactive count = 1
\tpath = /Users/u/Library/LaunchAgents/ai.hermes.gateway.plist
\tstate = running
\tpid = 59038
\tprogram = /Users/u/.hermes/bin/hermes
}
"""


class TestServicePidSweepExclusion:
    """Regression for the PR #75021 review: `_get_service_pids()` must not
    rely on `launchctl list` alone.

    In the session-scoped failure state (`list` exits non-zero while the
    domain-qualified `print` reports a positive PID) the launchd-owned
    gateway PID was missing from the exclusion set, so the post-update
    manual-gateway sweep could kill the process launchd just (re)started.
    """

    @pytest.fixture
    def macos_launchd(self, monkeypatch):
        import hermes_cli.gateway as gateway_mod

        state = {"list_rc": 1, "print_rc": 0, "print_out": _PRINT_OUTPUT_RUNNING}

        monkeypatch.setattr(gateway_mod, "supports_systemd_services", lambda: False)
        monkeypatch.setattr(gateway_mod, "is_macos", lambda: True)
        monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway")
        monkeypatch.setattr(gateway_mod, "_launchd_domain", lambda: "gui/501")

        def fake_run(argv, **kwargs):
            if argv[:2] == ["launchctl", "list"]:
                return subprocess.CompletedProcess(argv, state["list_rc"], stdout="", stderr="")
            if argv[:2] == ["launchctl", "print"]:
                return subprocess.CompletedProcess(
                    argv, state["print_rc"], stdout=state["print_out"], stderr=""
                )
            raise AssertionError(f"unexpected subprocess call: {argv}")

        monkeypatch.setattr(gateway_mod.subprocess, "run", fake_run)
        return state

    def test_list_failure_falls_back_to_domain_print(self, macos_launchd):
        """`list` rc=1, `print` reports pid 59038 → the PID is still excluded."""
        from hermes_cli.gateway import _get_service_pids

        assert 59038 in _get_service_pids()

    def test_both_interfaces_negative_means_no_pid(self, macos_launchd):
        macos_launchd["print_rc"] = 113  # job genuinely not found in the domain

        from hermes_cli.gateway import _get_service_pids

        assert _get_service_pids() == set()

    def test_registered_but_not_running_has_no_pid_line(self, macos_launchd):
        macos_launchd["print_out"] = _PRINT_OUTPUT_RUNNING.replace("\tpid = 59038\n", "")

        from hermes_cli.gateway import _get_service_pids

        assert _get_service_pids() == set()


class TestParseLaunchdPidFromPrintOutput:
    def test_running_service(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output(_PRINT_OUTPUT_RUNNING) == 59038

    def test_no_pid_line(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output("state = not running\n") is None

    def test_nonpositive_pid_is_ignored(self):
        from hermes_cli.gateway import _parse_launchd_pid_from_print_output

        assert _parse_launchd_pid_from_print_output("\tpid = -1\n") is None
