"""Regression tests for #100179: cron-update three-way restart deadlock.

When `hermes update` runs INSIDE the gateway's own process tree (the
hermes-auto-update cron job), waiting for that gateway to exit is a
circular wait:

  gateway waits on all in-flight work units (#77184)
    -> cron agent session waits on the `hermes update` process
      -> `hermes update` waits on the gateway to exit  [back to A]

The wedged-loop probe cannot break it: the cron session posts activity
every ~180s (process-tool poll return), so it is never marked wedged and
the gateway burns the full 1800s force-drain cap.

The fix: when the target gateway PID is an ancestor of this process,
fire-and-forget (SIGUSR1 + return) instead of drain-waiting.
"""

from unittest.mock import patch

import pytest

linux_only = pytest.mark.linux_only


class TestAncestorDetectionGuard:
    """_is_pid_ancestor_of_current_process is the deadlock discriminator."""

    def test_own_pid_is_ancestor(self):
        import os

        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        assert _is_pid_ancestor_of_current_process(os.getpid()) is True

    def test_parent_pid_is_ancestor(self):
        import os

        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        ppid = os.getppid()
        if ppid <= 1:
            pytest.skip("no meaningful parent in this environment")
        assert _is_pid_ancestor_of_current_process(ppid) is True

    def test_unrelated_pid_is_not_ancestor(self):
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        # PID 0 / negative are never ancestors; a very high unlikely PID isn't
        # either. Use the documented zero/negative contract for determinism.
        assert _is_pid_ancestor_of_current_process(0) is False
        assert _is_pid_ancestor_of_current_process(-5) is False


@linux_only
class TestSelfRestartFireAndForget:
    """_request_gateway_self_restart signals without waiting for exit."""

    def test_refuses_non_ancestor_pid(self):
        from hermes_cli.gateway import _request_gateway_self_restart

        # A non-ancestor must be refused — signalling an unrelated gateway
        # and returning immediately would skip its drain entirely.
        assert _request_gateway_self_restart(0) is False

    def test_signals_ancestor_and_returns_immediately(self):
        """The ancestor path sends SIGUSR1 and does NOT poll for exit."""
        import os
        import signal as _signal

        from hermes_cli import gateway as gw

        sent = []

        def _fake_kill(pid, sig):
            sent.append((pid, sig))

        with patch.object(gw.os, "kill", side_effect=_fake_kill), patch.object(
            gw, "_wait_for_pid_exit",
            side_effect=AssertionError(
                "fire-and-forget must NOT wait for the gateway to exit — "
                "that wait is the #100179 deadlock"
            ),
        ):
            ok = gw._request_gateway_self_restart(os.getpid())

        assert ok is True
        assert sent == [(os.getpid(), _signal.SIGUSR1)]

    def test_graceful_restart_does_wait(self):
        """Contrast: the non-ancestor path DOES drain-wait (unchanged)."""
        import signal as _signal

        from hermes_cli import gateway as gw

        waited = []

        with patch.object(gw.os, "kill"), patch.object(
            gw, "_wait_for_pid_exit",
            side_effect=lambda pid, t: waited.append((pid, t)) or True,
        ):
            ok = gw._graceful_restart_via_sigusr1(4242, drain_timeout=7.0)

        assert ok is True
        assert waited == [(4242, 7.0)], "drain path must still wait for exit"


class TestDrainOrSignalTriage:
    """_drain_or_signal_gateway_for_update routes the three cases correctly."""

    def _patched(self, monkeypatch, *, ancestor, wedged):
        from hermes_cli import gateway as gw

        calls = {"self_restart": [], "escalate": [], "drain": []}
        monkeypatch.setattr(
            gw, "_is_pid_ancestor_of_current_process", lambda pid: ancestor
        )
        monkeypatch.setattr(
            gw,
            "probe_gateway_loop_liveness",
            lambda pid: gw.GATEWAY_LOOP_WEDGED if wedged else "alive",
        )
        monkeypatch.setattr(
            gw,
            "_request_gateway_self_restart",
            lambda pid: calls["self_restart"].append(pid) or True,
        )
        monkeypatch.setattr(
            gw, "_escalate_wedged_gateway", lambda pid: calls["escalate"].append(pid)
        )
        monkeypatch.setattr(
            gw,
            "_graceful_restart_via_sigusr1",
            lambda pid, drain_timeout: calls["drain"].append((pid, drain_timeout))
            or True,
        )
        return calls

    def test_ancestor_gateway_is_signalled_fire_and_forget(self, monkeypatch):
        """In-tree gateway (#100179): SIGUSR1 request, NO drain wait."""
        from hermes_cli.update_cmd import _drain_or_signal_gateway_for_update

        calls = self._patched(monkeypatch, ancestor=True, wedged=False)
        assert _drain_or_signal_gateway_for_update(1234, 900.0, "svc") is True
        assert calls["self_restart"] == [1234]
        assert calls["drain"] == [], "drain-waiting on an ancestor IS the deadlock"
        assert calls["escalate"] == []

    def test_wedged_gateway_is_escalated(self, monkeypatch):
        """Provably-dead loop (#81642): bounded escalation, no drain wait."""
        from hermes_cli.update_cmd import _drain_or_signal_gateway_for_update

        calls = self._patched(monkeypatch, ancestor=False, wedged=True)
        assert _drain_or_signal_gateway_for_update(1234, 900.0, "svc") is True
        assert calls["escalate"] == [1234]
        assert calls["drain"] == []
        assert calls["self_restart"] == []

    def test_live_out_of_tree_gateway_still_drain_waits(self, monkeypatch):
        """Normal out-of-tree update keeps the full graceful drain semantics."""
        from hermes_cli.update_cmd import _drain_or_signal_gateway_for_update

        calls = self._patched(monkeypatch, ancestor=False, wedged=False)
        assert _drain_or_signal_gateway_for_update(1234, 900.0, "svc") is True
        assert calls["drain"] == [(1234, 900.0)]
        assert calls["self_restart"] == []
        assert calls["escalate"] == []
