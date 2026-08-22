"""Regression for #78574 — a crashed gateway-restart phase must not stay silent.

``hermes update`` wrapped its entire gateway auto-restart phase in a blanket
``except Exception`` that only logged at debug level. When the phase raised
early (e.g. importing ``hermes_cli.gateway`` from the freshly pulled checkout
inside a process that already loaded the pre-update modules), every drain and
restart line vanished from the update output, the update printed
"Update complete!" and exited 0 — while the still-running default-profile
gateway kept serving pre-update modules and died on the next turn with
``ImportError: cannot import name 'is_trivial_prompt'``.
"""

from __future__ import annotations

import sys
import types

from hermes_cli.main import (
    _pids_still_running,
    _restart_phase_failure_is_incomplete,
    _surviving_gateway_pids_after_failed_restart,
    _warn_gateway_restart_phase_aborted,
)


class TestSurvivingGatewayProbe:
    def test_reports_running_gateway_pids(self, monkeypatch):
        fake = types.ModuleType("hermes_cli.gateway")
        fake.find_gateway_pids = lambda **_kwargs: [4321]
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake)

        assert _surviving_gateway_pids_after_failed_restart() == [4321]

    def test_empty_when_no_gateway_is_running(self, monkeypatch):
        fake = types.ModuleType("hermes_cli.gateway")
        fake.find_gateway_pids = lambda **_kwargs: []
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake)

        # An empty list is the only "nothing to restart" proof; it must be
        # distinguishable from the undeterminable case below.
        assert _surviving_gateway_pids_after_failed_restart() == []

    def test_undeterminable_when_gateway_module_is_broken(self, monkeypatch):
        """The probe must not raise — a broken gateway module is the bug's cause."""
        fake = types.ModuleType("hermes_cli.gateway")

        def _boom(**_kwargs):
            raise ImportError("cannot import name 'is_trivial_prompt'")

        fake.find_gateway_pids = _boom
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake)

        assert _surviving_gateway_pids_after_failed_restart() is None


class TestRestartPhaseFailureIsIncomplete:
    """The fail-closed decision behind the survivor probe.

    An empty ``surviving`` probe is only proof-of-safety when nothing was
    running before the phase touched anything. A gateway that was discovered
    pre-restart, stopped, and never verified back up leaves the probe empty at
    exactly the unsafe moment — the fail-open contract #78574 exists to close.
    """

    def test_stale_when_a_gateway_still_survives(self):
        assert _restart_phase_failure_is_incomplete([4321], [4321]) is True

    def test_stale_when_survivor_probe_is_undeterminable(self):
        assert _restart_phase_failure_is_incomplete(None, []) is True

    def test_stale_when_preexisting_gateway_stopped_without_replacement(self):
        # The gap egilewski flagged: a gateway was running, we stopped it, and
        # the post-failure probe is empty because the replacement never came
        # back. `[]` here means "gone", not "safe".
        assert _restart_phase_failure_is_incomplete([], [4321]) is True

    def test_stale_when_pre_restart_state_could_not_be_read(self):
        # Unknown pre-state (probe raised before we recorded it) also fails
        # closed on an empty survivor set — we cannot prove nothing was running.
        assert _restart_phase_failure_is_incomplete([], None) is True

    def test_clean_only_when_nothing_ran_before_and_none_survive(self):
        # Positive control: truly no gateway anywhere, before or after.
        assert _restart_phase_failure_is_incomplete([], []) is False


class TestAbortedRestartWarning:
    def test_warns_with_recovery_command_and_cause(self, capsys):
        _warn_gateway_restart_phase_aborted(
            ImportError("cannot import name 'is_trivial_prompt'"),
            [4321],
        )
        out = capsys.readouterr().out

        assert "Update incomplete" in out
        assert "is_trivial_prompt" in out
        assert "4321" in out
        assert "hermes gateway restart" in out

    def test_warns_even_when_surviving_pids_are_unknown(self, capsys):
        _warn_gateway_restart_phase_aborted(RuntimeError("systemctl exploded"), None)
        out = capsys.readouterr().out

        assert "Update incomplete" in out
        assert "systemctl exploded" in out
        assert "hermes gateway restart" in out


class TestPreRestartSnapshotFallback:
    """Regression for #92145 -- the warning that knows nothing it already knew.

    The #78574 fix above made the aborted restart phase speak up. What it says
    on the runs that matter, though, is "any gateway still running is serving
    pre-update code", with no pid attached, because ``pids`` comes from
    :func:`_surviving_gateway_pids_after_failed_restart`, which re-imports
    ``hermes_cli.gateway`` from the checkout that just broke. The reporter's own
    log is the proof: it shows the no-pid branch, so the probe answered ``None``
    or ``[]`` on a box where a stale gateway was demonstrably alive (their next
    turn died importing ``opencode_provider_family`` from stale
    ``sys.modules``).

    The update was not actually ignorant. ``_pre_restart_gateway_pids``, taken
    before the phase touched anything, is a local at the abort site. These tests
    pin that the snapshot reaches the operator, and that it is liveness-filtered
    so a gateway that really did exit is never named as a survivor.
    """

    @staticmethod
    def _pid_exists_module(alive):
        """A stand-in ``gateway.status`` whose ``_pid_exists`` answers ``alive``."""
        fake = types.ModuleType("gateway.status")
        fake._pid_exists = lambda pid: pid in alive

        return fake

    def test_snapshot_pids_are_named_when_the_probe_cannot_answer(
        self, monkeypatch, capsys
    ):
        """The reporter's exact shape: probe undeterminable, gateway alive."""
        monkeypatch.setitem(
            sys.modules, "gateway.status", self._pid_exists_module({991, 992})
        )

        _warn_gateway_restart_phase_aborted(
            ImportError("cannot import name 'line_input' from 'hermes_cli.cli_output'"),
            None,
            [991, 992],
        )
        out = capsys.readouterr().out

        assert "991, 992" in out, (
            "the update recorded these pids before the restart phase; naming "
            "them is the difference between an actionable warning and a riddle"
        )
        assert "seen before the restart phase" in out, (
            "an operator must be able to tell a pre-restart snapshot from a "
            "fresh enumeration before acting on it"
        )
        assert "hermes gateway restart" in out

    def test_a_snapshot_pid_that_has_since_exited_is_not_named(
        self, monkeypatch, capsys
    ):
        """Liveness-filtered, not replayed.

        The snapshot is a list of pids that WERE running. Printing it back
        unfiltered would accuse a process that the restart phase successfully
        stopped, which is worse than the silence it replaces.
        """
        monkeypatch.setitem(
            sys.modules, "gateway.status", self._pid_exists_module({991})
        )

        _warn_gateway_restart_phase_aborted(RuntimeError("boom"), None, [991, 992])
        out = capsys.readouterr().out

        assert "991" in out
        assert "992" not in out

    def test_no_survivors_falls_back_to_the_original_message(
        self, monkeypatch, capsys
    ):
        """Every snapshot pid is gone: say nothing new, and still fail closed.

        The caller has already decided this run is incomplete (an empty
        survivor set with a non-empty pre-state is the #78574 case). The
        warning must not invent a pid list it cannot stand behind.
        """
        monkeypatch.setitem(sys.modules, "gateway.status", self._pid_exists_module(set()))

        _warn_gateway_restart_phase_aborted(RuntimeError("boom"), None, [991])
        out = capsys.readouterr().out

        assert "991" not in out
        assert "Any gateway still running is serving pre-update code" in out
        assert "hermes gateway restart" in out

    def test_a_live_survivor_probe_still_wins(self, monkeypatch, capsys):
        """The fresh answer beats the snapshot when there is one.

        The snapshot is a fallback, not a replacement: it is older, and it
        cannot see a gateway that started during the phase.
        """
        monkeypatch.setitem(
            sys.modules, "gateway.status", self._pid_exists_module({991})
        )

        _warn_gateway_restart_phase_aborted(RuntimeError("boom"), [4321], [991])
        out = capsys.readouterr().out

        assert "4321" in out
        assert "991" not in out
        assert "seen before the restart phase" not in out

    def test_no_snapshot_is_the_pre_fix_behaviour(self, capsys):
        """Guardrail for the callers that pass two arguments.

        ``pre_restart_pids`` defaults to ``None`` precisely so the existing
        call shape keeps working; this pins that it does.
        """
        _warn_gateway_restart_phase_aborted(RuntimeError("boom"), None)
        out = capsys.readouterr().out

        assert "Any gateway still running is serving pre-update code" in out


class TestPidsStillRunning:
    """The liveness helper itself.

    It must never hand-roll ``os.kill(pid, 0)``: on Windows CPython routes
    ``sig=0`` to ``GenerateConsoleCtrlEvent``, which Ctrl+C's the target's whole
    console process group (bpo-14484). ``update_lock._pid_alive`` documents this
    and delegates to ``gateway.status._pid_exists``; so does this.
    """

    def test_filters_to_the_living(self, monkeypatch):
        fake = types.ModuleType("gateway.status")
        fake._pid_exists = lambda pid: pid == 7
        monkeypatch.setitem(sys.modules, "gateway.status", fake)

        assert _pids_still_running([7, 8, 9]) == [7]

    def test_empty_input_is_empty_output_without_importing_anything(self, monkeypatch):
        """No pids means no probe -- and on this path an import can raise."""
        broken = types.ModuleType("gateway.status")

        def _boom(_pid):
            raise ImportError("checkout is mid-rewrite")

        broken._pid_exists = _boom
        monkeypatch.setitem(sys.modules, "gateway.status", broken)

        assert _pids_still_running([]) == []

    def test_unprobeable_is_none_not_empty(self, monkeypatch):
        """"We could not look" must stay distinguishable from "nothing is left".

        Same contract as ``_surviving_gateway_pids_after_failed_restart``. If
        this returned ``[]`` on a failed probe the caller would silently drop
        back to the vague message with no way to know why.
        """
        broken = types.ModuleType("gateway.status")

        def _boom(_pid):
            raise ImportError("checkout is mid-rewrite")

        broken._pid_exists = _boom
        monkeypatch.setitem(sys.modules, "gateway.status", broken)

        assert _pids_still_running([7]) is None
