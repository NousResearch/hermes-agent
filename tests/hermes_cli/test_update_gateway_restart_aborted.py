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

from hermes_cli.update_cmd import _restart_phase_failure_is_incomplete, _surviving_gateway_pids_after_failed_restart, _warn_gateway_restart_phase_aborted


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
