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

import os
import sys
import types

from hermes_cli import update_abort_recovery
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


class TestFreshRecoveryUserBusEnv:
    """#107614 — the fresh recovery child must inherit a user-bus env it cannot build itself.

    ``update_restart_recovery`` deliberately imports no gateway code (a broken freshly pulled
    import graph is what aborts the phase), so it can never call ``_ensure_user_systemd_env``
    itself. Under a bus-less dispatcher (``sudo -u``, cron) every ``systemctl --user`` probe of the
    child then fails: a healthy gateway reads ``relaunch_attempted`` and the update exits 1 — the
    same false negative #107477 fixed at the in-process listing helper.
    """

    def _spawn_capturing(self, monkeypatch):
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        captured = {}

        class _Completed:
            returncode = 0
            stdout = "{}"
            stderr = ""

        def fake_run(_command, **kwargs):
            captured["env"] = kwargs["env"]
            return _Completed()

        monkeypatch.setattr(update_abort_recovery.subprocess, "run", fake_run)
        result = update_abort_recovery._run_fresh_recovery_process(
            ["default"], {"default": "systemd"},
            gateway_mode=False, recover_serve=False, skip_units=())
        return result, captured

    def test_child_env_adopts_user_bus_before_spawn(self, monkeypatch):
        fake = types.ModuleType("hermes_cli.gateway")

        def _adopt():
            os.environ["XDG_RUNTIME_DIR"] = "/run/user/501"
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=/run/user/501/bus"

        fake._ensure_user_systemd_env = _adopt
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake)

        result, captured = self._spawn_capturing(monkeypatch)

        assert result is not None
        assert captured["env"]["XDG_RUNTIME_DIR"] == "/run/user/501"
        assert captured["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/501/bus"
        # The recovery self-identification markers stay independent of the adoption.
        assert captured["env"]["HERMES_UPDATE_RESTART_RECOVERY"] == "1"
        assert "_HERMES_GATEWAY" not in captured["env"]

    def test_spawn_survives_a_broken_gateway_module(self, monkeypatch):
        """A gateway module that cannot even import must not cancel the recovery spawn (#78574 shape)."""

        def _boom():
            raise ImportError("cannot import name '_ensure_user_systemd_env'")

        fake = types.ModuleType("hermes_cli.gateway")
        fake._ensure_user_systemd_env = _boom
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", fake)

        result, captured = self._spawn_capturing(monkeypatch)

        assert result is not None
        # Nothing was fabricated for a bus the host never advertised — fail closed stays intact.
        assert "XDG_RUNTIME_DIR" not in captured["env"]
        assert "DBUS_SESSION_BUS_ADDRESS" not in captured["env"]
