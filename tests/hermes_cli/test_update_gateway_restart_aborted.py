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

import importlib
import json
import logging
import os
import sys
import types
from types import SimpleNamespace

import pytest

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


class _Completed:
    """CompletedProcess stand-in for the faked systemctl / fresh-child calls."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _env_sensitive_systemctl(calls: list, restarted: list):
    """A ``systemctl`` faithful to the real one (test design credit: the #107623 review): every
    ``--user`` probe fails without a session bus — the observable separating a reachable user
    manager from an absent one (``Failed to connect to user scope bus``, rc=1)."""

    def fake_run(argv, **kwargs):
        argv = list(argv)
        if "hermes_cli.main" in argv:  # the per-profile relaunch
            calls.append(argv)
            return _Completed(0)
        if argv and str(argv[0]).endswith("systemctl"):
            calls.append(argv)
            user_scope = "--user" in argv
            if user_scope and not (
                os.environ.get("XDG_RUNTIME_DIR") and os.environ.get("DBUS_SESSION_BUS_ADDRESS")
            ):
                return _Completed(1, stderr="Failed to connect to user scope bus via local transport")
            if "is-active" in argv:
                return _Completed(0, stdout="active\n")
            if "list-units" in argv:
                # The serve unit exists in the user manager only (system scope: nothing to recover).
                return _Completed(0, stdout="hermes-serve.service loaded active running\n" if user_scope else "")
            if "show" in argv:
                return _Completed(0, stdout="5151\n" if restarted else "4242\n")
            if "restart" in argv:
                restarted.append({"user" if user_scope else "system": argv[-1]})
                return _Completed(0)
            return _Completed(0)
        return _Completed(0)

    return fake_run


class TestFreshRecoveryOutcome:
    """#107614 at the outcome level (test design credit: the #107623 review): what the child
    observes — not the env blob — decides the update's exit code. The fresh child is faked to run
    the REAL recovery passes in the environment the parent handed down, so the parent-side
    adoption is exercised end-to-end through the real ``_recover_gateway_restart_after_abort``
    entry and the real ``_abort_recovery_is_complete`` verdict."""

    def _busless_parent_with_fake_child(self, monkeypatch, gateway_fake):
        """Wire a bus-less parent (with the given fake adoption helper) to a faked fresh child that
        runs the real recovery passes in the environment it inherited; returns ``(invoke, calls,
        restarted)``."""
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        monkeypatch.setitem(sys.modules, "hermes_cli.gateway", gateway_fake)
        recovery = importlib.import_module("hermes_cli.update_restart_recovery")
        monkeypatch.setattr(recovery.shutil, "which", lambda name: f"/usr/bin/{name}")
        calls: list = []
        restarted: list = []
        child_run = _env_sensitive_systemctl(calls, restarted)

        def fake_child(_command, **kwargs):
            """The fresh child's output: the production passes, in the environment it inherited."""
            payload = json.loads(kwargs["input"])
            result = recovery.restart_profiles(
                payload["profiles"], supervisors=payload["supervisors"], run=child_run
            )
            if payload["serve_units"]["recover"]:
                result["serve_units"] = recovery.restart_serve_units(run=child_run)
            return _Completed(0, stdout=json.dumps(result))

        monkeypatch.setattr(update_abort_recovery.subprocess, "run", fake_child)
        plan = SimpleNamespace(
            runtimes=[SimpleNamespace(profile="default", supervisor="systemd", kind="gateway", pid=1234)]
        )

        def invoke():
            return update_abort_recovery._recover_gateway_restart_after_abort(plan, gateway_mode=False)

        return invoke, calls, restarted

    def test_adopted_env_verifies_and_completes_the_abort_path(self, monkeypatch):
        fake = types.ModuleType("hermes_cli.gateway")

        def _adopt():
            os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{os.getuid()}/bus"

        fake._ensure_user_systemd_env = _adopt

        invoke, _calls, _restarted = self._busless_parent_with_fake_child(monkeypatch, fake)
        result = invoke()

        assert result["verified"] == ["default"]
        assert result["relaunch_attempted"] == []
        # The consequence the issue reports: the abort path may clear ``incomplete`` (exit 0)
        # instead of failing the whole update for a fleet already on the new code.
        assert update_abort_recovery._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=result,
            stale_runtime_rows=[],
        )

    def test_unavailable_adoption_fails_closed_and_warns(self, monkeypatch, caplog):
        """The #107623 review's regression point: an adoption that cannot run (an import failure in
        the pre-pull interpreter) must not silently restore the bug — the loss is visible at
        warning level and the honest outcome (``relaunch_attempted``) keeps the update incomplete."""

        def _boom():
            raise ImportError("cannot import name '_ensure_user_systemd_env'")

        fake = types.ModuleType("hermes_cli.gateway")
        fake._ensure_user_systemd_env = _boom

        invoke, _calls, _restarted = self._busless_parent_with_fake_child(monkeypatch, fake)
        with caplog.at_level(logging.WARNING, logger="hermes_cli.update_abort_recovery"):
            result = invoke()

        assert result["verified"] == []
        assert result["relaunch_attempted"] == ["default"]
        assert "User-bus env normalization unavailable" in caplog.text
        assert "cannot import name '_ensure_user_systemd_env'" in caplog.text
        # Nothing was fabricated — completeness stays unprovable, the update keeps exit 1.
        assert not update_abort_recovery._abort_recovery_is_complete(
            planned_gateway_profiles={"default"},
            covered_gateway_profiles={"default"},
            recovery_result=result,
            stale_runtime_rows=[],
        )

    @pytest.mark.linux_only
    def test_adopted_env_recovers_the_user_scope_serve_unit(self, monkeypatch):
        """Serve half of the same outcome: the user manager lists and restarts ``hermes-serve``
        only once the parent-side adoption reaches the child's ``systemctl --user`` probes."""

        def _adopt():
            os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path=/run/user/{os.getuid()}/bus"

        fake = types.ModuleType("hermes_cli.gateway")
        fake._ensure_user_systemd_env = _adopt

        invoke, _calls, restarted = self._busless_parent_with_fake_child(monkeypatch, fake)
        result = invoke()

        assert result["serve_units"] == {"verified": ["user/hermes-serve"], "failed": []}
        assert restarted == [{"user": "hermes-serve.service"}]
