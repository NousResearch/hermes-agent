"""#107002: post-update Windows gateway liveness empty + registered
Scheduled Task → ``schtasks /Run`` recovery, then re-poll.

A parent Job Object can kill the respawned gateway during updater
teardown (#48820). When the first ``_wait_for_gateway_ready`` poll is
empty and a Hermes Scheduled Task is registered, the updater must try
``schtasks /Run`` so Task Scheduler starts the gateway outside that Job
Object, then poll again.

Failure-preserving recovery: no task / query error / non-zero ``/Run`` /
still-empty second poll → original RuntimeError, never a fake ✓. Ordinary
``hermes gateway start()`` is unchanged (Scheduled Task remains login
persistence only).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm
import hermes_cli.main_install_repair as main_install_repair
import hermes_cli.update_inventory as update_inventory
from hermes_cli import update_cmd, update_cmd_windows
from hermes_cli.update_cmd import _resume_windows_gateways_after_update
from hermes_cli.update_cmd_windows import _verify_relaunched_gateways_alive


_TASK = "Hermes_Gateway"


def _token(profiles: dict) -> dict:
    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": [],
        "unmapped": [],
    }


def _install_windows_resume_stubs(monkeypatch) -> None:
    monkeypatch.setattr(hm, "_is_windows", lambda: True)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda *_a, **_kw: _TASK)


def _install_ready_then_recover(monkeypatch, schtasks_calls: list, *, after_run_pids):
    """First liveness poll is empty; a later poll returns *after_run_pids* only
    after ``schtasks`` was invoked with ``/Run``."""

    def fake_wait(**_kw):
        if any(call and call[0] == "/Run" for call in schtasks_calls):
            return list(after_run_pids)
        return []

    def fake_schtasks(args):
        schtasks_calls.append(list(args))
        return (0, "", "")

    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
    monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)


class TestRelaunchSchtasksRecovery:
    def test_empty_liveness_plus_registered_task_recovers_via_run(
        self, monkeypatch
    ):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        token = _token({"default": 1111})
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(token)

        assert any(call and call[0] == "/Run" for call in schtasks_calls)
        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is False

    def test_verify_path_recovers_without_raising(self, monkeypatch):
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda *_a, **_kw: _TASK)
        monkeypatch.setattr(
            gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
        )
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        token = _token({"default": 1111})
        _verify_relaunched_gateways_alive(token, {"default": 1111}, [])

        assert ["/Run", "/TN", _TASK] in schtasks_calls

    def test_unregistered_task_raises_and_does_not_run(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: False)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)
        assert token["resume_needed"] is True

    def test_first_liveness_ready_never_touches_schtasks(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        schtasks_calls: list[list[str]] = []
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777]
        )

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(token)

        assert schtasks_calls == []
        assert token["resume_needed"] is False

    def test_schtasks_run_nonzero_stays_on_original_failure(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (1, "", "access denied")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is True

    def test_second_liveness_still_empty_raises(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[])

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is True

    def test_task_query_error_does_not_run(self, monkeypatch):
        _install_windows_resume_stubs(monkeypatch)

        def boom():
            raise OSError("schtasks query failed")

        monkeypatch.setattr(gateway_windows, "is_task_registered", boom)
        schtasks_calls: list[list[str]] = []

        def fake_wait(**_kw):
            return []

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        token = _token({"default": 1111})
        with patch("builtins.print"):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)


class TestColdStartSchtasksRecovery:
    def _install_cold_start_stubs(self, monkeypatch) -> None:
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(main_install_repair, "_is_windows", lambda: True)
        monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_k: [])
        monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
        monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda: 9001)
        monkeypatch.setattr(
            gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda *_a, **_kw: _TASK)

    def test_empty_liveness_plus_registered_task_recovers_via_run(
        self, monkeypatch
    ):
        self._install_cold_start_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: True)
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        with patch("builtins.print"):
            assert update_cmd._cold_start_windows_gateway_after_update() is True

        assert ["/Run", "/TN", _TASK] in schtasks_calls

    def test_unregistered_task_raises_and_does_not_run(self, monkeypatch):
        self._install_cold_start_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda **_kw: False)
        schtasks_calls: list[list[str]] = []
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: []
        )

        def fake_schtasks(args):
            schtasks_calls.append(list(args))
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)

        with pytest.raises(RuntimeError, match="did not become ready"):
            update_cmd._cold_start_windows_gateway_after_update()

        assert not any(call and call[0] == "/Run" for call in schtasks_calls)


class TestAttestedProfileSchtasksRecovery:
    """The updater runs under one profile, but each cold-start obligation owns another home."""

    def test_beta_recovery_uses_beta_task_and_beta_readiness(self, monkeypatch):
        default_home = Path("/homes/default")
        beta_home = Path("/homes/beta")
        task_homes: list[Path | None] = []
        readiness_homes: list[Path | None] = []
        spawned: list[Path | None] = []
        consumed: list[tuple[str, Path | None]] = []

        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda *, home: [])
        monkeypatch.setattr(
            gateway_windows,
            "_spawn_detached",
            lambda *, home: spawned.append(home) or 9001,
        )

        def ready(*, home, **_kwargs):
            readiness_homes.append(home)
            # A new default-profile PID must never satisfy beta's obligation.
            return [111] if home == default_home else ([] if len(readiness_homes) == 1 else [222])

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", ready)
        monkeypatch.setattr(
            gateway_windows,
            "is_task_registered",
            lambda *, home=None: task_homes.append(home) or home == beta_home,
        )
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: task_homes.append(home) or (0, "", ""),
        )
        monkeypatch.setattr(
            gateway_windows,
            "_consume_start_attestation",
            lambda generation, *, home=None: consumed.append((generation, home)),
        )
        monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None)

        token = {"cold_start_profiles": {"beta": "beta-generation"}}
        update_cmd_windows._cold_start_attested_profiles(token)

        assert spawned == [beta_home]
        assert task_homes == [beta_home, beta_home]
        assert readiness_homes == [beta_home, beta_home]
        assert consumed == [("beta-generation", beta_home)]
        assert "cold_start_profiles" not in token

    def test_stable_target_does_not_run_its_task(self, monkeypatch):
        beta_home = Path("/homes/beta")
        runs: list[Path | None] = []

        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda *, home: [222])
        monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda **_kw: pytest.fail("must not spawn"))
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda *, home, **_kw: [222])
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: runs.append(home) or pytest.fail("must not run task"),
        )
        monkeypatch.setattr(gateway_windows, "_consume_start_attestation", lambda *_a, **_kw: None)
        monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None)

        token = {"cold_start_profiles": {"beta": "beta-generation"}}
        update_cmd_windows._cold_start_attested_profiles(token)

        assert runs == []
        assert "cold_start_profiles" not in token

    def test_partial_profile_recovery_keeps_failed_generation_pending(self, monkeypatch):
        homes = {"alpha": Path("/homes/alpha"), "beta": Path("/homes/beta")}
        consumed: list[tuple[str, Path | None]] = []
        runs: list[Path | None] = []

        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
        monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda *, home: [])
        monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda *, home: 9001)
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda *, home=None: True)
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: runs.append(home) or (0, "", ""),
        )
        ready_attempts: dict[Path, int] = {}

        def ready(*, home, **_kwargs):
            ready_attempts[home] = ready_attempts.get(home, 0) + 1
            return [222] if home == homes["beta"] and ready_attempts[home] > 1 else []

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", ready)
        monkeypatch.setattr(
            gateway_windows,
            "_consume_start_attestation",
            lambda generation, *, home=None: consumed.append((generation, home)),
        )
        monkeypatch.setattr(gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None)

        token = {"cold_start_profiles": {"alpha": "alpha-generation", "beta": "beta-generation"}}
        with pytest.raises(RuntimeError, match="alpha"):
            update_cmd_windows._cold_start_attested_profiles(token)

        assert runs == [homes["alpha"], homes["beta"]]
        assert consumed == [("beta-generation", homes["beta"])]
        assert token["cold_start_profiles"] == {"alpha": "alpha-generation"}


class TestTargetedRecoveryContract:
    def test_task_name_resolves_from_explicit_target_home(self, monkeypatch):
        beta_home = Path("/homes/beta")
        resolved_homes: list[Path | None] = []

        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(
            gateway,
            "_profile_suffix",
            lambda home=None: resolved_homes.append(home) or "beta",
        )

        assert gateway_windows.get_task_name(beta_home) == "Hermes_Gateway_beta"
        assert resolved_homes == [beta_home]

    @pytest.mark.parametrize(
        ("registered", "run", "ready", "diagnostic"),
        [
            (False, (0, "", ""), [222], "No registered"),
            (OSError("query failed"), (0, "", ""), [222], "query failed"),
            (True, (1, "", "denied"), [222], "trigger rejected"),
            (True, OSError("run failed"), [222], "trigger rejected"),
            (True, (0, "", ""), [], "trigger accepted but target not ready"),
        ],
    )
    def test_failure_diagnostics_preserve_the_failure_path(
        self, monkeypatch, capsys, registered, run, ready, diagnostic
    ):
        beta_home = Path("/homes/beta")
        calls: list[tuple[str, Path | None]] = []

        def query(*, home=None):
            calls.append(("query", home))
            if isinstance(registered, Exception):
                raise registered
            return registered

        def trigger(*, home=None):
            calls.append(("run", home))
            if isinstance(run, Exception):
                raise run
            return run

        monkeypatch.setattr(gateway_windows, "is_task_registered", query)
        monkeypatch.setattr(gateway_windows, "_run_scheduled_task_once", trigger)
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda *, home=None, **_kw: calls.append(("ready", home)) or ready,
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(
            gateway_windows, home=beta_home
        ) == []
        assert diagnostic in capsys.readouterr().out
        assert all(home == beta_home for _kind, home in calls)

    def test_resume_refreshes_before_target_task_trigger(self, monkeypatch):
        beta_home = Path("/homes/beta")
        order: list[str] = []

        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: order.append("refresh"))
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda *, home=None: True)

        def trigger(*, home=None):
            assert home == beta_home
            assert order == ["refresh"]
            order.append("run")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_run_scheduled_task_once", trigger)
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda *, home=None, **_kw: [222] if order == ["refresh", "run"] else [],
        )

        _resume_windows_gateways_after_update(_token({"beta": 1111}))

        assert order == ["refresh", "run"]


class TestPlannedManualProfileRecovery:
    def test_manual_planned_profile_with_registered_task_reaches_recovery(
        self, monkeypatch, tmp_path
    ):
        """The real inventory calls this PID ``manual``; that label is not a
        recovery veto.  Drive the actual Windows pause/resume entrypoints,
        rather than adding an unused ``supervisor`` field to their token.

        This is the scheduler-shaped case from #107002: task ancestry is not
        SCM ownership, so the inventory's current classifier has no stronger
        supervisor evidence and deliberately says ``manual``.  Once the
        direct relaunch is not ready, an already registered *target* task is
        still eligible for the one-shot post-update recovery.
        """
        beta_home = tmp_path / "profiles" / "beta"
        beta_home.mkdir(parents=True)

        # Build a real plan row through the production inventory collector.
        # The test only stubs OS probes; it does not manufacture a plan field
        # that the pause/resume code never reads.
        monkeypatch.setattr(update_inventory, "_collect_install_shape", lambda plan: None)
        monkeypatch.setattr(
            "hermes_cli.build_info.get_code_identity",
            lambda refresh=False: {"sha": None, "version": None},
        )
        monkeypatch.setattr(
            "hermes_cli.update_receipt._profile_homes", lambda: [("beta", beta_home)]
        )
        monkeypatch.setattr("hermes_cli.update_receipt._socket_identity", lambda home: None)
        monkeypatch.setattr(
            "gateway.status.live_gateway_pid_for_home", lambda home: 777 if home == beta_home else None
        )
        monkeypatch.setattr("gateway.status.read_runtime_status", lambda path: {})
        monkeypatch.setattr(gateway, "_get_service_pids", lambda **_kw: set())
        monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda **_kw: [])
        monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **_kw: [])

        plan = update_inventory.collect_runtime_inventory()
        assert [(runtime.profile, runtime.supervisor, runtime.restart_via) for runtime in plan.runtimes] == [
            ("beta", "manual", "manual")
        ]

        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(update_cmd, "_desktop_owns_gateway_lifecycle", lambda: False)
        monkeypatch.setattr(hm, "_venv_launcher_ancestors", lambda _pids: [])
        monkeypatch.setattr(hm, "_wait_for_windows_update_gateway_exit", lambda _pids, timeout: set())
        monkeypatch.setattr(update_cmd_windows, "_gateway_drain_timeout", lambda _acks: 0.0)
        beta = SimpleNamespace(pid=777, profile="beta", path=beta_home)
        monkeypatch.setattr(
            update_cmd_windows,
            "_discover_windows_gateways",
            lambda: ({777: beta}, [], set(), [777]),
        )
        monkeypatch.setattr(
            update_cmd_windows,
            "_request_socket_pauses",
            lambda *_args: ({"beta": 777}, [777], []),
        )
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)

        run_homes: list[Path | None] = []
        readiness_attempts = 0

        def readiness(*, home=None, **_kwargs):
            nonlocal readiness_attempts
            assert home == beta_home
            readiness_attempts += 1
            return [222] if run_homes else []

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", readiness)
        monkeypatch.setattr(
            gateway_windows, "is_task_registered", lambda *, home=None: home == beta_home
        )
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: run_homes.append(home) or (0, "", ""),
        )

        token = update_cmd._pause_windows_gateways_for_update()
        assert token["profiles"] == {"beta": 777}
        with patch("builtins.print"):
            update_cmd._resume_windows_gateways_after_update(token)

        assert run_homes == [beta_home]
        assert readiness_attempts == 2
        assert token["resume_needed"] is False
