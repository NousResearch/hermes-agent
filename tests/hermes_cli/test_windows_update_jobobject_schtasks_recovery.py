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
    monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda *a, **kw: None)
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


def _install_managed_task_action(monkeypatch) -> None:
    """Give legacy recovery tests a registered action that passes the real predicate."""
    managed_xml = r'''<Task><Actions><Exec><Command>wscript.exe</Command><Arguments>//B //Nologo "C:\Hermes\gateway-service\Hermes_Gateway.vbs"</Arguments></Exec></Actions></Task>'''
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda *_a, **_kw: _TASK)
    monkeypatch.setattr(gateway_windows, "_query_scheduled_task_xml", lambda _task: managed_xml)
    monkeypatch.setattr(
        gateway_windows,
        "_scheduled_task_template",
        lambda _task, _home=None: managed_xml,
    )


class TestRelaunchSchtasksRecovery:
    def test_empty_liveness_plus_registered_task_recovers_via_run(
        self, monkeypatch
    ):
        _install_windows_resume_stubs(monkeypatch)
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        token = _token({"default": 1111})
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(token)

        assert any(call and call[0] == "/Run" for call in schtasks_calls)
        assert ["/Run", "/TN", _TASK] in schtasks_calls
        assert token["resume_needed"] is False

    def test_verify_path_recovers_without_raising(self, monkeypatch):
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
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
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "query-failed")
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
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
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
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
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
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
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

        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: (_ for _ in ()).throw(boom()))
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
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        schtasks_calls: list[list[str]] = []
        _install_ready_then_recover(monkeypatch, schtasks_calls, after_run_pids=[4242])

        with patch("builtins.print"):
            assert update_cmd._cold_start_windows_gateway_after_update() is True

        assert ["/Run", "/TN", _TASK] in schtasks_calls

    def test_unregistered_task_raises_and_does_not_run(self, monkeypatch):
        self._install_cold_start_stubs(monkeypatch)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "query-failed")
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
            "_task_registration_state",
            lambda *, home=None: task_homes.append(home) or ("registered" if home == beta_home else "query-failed"),
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
        assert task_homes == [beta_home]
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

        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
        monkeypatch.setattr(gateway_windows, "_live_gateway_pids", lambda *, home: [])
        monkeypatch.setattr(gateway_windows, "_spawn_detached", lambda *, home: 9001)
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda *, home=None: "registered")
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

        assert runs == [homes["alpha"]]
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
            (False, (0, "", ""), [222], "query failed"),
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

        _install_managed_task_action(monkeypatch)
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

        monkeypatch.setattr(
            gateway_windows, "_task_registration_state",
            lambda *, home=None: (query(home=home) and "registered") or "query-failed",
        )
        monkeypatch.setattr(gateway_windows, "_run_scheduled_task_once", trigger)
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda *, home=None, **_kw: calls.append(("ready", home)) or (
                ready if any(kind == "run" for kind, _home in calls) else []
            ),
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
        _install_managed_task_action(monkeypatch)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda *a, **kw: order.append("refresh"))
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda *, home=None: "registered")

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
        _install_managed_task_action(monkeypatch)
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
            gateway_windows, "_task_registration_state", lambda *, home=None: "registered" if home == beta_home else "query-failed"
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
        assert readiness_attempts == 3
        assert token["resume_needed"] is False


class TestRegisteredTaskRecoveryFollowups:
    def test_resume_refreshes_the_beta_launcher_not_active_default(
        self, monkeypatch, tmp_path
    ):
        """R1: the update's real resume path must render the launcher under
        beta before beta's registered task can be used; rendering default is
        not a refresh for beta."""
        default_home = tmp_path / "default"
        beta_home = default_home / "profiles" / "beta"
        default_home.mkdir(parents=True)
        beta_home.mkdir(parents=True)
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(
            hm, "_refresh_windows_gateway_launchers", update_cmd_windows._refresh_windows_gateway_launchers
        )
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(default_home))
        monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: default_home)
        monkeypatch.setattr(gateway, "_native_service_homes", lambda: set())
        monkeypatch.setattr(gateway, "_bare_unit_pinned_home", lambda: None)
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway, "get_python_path", lambda: r"C:\\Hermes\\venv\\Scripts\\python.exe")
        monkeypatch.setattr(gateway_windows, "is_task_registered", lambda *, home=None: home == beta_home)
        monkeypatch.setattr(gateway_windows, "is_startup_entry_installed", lambda **_kw: False)
        monkeypatch.setattr(gateway_windows, "reconcile_scheduled_task", lambda *_a, **_kw: False)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [222])

        _resume_windows_gateways_after_update(_token({"beta": 1111}))

        scripts = list((beta_home / "gateway-service").glob("*.cmd"))
        assert len(scripts) == 1
        beta_script = scripts[0]
        rendered = beta_script.read_text(encoding="utf-8")
        assert f'HERMES_HOME={beta_home}' in rendered
        assert "--profile beta gateway run" in rendered
        assert not (default_home / "gateway-service" / "Hermes_Gateway.cmd").exists()

    def test_custom_task_action_is_not_reconciled_over(self, monkeypatch, tmp_path):
        """R1: updater refresh may rewrite its launcher files, but must not
        replace an action it cannot identify as Hermes-managed."""
        script = tmp_path / "gateway.cmd"
        custom = gateway_windows._build_scheduled_task_xml(
            "Hermes_Gateway", tmp_path / "custom-launcher.vbs", r"PC\\me"
        )
        calls: list[list[str]] = []

        def schtasks(args):
            calls.append(list(args))
            if "/XML" in args and "/Query" in args:
                return (0, custom, "")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
        monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda *a, **kw: script)
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\\me")
        monkeypatch.setattr(gateway_windows, "_write_task_script", lambda *a, **kw: pytest.fail("must not overwrite custom action"))

        assert gateway_windows.reconcile_scheduled_task("Hermes_Gateway") is False
        assert not any(call[0] in ("/Delete", "/Create") for call in calls)

    def test_custom_vbs_with_quoted_hermes_target_argument_is_not_reconciled(
        self, monkeypatch, tmp_path
    ):
        """The first quoted VBS is the script wscript executes. A later
        Hermes-looking argument is not evidence of task ownership."""
        task_name = "Hermes_Gateway_beta"
        custom = gateway_windows._build_scheduled_task_xml(
            task_name, Path(r"C:\Custom\custom.vbs"), r"PC\me"
        )
        custom = custom.replace(
            r'"C:\Custom\custom.vbs"',
            r'"C:\Custom\custom.vbs" /target:"C:\Hermes\profiles\beta\gateway-service\Hermes_Gateway_beta.vbs"',
        ).replace('version="1.4"', 'version="1.3"')
        calls: list[list[str]] = []

        def schtasks(args):
            calls.append(list(args))
            if "/XML" in args and "/Query" in args:
                return (0, custom, "")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
        monkeypatch.setattr(
            gateway_windows,
            "get_task_script_path",
            lambda *a, **kw: tmp_path / "gateway-service" / f"{task_name}.cmd",
        )
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\me")
        monkeypatch.setattr(
            gateway_windows,
            "_write_task_script",
            lambda *a, **kw: pytest.fail("must not rewrite an unknown custom action"),
        )

        assert gateway_windows.reconcile_scheduled_task(task_name) is False
        assert not any(call[0] in ("/Delete", "/Create") for call in calls)

    def test_canonical_target_launcher_action_still_reconciles(self, monkeypatch, tmp_path):
        """A supported canonical launcher remains eligible for the existing
        task-drift repair path after unknown Action arguments are rejected."""
        task_name = "Hermes_Gateway_beta"
        script = tmp_path / "gateway-service" / f"{task_name}.cmd"
        script.parent.mkdir()
        launcher = script.with_suffix(".vbs")
        registered = gateway_windows._build_scheduled_task_xml(task_name, launcher, r"PC\me")
        registered = registered.replace('version="1.4"', 'version="1.3"')
        calls: list[list[str]] = []

        def schtasks(args):
            calls.append(list(args))
            if "/XML" in args and "/Query" in args:
                return (0, registered, "")
            return (0, "", "")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
        monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda *a, **kw: script)
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"PC\me")
        monkeypatch.setattr(gateway_windows, "_write_task_script", lambda *a, **kw: script)

        assert gateway_windows.reconcile_scheduled_task(task_name) is True
        assert any(call[0] == "/Create" for call in calls)

    def test_target_launcher_refresh_failure_is_not_reported_as_refreshed(self, monkeypatch, tmp_path, capsys):
        """R1: a target write failure is retained as best-effort failure;
        it cannot silently count as refreshing a different profile."""
        beta_home = tmp_path / "beta"
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(gateway_windows, "is_installed", lambda *, home=None: home == beta_home)
        writes: list[Path | None] = []

        def fail_write(*, home=None):
            writes.append(home)
            raise OSError("read-only")

        monkeypatch.setattr(gateway_windows, "_write_task_script", fail_write)
        monkeypatch.setattr(
            gateway_windows, "reconcile_scheduled_task", lambda *_a, **_kw: pytest.fail("no reconcile after failed render")
        )

        update_cmd_windows._refresh_windows_gateway_launchers(home=beta_home)

        assert writes == [beta_home]
        assert "Refreshed Windows gateway launcher scripts" not in capsys.readouterr().out

    def test_mixed_cold_start_refreshes_current_and_profile_homes_before_spawning(
        self, monkeypatch, tmp_path
    ):
        """The mixed plan cold-starts the actual current home first, so it
        and beta need one target-scoped refresh each before either spawn."""
        current_home = tmp_path / "configured-home"
        beta_home = current_home / "profiles" / "beta"
        refreshed: list[Path | None] = []
        order: list[str] = []
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(current_home))
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: beta_home)
        monkeypatch.setattr(
            hm,
            "_refresh_windows_gateway_launchers",
            lambda *, home=None: refreshed.append(home) or order.append(f"refresh:{home}"),
        )
        monkeypatch.setattr(update_cmd_windows, "_resume_windows_services", lambda _token: None)
        monkeypatch.setattr(
            hm,
            "_cold_start_windows_gateway_after_update",
            lambda _token: order.append("current-cold-start") or True,
        )
        monkeypatch.setattr(
            update_cmd_windows,
            "_cold_start_attested_profiles",
            lambda _token: order.append("beta-cold-start"),
        )

        token = {
            "resume_needed": True,
            "profiles": {},
            "unmapped": [],
            "cold_start_if_installed": True,
            "cold_start_profiles": {"beta": "beta-generation"},
        }
        _resume_windows_gateways_after_update(token)

        assert refreshed == [current_home, beta_home]
        assert refreshed.count(current_home) == 1
        assert order == [
            f"refresh:{current_home}",
            f"refresh:{beta_home}",
            "current-cold-start",
            "beta-cold-start",
        ]

    def test_partial_relaunch_attests_alpha_and_keeps_only_beta_pending(self, monkeypatch, tmp_path):
        """R2: a created watcher is not a recovered profile.  Alpha may be
        attested and reconciled even while beta remains retryable."""
        homes = {"alpha": tmp_path / "alpha", "beta": tmp_path / "beta"}
        for home in homes.values():
            home.mkdir()
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "query-failed")
        attestations: list[tuple[list[int], Path | None]] = []
        monkeypatch.setattr(
            gateway_windows,
            "_write_start_attestation",
            lambda pids, _via, home=None: attestations.append((list(pids), home)),
        )
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda *, home=None, **_kw: [101] if home == homes["alpha"] else [],
        )

        token = _token({"alpha": 11, "beta": 22})
        with pytest.raises(RuntimeError, match="not verified alive"):
            _resume_windows_gateways_after_update(token)

        assert attestations == [([101], homes["alpha"])]
        assert token["profiles"] == {"beta": 22}
        assert token["relaunched_profiles"] == ["alpha"]
        assert token["resume_needed"] is True

    def test_pure_unmapped_success_records_neutral_attestation(self, monkeypatch, tmp_path):
        """R3: a verified unmapped restart gets a durable diagnostic record,
        without putting an unknown gateway in the default profile marker."""
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777])

        _verify_relaunched_gateways_alive(
            _token({}), {}, [{"pid": 77, "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"]}]
        )

        assert (tmp_path / "state" / "gateway.unmapped-start-attestation.json").exists()
        assert not (tmp_path / "state" / "gateway.start-attestation.json").exists()
        warning = gateway_windows.check_unmapped_start_attestation(current_pids=[])
        assert warning is not None and "unmapped gateway" in warning
        assert "Task Scheduler" not in warning

    def test_pure_unmapped_failure_keeps_recovery_obligation(self, monkeypatch, tmp_path):
        """R3: no stable unmapped process means no neutral success marker and
        the original argv remains available for a later recovery attempt."""
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(tmp_path))
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [])
        entry = {"pid": 77, "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"]}
        token = _token({})

        with pytest.raises(RuntimeError, match="not verified alive"):
            _verify_relaunched_gateways_alive(token, {}, [entry])

        assert token["unmapped"] == [entry]
        assert not (tmp_path / "state" / "gateway.unmapped-start-attestation.json").exists()

    def test_mixed_mapped_and_unmapped_success_keeps_identities_separate(self, monkeypatch, tmp_path):
        """R3: a mixed recovery records the mapped profile normally and the
        unmapped process neutrally, without manufacturing a default profile."""
        default_home = tmp_path / "default"
        alpha_home = tmp_path / "alpha"
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: str(default_home))
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: alpha_home)
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda *, home=None, all_profiles=False, **_kw: [777] if all_profiles else ([101] if home == alpha_home else []),
        )
        token = _token({})

        _verify_relaunched_gateways_alive(
            token,
            {"alpha": 11},
            [{"pid": 77, "argv": ["python", "-m", "hermes_cli.main", "gateway", "run"]}],
        )

        assert (alpha_home / "state" / "gateway.start-attestation.json").exists()
        assert (default_home / "state" / "gateway.unmapped-start-attestation.json").exists()
        assert not (default_home / "state" / "gateway.start-attestation.json").exists()
        assert token["relaunched_profiles"] == ["alpha"]

    def test_partial_retry_does_not_drop_or_restart_verified_alpha(self, monkeypatch, tmp_path):
        """R2: after alpha was verified, a retry only attempts beta and
        retains alpha in the outcome bookkeeping if beta still cannot start."""
        beta_home = tmp_path / "beta"
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda _name: beta_home)
        attempted: list[str] = []
        monkeypatch.setattr(
            gateway,
            "launch_detached_profile_gateway_restart",
            lambda profile, _pid: attempted.append(profile) or False,
        )
        token = _token({"beta": 22})
        token["relaunched_profiles"] = ["alpha"]

        with pytest.raises(RuntimeError, match="not verified alive"):
            _resume_windows_gateways_after_update(token)

        assert attempted == ["beta"]
        assert token["relaunched_profiles"] == ["alpha"]
        assert token["profiles"] == {"beta": 22}
        assert token["resume_needed"] is True

    def test_direct_and_readiness_failures_are_combined_for_retry(self, monkeypatch, tmp_path):
        """R2: a beta watcher creation failure must not be overwritten when
        alpha's created watcher later fails its target readiness check."""
        homes = {"alpha": tmp_path / "alpha", "beta": tmp_path / "beta"}
        _install_windows_resume_stubs(monkeypatch)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda *a, **kw: None)
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
        monkeypatch.setattr(
            gateway,
            "launch_detached_profile_gateway_restart",
            lambda profile, _pid: profile == "alpha",
        )
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "query-failed")
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [])
        token = _token({"alpha": 11, "beta": 22})

        with pytest.raises(RuntimeError, match="not verified alive"):
            _resume_windows_gateways_after_update(token)

        assert token["profiles"] == {"alpha": 11, "beta": 22}
        assert token["relaunched_profiles"] == []
        assert token["resume_needed"] is True

    @pytest.mark.parametrize("failure", ["timeout", "oserror"])
    def test_schtasks_query_failure_is_not_reported_as_no_task(self, monkeypatch, capsys, failure):
        """R4: the real subprocess boundary returns 124/1 for a timeout or
        invocation failure; neither is evidence that the task is absent."""
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows.shutil, "which", lambda _name: "schtasks.exe")
        if failure == "timeout":
            monkeypatch.setattr(
                gateway_windows.subprocess,
                "run",
                lambda *_a, **_kw: (_ for _ in ()).throw(gateway_windows.subprocess.TimeoutExpired("schtasks", 1)),
            )
        else:
            monkeypatch.setattr(
                gateway_windows.subprocess,
                "run",
                lambda *_a, **_kw: (_ for _ in ()).throw(OSError("unavailable")),
            )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(gateway_windows) == []
        out = capsys.readouterr().out
        assert "query failed" in out
        assert "No registered" not in out

    def test_rechecks_target_before_run_when_task_query_races_recovery(self, monkeypatch):
        """R5: a target that becomes stable during task lookup must be
        returned directly; a second competing /Run is unnecessary."""
        runs: list[object] = []
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777])
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda **_kw: runs.append(True) or (0, "", ""),
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(gateway_windows) == [777]
        assert runs == []


class TestScheduledTaskActionOwnershipRecovery:
    """Recovery may run only the target profile's managed task Action."""

    @staticmethod
    def _managed_template(monkeypatch, home: Path) -> str:
        monkeypatch.setattr(gateway_windows, "_assert_windows", lambda: None)
        monkeypatch.setattr(gateway_windows, "get_task_name", lambda *_a, **_kw: _TASK)
        return gateway_windows._scheduled_task_template(_TASK, home)

    def test_custom_same_name_task_action_fails_closed_without_run(self, monkeypatch, tmp_path):
        home = tmp_path / "target"
        template = self._managed_template(monkeypatch, home)
        arguments_start = template.index("<Arguments>")
        arguments_end = template.index("</Arguments>", arguments_start) + len("</Arguments>")
        custom_xml = (
            template[:arguments_start]
            + '<Arguments>//B //Nologo "C:\\Custom\\custom.vbs"</Arguments>'
            + template[arguments_end:]
        )
        runs: list[Path | None] = []

        assert not gateway_windows._task_action_is_hermes_managed(
            custom_xml, template, task_name=_TASK
        )
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [])
        monkeypatch.setattr(gateway_windows, "_query_scheduled_task_xml", lambda _task: custom_xml)
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: runs.append(home) or (0, "", ""),
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(
            gateway_windows, home=home
        ) == []
        assert runs == []

    def test_unreadable_task_xml_fails_closed_without_run(self, monkeypatch):
        runs: list[object] = []

        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [])
        monkeypatch.setattr(gateway_windows, "_query_scheduled_task_xml", lambda _task: None)
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda **_kw: runs.append(True) or (0, "", ""),
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(gateway_windows) == []
        assert runs == []

    def test_managed_target_task_runs_once_and_recovers(self, monkeypatch, tmp_path):
        home = tmp_path / "target"
        template = self._managed_template(monkeypatch, home)
        runs: list[Path | None] = []

        assert gateway_windows._task_action_is_hermes_managed(
            template, template, task_name=_TASK
        )
        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        monkeypatch.setattr(
            gateway_windows,
            "_wait_for_gateway_ready",
            lambda **_kw: [4242] if runs else [],
        )
        monkeypatch.setattr(gateway_windows, "_query_scheduled_task_xml", lambda _task: template)
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda *, home=None: runs.append(home) or (0, "", ""),
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(
            gateway_windows, home=home
        ) == [4242]
        assert runs == [home]

    def test_recovery_race_skips_action_query_and_run(self, monkeypatch):
        query_calls: list[str] = []
        runs: list[object] = []

        monkeypatch.setattr(gateway_windows, "_task_registration_state", lambda **_kw: "registered")
        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777])
        monkeypatch.setattr(
            gateway_windows,
            "_query_scheduled_task_xml",
            lambda task: query_calls.append(task) or pytest.fail("must not query task Action"),
        )
        monkeypatch.setattr(
            gateway_windows,
            "_run_scheduled_task_once",
            lambda **_kw: runs.append(True) or pytest.fail("must not run task"),
        )

        assert update_cmd_windows._recover_windows_gateway_via_schtasks(gateway_windows) == [777]
        assert query_calls == []
        assert runs == []
