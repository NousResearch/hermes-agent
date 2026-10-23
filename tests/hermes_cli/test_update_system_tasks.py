"""Protected SYSTEM tasks retain their identity through update and recovery (#63743).

The old PR accepted arbitrary SYSTEM venv processes. These tests enter through
strict profile discovery and require the exact home's registered launcher instead.
No real task is registered, stopped or run, and no UAC prompt is displayed.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import psutil
import pytest

from hermes_cli import main, update_cmd, update_cmd_windows, update_pause_record
from hermes_cli import update_windows_tasks as tasks


def definition(home, name="Hermes_Gateway", principal="S-1-5-18", action="wscript.exe"):
    script = Path(home) / "gateway-service" / f"{name}.vbs"
    return (
        f'<Task xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">'
        f"<Principals><Principal><UserId>{principal}</UserId></Principal></Principals>"
        f"<Actions><Exec><Command>{action}</Command>"
        f'<Arguments>//B //Nologo "{escape(str(script))}"</Arguments></Exec></Actions></Task>'
    )


@pytest.mark.parametrize(
    "change", ["foreign_home", "user_principal", "other_command", "multiple_actions"]
)
def test_task_name_and_system_account_cannot_substitute_for_home_ownership(
    tmp_path, change
):
    xml = definition(tmp_path)
    assert tasks.definition_matches_home(xml, tmp_path, "Hermes_Gateway")
    if change == "foreign_home":
        xml = definition(tmp_path / "another-install")
    elif change == "user_principal":
        xml = definition(tmp_path, principal="fixture-user")
    elif change == "other_command":
        xml = definition(tmp_path, action="python.exe")
    else:
        xml = xml.replace(
            "</Actions>", "<Exec><Command>other.exe</Command></Exec></Actions>"
        )
    assert not tasks.definition_matches_home(xml, tmp_path, "Hermes_Gateway")


@pytest.fixture
def system_gateway(monkeypatch, tmp_path):
    from hermes_cli import gateway_windows

    profile = SimpleNamespace(profile="ops", path=tmp_path, pid=321, create_time=12.5)
    process = SimpleNamespace(create_time=lambda: 12.5)
    monkeypatch.setattr(psutil, "Process", lambda _pid: process)
    monkeypatch.setattr(tasks, "process_is_system", lambda _: True)
    monkeypatch.setattr(gateway_windows, "get_task_name", lambda: "Hermes_Gateway_ops")
    calls = []

    def operation(name, op, expected=""):
        calls.append((name, op, expected))
        return definition(tmp_path, name)

    monkeypatch.setattr(tasks, "task_operation", operation)
    return profile, process, calls


@pytest.mark.platforms("windows")
def test_only_strict_profile_gateways_are_candidates_and_service_gateways_stay_separate(
    system_gateway,
    monkeypatch,
):
    profile, _process, calls = system_gateway
    assert tasks.discover_profile_tasks({}, set()) == {}
    assert tasks.discover_profile_tasks({321: profile}, {321}) == {}
    assert calls == []
    found = tasks.discover_profile_tasks({321: profile}, set())
    assert found["ops"]["pid"] == 321
    assert found["ops"]["name"] == "Hermes_Gateway_ops"
    assert calls == [("Hermes_Gateway_ops", "query", "")]
    calls.clear()
    monkeypatch.setattr(tasks, "process_is_system", lambda _: False)
    assert tasks.discover_profile_tasks({321: profile}, set()) == {}
    assert calls == []


@pytest.mark.platforms("windows")
def test_reused_pid_is_rejected_before_task_query(system_gateway):
    profile, _process, calls = system_gateway
    profile.create_time = 9.0
    with pytest.raises(RuntimeError, match="identity changed"):
        tasks.discover_profile_tasks({321: profile}, set())
    assert calls == []


def test_task_stop_rechecks_birth_and_only_touches_survivors(
    monkeypatch, system_gateway
):
    _profile, process, calls = system_gateway
    recorded = {
        "ops": {
            "pid": 321,
            "created": 12.5,
            "name": "Hermes_Gateway_ops",
            "definition": "f" * 64,
        }
    }
    tasks.stop_profile_tasks(recorded, set())
    assert calls == []
    tasks.stop_profile_tasks(recorded, {321})
    assert calls == [("Hermes_Gateway_ops", "stop", "f" * 64)]
    calls.clear()
    monkeypatch.setattr(process, "create_time", lambda: 13.0)
    with pytest.raises(RuntimeError, match="identity changed"):
        tasks.stop_profile_tasks(recorded, {321})
    assert calls == []


def test_task_resume_uses_exact_registration_and_never_replays_argv(
    monkeypatch, system_gateway
):
    from hermes_cli import gateway, profiles

    _profile, _process, calls = system_gateway
    monkeypatch.setattr(
        gateway,
        "launch_detached_profile_gateway_restart",
        lambda *_: pytest.fail("SYSTEM task fell back to current-user replay"),
    )
    monkeypatch.setattr(tasks, "refresh_task_launcher", lambda _: None)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _: _profile.path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)
    recorded = {
        "ops": {
            "pid": 321,
            "home": str(_profile.path),
            "name": "Hermes_Gateway_ops",
            "definition": "a" * 64,
        }
    }
    assert tasks.relaunch_profile("ops", 321, recorded)
    assert calls == [
        ("Hermes_Gateway_ops", "query", "a" * 64),
        ("Hermes_Gateway_ops", "run", "a" * 64),
    ]
    calls.clear()
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _: _profile.path / "other")
    with pytest.raises(RuntimeError, match="home differs"):
        tasks.relaunch_profile("ops", 321, recorded)
    assert calls == []
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _: _profile.path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _: False)
    with pytest.raises(RuntimeError, match="no longer exists"):
        tasks.relaunch_profile("ops", 321, recorded)
    assert calls == []
    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)

    def cancelled(*_):
        raise RuntimeError("UAC cancelled")

    monkeypatch.setattr(tasks, "task_operation", cancelled)
    with pytest.raises(RuntimeError, match="cancelled"):
        tasks.relaunch_profile("ops", 321, recorded)


@pytest.mark.parametrize(
    "code,requested", [(0, True), (1223, False), (13, False), (124, True), (1, True)]
)
def test_async_stop_request_is_durable_but_cancellation_does_not_claim_a_stop(
    monkeypatch, system_gateway, code, requested
):
    _profile, _process, _calls = system_gateway
    notified = []
    recorded = {
        "ops": {
            "pid": 321,
            "created": 12.5,
            "name": "Hermes_Gateway_ops",
            "definition": "a" * 64,
        }
    }

    def operation(*_):
        if code:
            raise tasks.TaskOperationError(code, "fixture failure")
        return "<Task/>"

    monkeypatch.setattr(tasks, "task_operation", operation)
    if code:
        with pytest.raises(tasks.TaskOperationError):
            tasks.stop_profile_tasks(recorded, {321}, on_request=notified.append)
    else:
        tasks.stop_profile_tasks(recorded, {321}, on_request=notified.append)
    assert notified == ([321] if requested else [])


@pytest.mark.platforms("windows")
def test_task_exit_must_be_verified_before_update_continues(
    monkeypatch, system_gateway
):
    from gateway import status

    profile, _process, calls = system_gateway
    sent = []
    recorded = {
        "ops": {
            "pid": 321,
            "home": str(profile.path),
            "created": 12.5,
            "name": "Hermes_Gateway_ops",
            "definition": "a" * 64,
        }
    }
    monkeypatch.setattr(update_cmd, "_venv_launcher_ancestors", lambda _: [])
    monkeypatch.setattr(
        update_cmd, "_wait_for_windows_update_gateway_exit", lambda *a, **k: {321}
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_request_socket_pauses",
        lambda *a, **k: ({"ops": 321}, [321], []),
    )
    monkeypatch.setattr(update_cmd_windows, "_gateway_drain_timeout", lambda _: 0)
    monkeypatch.setattr(
        status,
        "terminate_pid",
        lambda *a, **k: pytest.fail("unverified task exit reached force kill"),
    )
    with pytest.raises(RuntimeError, match="did not exit"):
        update_cmd_windows._stop_windows_gateways(
            [321],
            {321: profile},
            set(),
            [],
            [],
            born={321: 1250},
            profile_tasks=recorded,
            on_request=sent.append,
        )
    assert calls == [("Hermes_Gateway_ops", "stop", "a" * 64)]
    assert sent == [321]


def test_task_identity_survives_pause_publication_and_orphan_adoption(
    monkeypatch, tmp_path
):
    from hermes_cli import gateway

    pid = os.getpid()
    profile = SimpleNamespace(
        profile="ops",
        path=tmp_path,
        pid=pid,
        create_time=psutil.Process(pid).create_time(),
    )
    recorded = {
        "ops": {"pid": pid, "name": "Hermes_Gateway_ops", "definition": "a" * 64}
    }
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(
        gateway, "find_profile_gateway_processes", lambda **_: [profile]
    )
    monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda **_: [])
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_: [pid])
    monkeypatch.setattr(tasks, "discover_profile_tasks", lambda *_: recorded)
    monkeypatch.setattr(
        update_cmd_windows, "_record_attested_cold_start_profiles", lambda *_: None
    )

    def stop(*_args, **kwargs):
        saved = update_pause_record.read(update_pause_record.record_path())
        assert saved is not None
        assert saved["token"]["profile_tasks"] == recorded
        assert kwargs["profile_tasks"] == recorded
        return {"ops": pid}

    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", stop)
    result = update_cmd_windows._pause_windows_gateways_for_update()
    assert result is not None
    assert result["profile_tasks"] == recorded
    adopted = update_pause_record.merge_into(None, result)
    assert adopted["profile_tasks"] == recorded
    newer = update_pause_record.merge_into({"profiles": {"ops": pid + 1}}, result)
    assert not newer["profile_tasks"]


@pytest.mark.platforms("windows")
def test_protected_resume_does_not_rewrite_task_or_discard_unready_debt(monkeypatch):
    from hermes_cli import gateway, profiles

    token = {
        "profiles": {"ops": 321},
        "profile_tasks": {
            "ops": {
                "pid": 321,
                "home": os.environ["HERMES_HOME"],
                "name": "Hermes_Gateway_ops",
                "definition": "a" * 64,
            }
        },
    }
    monkeypatch.setattr(
        update_cmd,
        "_refresh_windows_gateway_launchers",
        lambda: pytest.fail("task rewritten"),
    )
    monkeypatch.setattr(update_cmd_windows, "_resume_windows_services", lambda _: None)
    monkeypatch.setattr(
        update_cmd_windows, "_cold_start_attested_profiles", lambda _: None
    )
    monkeypatch.setattr(
        gateway,
        "launch_detached_profile_gateway_restart",
        lambda *_: pytest.fail("argv replay"),
    )
    monkeypatch.setattr(tasks, "task_operation", lambda *_: "")
    monkeypatch.setattr(tasks, "refresh_task_launcher", lambda _: None)
    monkeypatch.setattr(
        profiles, "get_profile_dir", lambda _: os.environ["HERMES_HOME"]
    )
    monkeypatch.setattr(profiles, "profile_exists", lambda _: True)

    def unready(*_):
        raise RuntimeError("gateway is not ready")

    monkeypatch.setattr(
        update_cmd_windows, "_verify_relaunched_gateways_alive", unready
    )
    with pytest.raises(RuntimeError, match="not ready"):
        update_cmd_windows._resume_paused_set(token)
    assert token["profiles"] == {"ops": 321}
    assert token["profile_tasks"]["ops"]["definition"] == "a" * 64


@pytest.mark.platforms("windows")
def test_legacy_launcher_is_refreshed_on_its_own_home_without_reregistering_task(
    monkeypatch, tmp_path
):
    from hermes_cli import gateway_windows
    from hermes_constants import get_hermes_home

    service = tmp_path / "gateway-service"
    service.mkdir()

    def settings(home):
        assert home == get_hermes_home() == tmp_path
        return ("fixture-python.exe", str(tmp_path), str(tmp_path), "")

    monkeypatch.setattr(gateway_windows, "_launcher_settings", settings)
    monkeypatch.setattr(
        gateway_windows, "_exec_schtasks", lambda *_: pytest.fail("task re-registered")
    )
    tasks.refresh_task_launcher({"home": str(tmp_path), "name": "Hermes_Gateway"})
    assert "fixture-python.exe" in (service / "Hermes_Gateway.vbs").read_text(
        encoding="utf-8"
    )
    assert "fixture-python.exe" in (service / "Hermes_Gateway.cmd").read_text(
        encoding="utf-8"
    )
