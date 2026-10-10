"""A failed service pause keeps its record exactly when its rollback left gateways down.

``_pause_windows_gateways_for_update`` decides whether to abandon the durable pause record from the
service-pause failure. That decision is read from a typed ``rollback_failures`` list, never from the
error message's wording: a reworded message must not delete the record while gateways stay stopped.

Real pause producer and durable record; seams are the Windows-only discovery, the SCM stop/restore
calls and the ordinary-gateway stop (nothing is signalled).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, gateway_windows, update_cmd, update_cmd_windows
from hermes_cli import update_pause_record as pause_record
from hermes_cli.update_cmd_windows import ServicePauseFailed, _pause_windows_gateways_for_update


@pytest.mark.parametrize("restore_fails", [False, True])
def test_the_pause_record_survives_exactly_when_the_service_rollback_failed(monkeypatch, restore_fails):
    from hermes_cli import main
    pid = os.getpid()  # a live identity for the service's gateway; never signalled
    service = SimpleNamespace(name="hermes-gw", profile="default", gateway_pid=pid, gateway_create_time=1.0,
                              service_pid=pid, service_create_time=1.0, descendant_identities=())
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda strict=False: [])
    monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda profile_processes=(): [service])
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *a, **k: {})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None)

    def stop(*_a, **_k):
        raise OSError("access denied")

    def restore(_name):
        if restore_fails:
            raise OSError("service will not start")

    monkeypatch.setattr(update_cmd, "_stop_windows_gateway_service", stop)
    monkeypatch.setattr(update_cmd, "_restore_windows_gateway_service", restore)

    with pytest.raises(ServicePauseFailed) as failed:
        _pause_windows_gateways_for_update()
    assert bool(failed.value.rollback_failures) is restore_fails
    saved = pause_record.read(pause_record.record_path())
    assert (saved is not None) is restore_fails, \
        "a rolled-back pause kept its record" if saved else "the record of still-stopped gateways was deleted"


def test_update_pause_stops_supervisor_before_force_stopping_its_child(monkeypatch):
    """A retrying supervisor cannot replace a force-killed child during mutation."""
    from hermes_cli import main

    pid = os.getpid()
    process = SimpleNamespace(pid=pid, profile="default", path=os.environ["HERMES_HOME"], create_time=1.0)
    order = []
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda all_profiles=False: [pid])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda strict=False: [process])
    monkeypatch.setattr(gateway, "find_windows_gateway_services", lambda profile_processes=(): [])
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *a, **k: order.append("force-child") or {"default": pid})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *a: None)
    monkeypatch.setattr(gateway_windows, "pause_supervisor_for_update", lambda *, before_stop: before_stop() or order.append("arm-stop") or "nonce")
    monkeypatch.setattr(gateway_windows, "wait_for_supervisor_pause", lambda token: order.append(("ack-stop", token)))
    token = _pause_windows_gateways_for_update()

    assert order == ["arm-stop", "force-child", ("ack-stop", "nonce")]
    assert token["supervisor_paused_profiles"] == {"default": os.environ["HERMES_HOME"]}


def test_update_pauses_every_profile_supervisor_before_stopping_the_host_fleet(monkeypatch, tmp_path):
    """A sibling Task must not respawn its force-stopped child during a host-wide update."""
    from hermes_cli import main, profiles
    from hermes_constants import get_hermes_home_override

    default_home, sibling_home = tmp_path / "default", tmp_path / "profiles" / "sibling"
    default_home.mkdir(parents=True)
    sibling_home.mkdir(parents=True)
    processes = {
        101: SimpleNamespace(pid=101, profile="default", path=default_home, create_time=1.0),
        202: SimpleNamespace(pid=202, profile="sibling", path=sibling_home, create_time=1.0),
    }
    calls = []
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *_a, **_k: [("default", default_home), ("sibling", sibling_home)])
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: (processes, [], set(), [101, 202]))
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateways", lambda *_a, **_k: calls.append("stop") or {"default": 101, "sibling": 202})
    monkeypatch.setattr(update_cmd_windows, "_record_attested_cold_start_profiles", lambda *_a: None)
    monkeypatch.setattr(gateway_windows, "pause_supervisor_for_update", lambda *, before_stop: before_stop() or calls.append(("arm", get_hermes_home_override())) or "nonce")
    monkeypatch.setattr(gateway_windows, "wait_for_supervisor_pause", lambda nonce: calls.append(("ack", get_hermes_home_override(), nonce)))

    token = _pause_windows_gateways_for_update()

    assert calls == [
        ("arm", str(default_home)), ("arm", str(sibling_home)), "stop",
        ("ack", str(default_home), "nonce"), ("ack", str(sibling_home), "nonce"),
    ]
    assert token["supervisor_paused_profiles"] == {"default": str(default_home), "sibling": str(sibling_home)}


def test_update_does_not_arm_a_supervisor_before_its_pause_is_durable(monkeypatch, tmp_path):
    """A record-write abort cannot leave a one-shot Task stop marker stranded."""
    from hermes_cli import main

    home = tmp_path / "default"
    home.mkdir()
    process = SimpleNamespace(pid=101, profile="default", path=home, create_time=1.0)
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({101: process}, [], set(), [101]))
    monkeypatch.setattr(update_cmd_windows, "_windows_supervisor_profile_homes", lambda *_a: {"default": str(home)})
    monkeypatch.setattr(pause_record, "record_pause", lambda *_a: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(gateway_windows, "pause_supervisor_for_update", lambda **_kw: pytest.fail("marker armed before record"))

    with pytest.raises(RuntimeError, match="Could not record"):
        _pause_windows_gateways_for_update()


def test_update_pauses_a_retrying_supervisor_even_when_its_child_is_absent(monkeypatch, tmp_path):
    """The retry-delay window is live ownership, not a cold-start case."""
    from hermes_cli import main, profiles
    from hermes_constants import get_hermes_home_override

    home = tmp_path / "default"
    home.mkdir()
    calls = []
    monkeypatch.setattr(main, "_is_windows", lambda: True)
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda *_a, **_k: [("default", home)])
    monkeypatch.setattr(update_cmd_windows, "_discover_windows_gateways", lambda: ({}, [], set(), []))
    monkeypatch.setattr(update_cmd_windows, "_cold_start_pause_token", lambda *_a: None)
    monkeypatch.setattr(gateway_windows, "pause_supervisor_for_update", lambda *, before_stop: before_stop() or calls.append(("arm", get_hermes_home_override())) or "nonce")
    monkeypatch.setattr(gateway_windows, "wait_for_supervisor_pause", lambda nonce: calls.append(("ack", get_hermes_home_override(), nonce)))

    token = _pause_windows_gateways_for_update()

    assert calls == [("arm", str(home)), ("ack", str(home), "nonce")]
    assert token["supervisor_paused_profiles"] == {"default": str(home)}


def test_update_resume_restarts_a_paused_supervisor_through_its_task(monkeypatch):
    """Update resume hands ownership back to the Scheduled Task, never a replay child."""
    from hermes_cli import main

    started = []
    monkeypatch.setattr(main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(gateway_windows, "start", lambda: started.append("task"))
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_k: [20])
    monkeypatch.setattr(
        update_cmd_windows, "_relaunch_paused_gateways",
        lambda *_a: pytest.fail("must not replay a direct child beside supervisor"),
    )

    token = {"supervisor_paused": True, "profiles": {"default": 10}, "unmapped": []}
    update_cmd_windows._resume_paused_set(token)

    assert started == ["task"]
    assert token["supervisor_paused"] is False
    assert token["profiles"] == {}


def test_update_resume_keeps_supervisor_debt_when_task_child_never_becomes_ready(monkeypatch):
    """A successful schtasks request is not a restart until its gateway survives readiness."""
    from hermes_cli import main

    monkeypatch.setattr(main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(gateway_windows, "start", lambda: None)
    monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", lambda **_k: [])

    token = {"resume_needed": True, "supervisor_paused_profiles": {"default": "C:/home"},
             "profiles": {"default": 10}, "unmapped": []}
    with pytest.raises(RuntimeError, match="did not become ready"):
        update_cmd_windows._resume_paused_set(token)

    assert token["resume_needed"] is True
    assert token["supervisor_paused_profiles"] == {"default": "C:/home"}
    assert token["profiles"] == {"default": 10}
