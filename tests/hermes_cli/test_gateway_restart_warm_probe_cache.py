"""``hermes gateway restart`` of a manual gateway must not attach to the gateway it just stopped.

The host-gateway probe is memoized for ``HOST_GATEWAY_CACHE_TTL_S``. The restart probes before the
stop, the stop finishes well inside the TTL, and the in-process ``run_gateway`` then read the memo
("PID N serves this profile"), printed "nothing to start" and exited 0 with no gateway running.
"""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway import control_socket
from gateway import host_attach
from gateway import host_rendezvous as hr
from gateway import status as gateway_status
from hermes_cli import gateway as gw
from hermes_constants import get_hermes_home

PLAIN_ARGV = ["/usr/bin/python", "-m", "hermes_cli.main", "gateway", "run"]


class _ReachedStart(Exception):
    """The start path got past the host attach decision: a new gateway would be launched."""


@pytest.fixture(autouse=True)
def _cold_memo():
    host_attach.invalidate_host_gateway_cache()
    yield
    host_attach.invalidate_host_gateway_cache()


def _write_record(pid: int, home) -> None:
    payload = {
        "role": hr.ROLE_GATEWAY, "home": str(home), "pid": pid,
        "createTime": hr.process_create_time(pid), "host": "", "port": None,
        "protocolVersion": hr.HOST_PROTOCOL_VERSION, "tokenFingerprint": "",
        "profiles": [], "updatedAt": "",
    }
    path = hr.record_path(hr.ROLE_GATEWAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def manual_host_gateway(tmp_path, monkeypatch):
    """A REAL live process published as this home's manual host gateway, serving this profile."""
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    home = get_hermes_home()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _write_record(child.pid, home)
    identity = {"pid": child.pid, "hermes_home": str(home), "supervisor": "none",
                "launch_supervisor": "manual", "served_profiles": [{"home": str(home)}]}
    monkeypatch.setattr(control_socket, "identify_gateway", lambda dialled, **kw: identity)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)


def test_restart_with_a_warm_probe_does_not_attach_to_the_stopped_gateway(manual_host_gateway, monkeypatch):
    child = manual_host_gateway
    profile = host_attach.profile_name_for_home(get_hermes_home())
    # Warm the memo through the real probe, exactly as a pre-stop lookup would.
    warm = host_attach.host_gateway()
    assert warm is not None and warm.pid == child.pid and warm.serves(profile)

    def stop():
        child.terminate()
        child.wait(timeout=10)
        return True

    def reached_start(**_kw):
        raise _ReachedStart

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_guard_named_profile_under_multiplexer", lambda **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_reap_unsupervised_gateway_orphans", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gateway_status, "get_running_pid", lambda *a, **k: child.pid)
    monkeypatch.setattr(gw, "_capture_gateway_argv", lambda pid: PLAIN_ARGV)
    monkeypatch.setattr("hermes_cli.main_dashboard._get_systemd_service_for_pid", lambda pid: None)
    monkeypatch.setattr(gw, "stop_profile_gateway", stop)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda *a, **k: None)
    # The real run_gateway runs up to the host attach decision; the first guard past it stands in
    # for the launch.
    monkeypatch.setattr(gw, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gw, "_guard_supervised_gateway_conflict", reached_start)

    with pytest.raises(_ReachedStart):
        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=False))
    assert child.poll() is not None
