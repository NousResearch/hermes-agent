"""E2E: ``hermes gateway stop`` inside the s6 image is a PLANNED stop.

``S6ServiceManager.stop()`` must find the supervised PID through the real s6
binaries and write the planned-stop marker before ``s6-svc -d``. Otherwise the
gateway reads the SIGTERM as an unexpected kill: it exits 1, persists
``gateway_state=running`` (so the next boot auto-starts it, undoing the stop)
and skips the clean-shutdown marker. Unit fakes cannot prove this; only the
real ``s6-svstat`` shipped in the image can.

Every ``docker exec`` runs as ``hermes`` per the conftest module docstring.
"""
from __future__ import annotations

import json
import time

from tests.docker.conftest import docker_exec_sh, start_container, wait_for_docker_logs

SLOT = "/run/service/gateway-default"


def _svstat_field(container: str, field: str) -> str:
    r = docker_exec_sh(container, f"/command/s6-svstat -o {field} {SLOT}")
    return r.stdout.strip() if r.returncode == 0 else ""


def _state(container: str) -> dict:
    r = docker_exec_sh(container, "cat /opt/data/gateway_state.json 2>/dev/null")
    if r.returncode != 0 or not r.stdout.strip():
        return {}
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _gateway_state(container: str) -> str:
    return str(_state(container).get("gateway_state", ""))


def _agent_log(container: str) -> str:
    return docker_exec_sh(container, "cat /opt/data/logs/agent.log 2>/dev/null").stdout


def _wait(predicate, what: str, deadline_s: float = 90.0, interval_s: float = 0.5) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError(f"timed out after {deadline_s}s waiting for {what}")


def test_gateway_stop_under_s6_is_planned(built_image: str, container_name: str) -> None:
    start_container(built_image, container_name, cmd="gateway run")
    wait_for_docker_logs(container_name, "s6 supervision", deadline_s=60.0)

    # The supervised gateway (no platforms: it stays up for cron) must be up and
    # past startup, so its signal handlers and planned-stop watcher are installed.
    # Boot seeds gateway_state=running before the process exists, so readiness is
    # the gateway's OWN state write (it stamps its pid) plus its startup-done line.
    _wait(lambda: _svstat_field(container_name, "up") == "true", "gateway-default up")
    supervised_pid = _svstat_field(container_name, "pid")
    assert supervised_pid.isdigit() and int(supervised_pid) > 0, supervised_pid
    _wait(
        lambda: str(_state(container_name).get("pid")) == supervised_pid
        and _gateway_state(container_name) == "running",
        "supervised gateway's own gateway_state=running write",
    )
    _wait(lambda: "Press Ctrl+C to stop" in _agent_log(container_name), "gateway startup done")

    r = docker_exec_sh(container_name, "hermes gateway stop", timeout=90)
    assert r.returncode == 0, f"gateway stop failed: stdout={r.stdout!r} stderr={r.stderr!r}"

    _wait(lambda: _svstat_field(container_name, "up") == "false", "gateway-default down")
    logs = docker_exec_sh(
        container_name, "cat /opt/data/logs/agent.log /opt/data/logs/gateway.log 2>/dev/null"
    ).stdout

    assert _svstat_field(container_name, "exitcode") == "0", (
        f"supervised gateway exit: {_svstat_field(container_name, 'exitcode')!r}\n"
        f"svstat: {docker_exec_sh(container_name, f'/command/s6-svstat {SLOT}').stdout!r}\n{logs[-4000:]}"
    )
    _wait(lambda: _gateway_state(container_name) == "stopped", "gateway_state=stopped", deadline_s=15.0)
    assert "as a planned gateway stop" in logs, logs[-4000:]
    assert "unexpected signal" not in logs, logs[-4000:]
