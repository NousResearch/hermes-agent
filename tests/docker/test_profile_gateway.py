"""Harness: per-profile gateway start/stop inside the container.

Phase 4 wires `hermes -p <profile> gateway start/stop` through the s6
ServiceManager dispatch path inside the container — so the lifecycle
commands now bring up an s6-supervised gateway rather than refusing
with the pre-Phase-4 informational message.

These tests were marked ``xfail(strict=True)`` through Phase 0–3 and
flip to plain ``test_…`` once Phase 4 lands (now).

NB: The harness profile created here has no model/auth configured,
so the gateway process itself will exit with code 1 on every start
attempt (s6 will keep restarting it). We assert against s6's
``want up`` / ``want down`` state — which reflects the lifecycle
command's intent, not the supervised process's health.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec_sh` in conftest); see the conftest module
docstring.
"""
from __future__ import annotations

import subprocess
import time

from tests.docker.conftest import docker_exec_sh

PROFILE = "test-harness-profile"


def _sh(
    container: str, command: str, timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return docker_exec_sh(container, command, timeout=timeout)


def _svstat(container: str) -> str:
    """Returns the raw s6-svstat output for the test profile's slot.
    /command/s6-svstat is called by absolute path because /command/
    isn't on PATH for docker-exec sessions."""
    r = _sh(container, f"/command/s6-svstat /run/service/gateway-{PROFILE}")
    return r.stdout if r.returncode == 0 else ""


def test_profile_create_then_gateway_start(
    built_image: str, container_name: str,
) -> None:
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)

    r = _sh(container_name, f"hermes profile create {PROFILE}")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    # Profile create's s6-register hook should have produced a service slot.
    r = _sh(container_name, f"test -d /run/service/gateway-{PROFILE}")
    assert r.returncode == 0, "s6 service slot not created on profile create"

    r = _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    assert r.returncode == 0, (
        f"gateway start failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )

    # After start, s6's intent is "up" — even if the supervised gateway
    # process spin-fails (no model/auth in the test profile), the
    # supervision-state contract holds.
    time.sleep(2)
    state = _svstat(container_name)
    assert "want up" in state, f"want up not in svstat: {state!r}"

    r = _sh(container_name, f"hermes -p {PROFILE} gateway stop", timeout=30)
    assert r.returncode == 0

    time.sleep(2)
    state = _svstat(container_name)
    assert "want up" not in state, f"want up still in svstat: {state!r}"


def test_profile_delete_stops_gateway(
    built_image: str, container_name: str,
) -> None:
    """Deleting a profile should stop its gateway and remove the s6
    service slot."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(3)

    _sh(container_name, f"hermes profile create {PROFILE}")
    _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    time.sleep(3)

    r = _sh(
        container_name,
        f"hermes profile delete {PROFILE} --yes",
        timeout=30,
    )
    assert r.returncode == 0, f"profile delete failed: {r.stderr}"

    time.sleep(2)
    # Service slot should be gone.
    r = _sh(container_name, f"test -d /run/service/gateway-{PROFILE}")
    assert r.returncode != 0, "s6 service slot still present after profile delete"
