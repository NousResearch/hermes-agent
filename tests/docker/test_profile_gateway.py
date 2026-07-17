"""Harness: per-profile gateway start/stop inside the container.

Phase 4 wires `hermes -p <profile> gateway start/stop` through the s6
ServiceManager dispatch path inside the container — so the lifecycle
commands now bring up an s6-supervised gateway rather than refusing
with the pre-Phase-4 informational message.

These tests were marked ``xfail(strict=True)`` through Phase 0–3 and
flip to plain ``test_…`` once Phase 4 lands (now).

NB: The harness profile has no model/auth configured. Depending on
how the gateway run script handles missing config, the supervised
process may either spin up successfully (and svstat reports ``up``)
or exit fast and get throttled by s6 (and svstat reports ``down …,
want up``). Both states are valid "user asked for gateway up" results
— what we assert is the *want* intent the lifecycle command set, NOT
the supervised process's health. ``s6-svc -u`` records ``want up`` in
the supervise/status file regardless of the run-script outcome.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec_sh` in conftest); see the conftest module
docstring.
"""
from __future__ import annotations

import subprocess
import time

from tests.docker.conftest import docker_exec_sh, start_container

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


def _svstat_wants_up(container: str) -> bool:
    """Read the slot's want-state from s6-svstat output.

    s6-svstat formats the output to elide redundancies — when the
    service is currently up AND s6 wants it up, the literal token
    ``want up`` doesn't appear (it's implicit from the leading ``up``).
    When the service is down but s6 wants it back up, ``, want up``
    appears explicitly. So a comprehensive "is the want-intent set to
    up" check has to accept both spellings.
    """
    state = _svstat(container)
    if not state:
        return False
    head = state.split()[0] if state.split() else ""
    if head == "up":
        # Currently up implies wanted-up unless ``want down`` is set.
        return "want down" not in state
    # Currently down — ``want up`` only shows up when explicitly set.
    return "want up" in state



def _wait_for_want_state(container_name: str, want_up: bool, timeout: float = 15.0) -> None:
    """Poll s6 want-state until it matches, instead of a fixed sleep.

    s6 state transitions are asynchronous; fixed two-second sleeps flaked
    on loaded CI hosts.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _svstat_wants_up(container_name) == want_up:
            return
        time.sleep(0.5)
    state = "up" if want_up else "down"
    raise AssertionError(
        f"slot want-state never became {state} within {timeout}s: "
        f"{_svstat(container_name)!r}"
    )


def test_profile_create_then_gateway_start(
    built_image: str, container_name: str,
) -> None:
    start_container(built_image, container_name, cmd="sleep 120")

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
    # supervision-state contract holds. See ``_svstat_wants_up`` for
    # why we accept both ``up …`` (currently up) and ``down …, want
    # up`` (down but s6 wants up).
    _wait_for_want_state(container_name, want_up=True)

    r = _sh(container_name, f"hermes -p {PROFILE} gateway stop", timeout=30)
    assert r.returncode == 0

    _wait_for_want_state(container_name, want_up=False)


def test_profile_delete_stops_gateway(
    built_image: str, container_name: str,
) -> None:
    """Deleting a profile should stop its gateway and remove the s6
    service slot."""
    start_container(built_image, container_name, cmd="sleep 120")

    _sh(container_name, f"hermes profile create {PROFILE}")
    _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    _wait_for_want_state(container_name, want_up=True)

    r = _sh(
        container_name,
        f"hermes profile delete {PROFILE} --yes",
        timeout=30,
    )
    assert r.returncode == 0, f"profile delete failed: {r.stderr}"

    # Poll for slot removal instead of a fixed sleep.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        r = _sh(container_name, f"test -d /run/service/gateway-{PROFILE}")
        if r.returncode != 0:
            break
        time.sleep(0.5)
    assert r.returncode != 0, "s6 service slot still present after profile delete"