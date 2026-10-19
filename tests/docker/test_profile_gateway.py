"""Harness: a named profile does not get a gateway of its own inside the container.

Multiplex-only (#118273, #118433): ONE gateway per container serves every profile. ``hermes
profile create`` still registers the profile's s6 slot (so ``--force`` has something to drive),
but ``hermes -p <profile> gateway start`` without ``--force`` refuses, names the escape, and
leaves the slot's want-state DOWN. ``--force`` is the only path to a second, supervised
per-profile gateway, and that path keeps working (its slot wants up, and stop takes it down).

NB: The harness profile has no model/auth configured. A ``--force``-started supervised
process may spin up (svstat ``up``) or exit fast and be throttled (``down …, want up``). We
assert the *want* intent the lifecycle command set, NOT the supervised process's health.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user (via
:func:`docker_exec_sh` in conftest); see the conftest module docstring.
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


def test_named_profile_gateway_start_refuses_without_force(
    built_image: str, container_name: str,
) -> None:
    start_container(built_image, container_name, cmd="sleep 120")

    r = _sh(container_name, f"hermes profile create {PROFILE}")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    # Profile create's s6-register hook still produces the slot (registered DOWN).
    r = _sh(container_name, f"test -d /run/service/gateway-{PROFILE}")
    assert r.returncode == 0, "s6 service slot not created on profile create"

    r = _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    assert r.returncode != 0, f"a named profile started its own gateway: {r.stdout!r}"
    assert not _svstat_wants_up(container_name), (
        f"refused start still flipped the slot's want-state: {_svstat(container_name)!r}")


def test_named_profile_gateway_force_start_then_stop(
    built_image: str, container_name: str,
) -> None:
    start_container(built_image, container_name, cmd="sleep 120")
    r = _sh(container_name, f"hermes profile create {PROFILE}")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    r = _sh(container_name, f"hermes -p {PROFILE} gateway start --force", timeout=60)
    assert r.returncode == 0, (
        f"--force gateway start failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )
    _wait_for_want_state(container_name, want_up=True)

    r = _sh(container_name, f"hermes -p {PROFILE} gateway stop", timeout=30)
    assert r.returncode == 0
    _wait_for_want_state(container_name, want_up=False)
