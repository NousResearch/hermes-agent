"""Harness: per-profile gateway start/stop inside the container.

Phase 4 will change the *implementation* of these commands inside the
container — they'll talk to s6 instead of refusing. The user-visible
surface that should result is locked here.

NOTE: These tests are marked ``xfail(strict=True)`` until Phase 4 lands.
The current tini image deliberately refuses gateway start/stop inside
containers — ``pgrep`` finds nothing and the tests fail. After Phase 4
they should flip to passing automatically; ``strict=True`` means an
unexpected pass also fails the test, protecting against side-channel
fixes outside the planned Phase 4 mechanism.
"""
from __future__ import annotations

import subprocess
import time

import pytest

PROFILE = "test-harness-profile"

_PHASE4_REASON = (
    "Phase 4 not yet landed: container-side `hermes gateway start` "
    "currently exits 0 with an informational message instead of "
    "spawning/supervising a gateway. Remove this marker after Task 4.3."
)


def _sh(
    container: str, command: str, timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", command],
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.xfail(reason=_PHASE4_REASON, strict=True)
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

    r = _sh(container_name, f"hermes -p {PROFILE} gateway start", timeout=60)
    assert r.returncode == 0, (
        f"gateway start failed: stderr={r.stderr!r} stdout={r.stdout!r}"
    )

    time.sleep(3)

    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode == 0, "gateway process not running"

    r = _sh(container_name, f"hermes -p {PROFILE} gateway stop", timeout=30)
    assert r.returncode == 0

    time.sleep(2)

    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode != 0, "gateway process still running after stop"


@pytest.mark.xfail(reason=_PHASE4_REASON, strict=True)
def test_profile_delete_stops_gateway(
    built_image: str, container_name: str,
) -> None:
    """Deleting a profile should stop its gateway if running."""
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
    assert r.returncode == 0

    time.sleep(2)
    r = _sh(container_name, f"pgrep -f 'gateway.*{PROFILE}'")
    assert r.returncode != 0, "gateway still running after profile delete"
