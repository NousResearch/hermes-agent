"""Container-restart survives per-profile gateway registrations.

The s6 dynamic scandir at /run/service/ lives on tmpfs and is wiped
on every container restart. Phase 4 Task 4.0's container_boot module
+ cont-init.d/02-reconcile-profiles regenerate the service slots from
$HERMES_HOME/profiles/<name>/gateway_state.json on every boot and
auto-start only those whose last state was `running`.

These tests stand up a container with a named volume, create profiles
inside it in various gateway states, restart the container, and
assert the reconciler did the right thing.
"""
from __future__ import annotations

import subprocess
import time

import pytest


def _docker(*args: str, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True, text=True, timeout=kw.pop("timeout", 60),
        **kw,
    )


def _exec(container: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return _docker("exec", container, *args, timeout=timeout)


def _sh(container: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return _docker("exec", container, "sh", "-c", cmd, timeout=timeout)


@pytest.fixture
def restart_container(request, built_image: str):
    """A long-running container with a named volume so docker restart
    preserves $HERMES_HOME/profiles/."""
    safe = request.node.name.replace("[", "_").replace("]", "_")
    name = f"hermes-restart-{safe}"
    volume = f"hermes-restart-vol-{safe}"
    _docker("rm", "-f", name)
    _docker("volume", "rm", "-f", volume)
    _docker("volume", "create", volume, timeout=10).check_returncode()
    r = _docker(
        "run", "-d", "--name", name,
        "-v", f"{volume}:/opt/data",
        built_image, "sleep", "infinity",
        timeout=30,
    )
    r.check_returncode()
    # Give s6 + stage2 + 02-reconcile a moment to come up cleanly on
    # the fresh volume.
    time.sleep(5)
    yield name
    _docker("rm", "-f", name)
    _docker("volume", "rm", "-f", volume)


def test_running_gateway_survives_container_restart(restart_container: str) -> None:
    container = restart_container

    # Create the profile + start its gateway. The Phase 4 hooks
    # register the s6 service slot during create and the dispatch
    # path brings it up via s6-svc -u.
    r = _exec(container, "hermes", "profile", "create", "coder")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    r = _exec(container, "hermes", "-p", "coder", "gateway", "start", timeout=60)
    assert r.returncode == 0, f"gateway start failed: {r.stderr}"

    # Give the service time to actually come up under supervision.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        r = _sh(container, "/command/s6-svstat /run/service/gateway-coder")
        if r.returncode == 0 and "up " in r.stdout:
            break
        time.sleep(0.5)
    assert "up " in r.stdout, f"gateway never came up pre-restart: {r.stdout!r}"

    # Persist state so the reconciler will treat the slot as 'running'
    # post-restart. The gateway process itself writes gateway_state.json
    # via gateway/status.py — but we don't want to wait for or assert
    # against the live process here; just stamp the file directly to
    # exercise the reconciler's contract.
    write_state = (
        "import json, pathlib; "
        "p = pathlib.Path('/opt/data/profiles/coder/gateway_state.json'); "
        "p.write_text(json.dumps({'gateway_state': 'running', 'timestamp': 1}))"
    )
    _exec(container, "python3", "-c", write_state, timeout=10).check_returncode()

    # Restart. After this, /run/service/ is empty until cont-init.d
    # runs the reconciler.
    _docker("restart", container, timeout=60).check_returncode()
    time.sleep(8)  # stage2 + reconcile + svscan rescan

    # Reconciler logged the action.
    r = _sh(container, "cat /opt/data/logs/container-boot.log")
    assert r.returncode == 0, f"reconcile log missing: {r.stderr}"
    assert "profile=coder" in r.stdout
    assert "action=started" in r.stdout

    # Service slot exists.
    r = _sh(container, "test -d /run/service/gateway-coder")
    assert r.returncode == 0, "slot not recreated after restart"

    # No `down` marker — we asked for auto-start.
    r = _sh(container, "test -f /run/service/gateway-coder/down")
    assert r.returncode != 0, "down marker present despite prior_state=running"


def test_stopped_gateway_stays_stopped_after_restart(restart_container: str) -> None:
    container = restart_container

    _exec(container, "hermes", "profile", "create", "writer").check_returncode()

    # Write 'stopped' directly so we don't have to race against the
    # gateway's own state writes.
    write_state = (
        "import json, pathlib; "
        "p = pathlib.Path('/opt/data/profiles/writer/gateway_state.json'); "
        "p.write_text(json.dumps({'gateway_state': 'stopped', 'timestamp': 1}))"
    )
    _exec(container, "python3", "-c", write_state, timeout=10).check_returncode()

    _docker("restart", container, timeout=60).check_returncode()
    time.sleep(8)

    # Slot exists.
    r = _sh(container, "test -d /run/service/gateway-writer")
    assert r.returncode == 0

    # Down marker present.
    r = _sh(container, "test -f /run/service/gateway-writer/down")
    assert r.returncode == 0, "down marker missing despite prior_state=stopped"


def test_stale_gateway_pid_cleaned_up_on_restart(restart_container: str) -> None:
    """A dead container's gateway.pid + processes.json must NOT
    survive the restart — a numerically-equal live PID in the new
    container is a different process and would confuse the gateway
    process-mismatch checks."""
    container = restart_container

    _exec(container, "hermes", "profile", "create", "ghost").check_returncode()

    # Stamp stale runtime files alongside a 'running' state so the
    # reconciler walks this profile.
    stamp = (
        "import json, pathlib; "
        "p = pathlib.Path('/opt/data/profiles/ghost'); "
        "(p / 'gateway_state.json').write_text(json.dumps({'gateway_state': 'stopped', 'timestamp': 1})); "
        "(p / 'gateway.pid').write_text(json.dumps({'pid': 99999, 'host': 'old'})); "
        "(p / 'processes.json').write_text('[]')"
    )
    _exec(container, "python3", "-c", stamp, timeout=10).check_returncode()

    _docker("restart", container, timeout=60).check_returncode()
    time.sleep(8)

    # Stale runtime files swept.
    r = _sh(container, "test -f /opt/data/profiles/ghost/gateway.pid")
    assert r.returncode != 0, "stale gateway.pid survived restart"
    r = _sh(container, "test -f /opt/data/profiles/ghost/processes.json")
    assert r.returncode != 0, "stale processes.json survived restart"
