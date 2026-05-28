"""Container-restart survives per-profile gateway registrations.

The s6 dynamic scandir at /run/service/ lives on tmpfs and is wiped
on every container restart. Phase 4 Task 4.0's container_boot module
+ cont-init.d/02-reconcile-profiles regenerate the service slots from
$HERMES_HOME/profiles/<name>/gateway_state.json on every boot and
auto-start only those whose last state was `running`.

These tests stand up a container with a named volume, create profiles
inside it in various gateway states, restart the container, and
assert the reconciler did the right thing.

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec` / :func:`docker_exec_sh` in conftest); see
the conftest module docstring.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from tests.docker.conftest import docker_exec, docker_exec_sh


def _docker(*args: str, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True, text=True, timeout=kw.pop("timeout", 60),
        **kw,
    )


def _exec(container: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return docker_exec(container, *args, timeout=timeout)


def _sh(container: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return docker_exec_sh(container, cmd, timeout=timeout)


def _wait_for_path(
    container: str,
    path: str,
    *,
    kind: str = "f",
    deadline_s: float = 30.0,
    interval_s: float = 0.25,
) -> bool:
    """Poll `test -<kind> <path>` inside container until success or timeout.

    `kind` is the `test` flag: 'f' for file, 'd' for directory, 'e' for
    existence. Returns True on success, False on timeout. Strictly
    better than a fixed `time.sleep()` because:

      * we don't wait the full budget when the path appears early, and
      * the test fails with a precise "waited N seconds" assertion
        instead of a confusing one-line failure mid-test when the
        sleep was too short.
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        r = _sh(container, f"test -{kind} {path}", timeout=5)
        if r.returncode == 0:
            return True
        time.sleep(interval_s)
    return False


def _wait_for_reconcile_log_mention(
    container: str,
    profile: str,
    *,
    deadline_s: float = 30.0,
    interval_s: float = 0.25,
) -> str:
    """Poll until /opt/data/logs/container-boot.log mentions `profile`.

    Returns the matching log content on success. On timeout, returns
    the last observed contents so the assertion can render a
    meaningful diagnostic. The container-boot.log is the explicit
    signal that the reconciler has finished — much more reliable
    than a fixed sleep that hopes 8 seconds is enough.
    """
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        r = _sh(container, "cat /opt/data/logs/container-boot.log", timeout=5)
        if r.returncode == 0:
            last = r.stdout
            if f"profile={profile}" in last:
                return last
        time.sleep(interval_s)
    return last


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
    # Wait for s6 + stage2 + 02-reconcile to publish the boot log so
    # the test can rely on the default slot being registered before
    # it starts issuing commands. The reconciler always writes one
    # 'default' line on every boot (PR #30136 item I1) — that's our
    # readiness signal.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        r = _docker(
            "exec", "-u", "hermes", name, "sh", "-c",
            "cat /opt/data/logs/container-boot.log 2>/dev/null",
            timeout=5,
        )
        if r.returncode == 0 and "profile=default" in r.stdout:
            break
        time.sleep(0.25)
    else:
        # Defensive: surface a timeout from the fixture itself so the
        # test failure points at "container never finished cont-init"
        # rather than mid-test where the symptom would be obscure.
        raise RuntimeError(
            f"container {name} did not finish cont-init within 30s"
        )
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
    # runs the reconciler. We need to wait long enough for the
    # reconciler to write coder's entry to the boot log AND for
    # s6-svscan to spin up the service supervise tree from the
    # restored slot. Polling the boot log gives us the first signal.
    _docker("restart", container, timeout=60).check_returncode()
    log = _wait_for_reconcile_log_mention(container, "coder", deadline_s=30.0)
    assert "profile=coder" in log, (
        f"reconciler never logged coder after restart: {log!r}"
    )
    assert "action=started" in log

    # Service slot exists.
    assert _wait_for_path(
        container, "/run/service/gateway-coder", kind="d", deadline_s=10.0,
    ), "slot not recreated after restart"

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
    log = _wait_for_reconcile_log_mention(container, "writer", deadline_s=30.0)
    assert "profile=writer" in log

    # Slot exists.
    assert _wait_for_path(
        container, "/run/service/gateway-writer", kind="d", deadline_s=10.0,
    )

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
    _wait_for_reconcile_log_mention(container, "ghost", deadline_s=30.0)

    # Stale runtime files swept.
    r = _sh(container, "test -f /opt/data/profiles/ghost/gateway.pid")
    assert r.returncode != 0, "stale gateway.pid survived restart"
    r = _sh(container, "test -f /opt/data/profiles/ghost/processes.json")
    assert r.returncode != 0, "stale processes.json survived restart"
