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

from tests.docker.conftest import docker_exec, docker_exec_sh, wait_for_path, wait_for_log, wait_for_docker_logs, poll_container


def _docker(*args: str, **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True, text=True, timeout=kw.pop("timeout", 60),
        **kw,
    )





def _wait_for_reconcile_log_mention(
    container: str,
    profile: str,
    *,
    deadline_s: float = 30.0,
    interval_s: float = 0.25,
) -> str:
    """Poll until /opt/data/logs/container-boot.log mentions `profile`.
    """
    return wait_for_log(container, "/opt/data/logs/container-boot.log",  f"profile={profile}")


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
    wait_for_log(name, "/opt/data/logs/container-boot.log", "profile=default")
    yield name
    _docker("rm", "-f", name)
    _docker("volume", "rm", "-f", volume)


def test_running_gateway_survives_container_restart(restart_container: str) -> None:
    container = restart_container

    # Create the profile + start its gateway. The Phase 4 hooks
    # register the s6 service slot during create and the dispatch
    # path brings it up via s6-svc -u.
    r = docker_exec(container, "hermes", "profile", "create", "coder")
    assert r.returncode == 0, f"profile create failed: {r.stderr}"

    r = docker_exec(container, "hermes", "-p", "coder", "gateway", "start", timeout=60)
    assert r.returncode == 0, f"gateway start failed: {r.stderr}"

    # Give the service time to actually come up under supervision.
    poll_container(container, "/command/s6-svstat /run/service/gateway-coder | grep -q 'up '")

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
    docker_exec(container, "python3", "-c", write_state, timeout=10).check_returncode()

    # Restart. After this, /run/service/ is empty until cont-init.d
    # runs the reconciler. We need to wait long enough for the
    # reconciler to write coder's entry to the boot log AND for
    # s6-svscan to spin up the service supervise tree from the
    # restored slot. Polling the boot log gives us the first signal.
    _docker("restart", container, timeout=60).check_returncode()
    log = _wait_for_reconcile_log_mention(container, "coder", deadline_s=30.0)
    assert "action=started" in log

    # Service slot exists.
    assert wait_for_path(
        container, "/run/service/gateway-coder", kind="d", deadline_s=10.0,
    ), "slot not recreated after restart"

    # No `down` marker — we asked for auto-start.
    r = docker_exec_sh(container, "test -f /run/service/gateway-coder/down")
    assert r.returncode != 0, "down marker present despite prior_state=running"


def test_stopped_gateway_stays_stopped_after_restart(restart_container: str) -> None:
    container = restart_container

    docker_exec(container, "hermes", "profile", "create", "writer").check_returncode()

    # Write 'stopped' directly so we don't have to race against the
    # gateway's own state writes.
    write_state = (
        "import json, pathlib; "
        "p = pathlib.Path('/opt/data/profiles/writer/gateway_state.json'); "
        "p.write_text(json.dumps({'gateway_state': 'stopped', 'timestamp': 1}))"
    )
    docker_exec(container, "python3", "-c", write_state, timeout=10).check_returncode()

    _docker("restart", container, timeout=60).check_returncode()
    _wait_for_reconcile_log_mention(container, "writer", deadline_s=30.0)

    # Slot exists.
    assert wait_for_path(
        container, "/run/service/gateway-writer", kind="d", deadline_s=10.0,
    )

    # Down marker present.
    r = docker_exec_sh(container, "test -f /run/service/gateway-writer/down")
    assert r.returncode == 0, "down marker missing despite prior_state=stopped"


def test_stale_gateway_pid_cleaned_up_on_restart(restart_container: str) -> None:
    """A dead container's gateway.pid + processes.json must NOT
    survive the restart — a numerically-equal live PID in the new
    container is a different process and would confuse the gateway
    process-mismatch checks."""
    container = restart_container

    docker_exec(container, "hermes", "profile", "create", "ghost").check_returncode()

    # Stamp stale runtime files alongside a 'running' state so the
    # reconciler walks this profile.
    stamp = (
        "import json, pathlib; "
        "p = pathlib.Path('/opt/data/profiles/ghost'); "
        "(p / 'gateway_state.json').write_text(json.dumps({'gateway_state': 'stopped', 'timestamp': 1})); "
        "(p / 'gateway.pid').write_text(json.dumps({'pid': 99999, 'host': 'old'})); "
        "(p / 'processes.json').write_text('[]')"
    )
    docker_exec(container, "python3", "-c", stamp, timeout=10).check_returncode()

    _docker("restart", container, timeout=60).check_returncode()
    _wait_for_reconcile_log_mention(container, "ghost", deadline_s=30.0)

    # Stale runtime files swept.
    r = docker_exec_sh(container, "test -f /opt/data/profiles/ghost/gateway.pid")
    assert r.returncode != 0, "stale gateway.pid survived restart"
    r = docker_exec_sh(container, "test -f /opt/data/profiles/ghost/processes.json")
    assert r.returncode != 0, "stale processes.json survived restart"


def test_live_gateway_autostarts_after_real_restart_without_manual_state_stamp(
    restart_container: str,
) -> None:
    """End-to-end guard for issue #42675.

    The other tests in this module stamp gateway_state.json directly to
    exercise the reconciler's READ side. This one exercises the WRITE
    side: a real, live gateway is killed by the container/s6 SIGTERM that
    `docker restart` sends — no manual state stamp — and must come back up
    on the next boot.

    Before the fix, the shutdown handler unconditionally persisted
    gateway_state=stopped on that SIGTERM, so the reconciler saw 'stopped'
    and registered the slot DOWN — the gateway silently stayed dark after
    every container restart. The fix classifies an unmarked SIGTERM as
    signal-initiated and persists 'running' instead, so auto-start works.
    """
    container = restart_container

    docker_exec(container, "hermes", "profile", "create", "live").check_returncode()
    r = docker_exec(container, "hermes", "-p", "live", "gateway", "start", timeout=60)
    assert r.returncode == 0, f"gateway start failed: {r.stderr}"

    # Wait for the gateway to actually come up under supervision AND write
    # its own gateway_state=running (we do NOT stamp it ourselves).
    poll_container(container, "/command/s6-svstat /run/service/gateway-live |  grep -q 'up '")

    # Confirm the gateway persisted its own 'running' state. The gateway has
    # to boot Python, discover ~50 plugins, construct GatewayRunner, and
    # reach write_runtime_status("running") at run.py start() — on a loaded
    # CI runner with parallel docker test containers competing for CPU, this
    # can take a while.
    wait_for_log(container, "/opt/data/profiles/live/gateway_state.json", '"running"', deadline_s=45, interval_s=1)

    # Real restart — Docker sends SIGTERM to PID 1; s6 propagates it to the
    # supervised gateway. No planned-stop marker is written (this is not an
    # operator `hermes gateway stop`), so the shutdown is signal-initiated.
    _docker("restart", container, timeout=60).check_returncode()

    log = _wait_for_reconcile_log_mention(container, "live", deadline_s=30.0)
    # The crux: the reconciler must AUTO-START it, not register it down.
    assert "action=started" in log, (
        f"gateway did NOT auto-start after a real restart (issue #42675 "
        f"regression): {log!r}"
    )

    # Slot recreated, and NO down marker (we expect auto-start).
    assert wait_for_path(
        container, "/run/service/gateway-live", kind="d", deadline_s=10.0,
    ), "slot not recreated after restart"
    r = docker_exec_sh(container, "test -f /run/service/gateway-live/down")
    assert r.returncode != 0, (
        "down marker present despite a live gateway being restarted — "
        "the signal-initiated shutdown wrongly persisted 'stopped' (#42675)"
    )
