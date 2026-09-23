"""Harness: `docker run <image> gateway run` redirects to supervised mode.

Before the s6 migration, ``docker run nousresearch/hermes-agent gateway
run`` was the standard pattern — the gateway ran as the container's
main process, container exit code matched gateway exit code, no
supervision. With s6 as PID 1, the same invocation now auto-redirects
to the supervised path (`gateway start`) so users get auto-restart on
crash and a supervised dashboard alongside (when ``HERMES_DASHBOARD=1``).

These tests verify the three load-bearing properties of that redirect:

  1. The default invocation **does** redirect (container stays up via
     ``sleep infinity`` while s6 supervises ``gateway-default``).
  2. ``--no-supervise`` / ``HERMES_GATEWAY_NO_SUPERVISE=1`` opts out.
  3. The supervised process itself does NOT recurse — the
     ``HERMES_S6_SUPERVISED_CHILD`` sentinel breaks the loop.

Every ``docker exec`` runs as ``hermes`` per the conftest module
docstring; see ``tests/docker/conftest.py`` for rationale.
"""
from __future__ import annotations

import subprocess
import time

from tests.docker.conftest import (
    docker_exec_sh,
    start_container,
    wait_for_docker_logs,
)


def _svstat(container: str, slot: str = "gateway-default") -> str:
    r = docker_exec_sh(container, f"/command/s6-svstat /run/service/{slot}")
    return r.stdout if r.returncode == 0 else ""


def _svstat_wants_up(container: str, slot: str = "gateway-default") -> bool:
    """See test_profile_gateway._svstat_wants_up for the format rules."""
    state = _svstat(container, slot)
    if not state:
        return False
    head = state.split()[0] if state.split() else ""
    if head == "up":
        return "want down" not in state
    return "want up" in state


def _wait_for_gateway_or_exit(
    container: str,
    *,
    deadline_s: float = 60.0,
) -> str:
    """Poll until the container is either running a foreground gateway
    process or has exited.  Returns the final container status.

    Used by the ``--no-supervise`` tests where the gateway runs as the
    CMD process (not supervised by s6).  Under CI load the gateway can
    take well over 6s to finish Python imports and reach the gateway
    entrypoint — a fixed ``time.sleep(6)`` races.  Polling for
    ``pgrep -f 'hermes.*gateway'`` (the gateway is running) or
    ``docker inspect`` returning ``exited`` is both faster on quick
    machines and flake-free on slow ones.
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            capture_output=True, text=True, timeout=10,
        )
        status = r.stdout.strip()
        if status == "exited":
            return "exited"
        if status == "running":
            # Check if the gateway process is actually running in the
            # foreground (the no-supervise path).  If it is, we're done.
            pgrep = docker_exec_sh(
                container, "pgrep -f 'hermes.*gateway' >/dev/null 2>&1",
            )
            if pgrep.returncode == 0:
                return "running"
        time.sleep(0.5)
    return status


def test_gateway_run_redirects_to_supervised(
    built_image: str, container_name: str,
) -> None:
    """``docker run <image> gateway run`` (the historical invocation)
    should now register and start the ``gateway-default`` s6 slot.

    The CMD process itself shouldn't be the gateway — it should be
    blocked on ``sleep infinity``, leaving s6 to supervise the actual
    gateway process. We verify by:

      * Confirming the CMD process is sleeping (not python/gateway).
      * Confirming ``s6-svstat gateway-default`` reports want-up.
    """
    # Start the container detached using the historical gateway-run
    # pattern. The redirect should fire and the container should NOT
    # exit immediately (which is what would happen pre-this-PR on the
    # s6 image — the foreground gateway would crash without config,
    # the CMD would exit, /init would shut down).
    start_container(built_image, container_name, cmd="gateway run")

    # Wait for the redirect breadcrumb to appear in docker logs.
    # Under heavy parallel load (32-way docker test fan-out), the CMD
    # process (main-wrapper.sh → python → hermes gateway run) can take
    # well over 5s to reach the redirect logic. The breadcrumb is the
    # definitive signal that the redirect fired — polling for it is
    # both faster on quick machines and flake-free on slow ones.
    # Under heavy parallel docker load (32-way fan-out), the CMD process
    # (main-wrapper.sh → python → hermes gateway run) can take well over
    # 30s to import the codebase, load config, and reach the redirect
    # logic. 60s matches the deadline other boot-readiness polls use.
    logs = wait_for_docker_logs(
        container_name, "s6 supervision", deadline_s=60.0,
    )
    assert "s6 supervision" in logs, (
        f"expected loud breadcrumb in docker logs; got:\n{logs}"
    )
    assert "--no-supervise" in logs, (
        f"breadcrumb missing opt-out hint; got:\n{logs}"
    )

    # Container should still be running. If the redirect didn't fire,
    # the foreground gateway would have crashed and the container
    # would be in `Exited` state by now.
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0 and r.stdout.strip() == "running", (
        f"container exited prematurely: {r.stdout!r}; "
        f"docker logs:\n{logs}"
    )

    # s6's intent for the default-profile gateway slot should be up.
    # Same accept-either rule as test_profile_gateway: the supervised
    # gateway may or may not be currently up depending on whether the
    # harness profile has a configured model, but the want-intent
    # contract holds either way.
    assert _svstat_wants_up(container_name), (
        f"gateway-default slot want-state not up: {_svstat(container_name)!r}"
    )

    # The CMD process (PID under /init that the wrapper exec'd into)
    # should be sleeping, not the gateway. We count `sleep infinity`
    # processes parented to the CMD wrapper (main-wrapper.sh / rc.init
    # top), NOT the static main-hermes service's sleep — a bare grep
    # for `sleep infinity` would false-positive on the main-hermes
    # sleep and pass even before the redirect fires.
    r = docker_exec_sh(
        container_name,
        "ps -eo pid,ppid,cmd | grep -v grep | awk "
        "'/main-wrapper.sh|rc.init top/ { wrapper_pid=$1 } "
        "$3==\"sleep\" && $4==\"infinity\" && $2==wrapper_pid { c++ } "
        "END { print c+0 }'",
    )
    assert r.returncode == 0
    redirected_sleeps = int(r.stdout.strip() or 0)
    assert redirected_sleeps == 1, (
        f"expected one `sleep infinity` heartbeat parented to the CMD "
        f"wrapper (the redirect); found {redirected_sleeps}. "
        f"ps:\n{docker_exec_sh(container_name, 'ps -eo pid,ppid,cmd').stdout}"
    )


    # If status == "exited" instead, the gateway exited (also valid
    # pre-s6 semantics). The breadcrumb-absence check above is
    # already enough to confirm the redirect didn't fire.




def test_supervised_gateway_does_not_recurse(
    built_image: str, container_name: str,
) -> None:
    """The HERMES_S6_SUPERVISED_CHILD sentinel must prevent the
    supervised ``hermes gateway run`` from re-entering the redirect.

    If recursion happened, every supervised gateway start would itself
    re-dispatch to s6 and exec ``sleep infinity`` — so the supervised
    gateway slot would never actually run a python ``hermes gateway
    run`` process. The slot would oscillate or settle into a state
    with no python in the supervise tree at all.

    We verify by counting python processes whose argv contains
    ``gateway run``: there should be at most one (the legitimately
    supervised gateway). Two or more would imply recursive spawning
    via the redirect → start → run → redirect → ... loop.
    """
    start_container(built_image, container_name, cmd="gateway run")

    # Wait for the redirect to fire by polling for the breadcrumb.
    # Under CI parallel docker test fan-out, the CMD process
    # (main-wrapper.sh → python → hermes gateway run) can take well
    # over 6s to reach the redirect logic. A fixed sleep would race:
    # if we check too early, the CMD process hasn't exec'd into
    # `sleep infinity` yet and the s6-supervised gateway hasn't
    # started either — so we'd see the CMD's `hermes gateway run`
    # AND the supervised one (2 processes) and falsely conclude
    # recursion. Polling the breadcrumb is the definitive signal
    # that the redirect fired and the CMD process is now `sleep`.
    wait_for_docker_logs(container_name, "s6 supervision")

    # Now that the redirect fired, count python processes running
    # `hermes gateway run`. If the recursion guard fails, s6 would
    # respawn fresh `gateway run` processes on every cycle, leaving
    # multiple Python-process descendants under the gateway-default
    # supervise tree.
    r = docker_exec_sh(container_name, "ps -eo pid,cmd | grep -v grep | grep -E 'python.*hermes.*gateway run' | wc -l")
    assert r.returncode == 0
    n = int(r.stdout.strip() or 0)
    assert n <= 1, (
        f"expected at most one supervised python `hermes gateway run` "
        f"process (the legitimately-supervised gateway); found {n}. "
        f"Recursion guard may have failed. "
        f"ps:\n{docker_exec_sh(container_name, 'ps -eo pid,ppid,cmd').stdout}"
    )

    # Stronger positive assertion: there should be exactly one
    # `sleep infinity` process whose parent is the main-wrapper.sh
    # CMD process (PID 17 typically). The static `main-hermes`
    # service has its own `sleep infinity` child; THAT one is fine
    # and unrelated to our redirect.
    r = docker_exec_sh(
        container_name,
        # Find PID of the CMD process (main-wrapper.sh or its sh
        # parent), then count `sleep infinity` children.
        "ps -eo pid,ppid,cmd | grep -v grep | awk '/main-wrapper.sh|rc.init top/ { wrapper_pid=$1 } "
        "$3==\"sleep\" && $4==\"infinity\" && $2==wrapper_pid { c++ } END { print c+0 }'",
    )
    assert r.returncode == 0
    redirected = int(r.stdout.strip() or 0)
    assert redirected == 1, (
        f"expected exactly one `sleep infinity` parented to the CMD "
        f"wrapper (the redirect heartbeat); found {redirected}. "
        f"ps:\n{docker_exec_sh(container_name, 'ps -eo pid,ppid,cmd').stdout}"
    )


def test_dashboard_supervised_when_env_set(
    built_image: str, container_name: str,
) -> None:
    """When ``HERMES_DASHBOARD=1`` is set, ``docker run <image> gateway
    run`` should result in BOTH the gateway and the dashboard being
    supervised by s6 — the dashboard slot was always there but only
    activates with the env var. This is the headline benefit of the
    redirect: one container = supervised gateway + supervised
    dashboard, with zero extra user effort.
    """
    start_container(
        built_image, container_name,
        "HERMES_DASHBOARD=1",
        cmd="gateway run",
    )

    # Wait for the redirect to fire (the breadcrumb appears in docker
    # logs when the CMD process reaches the redirect logic). This is
    # the same signal the other gateway-run tests use.
    # A fixed time.sleep(5) was racing: start_container returns when
    # cont-init finishes, but the redirect (which creates the
    # gateway-default s6 slot) happens later in the CMD process.
    wait_for_docker_logs(
        container_name, "s6 supervision", deadline_s=60.0,
    )

    # Poll for both slots to report want-up, using the same
    # _svstat_wants_up helper the other tests use. A simple
    # `grep 'want up'` is wrong: when the service is already up,
    # s6-svstat output is "up (pid ...) Ns" with no literal "want up"
    # — the want-up intent is implied by the absence of "want down".
    ok_gateway = False
    end = time.monotonic() + 30.0
    while time.monotonic() < end:
        if _svstat_wants_up(container_name, "gateway-default"):
            ok_gateway = True
            break
        time.sleep(0.5)
    assert ok_gateway, (
        f"gateway-default slot not want-up: {_svstat(container_name)!r}"
    )

    ok_dash = False
    end = time.monotonic() + 30.0
    while time.monotonic() < end:
        if _svstat_wants_up(container_name, "dashboard"):
            ok_dash = True
            break
        time.sleep(0.5)
    assert ok_dash, (
        f"dashboard slot not want-up: {_svstat(container_name, 'dashboard')!r}"
    )


