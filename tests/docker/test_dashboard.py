"""Harness: dashboard opt-in via HERMES_DASHBOARD.

Today (tini): dashboard starts once when HERMES_DASHBOARD=1; if it crashes
it stays dead. After Phase 2 (s6): dashboard starts once; if it crashes
it is restarted under supervision. The restart-after-crash test lives in
Phase 2 Task 2.5; this file only locks the opt-in surface (which must
not change between tini and s6).

Every ``docker exec`` here runs as the unprivileged ``hermes`` user
(via :func:`docker_exec`/:func:`docker_exec_sh` in conftest), matching
the realistic runtime context. See the conftest module docstring.
"""
from __future__ import annotations

import json
import subprocess
import time

from tests.docker.conftest import docker_exec, docker_exec_sh


def _poll(container: str, probe: str, *, deadline_s: float = 30.0,
          interval_s: float = 0.5) -> tuple[bool, str]:
    """Repeatedly run ``probe`` inside the container until it exits 0 or
    ``deadline_s`` elapses. Returns (success, last stdout)."""
    end = time.monotonic() + deadline_s
    last = ""
    while time.monotonic() < end:
        r = docker_exec_sh(container, probe, timeout=10)
        last = r.stdout
        if r.returncode == 0:
            return True, last
        time.sleep(interval_s)
    return False, last


def test_dashboard_not_running_by_default(
    built_image: str, container_name: str,
) -> None:
    """Without HERMES_DASHBOARD, no dashboard process should be running."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    # Give the entrypoint enough time to finish bootstrap; if a dashboard
    # were going to start it'd be visible by now.
    time.sleep(5)
    r = docker_exec(container_name, "pgrep", "-f", "hermes dashboard")
    # pgrep exits non-zero when no match found
    assert r.returncode != 0, (
        "Dashboard should not be running without HERMES_DASHBOARD"
    )


def test_dashboard_slot_reports_down_when_disabled(
    built_image: str, container_name: str,
) -> None:
    """Without HERMES_DASHBOARD, s6-svstat should report the dashboard
    slot as DOWN (not up-with-sleep-infinity, which would
    false-positive `hermes doctor` and any other health check).

    Locks the PR #30136 review item I3 fix: cont-init.d/03-dashboard-toggle
    writes a `down` marker file in the live service-dir when
    HERMES_DASHBOARD is unset, so the slot reflects reality.
    """
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, built_image,
         "sleep", "60"],
        check=True, capture_output=True, timeout=30,
    )
    time.sleep(5)
    # /command/ isn't on PATH for docker-exec sessions, so call by
    # absolute path.
    r = docker_exec(
        container_name, "/command/s6-svstat", "/run/service/dashboard",
    )
    assert r.returncode == 0, f"s6-svstat failed: {r.stderr!r} / {r.stdout!r}"
    assert "down" in r.stdout, (
        f"Dashboard slot should be 'down' without HERMES_DASHBOARD; "
        f"svstat reports: {r.stdout!r}"
    )


def test_dashboard_slot_reports_up_when_enabled(
    built_image: str, container_name: str,
) -> None:
    """Symmetry: with HERMES_DASHBOARD=1, s6-svstat reports the slot as up."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # uvicorn takes a moment to bind; poll svstat.
    deadline = time.monotonic() + 30.0
    last = ""
    while time.monotonic() < deadline:
        r = docker_exec(
            container_name, "/command/s6-svstat", "/run/service/dashboard",
        )
        last = r.stdout
        if r.returncode == 0 and "up " in r.stdout:
            return  # success
        time.sleep(0.5)
    raise AssertionError(
        f"Dashboard slot never reached up state; last svstat: {last!r}"
    )


def test_dashboard_opt_in_starts(
    built_image: str, container_name: str,
) -> None:
    """With HERMES_DASHBOARD=1, a dashboard process should be visible."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # Poll for the dashboard subprocess to appear — the entrypoint
    # backgrounds it and bootstrap (skills sync etc.) can take a few
    # seconds before the python process actually launches.
    ok, _ = _poll(
        container_name, "pgrep -f 'hermes dashboard'", deadline_s=30.0,
    )
    assert ok, "Dashboard should be running with HERMES_DASHBOARD=1"


def test_dashboard_port_override(
    built_image: str, container_name: str,
) -> None:
    """HERMES_DASHBOARD_PORT changes the dashboard's listen port."""
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", "-e", "HERMES_DASHBOARD_PORT=9120",
         built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # The dashboard process appearing in pgrep doesn't mean it's bound
    # to the port yet — uvicorn takes another second or two to come up.
    # The image doesn't ship ss/netstat, so probe /proc/net/tcp directly:
    # port 9120 = 0x23A0, state 0A = LISTEN.
    ok, stdout = _poll(
        container_name,
        "grep -E ' 0+:23A0 .* 0A ' /proc/net/tcp /proc/net/tcp6 "
        "2>/dev/null",
        deadline_s=60.0,
    )
    assert ok, f"Dashboard not listening on port 9120: stdout={stdout!r}"


def test_dashboard_restarts_after_crash(
    built_image: str, container_name: str,
) -> None:
    """Phase 2 invariant: under s6 supervision, killing the dashboard
    process should be recovered automatically.

    Pre-s6 (tini) behavior was "stays dead" — the test wouldn't have
    passed against that image. After the s6-overlay migration the
    dashboard runs as a longrun s6-rc service and s6-supervise restarts
    it after a ~1s backoff (the default).
    """
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1", built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    # Wait for the first dashboard to come up.
    ok, _ = _poll(
        container_name, "pgrep -f 'hermes dashboard'", deadline_s=30.0,
    )
    assert ok, "Dashboard never started initially"

    # Grab the initial PID. s6 may briefly transition through restart
    # state between our poll-success and the follow-up pgrep, so retry
    # a couple of times before giving up.
    first_pid: str | None = None
    for _attempt in range(10):
        first_pid_result = docker_exec(
            container_name, "pgrep", "-f", "hermes dashboard",
        )
        first_pids = first_pid_result.stdout.strip().split()
        if first_pids:
            first_pid = first_pids[0]
            break
        time.sleep(0.5)
    assert first_pid is not None, "Could not capture initial dashboard PID"

    # Kill the dashboard. The dashboard process runs as hermes, so the
    # hermes user can kill it (same UID).
    docker_exec(container_name, "kill", "-9", first_pid)

    # s6 backs off ~1s before restart; allow up to 15s for the new
    # process to appear with a different PID.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        r = docker_exec(container_name, "pgrep", "-f", "hermes dashboard")
        pids = r.stdout.strip().split() if r.returncode == 0 else []
        if pids and pids[0] != first_pid:
            return  # success
        time.sleep(0.5)

    raise AssertionError(
        f"Dashboard not restarted after kill (first_pid={first_pid})"
    )


# ---------------------------------------------------------------------------
# OAuth auth-gate behaviour — regression guard for the dashboard-insecure
# auto-injection bug. Pre-fix, the s6 run script appended `--insecure`
# whenever `HERMES_DASHBOARD_HOST` was non-loopback, silently disabling
# the OAuth gate on every container-deployed dashboard. The matching
# static-text guard lives in tests/test_docker_home_override_scripts.py;
# this is the behavioural end-to-end check.
# ---------------------------------------------------------------------------


def _fetch_api_status(container: str, *, deadline_s: float = 60.0) -> dict:
    """Poll ``/api/status`` from inside the container via the venv python.

    The dashboard binds to ``HERMES_DASHBOARD_HOST`` (typically ``0.0.0.0``)
    so loopback inside the container works. The image doesn't ship
    ``curl`` but Python's stdlib ``urllib`` is good enough.

    Returns the decoded JSON dict on success; raises AssertionError on
    timeout.
    """
    probe = (
        "/opt/hermes/.venv/bin/python -c "
        "'import json,urllib.request as u;"
        "print(u.urlopen(\"http://127.0.0.1:9119/api/status\",timeout=5)"
        ".read().decode())'"
    )
    end = time.monotonic() + deadline_s
    last_err = ""
    while time.monotonic() < end:
        r = docker_exec_sh(container, probe, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            try:
                return json.loads(r.stdout)
            except (ValueError, json.JSONDecodeError) as exc:  # noqa: F841
                last_err = f"json parse: {exc!r} / stdout={r.stdout!r}"
        else:
            last_err = f"rc={r.returncode} stderr={r.stderr!r}"
        time.sleep(0.5)
    raise AssertionError(
        f"/api/status never returned valid JSON within {deadline_s}s; "
        f"last error: {last_err}"
    )


def test_dashboard_oauth_gate_engages_on_non_loopback_bind(
    built_image: str, container_name: str,
) -> None:
    """The s6 dashboard run script must NOT auto-add ``--insecure`` when the
    dashboard binds to ``0.0.0.0``. The OAuth auth gate engages on its own
    when a ``DashboardAuthProvider`` is registered (the bundled nous
    provider activates whenever ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` is
    set).

    Regression guard for the wildcard-subdomain rollout where every
    portal-provisioned agent binds ``0.0.0.0`` and relies on the OAuth
    gate to authenticate browser callers. Before this fix, the run script
    flipped ``--insecure`` on for any non-loopback bind, which routed
    ``start_server`` straight back into the legacy ``allow_public=True``
    branch and disabled the gate every time.
    """
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1",
         "-e", "HERMES_DASHBOARD_HOST=0.0.0.0",
         "-e", "HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test-instance",
         built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    status = _fetch_api_status(container_name)
    assert status.get("auth_required") is True, (
        "OAuth gate must be engaged on 0.0.0.0 bind when a provider is "
        "registered and HERMES_DASHBOARD_INSECURE is unset. Got: "
        f"{status!r}"
    )
    assert "nous" in status.get("auth_providers", []), (
        "Bundled dashboard_auth/nous provider should register when "
        f"HERMES_DASHBOARD_OAUTH_CLIENT_ID is set. Got: {status!r}"
    )


def test_dashboard_insecure_env_var_opts_out_of_gate(
    built_image: str, container_name: str,
) -> None:
    """``HERMES_DASHBOARD_INSECURE=1`` re-enables the legacy no-gate mode
    for operators running on trusted LANs behind a reverse proxy without
    the OAuth contract. Same opt-out shape as the rest of the s6 boolean
    envs (``HERMES_DASHBOARD``, ``HERMES_DASHBOARD_TUI``).
    """
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name,
         "-e", "HERMES_DASHBOARD=1",
         "-e", "HERMES_DASHBOARD_HOST=0.0.0.0",
         "-e", "HERMES_DASHBOARD_INSECURE=1",
         built_image, "sleep", "120"],
        check=True, capture_output=True, timeout=30,
    )
    status = _fetch_api_status(container_name)
    assert status.get("auth_required") is False, (
        "HERMES_DASHBOARD_INSECURE=1 must disable the auth gate (explicit "
        f"opt-in for trusted-LAN deployments). Got: {status!r}"
    )
