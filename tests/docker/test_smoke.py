"""Runtime smoke tests for the Docker image entrypoint and subcommands.

Converted from the former ``.github/actions/hermes-smoke-test`` composite
action.  These tests exercise the image's real ENTRYPOINT (``/init`` +
``main-wrapper.sh``) via ``docker run --rm <image> --help`` and
``docker run --rm <image> dashboard --help`` to catch basic runtime
regressions before publishing.

The harness expects the ``built_image`` fixture from
``tests/docker/conftest.py``.  When Docker isn't available every test
here is skipped at collection time.
"""
from __future__ import annotations

import subprocess


def test_hermes_help(built_image: str) -> None:
    """``docker run --rm <image> --help`` must exit 0.

    Uses the image's real ENTRYPOINT (``/init`` + ``main-wrapper.sh``)
    so this exercises the actual production startup path.  PR #30136
    review caught that an ``--entrypoint`` override in the old composite
    action had been silently neutered by the s6-overlay migration —
    ``stage2-hook`` ignores CMD args passed after an overridden
    entrypoint, so the smoke test was a no-op.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "--help"],
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert r.returncode == 0, (
        f"hermes --help failed (exit {r.returncode}): "
        f"stdout={r.stdout[-2000:]!r} stderr={r.stderr[-2000:]!r}"
    )
    assert "Traceback" not in r.stderr, (
        f"hermes --help produced a traceback: {r.stderr[-2000:]!r}"
    )


def test_dashboard_subcommand_present(built_image: str) -> None:
    """``docker run --rm <image> dashboard --help`` must exit 0.

    Regression guard for #9153: the ``dashboard`` subcommand was present
    in source but missing from the published image.  If this fails,
    something in the Dockerfile is excluding the dashboard subcommand
    from the installed package.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "dashboard", "--help"],
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert r.returncode == 0, (
        f"hermes dashboard --help failed (exit {r.returncode}): "
        f"stdout={r.stdout[-2000:]!r} stderr={r.stderr[-2000:]!r}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "dashboard" in combined or "usage" in combined, (
        f"dashboard --help output unexpected: {combined[-2000:]!r}"
    )


def test_hermes_help_under_wrapped_init(built_image: str) -> None:
    """``docker run --init --rm <image> --help`` must exit 0.

    Regression guard for #38349: platforms whose own init owns PID 1
    (Fly Machines, ``docker run --init``, podman on some hosts) exec the
    image entrypoint as a child.  s6-overlay's ``/init`` hard-aborts
    there with ``s6-overlay-suexec: fatal: can only run as pid 1``.  The
    entrypoint dispatcher must detect the non-PID-1 case and fall back
    to the direct bootstrap path so the requested command still runs.
    """
    r = subprocess.run(
        ["docker", "run", "--init", "--rm", built_image, "--help"],
        capture_output=True, text=True, timeout=120,
        check=False,
    )
    assert "can only run as pid 1" not in (r.stdout + r.stderr), (
        f"s6-overlay-suexec aborted under a wrapped init (#38349): "
        f"stderr={r.stderr[-2000:]!r}"
    )
    assert r.returncode == 0, (
        f"hermes --help failed under `docker run --init` (exit {r.returncode}): "
        f"stdout={r.stdout[-2000:]!r} stderr={r.stderr[-2000:]!r}"
    )
    assert "Traceback" not in r.stderr, (
        f"hermes --help produced a traceback under --init: {r.stderr[-2000:]!r}"
    )



def test_detached_start_with_no_command_stays_up_as_the_gateway(built_image: str) -> None:
    """``docker run -d <image>`` (no command, no TTY) keeps running the supervised gateway.

    Regression for #159590: the no-argument path exec'd the interactive CLI, which quits on a
    non-TTY stdin, so the container exited and ``--restart unless-stopped`` looped every ~65 s.
    """
    from tests.waiting import wait_until

    cid = subprocess.run(["docker", "run", "-d", built_image], capture_output=True, text=True,
                         timeout=60, check=True).stdout.strip()

    def _docker(*args: str) -> str:
        r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30, check=False)
        return r.stdout + r.stderr

    def _settled() -> str:
        # Either the old CLI path ended the container, or the main program handed off to the
        # supervised gateway (its handoff notice; s6's own profile service alone never prints it).
        if _docker("inspect", cid, "--format", "{{.State.Running}}").strip() != "true":
            return "exited"
        return "gateway" if "running under s6 supervision" in _docker("logs", cid) else ""

    try:
        outcome = wait_until(_settled, timeout=150, interval=2, what="container exit or a running gateway")
        logs = _docker("logs", cid)
        assert outcome == "gateway", f"container exited on a detached start: {logs[-2000:]!r}"
        assert "Input is not a terminal" not in logs
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60, check=False)
