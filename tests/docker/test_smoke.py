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
    )
    assert r.returncode == 0, (
        f"hermes dashboard --help failed (exit {r.returncode}): "
        f"stdout={r.stdout[-2000:]!r} stderr={r.stderr[-2000:]!r}"
    )
    combined = (r.stdout + r.stderr).lower()
    assert "dashboard" in combined or "usage" in combined, (
        f"dashboard --help output unexpected: {combined[-2000:]!r}"
    )
