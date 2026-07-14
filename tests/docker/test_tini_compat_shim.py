"""Runtime smoke test for the Docker tini compatibility shim (#34192).

Build the real image and verify:

  1. /usr/bin/tini exists and is a symlink to /init (the compat shim
     for orchestration templates that still reference /usr/bin/tini)
  2. The actual ENTRYPOINT is /init (s6-overlay), not /usr/bin/tini
"""
from __future__ import annotations

import subprocess


def test_tini_compat_symlink_exists(built_image: str) -> None:
    """/usr/bin/tini must exist as a symlink to /init.

    Regression for #34192: orchestration templates (e.g. Hostinger's
    'Hermes WebUI' catalog) still pin /usr/bin/tini as the entrypoint.
    The shim symlinks it to /init so legacy wrappers exec the right
    PID-1 reaper without behavior change.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh",
         built_image, "-c",
         'test -L /usr/bin/tini && '
         'test "$(readlink -f /usr/bin/tini)" = "/init"'],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, (
        f"/usr/bin/tini is not a symlink to /init: {r.stderr[-500:]}"
    )


def test_entrypoint_is_init_not_tini(built_image: str) -> None:
    """The image's actual ENTRYPOINT must be /init (s6-overlay).

    The tini shim is only for legacy external wrappers; the image's own
    runtime must continue to use the canonical /init.
    """
    r = subprocess.run(
        ["docker", "inspect", built_image,
         "--format", "{{json .Config.Entrypoint}}"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"docker inspect failed: {r.stderr}"
    entrypoint = r.stdout.strip()
    assert "/init" in entrypoint, (
        f"ENTRYPOINT is not /init: {entrypoint!r}"
    )
    # The entrypoint array should be ["/init", "/opt/hermes/docker/main-wrapper.sh"]
    # /usr/bin/tini should NOT be in the entrypoint.
    assert "tini" not in entrypoint.lower(), (
        f"ENTRYPOINT references tini instead of /init: {entrypoint!r}"
    )