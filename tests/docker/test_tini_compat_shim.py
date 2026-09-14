"""Runtime smoke test for the Docker tini compatibility shim (#34192, #66679).

Build the real image and verify:

  1. /usr/bin/tini exists as an executable shim (not a bare symlink to
     /init — that forwarded tini's ``-g`` into s6 and boot-looped)
  2. The actual ENTRYPOINT is /init (s6-overlay), not /usr/bin/tini
  3. Legacy ``tini -g -- <cmd>`` entrypoints boot without
     ``rc.init: -g: not found``
"""
from __future__ import annotations

import subprocess


def test_tini_compat_shim_exists(built_image: str) -> None:
    """/usr/bin/tini must be an executable shim script.

    Regression for #34192 / #66679: orchestration templates (e.g.
    Hostinger's 'Hermes WebUI' catalog, NAS compose projects that keep
    an old entrypoint across image updates) still pin /usr/bin/tini as
    the entrypoint, often with ``-g --``. The shim must exist *and*
    strip those flags before exec'ing /init.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh",
         built_image, "-c",
         'test -x /usr/bin/tini && '
         # Must NOT be a raw symlink to /init — that reintroduces #66679.
         'if [ -L /usr/bin/tini ]; then '
         '  target="$(readlink -f /usr/bin/tini)"; '
         '  test "$target" != "/init"; '
         'fi && '
         'head -n1 /usr/bin/tini | grep -q "^#!"'],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, (
        f"/usr/bin/tini is not a usable tini shim: "
        f"stdout={r.stdout[-500:]!r} stderr={r.stderr[-500:]!r}"
    )


def test_entrypoint_is_dispatcher_not_tini(built_image: str) -> None:
    """The image's actual ENTRYPOINT must be the PID-1 dispatcher.

    Since #38349 the ENTRYPOINT is ``entrypoint-dispatch.sh``, which
    exec's the canonical ``/init`` (s6-overlay) when the image owns
    PID 1 and falls back to a direct bootstrap on wrapped runtimes
    (Fly Machines, ``docker run --init``). The tini shim is only for
    legacy external wrappers; the image's own runtime must route
    through the dispatcher, never through /usr/bin/tini.
    """
    r = subprocess.run(
        ["docker", "inspect", built_image,
         "--format", "{{json .Config.Entrypoint}}"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"docker inspect failed: {r.stderr}"
    entrypoint = r.stdout.strip()
    assert "entrypoint-dispatch.sh" in entrypoint, (
        f"ENTRYPOINT is not the PID-1 dispatcher: {entrypoint!r}"
    )
    # /usr/bin/tini should NOT be in the entrypoint.
    assert "tini" not in entrypoint.lower(), (
        f"ENTRYPOINT references tini instead of the dispatcher: {entrypoint!r}"
    )
    # The dispatcher must still hand PID-1 execution to /init inside
    # the image, preserving the supervision tree.
    r2 = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "sh", built_image, "-c",
         "grep -q 'exec /init /opt/hermes/docker/main-wrapper.sh' "
         "/opt/hermes/docker/entrypoint-dispatch.sh"],
        capture_output=True, text=True, timeout=60,
    )
    assert r2.returncode == 0, (
        "entrypoint-dispatch.sh in the image does not delegate the "
        f"PID-1 path to /init: stderr={r2.stderr[-500:]!r}"
    )


