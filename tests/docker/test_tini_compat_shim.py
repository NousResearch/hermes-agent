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


def test_legacy_tini_g_entrypoint_does_not_boot_loop(built_image: str) -> None:
    """``docker run --entrypoint /usr/bin/tini … -g -- --help`` must work.

    Exact failure from #66679: after update, NAS templates still invoke
    ``/usr/bin/tini -g -- …``. The old symlink turned that into
    ``/init -g -- …``, rc.init tried to exec ``-g``, and the container
    restart-looped. The shim must strip ``-g`` / ``--`` and reach hermes.
    """
    r = subprocess.run(
        [
            "docker", "run", "--rm",
            "--entrypoint", "/usr/bin/tini",
            built_image,
            "-g", "--", "--help",
        ],
        capture_output=True, text=True, timeout=120,
    )
    combined = r.stdout + r.stderr
    assert "-g: not found" not in combined, (
        f"tini -g leaked into rc.init (boot-loop regression):\n{combined[-3000:]}"
    )
    assert r.returncode == 0, (
        f"legacy tini -g -- --help failed (exit {r.returncode}):\n"
        f"stdout={r.stdout[-2000:]!r}\nstderr={r.stderr[-2000:]!r}"
    )
