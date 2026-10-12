"""Runtime smoke test for the Docker tini compatibility shim (#34192, #66679).

Build the real image and verify that legacy ``/usr/bin/tini -g -- <cmd>``
entrypoints still boot the image (no ``rc.init: -g: not found`` boot loop).
"""
from __future__ import annotations

import subprocess


def test_legacy_tini_entrypoint_boots(built_image: str) -> None:
    """``--entrypoint /usr/bin/tini <image> -g -- --help`` must run hermes.

    Regression for #34192 / #66679: orchestration templates (e.g.
    Hostinger's 'Hermes WebUI' catalog, NAS compose projects that keep
    an old entrypoint across image updates) still pin /usr/bin/tini as
    the entrypoint, often with ``-g --``. A missing shim fails to exec; a
    bare symlink to /init forwards ``-g`` into s6 and boot-loops. The shim
    must strip the tini flags and hand the remaining args to /init +
    main-wrapper, so the requested command runs normally.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/usr/bin/tini",
         built_image, "-g", "--", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    combined = r.stdout + r.stderr
    assert "-g: not found" not in combined, (
        f"tini flags leaked into s6 rc.init (#66679): {combined[-2000:]!r}"
    )
    assert r.returncode == 0, (
        f"legacy tini entrypoint failed (exit {r.returncode}): "
        f"stdout={r.stdout[-2000:]!r} stderr={r.stderr[-2000:]!r}"
    )
    assert "Traceback" not in r.stderr
