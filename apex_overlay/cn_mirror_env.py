"""ApexNodes runtime CN mirror env — self-apply the download-mirror env vars
that the installer's CN mode exports, into the *running* Hermes process.

Why this exists (hc-476 follow-up)
----------------------------------
``scripts/lib/apexnodes-region-detect.{sh,ps1}`` export the CN mirror env
(``HERMES_NODE_DIST_BASE``, ``PLAYWRIGHT_DOWNLOAD_HOST``, ...) **inside the
install-script process only**. Nothing persists them: the desktop
(bootstrap-runner.cjs ``cnInstallEnv``) threads just ``HERMES_CN_MIRRORS`` /
``HERMES_RUNTIME_COS_BASE`` to that one-shot child, and the long-lived gateway
is later spawned from the Electron main process's own ``process.env``
(apps/desktop/electron/main.cjs), which never saw the derived mirror values.
So the RUNTIME lazy-download paths — ``tools/browser_tool.py::
_maybe_autoinstall_chromium`` (Chromium ~170MB) and ``hermes_constants.py::
_heal_managed_node_windows`` / the bash ``node-bootstrap.sh`` paths — ran with
an empty mirror env on a mainland-China machine and pulled from the foreign
default hosts. That is exactly the "installed fine, feature broke on day N"
failure hc-476 is about.

What this seam does
-------------------
At plugin load (CLI deferred-startup / gateway boot — before any tool can
trigger a lazy download), when :func:`apex_overlay.region.is_cn_mode` says this
is a CN deployment, ``setdefault`` the runtime-relevant mirror env vars into
``os.environ``. Every child then inherits them: ``hermes_subprocess_env()``
starts from ``os.environ.copy()`` and only strips credentials, and the bash
node-bootstrap invocations pass ``{**os.environ}`` explicitly.

``setdefault`` keeps operator overrides authoritative (a pre-set value is
never replaced), and on global machines this is a strict no-op.

KEEP THE VALUES IN STEP with the single source of truth for install-time:
``apexnodes_apply_cn_mirror_env`` in scripts/lib/apexnodes-region-detect.sh and
``Set-ApexCnMirrorEnv`` in scripts/lib/apexnodes-region-detect.ps1. A seam test
(tests/apex_overlay/test_cn_mirror_env_seam.py) pins the three copies together.

This lives in ``apex_overlay/`` (a namespace upstream never creates) so it has
a zero-conflict merge surface, and is import-safe / fail-safe like every seam.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Only the mirrors a RUNTIME download path actually consumes. The full
# install-time set (pypi/npm/electron/...) stays install-only: runtime never
# resolves those sources itself, and injecting package-manager env into a
# long-lived agent process would leak into user-facing terminal subprocesses.
RUNTIME_CN_MIRROR_ENV: dict[str, str] = {
    # Node dist tarballs — consumed by scripts/lib/node-bootstrap.sh
    # (_nb_install_bundled_node) and hermes_constants._heal_managed_node_windows.
    "HERMES_NODE_DIST_BASE": "https://registry.npmmirror.com/-/binary/node",
    # Playwright browser downloads — read natively by `playwright install`
    # (playwright.dev/docs/browsers); reaches agent-browser's Chromium install
    # through the inherited subprocess env.
    "PLAYWRIGHT_DOWNLOAD_HOST": "https://cdn.npmmirror.com/binaries/playwright",
}

_APPLIED = False


def apply() -> bool:
    """Set the runtime CN mirror env (CN deployments only). Idempotent.

    Returns True on success (including the global-machine no-op), False only
    when something unexpected raised — never propagates.
    """
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from apex_overlay.region import is_cn_mode

        if not is_cn_mode():
            _APPLIED = True
            return True
        applied = []
        for key, value in RUNTIME_CN_MIRROR_ENV.items():
            if not (os.environ.get(key) or "").strip():
                os.environ[key] = value
                applied.append(key)
        if applied:
            logger.info(
                "apex-overlay: CN runtime mirror env applied: %s",
                ", ".join(applied),
            )
        _APPLIED = True
        return True
    except Exception:
        logger.warning("apex-overlay: cn_mirror_env seam failed", exc_info=True)
        return False
