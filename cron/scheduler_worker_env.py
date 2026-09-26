"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env -- the sanitizer's other decisions (dropped runtime
site-packages, dropped venv markers) stand.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _activated_site_packages() -> Path | None:
    """Find the activated dependency site-packages directory from sys.path.

    When running under a self-managed install, the gateway activates dependencies
    by placing the selected generation's site-packages on sys.path. The external
    cron worker spawned with the store Python needs this path restored next to
    the tree, while excluding the interpreter's own purelib and requiring a real
    venv ancestor (pyvenv.cfg).
    """
    own_purelib = _installed_purelib()
    for entry in sys.path:
        if not entry:
            continue
        try:
            p = Path(entry).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if p.name != "site-packages":
            continue
        if own_purelib is not None and p == own_purelib:
            continue
        if any((ancestor / "pyvenv.cfg").is_file() for ancestor in p.parents):
            return p
    return None




def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` and activated dependency site-packages to the worker env's own PYTHONPATH.

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    added = [root]
    activated = _activated_site_packages()
    if activated is not None:
        added.append(str(activated))
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*added, *existing]))
    return worker_env

