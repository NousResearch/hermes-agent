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
import sysconfig
from pathlib import Path


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _committed_site_packages(repo_root: Path) -> Path | None:
    """The PM dependency environment's site-packages for this checkout, or None.

    The worker entry point (``python -m cron.scheduler``) never runs the launcher
    bootstrap, so unlike the gateway process it does not get the committed
    dependency environment on ``sys.path``; combined with the sanitizer stripping
    the launch PYTHONPATH the worker dies on ``import hermes_yaml`` ->
    ``ModuleNotFoundError: No module named 'ruamel'`` before its ownership ack.
    """
    try:
        from pm.environments import selected_venv, site_packages

        return site_packages(selected_venv(repo_root))
    except Exception:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` (and the committed dependency environment's
    site-packages) to the worker env's own PYTHONPATH (never ``os.environ``'s).

    The repo-root pin is skipped when ``repo_root`` is the interpreter's
    ``purelib``: under a wheel / pipx / uv-tool install ``cron/`` lives in
    site-packages itself, which is already importable, and pinning it would move
    site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    entries = []
    if _installed_purelib() != Path(root).resolve():
        entries.append(root)
    dep_site = _committed_site_packages(repo_root)
    if dep_site is not None:
        entries.append(str(dep_site))
    if not entries:
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*entries, *existing]))
    return worker_env
