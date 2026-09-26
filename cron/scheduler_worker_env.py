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

The pin covers BOTH halves of the gateway's own import path: the checkout (``cron.*``,
``tools.*``) AND the committed dependency environment's site-packages (``ruamel``,
``pydantic``, ...). The gateway process itself runs on the bundled store Python with
dependencies activated onto ``sys.path``/``PYTHONPATH`` at boot
(``pm.environments.activate_dependencies``); a bare ``-m`` child never runs that
bootstrap, so without the second half it dies with ``ModuleNotFoundError: No module
named 'ruamel'`` before its ownership ack -- every managed-topology cron job fails
forever while in-process jobs look fine.
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
    """Site-packages of the dependency environment PM committed for this checkout.

    ``None`` when nothing is committed yet (fresh install), the record is unreadable,
    or the tree is gone -- the caller then keeps the historical repo-root-only pin.
    Never the in-tree ``venv``/``.venv``: that tree predates PM and is built for
    whichever interpreter created it (``committed_venv`` already excludes it).
    """
    try:
        from pm.environments import committed_venv, site_packages
    except ImportError:
        return None
    try:
        venv = committed_venv(repo_root)
        site = site_packages(venv) if venv is not None else None
    except (OSError, RuntimeError, ValueError):
        return None
    if site is None:
        return None
    try:
        return site if site.is_dir() else None
    except OSError:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend the checkout AND its committed site-packages to the worker env's PYTHONPATH.

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    pins = [root]
    site_packages = _committed_site_packages(repo_root)
    if site_packages is not None and not _same_tree(site_packages, repo_root):
        pins.append(str(site_packages))
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pins, *existing]))
    return worker_env


def _same_tree(left: Path, right: Path) -> bool:
    """True when both paths resolve to the same tree (wheel-style layout guard)."""
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
