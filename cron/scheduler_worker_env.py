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
site-packages, dropped venv markers) stand, EXCEPT that we must re-pin the PM-selected
site-packages dir ourselves (see below) or every third-party import the worker needs
(``ruamel.yaml``, etc.) is missing and the worker dies before its ownership ack.
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


def _pm_selected_site_packages(repo_root: Path) -> Path | None:
    """The PM-committed dependency environment's site-packages dir, if any.

    Under a PM-managed install (``hermes update``'s current layout) the gateway process's
    own third-party packages live in a generation directory under
    ``<install>/environments/<hash>/venv/lib/pythonX.Y/site-packages`` -- NOT beside
    ``repo_root`` and NOT on the bare interpreter's default ``sys.path``. The subprocess
    sanitizer strips the parent's PYTHONPATH (by design, for user-spawned children), so a
    worker launched as ``sys.executable -m cron.scheduler`` needs this re-pinned explicitly
    or imports like ``ruamel.yaml`` (via ``hermes_yaml`` -> ``utils`` ->
    ``agent.secret_scope`` -> ``cron.env_settings`` -> ``cron.jobs``) fail before the
    worker's ownership ack (#124713, hypothesised cause -- mirrors #112729 for ``cron/``
    itself, one layer up the dependency graph).
    """
    try:
        from pm.environments import selected_venv, site_packages
    except ImportError:
        return None  # Not a PM-managed install; nothing to add.
    try:
        selected = site_packages(selected_venv(repo_root))
    except (OSError, RuntimeError, ValueError):
        return None
    return selected if selected.is_dir() else None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` (and, when PM-managed, its selected site-packages) to the
    worker env's own PYTHONPATH (never ``os.environ``'s).

    Skipped for ``repo_root`` when it is the interpreter's ``purelib``: under a wheel /
    pipx / uv-tool install ``cron/`` lives in site-packages itself, which is already
    importable, and pinning it would move site-packages ahead of the stdlib on
    ``sys.path``. The PM site-packages pin is independent of that check -- it supplies
    third-party deps, not the ``cron`` package, and is a no-op when there is nothing
    PM-committed to find.
    """
    root = str(repo_root)
    entries = [] if _installed_purelib() == Path(root).resolve() else [root]
    if (selected := _pm_selected_site_packages(repo_root)) is not None:
        entries.append(str(selected))
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*entries, *existing]))
    return worker_env
