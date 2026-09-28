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


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` to the worker env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *existing]))
    return worker_env


def ensure_worker_dependencies(worker_env: dict, repo_root: Path, *, python: str | None = None) -> dict:
    """Append the install's selected dependency tree to the worker's PYTHONPATH.

    PM launchers run the gateway on the *store* Python, which owns the ABI but ships no
    third-party packages; the selected generation is activated into ``sys.path`` at boot
    (``hermes_bootstrap`` -> ``pm.environments.activate_dependencies``) and never lands in
    ``os.environ``, so the sanitizer cannot carry it to the child. This worker's entry
    module (``cron.scheduler``) has no bootstrap, so it must be handed the generation
    explicitly or it dies on its first third-party import (``No module named 'ruamel'``).

    Only applies when the child would run on that dependency-less store interpreter; a
    venv/wheel launch (the interpreter already owns its site-packages) is left untouched.
    """
    from hermes_cli._launchers import resolve_store_python
    from pm.environments import selected_venv, site_packages

    store = resolve_store_python(repo_root)
    if store is None:
        return worker_env
    if Path(python or sys.executable).resolve() != Path(store).resolve():
        return worker_env
    try:
        dependencies = site_packages(selected_venv(Path(repo_root)))
    except (OSError, RuntimeError, ValueError):
        return worker_env
    if not dependencies.is_dir():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*existing, str(dependencies)]))
    return worker_env
