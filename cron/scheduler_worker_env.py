"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env. On a self-managed (shell-installer / PM) install the
sanitizer's drop of the runtime site-packages cannot stand this time: the worker's
interpreter is the store Python, which owns no third-party dependencies, so the activated
dependency environment's site-packages must be restored here too or the child dies at its
first import (``No module named 'ruamel'``, #122222) before it can publish its ownership
acknowledgement.
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


def _in_real_venv(path: Path) -> bool:
    """True when *path* sits under a directory carrying ``pyvenv.cfg``.

    ``activate_dependencies`` exposes the selected generation by path (no ``sys.prefix``
    switch), so the only proof that a ``site-packages`` entry is a dependency environment
    rather than a directory that merely happens to be named that is the venv marker in an
    ancestor.
    """
    current = path
    while current != current.parent:
        try:
            if (current / "pyvenv.cfg").is_file():
                return True
        except OSError:
            return False
        current = current.parent
    return False


def _activated_dependency_site_packages() -> Path | None:
    """The site-packages this process imports Hermes's dependencies from, or ``None``.

    On a self-managed install the gateway runs on the store Python and
    ``activate_dependencies`` puts the dependency generation on ``sys.path`` -- the same
    signal ``pm.environments.running_from_selected_environment`` reads. Derived from
    ``sys.path`` (never via ``selected_venv()``/``site_packages()``) so the child-spawn
    path performs no home-scoped filesystem reads. The interpreter's own ``purelib`` is
    excluded: a runner that owns its dependencies (wheel / pipx / test / developer venv)
    hands the child its own interpreter and needs nothing added. ``None`` therefore means
    "pin the tree only", and nothing is invented.
    """
    purelib = _installed_purelib()
    purelib_key = os.path.normcase(str(purelib)) if purelib is not None else None
    seen: set[str] = set()
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        if path.name not in ("site-packages", "dist-packages"):
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        if purelib_key is not None and key == purelib_key:
            continue
        if not resolved.is_dir():
            continue
        if not _in_real_venv(resolved):
            continue
        return resolved
    return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` -- and, when the worker's interpreter cannot import Hermes's
    dependencies otherwise, the activated environment's ``site-packages`` -- to the worker
    env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    pinned = [root]
    dependency = _activated_dependency_site_packages()
    if dependency is not None:
        pinned.append(str(dependency))
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env