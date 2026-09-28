"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env -- the sanitizer's other decisions (dropped venv
markers, user-owned entries) stand. The activated *dependency* site-packages are carried
over as well: unlike ``hermes_cli.main`` the bare ``-m cron.scheduler`` worker never runs
``hermes_bootstrap``/``activate_dependencies``, yet imports third-party dependencies
(``ruamel``) at import time, so without them it dies with ``ModuleNotFoundError`` before
the ownership ack (#126609).
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

# Only dependency layouts are recognized, by directory name; every other PYTHONPATH /
# sys.path entry belongs to the user or the interpreter and stays out of the worker env.
_DEPENDENCY_DIR_NAMES = ("site-packages", "dist-packages")


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _activated_dependency_dirs() -> list[str]:
    """Site/dist-packages dirs this process was activated with, in activation order.

    Sources are the gateway's own activation state -- the ``PYTHONPATH``
    ``activate_dependencies`` wrote and the ``sys.path`` it rebuilt -- never the sanitized
    child env. The interpreter's own ``purelib`` is excluded: it is importable by the
    worker anyway (same ``sys.executable``), and pinning it ahead of the checkout would
    shadow the checkout's own modules.
    """
    own_purelib = _installed_purelib()
    candidates = [e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]
    candidates += [p for p in sys.path if p]
    deps: list[str] = []
    for entry in candidates:
        path = Path(entry)
        if path.name not in _DEPENDENCY_DIR_NAMES or not path.is_dir():
            continue
        if own_purelib is not None and path.resolve() == own_purelib:
            continue
        deps.append(entry)
    return deps


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` and the activated dependency dirs to the worker env's own
    PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *_activated_dependency_dirs(), *existing]))
    return worker_env
