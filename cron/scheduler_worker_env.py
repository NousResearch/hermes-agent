"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env. If a gateway was launched by a bare interpreter
with its dependencies supplied by an external managed venv via PYTHONPATH, the worker
also needs that already-loaded dependency tree: the sanitizer strips it, and the bare
interpreter otherwise dies importing ruamel before acknowledging ownership.
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


def _gateway_dependency_path(repo_root: Path) -> str | None:
    """Find the external runtime that supplied this already-imported Hermes process.

    Only a bare interpreter needs this bridge. A venv has its dependencies on its
    own sys.path, and adding another venv's site-packages can mix binary wheels.
    The path must be a *pre-existing* sys.path entry and contain the dependency we
    need, never a path invented from VIRTUAL_ENV or blindly copied from the launcher.
    """
    if sys.prefix != sys.base_prefix:
        return None
    try:
        import ruamel.yaml
    except ImportError:
        return None
    origin = Path(ruamel.yaml.__file__).resolve()
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry).resolve()
        if candidate == repo_root.resolve() or candidate == _installed_purelib():
            continue
        if candidate.is_dir() and origin.is_relative_to(candidate):
            return str(candidate)
    return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Pin checkout and, for bare-interpreter gateways, its loaded dependency tree.

    Operates on the sanitized worker env only; never restore unrelated launcher paths.
    A wheel/pipx checkout already in purelib must not hoist site-packages above stdlib.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    dependency_path = _gateway_dependency_path(repo_root)
    pinned = [root, *([dependency_path] if dependency_path else []), *existing]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pinned))
    return worker_env
