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

Third-party deps (ruamel.yaml, etc.) need the same treatment: under the PM-managed
environment layout, ``sys.executable`` is the bare bundled interpreter -- the gateway's
own third-party packages live in a separate PM-selected venv whose site-packages is
spliced onto ``sys.path`` at boot (``pm.environments.activate_dependencies``), never
recorded on the interpreter's own default path. A worker launched via ``-m`` skips that
splice entirely, so without also pinning that site-packages dir here it fails on the
first third-party import (observed: ``ModuleNotFoundError: No module named 'ruamel'``,
since the 2026-09-25 multiplex migration moved the gateway onto that interpreter).
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


def _pm_dependency_site_packages(repo_root: Path) -> Path | None:
    """The PM-selected environment's site-packages for *repo_root*, or ``None``.

    Best-effort: any failure (no PM environment committed, corrupt facts.json, plain
    in-tree venv install) just skips the pin -- the worker falls back to whatever
    ``sys.executable`` already carries, same as before this existed.
    """
    try:
        from pm.environments import selected_venv, site_packages
        candidate = site_packages(selected_venv(repo_root))
        return candidate if candidate.is_dir() else None
    except Exception:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` (and the PM dependency environment, if any) to the worker
    env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    entries: list[str] = []
    root = str(repo_root)
    if _installed_purelib() != Path(root).resolve():
        entries.append(root)
    dep_site_packages = _pm_dependency_site_packages(repo_root)
    if dep_site_packages is not None:
        entries.append(str(dep_site_packages))
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*entries, *existing]))
    return worker_env
