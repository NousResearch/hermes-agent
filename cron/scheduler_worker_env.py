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

    Under a PM store-interpreter launch the worker's ``sys.executable`` is the bare store
    python: the committed dependency venv reaches the gateway only through its in-process
    ``sys.path`` (``activate_dependencies``), and the sanitizer strips that venv's
    site-packages entry from child envs by provenance -- the right call for a child of a
    DIFFERENT Python version. This worker is Hermes on the SAME interpreter the committed
    venv is built for, so the venv's site-packages are pinned alongside the repo root;
    without them the worker dies on its first third-party import before the ownership
    ack. Launches already running on the committed venv (``sys.prefix`` == that venv)
    carry its site-packages natively and gain nothing from re-pinning them.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    pinned = [root]
    from pm.environments import runtime_facts_path, selected_venv, site_packages

    if runtime_facts_path(Path(repo_root)).is_file():
        venv = selected_venv(Path(repo_root))
        if venv is not None and Path(sys.prefix).resolve() != Path(venv).resolve():
            deps = site_packages(venv)
            if deps.is_dir():
                pinned.append(str(deps))
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
