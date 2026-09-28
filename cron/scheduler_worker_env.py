"""Cron: import path and interpreter of the restart-safe external worker.

The worker is spawned as ``<python> -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env -- the sanitizer's other decisions (dropped runtime
site-packages, dropped venv markers) stand.

PM-generation installs need the worker to run on the committed generation's interpreter.
The gateway attaches that generation in-process at boot (``activate_dependencies``), so
``sys.executable`` may be a bare base interpreter with no third-party deps on disk; a
worker spawned from it dies at the first third-party import (``ruamel.yaml`` via
``utils``) before its ownership ack. ``worker_python()`` returns the generation venv's
own interpreter when the checkout boots via PM runtime facts, so the worker reproduces
the gateway's import surface by construction; ``sys.executable`` remains the answer for
venv / wheel / pipx installs.
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


def worker_python(repo_root: Path) -> str:
    """The interpreter for the restart-safe external worker.

    PM-generation install: the committed generation venv's interpreter (what
    ``activate_dependencies`` attached in-process), so the worker sees the same
    third-party deps the gateway does. Any other install kind: ``sys.executable``.
    Falls back to ``sys.executable`` whenever PM state or the interpreter is missing.
    """
    try:
        from hermes_constants import venv_python_path
        from pm.environments import runtime_facts_path, selected_venv

        for root in (Path(repo_root), Path(__file__).resolve().parents[1]):
            if runtime_facts_path(root).is_file():
                python = venv_python_path(selected_venv(root))
                if python.is_file():
                    return str(python)
    except Exception:
        pass
    return sys.executable


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
