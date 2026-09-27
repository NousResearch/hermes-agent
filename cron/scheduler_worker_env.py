"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``<worker_interpreter> -m cron.scheduler`` -- the selected dependency
environment's interpreter when this process is an install with recorded dependency facts, else
``sys.executable`` (see :func:`worker_interpreter`). Its entry module is
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

import logging
import os
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)


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


def worker_interpreter(repo_root: Path) -> str:
    """Interpreter to spawn the restart-safe external worker on.

    ``sys.executable`` is the *gateway's* interpreter, which is not guaranteed to carry the
    tree's dependencies. An update can leave the gateway running a bare provisioned runtime
    (a store Python whose site-packages holds pip and nothing else) while the real dependency
    environment lives in the venv PM selected for the install. The worker then dies at import
    with ``ModuleNotFoundError`` (``ruamel.yaml``, ``dotenv``, ...) *before* its ownership ack,
    so the scheduler reports every agent job failed while every systemd unit still reads
    ``active`` -- and only jobs that import nothing (``no_agent`` scripts) keep working.

    Prefer the selected dependency environment's interpreter (``pm.environments``). Fall back
    to ``sys.executable`` when this process is not an install with recorded dependency facts --
    a source checkout, a wheel/pipx install, or a test.

    The sanitized worker env cannot compensate: it deliberately drops Hermes-owned PYTHONPATH
    entries, the runtime site-packages and the venv markers, so the interpreter's OWN
    site-packages is the only source of third-party imports for this child.
    """
    try:
        from pm.environments import project_python, runtime_facts_path

        if not runtime_facts_path(repo_root).is_file():
            return sys.executable
        candidate = project_python(repo_root)
    except Exception:
        # A broken or absent dependency record must not stop the worker from being spawned:
        # sys.executable is what we used before, so this is never worse than the status quo.
        logger.debug(
            "No usable dependency environment for %s; spawning the cron worker on sys.executable",
            repo_root, exc_info=True,
        )
        return sys.executable
    if not candidate.is_file():
        logger.warning(
            "Dependency environment for %s has no interpreter at %s; falling back to %s. "
            "Cron agent jobs may fail to import the tree.",
            repo_root, candidate, sys.executable,
        )
        return sys.executable
    return str(candidate)
