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

import logging
import os
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
    # The worker also needs third-party dependencies (#122222): on self-managed
    # installs ``sys.executable`` is the PM store Python, which owns none — they
    # live in the committed dependency generation, activated by path. Pin its
    # site-packages too (after the tree, so our own modules always win).
    try:
        from pm.environments import committed_venv, site_packages
        deps_dir = site_packages(committed_venv(Path(root)))
    except Exception as exc:
        logger.debug("Could not resolve committed dependency dir: %s", exc)
        deps_dir = None
    if deps_dir is not None:
        try:
            is_dir = deps_dir.is_dir()
        except OSError:
            is_dir = False
        if is_dir:
            entries = worker_env.get("PYTHONPATH", "").split(os.pathsep)
            if str(deps_dir) not in entries:
                entries.insert(1, str(deps_dir))
            worker_env["PYTHONPATH"] = os.pathsep.join(
                dict.fromkeys(e for e in entries if e))
    return worker_env
