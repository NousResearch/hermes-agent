"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env -- and because the worker is a Hermes child that must
also import the dependency graph, the pin restores the runtime site-packages the stripper
classified as Hermes-owned (see ``pin_hermes_tree_on_pythonpath``). The sanitizer's
dropped venv markers still stand, and nothing here widens the env for user children.
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


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` to the worker env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    # The shared sanitizer stripped BOTH Hermes-owned entries — the checkout and the
    # runtime site-packages that actually hold the dependency graph (the interpreter
    # itself carries only pip). The worker is a Hermes child that must import the tree,
    # so restore exactly the site-packages the stripper itself classifies as owned:
    # the same helper is the stripper's own ownership test, which keeps the restored set
    # the exact inverse of the strip and widens nothing for user children.
    from tools.environments.local_pythonpath import _get_hermes_site_packages

    owned = [str(p) for p in _get_hermes_site_packages(worker_env) if Path(p).is_dir()]
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *owned, *existing]))
    return worker_env
