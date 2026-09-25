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
import sysconfig
from pathlib import Path


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def resolved_worker_python(repo_root: Path) -> Path | None:
    """The committed PM venv's interpreter for *repo_root*, or None.

    A managed gateway boots on the bare store Python: PM activates the committed
    venv's site-packages *in-process* only (``pm.environments.activate_dependencies``),
    so ``sys.executable`` never points at the interpreter that owns the
    dependencies.  A worker spawned from it dies at first third-party import
    (``No module named 'ruamel'``) before its ownership ack (#112729's sibling).
    The venv python imports the same checkout through its editable mapping;
    callers must still pin the checkout on PYTHONPATH in case that mapping
    outlives a moved/deleted checkout.
    """
    try:
        from pm.environments import committed_venv

        environment = committed_venv(Path(repo_root))
        if environment is None:
            return None
        rel = "python.exe" if os.name == "nt" else "bin/python3"
        candidate = Path(environment) / rel
        return candidate if candidate.is_file() else None
    except Exception:
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
