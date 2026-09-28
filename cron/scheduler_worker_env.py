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


def _committed_environment_purelib() -> Path | None:
    """site-packages of the dependency generation PM committed for this checkout.

    A store-Python gateway carries its third-party packages (``ruamel``, ...) in
    that generation, activated in-process by ``pm`` at boot — invisible to a fresh
    ``sys.executable`` interpreter. The worker is Hermes itself and must import the
    same dependency graph the gateway runs, so the pin adds this directory next to
    the checkout (#cron-worker-store-python: worker died on
    ``ModuleNotFoundError: No module named 'ruamel'`` after the PM migration when
    ``sys.executable`` stopped being the in-tree venv python).
    """
    try:
        from pathlib import Path as _Path

        from pm.environments import committed_venv, site_packages

        root = _Path(__file__).resolve().parent.parent
        environment = committed_venv(root)
        if environment is None:
            return None
        selected = site_packages(environment)
        return selected.resolve() if selected.is_dir() else None
    except Exception:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend the committed dependency generation and ``repo_root`` to the worker env's
    own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    entries: list[str] = [root]
    generation_purelib = _committed_environment_purelib()
    if generation_purelib is not None and generation_purelib not in map(Path, existing):
        # Boot activation order: checkout first, committed generation second.
        entries.append(str(generation_purelib))
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*entries, *existing]))
    return worker_env
