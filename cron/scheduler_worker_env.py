"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env. The dropped *venv markers* still stand, but the
runtime ``site-packages`` must be restored too: the worker is spawned as
``sys.executable -m cron.scheduler`` with the store interpreter, which has no venv of its
own, so a stripped ``site-packages`` leaves it unable to import any third-party dependency
(``ruamel.yaml`` via ``hermes_yaml`` first) and it dies before its ownership ack. Restoring
the tree alone fixes "No module named 'cron'" and leaves "No module named 'ruamel'".
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


def _committed_site_packages(repo_root: Path) -> Path | None:
    """The install's committed venv ``site-packages``, or None if none is recorded.

    Mirrors the launcher, which selects this same generation before any third-party
    import; the worker must resolve to the SAME directory or it can import a different
    generation's packages than the gateway that spawned it. Never consults
    ``VIRTUAL_ENV``: on a store-interpreter spawn that describes the caller's shell, not
    this install.
    """
    try:
        from pm.environments import committed_venv, site_packages
    except Exception:
        return None
    try:
        environment = committed_venv(repo_root)
        if environment is None:
            return None
        selected = site_packages(environment)
    except Exception:
        return None
    return selected if selected.is_dir() else None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` — and the committed runtime ``site-packages`` — to the worker
    env's own PYTHONPATH (never ``os.environ``'s).

    The repo alone is not enough: the sanitizer that built ``worker_env`` stripped the
    Hermes-owned runtime ``site-packages`` (see the module docstring), and the worker runs
    under the store interpreter rather than the venv's own, so without it every third-party
    import (``ruamel.yaml``, ``httpx``, ``pydantic``) fails.

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    pinned = [root]
    site_dir = _committed_site_packages(Path(root))
    if site_dir is not None and str(site_dir) not in pinned:
        pinned.append(str(site_dir))
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
