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


def _committed_site_packages(repo_root: Path) -> Path | None:
    """Site-packages of the dependency generation committed for this install.

    Mirrors what ``activate_dependencies`` puts on ``PYTHONPATH`` at boot
    (repo root + selected site dir). Pinned only when the generation's
    interpreter matches THIS process (the worker is spawned via our own
    ``sys.executable``): a foreign-version site dir would shadow the child's
    own C extensions and crash it. ``None`` when nothing compatible is
    committed: the worker keeps today's behavior (no regression).
    """
    import sys

    try:
        from pm.environments import selected_venv, site_packages, venv_python_version
    except Exception:
        return None
    try:
        venv = selected_venv(repo_root)
    except Exception:
        return None
    try:
        version = venv_python_version(venv)
    except Exception:
        return None
    if version is not None and tuple(version) != (sys.version_info.major, sys.version_info.minor):
        return None
    try:
        site = site_packages(venv)
    except Exception:
        return None
    try:
        return site if site.is_dir() else None
    except OSError:
        return None


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` to the worker env's own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.

    Next to the root, the committed dependency generation's site dir is pinned
    (t_7b0df4cf): the shared sanitizer strips Hermes-owned PYTHONPATH entries —
    root AND site — so root alone leaves the external worker dying on
    ``ModuleNotFoundError: No module named 'ruamel'`` at import.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    pinned: list[str] = [root]
    site = _committed_site_packages(Path(root))
    if site is not None:
        pinned.append(str(site))
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
