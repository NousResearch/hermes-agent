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


def _parent_hermes_site_packages() -> list[str]:
    """site-packages owned by the Hermes runtime, taken from the live parent.

    The gateway resolves its dependencies at runtime (standalone interpreter +
    bootstrap), so the external cron worker — spawned as plain
    ``sys.executable -m cron.scheduler`` — must inherit the same entries or it
    dies importing third-party deps (e.g. ``ruamel`` via ``hermes_yaml``).
    Only existing dirs under a ``.hermes`` tree are returned; when none match,
    behaviour is unchanged (tree pin only).
    """
    import sys

    found: list[str] = []
    for entry in sys.path:
        if "site-packages" not in entry or ".hermes" not in entry:
            continue
        if entry in found:
            continue
        try:
            if Path(entry).is_dir():
                found.append(entry)
        except OSError:
            continue
    return found


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
    ordered = dict.fromkeys([root, *_parent_hermes_site_packages(), *existing])
    worker_env["PYTHONPATH"] = os.pathsep.join(ordered)
    return worker_env
