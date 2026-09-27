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
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *existing]))
    return worker_env


def external_worker_command_prefix(repo_root: Path) -> list[str]:
    """Interpreter argv prefix that can import the external worker's dependencies.

    Under a managed store install ``sys.executable`` is the bare store Python: the repo
    and managed site-packages exist only on the parent's in-process ``sys.path``
    (installed by ``activate_dependencies`` at bootstrap) and a fresh child re-runs no
    dependency activation, so it dies on its first third-party import -- "No module
    named 'ruamel'" (#125269). Cron scripts already run the committed dependency venv's
    own interpreter for exactly this (#123044/#123440); the worker takes the same
    interpreter. Elsewhere (venv checkout, wheel/pipx) the current interpreter is
    already the right one.
    """
    from hermes_cli._launchers import resolve_store_python
    from pm.environments import project_python

    if resolve_store_python(repo_root) is None:
        return [sys.executable]
    python = project_python(repo_root)
    if not python.is_file():
        # Caller's interpreter is the bare store Python here (the #123044 mode): a named
        # failure beats a spawn that dies before its ownership ack.
        raise RuntimeError(f"dependency environment interpreter is missing: {python}")
    return [str(python)]
