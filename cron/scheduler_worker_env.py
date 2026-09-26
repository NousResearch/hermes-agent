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


def managed_runtime_python(repo_root: Path, *, windows: bool | None = None) -> Path | None:
    """The committed dependency environment's own interpreter, or ``None``.

    Under the PM runtime the gateway process runs the bare store Python with the
    environment's site-packages activated in-process — so a child spawned with
    ``sys.executable`` sees NEITHER: the subprocess sanitizer strips PYTHONPATH and
    ``pin_hermes_tree_on_pythonpath`` re-adds only the checkout root, and the worker
    dies importing ``ruamel`` before its ownership ack (#123400, #123440). The venv
    interpreter carries its own site-packages (``pyvenv.cfg``), so it imports Hermes
    modules and managed dependencies directly, and its children resolve their own
    dependencies and stay clean.

    ``None`` on Windows (its cron invocation overlays venv paths instead), when no
    environment is committed, or when the interpreter is missing. *windows* lets a
    POSIX process answer the POSIX layout (pure data, never ``sys.platform``).
    """
    import sys

    if windows is None:
        windows = os.name == "nt"
    if windows:
        return None
    try:
        from pm.environments import committed_venv, venv_python

        python = venv_python(committed_venv(repo_root), windows=windows)
    except Exception:
        return None
    if python == Path(sys.executable):
        return None  # already the running interpreter — keep the launch contract
    if not python.is_file():
        return None
    return python
