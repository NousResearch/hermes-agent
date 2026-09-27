"""Cron: import path of the restart-safe external worker.

The worker is spawned through the installation launcher (store Python plus
``hermes_bootstrap``), not a bare ``sys.executable -m cron.scheduler``. The
gateway's executable is that store interpreter: a bare CPython with no
third-party packages until bootstrap activates the committed dependency
generation. A raw ``-m`` dies on the first import (``ruamel``, ``requests``)
before the job script runs.

``pin_hermes_tree_on_pythonpath`` still prepends this checkout. The launcher
clears ``PYTHONPATH`` and inserts the repo itself; the pin covers a process
that reaches the module without that launcher. Historically the entry was
only the implicit ``-m`` cwd, which is gone under ``PYTHONSAFEPATH`` and
useless when an editable install maps a moved checkout (#112729).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because
user children must not see our tree. This child IS Hermes, so the pin is
applied *after* the env is built, on the sanitized env.
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


def external_worker_command(repo_root: Path, payload_path: Path, ack_path: Path) -> list[str]:
    """Argv for the external worker: the same launcher the ``hermes`` command uses.

    Store Python plus ``hermes_bootstrap`` selects the committed dependency
    generation before ``cron.scheduler`` imports. A bare ``-m`` keeps the store
    interpreter's empty site-packages, so the worker dies on ``ruamel`` before
    the job script runs.
    """
    from hermes_cli._launchers import runtime_command

    return runtime_command(
        repo_root,
        ["--external-worker-file", str(payload_path), "--ack-file", str(ack_path)],
        module="cron.scheduler",
    )


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
