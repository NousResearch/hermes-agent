"""Cron: import path and launch contract of the restart-safe external worker.

The worker is spawned as ``cron.scheduler``, not ``hermes_cli.main``, so nothing
bootstraps the gateway's checkout onto its ``sys.path``; historically it imported
``cron`` only through the implicit ``-m`` cwd entry. That entry is gone under
``PYTHONSAFEPATH`` and useless when the venv's editable install maps a moved/deleted
checkout -- the worker then dies with "No module named 'cron'" before its ownership
ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. For children whose entry module is
``hermes_cli.main`` that is enough: their own bootstrap owns dependency activation,
so the tree pin (``pin_hermes_tree_on_pythonpath``, used by the kanban dispatcher
and the bot-chat delivery child) only has to make the package findable.

``cron.scheduler`` has no such bootstrap of its own, and its module-level imports
need the application's dependencies before any entry-point code could run. Pinning
only the tree left the worker importing third-party code from the store
interpreter's own site-packages -- a tree nothing synchronizes, so a code advance
that raises the startup import surface killed every cron worker until a hand
repair installed the package into the store interpreter (28.09.2026: workers down
~9h between the update that advanced the code and the lazy ``dotenv`` repair).
The worker therefore launches through the SAME bootstrap every Hermes entry point
uses (``external_worker_command``): isolated mode, repo root pinned by the
bootstrap, ``import hermes_bootstrap`` selecting and leasing the committed
dependency generation at child start -- the generation the very same update run
synchronized and startup-validated.
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

    Only for children whose entry module bootstraps dependency activation itself
    (``hermes_cli.main``); the external cron worker launches through
    :func:`external_worker_command` instead.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([root, *existing]))
    return worker_env


def external_worker_command(repo_root: Path, *, payload_path: Path, ack_path: Path) -> list[str]:
    """Bootstrap-embedded argv for the external cron worker.

    A bare ``sys.executable -m cron.scheduler`` boots with no dependency environment:
    the sanitized env strips the selected generation's site-packages, and the store
    interpreter's own site-packages carries nothing the updater synchronizes. The
    shared launcher contract fixes that: ``-I`` ignores the sanitized env's
    ``PYTHON*`` residue, the bootstrap pins the checkout and ``import
    hermes_bootstrap`` selects and leases the committed dependency generation before
    ``cron.scheduler``'s module-level imports run. ``cron.scheduler``'s argparse reads
    ``sys.argv[1:]``, which ``python -I -c <code> <args>...`` spells exactly.
    """
    from hermes_cli._launchers import runtime_command

    return runtime_command(
        Path(repo_root),
        ["--external-worker-file", str(payload_path), "--ack-file", str(ack_path)],
        module="cron.scheduler",
    )
