"""Cron: import path of the restart-safe external worker.

The worker is spawned as ``sys.executable -m cron.scheduler``. Its entry module is
``cron.scheduler``, not ``hermes_cli.main``, so nothing bootstraps the gateway's checkout
onto its ``sys.path``; historically it imported ``cron`` only through the implicit ``-m``
cwd entry. That entry is gone under ``PYTHONSAFEPATH`` and useless when the venv's
editable install maps a moved/deleted checkout -- the worker then dies with
"No module named 'cron'" before its ownership ack (#112729, hypothesised cause).

The shared subprocess sanitizer strips Hermes-owned PYTHONPATH entries because user
children must not see our tree. This child IS Hermes, so the pin is applied *after* the
env is built, on the sanitized env. On a self-managed (shell-installer / PM) install the
sanitizer's drop of the runtime site-packages cannot stand this time: the worker inherits
this process's interpreter, which is PM's store Python and owns no third-party
dependencies, so the committed generation's site-packages is restored here too or the
child dies at its first dependency import (``No module named 'ruamel'``, #122222) before
it can publish its ownership acknowledgement.
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _committed_dependency_site_packages(project_root: Path) -> Path | None:
    """The dependency ``site-packages`` PM committed for this install, or ``None``.

    Asked of PM's own committed selection -- the record ``activate_dependencies`` resolves
    at process boot -- rather than re-derived from this process's ``sys.path``: the record
    belongs to this install (``install_state_dir``, shared by every profile it serves).
    ``None`` (no committed generation, unreadable record, or missing tree) lets the
    caller fall back to :func:`_active_runtime_site_packages`; only when that is also
    empty does the pin reduce to the tree itself.
    """
    try:
        from pm.environments import committed_venv, site_packages

        environment = committed_venv(project_root)
    except Exception as exc:
        # An unreadable record: pin the tree only. The cron worker's own boot re-reads it and
        # fails the dispatch with PM's error.
        logger.warning(
            "cron worker: could not read the committed dependency environment: %s", exc
        )
        return None
    if environment is None:
        return None
    selected = site_packages(environment)
    return selected if selected.is_dir() else None


def _active_runtime_site_packages() -> list[Path]:
    """Site-packages dirs this process actually imports from, in priority order.

    Fallback when PM has no committed generation for the checkout (non-PM installs,
    test checkouts, fallback paths): the worker inherits this interpreter
    (``sys.executable``), so handing it the same third-party dirs keeps imports like
    ``ruamel`` working. Sources, in order:

    1. ``sys.path`` entries named ``site-packages``/``dist-packages`` that exist —
       this covers the usual venv layout, ``PYTHONPATH``-injected generations, and
       system site dirs alike, without trusting leaked ``VIRTUAL_ENV``/``PYTHONPATH``
       provenance.
    2. The ``sys.prefix``-derived site-packages (covers ``-S``/isolated launches
       where ``site`` never added it to ``sys.path``).
    3. ``VIRTUAL_ENV``-derived site-packages when it names a different, existing
       tree (a venv-activated gateway whose markers the sanitizer stripped).

    Empty when nothing usable exists — the caller then pins the tree only and
    invents nothing.
    """
    ordered: list[Path] = []
    seen: set[str] = set()

    def _append(candidate: Path) -> None:
        try:
            if not candidate.is_dir():
                return
            key = str(candidate.resolve())
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        ordered.append(candidate)

    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry)
        except (OSError, ValueError):
            continue
        if candidate.name not in ("site-packages", "dist-packages"):
            continue
        _append(candidate)

    try:
        pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
        if os.name == "nt":
            _append(Path(sys.prefix) / "Lib" / "site-packages")
        else:
            _append(Path(sys.prefix) / "lib" / pyver / "site-packages")
    except (OSError, ValueError):
        pass

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        try:
            venv_path = Path(venv)
            if os.name == "nt":
                _append(venv_path / "Lib" / "site-packages")
            else:
                _append(venv_path / "lib" / pyver / "site-packages")
        except (OSError, ValueError):
            pass

    return ordered


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` -- and the dependency ``site-packages`` -- to the worker env's
    own PYTHONPATH (never ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.

    Order (checkout, generation, sanitizer-kept) mirrors ``activate_dependencies``' own
    ``sys.path``. The generation is the committed PM selection when one exists; otherwise
    the current active runtime site-packages from ``sys.path``/venv (``#127016``) so a
    bare store Python worker still finds third-party deps like ``ruamel``. For the cron
    worker, its boot (``cron/worker_bootstrap.py``) then re-selects and leases the
    committed generation before any third-party import, and exits the worker if it cannot.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    dependency = _committed_dependency_site_packages(Path(root))
    if dependency is not None:
        dependencies = [str(dependency)]
    else:
        dependencies = [str(p) for p in _active_runtime_site_packages()]
    pinned = [root, *dependencies]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env
