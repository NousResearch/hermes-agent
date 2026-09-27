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

The gateway process itself runs on the bundled store Python with dependencies activated
onto ``sys.path`` at boot (``pm.environments.activate_dependencies``). A bare ``-m``
child never runs that bootstrap, so without the same activation it dies with
``ModuleNotFoundError: No module named 'ruamel'`` before its ownership ack -- every
managed-topology cron job fails forever while in-process jobs look fine. The fix is a
real bootstrap in the child (``bootstrap_worker_environment``), called FIRST in the
``__main__`` external-worker branch: repo-root pin + ``site.addsitedir`` of the
committed generation (which runs ``.pth`` hooks, unlike PYTHONPATH) + a generation
lease so GC cannot reap the tree mid-job after a gateway restart reselects.
"""

from __future__ import annotations

import os
import site
import sysconfig
from pathlib import Path


def _installed_purelib() -> Path | None:
    try:
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    except (KeyError, OSError):
        return None


def _committed_venv(repo_root: Path):
    """The committed dependency venv for this checkout, or ``None``.

    ``None`` when nothing is committed yet (fresh install), the record is
    unreadable, or the tree is gone -- the caller then keeps the historical
    repo-root-only behaviour. Never the in-tree ``venv``/``.venv``: that tree
    predates PM and is built for whichever interpreter created it
    (``committed_venv`` already excludes it).
    """
    try:
        from pm.environments import committed_venv
    except ImportError:
        return None
    try:
        return committed_venv(repo_root)
    except (OSError, RuntimeError, ValueError):
        return None


def _committed_site_packages(repo_root: Path) -> Path | None:
    """Site-packages of the committed dependency environment, or ``None``."""
    venv = _committed_venv(repo_root)
    if venv is None:
        return None
    try:
        from pm.environments import site_packages
    except ImportError:
        return None
    try:
        selected = site_packages(venv)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        return selected if selected.is_dir() else None
    except OSError:
        return None


def bootstrap_worker_environment(repo_root: Path) -> None:
    """Activate the gateway's import path inside the external worker process.

    Must run before any third-party import in the ``--external-worker-file``
    branch: prepends the checkout to ``sys.path`` (covers ``PYTHONSAFEPATH`` and
    rotted editable mappings, #112729), ``addsitedir``s the committed
    generation's site-packages (runs ``.pth`` hooks -- plain PYTHONPATH does
    not), and leases the generation so ``collect_generations`` cannot delete
    the tree mid-job. Best-effort throughout: a worker that cannot fully
    activate still tries the historical repo-root-only path rather than dying
    in the bootstrap itself.
    """
    try:
        root = repo_root.resolve()
    except OSError:
        return
    import sys

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    selected = _committed_site_packages(repo_root)
    if selected is None:
        return
    try:
        same = selected.resolve() == root
    except OSError:
        return
    if same:
        return
    try:
        site.addsitedir(str(selected))
    except (OSError, ValueError):
        return
    try:
        import hermes_cli.runtime_state as runtime_state

        venv = _committed_venv(repo_root)
        if venv is not None:
            runtime_state.lease_generation(venv)
    except Exception:
        pass


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend the checkout to the worker env's own PYTHONPATH (never ``os.environ``'s).

    The dependency half of the activation happens in-process via
    ``bootstrap_worker_environment`` (``site.addsitedir`` + lease); the parent
    cannot lease a generation it does not hold past its own restart, and
    PYTHONPATH alone skips ``.pth`` hooks. So the parent pins only the
    checkout here -- enough for the child to find this module and bootstrap
    itself.

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


def _same_tree(left: Path, right: Path) -> bool:
    """True when both paths resolve to the same tree (wheel-style layout guard)."""
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
