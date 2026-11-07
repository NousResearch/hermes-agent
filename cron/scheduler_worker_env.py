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
    belongs to this install (``install_state_dir``, shared by every profile it serves). A
    runner that owns its dependencies (wheel / pipx / developer venv / Nix: no committed
    generation) has nothing to restore, so ``None`` means "pin the tree only" and nothing
    is invented.
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


def pin_hermes_tree_on_pythonpath(worker_env: dict, repo_root: Path) -> dict:
    """Prepend ``repo_root`` -- and, when this runner has one, the committed dependency
    generation's ``site-packages`` -- to the worker env's own PYTHONPATH (never
    ``os.environ``'s).

    Skipped when ``repo_root`` is the interpreter's ``purelib``: under a wheel / pipx /
    uv-tool install ``cron/`` lives in site-packages itself, which is already importable,
    and pinning it would move site-packages ahead of the stdlib on ``sys.path``.

    Order (checkout, generation, sanitizer-kept) mirrors ``activate_dependencies``' own
    ``sys.path``. For the cron worker, its boot (``cron/worker_bootstrap.py``) then re-selects
    and leases the committed generation before any third-party import, and exits the worker
    if it cannot.
    """
    root = str(repo_root)
    if _installed_purelib() == Path(root).resolve():
        return worker_env
    existing = [e for e in worker_env.get("PYTHONPATH", "").split(os.pathsep) if e]
    dependency = _committed_dependency_site_packages(Path(root))
    pinned = [root, *([str(dependency)] if dependency is not None else [])]
    worker_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([*pinned, *existing]))
    return worker_env


# --- Between dependency generations: hold the slot, do not book a failure -------------
#
# PM unlinks ``facts.json`` while it rebuilds a generation, so a worker booted in that window
# raises ``pm.environments.NoCommittedEnvironment`` and exits BEFORE its ownership
# acknowledgement. The dispatcher reports that as a failed run (#123401 made a real outage
# loud on purpose, and it must stay loud for a real one) — but a rebuild is transient: the
# same install commits a generation seconds to minutes later, so a fire landing inside the
# window is a missed slot, not a broken job.
#
# Hold it the way the headless gateway gate does: no run is recorded, the due instant does
# not move, and the job fires on the next tick once a generation is committed. Bounded by
# ``GRACE_SECONDS``: past it the normal loud failure path resumes, so an install that is
# genuinely broken still opens an incident and still pages.

GRACE_SECONDS = 600.0


def dependency_environment_uncommitted(project_root: Path) -> bool:
    """True when a worker booted for this install would refuse to start.

    Asked of PM, not re-derived: ``committed_venv`` is empty for a venv / wheel / Nix install
    too, where nothing is wrong and a worker boots on its own interpreter. Only a store-managed
    install with no committed generation raises ``NoCommittedEnvironment`` — the exact refusal
    that kills the dispatch before its acknowledgement.
    """
    root = Path(project_root)
    try:
        from pm.environments import (
            NoCommittedEnvironment,
            _require_own_dependencies,
            committed_venv,
        )
    except Exception:  # a tree without PM owns its dependencies: dispatch
        return False
    try:
        if committed_venv(root) is not None:
            return False
        _require_own_dependencies(root)
    except NoCommittedEnvironment:
        return True
    except Exception:  # unreadable record: let the dispatch fail loudly, as it does today
        return False
    return False


def _defer_marker(home: Path) -> Path:
    return Path(home) / "cron" / "dependency-uncommitted-since"


def _defer_enabled(cfg: dict | None = None) -> bool:
    """``cron.defer_uncommitted_dependencies`` — default ON (it can only skip a doomed slot)."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
        except Exception:  # config unreadable: keep the reliability default
            return True
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("defer_uncommitted_dependencies") is not False


def defer_while_dependencies_uncommitted(
    due_jobs: list,
    *,
    home: Path,
    project_root: Path | None = None,
    grace_seconds: float = GRACE_SECONDS,
    now: float | None = None,
) -> list:
    """Due jobs a tick may dispatch; all of them are held while PM is between generations.

    The caller must invoke this before advancing ``next_run_at``: holding here leaves the due
    instant where it is, so the job is retried by the next tick rather than lost.
    """
    if not due_jobs:
        return due_jobs
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    if not dependency_environment_uncommitted(root) or not _defer_enabled():
        # Committed (or undeterminable): dispatch, and clear a stale marker so a later window
        # is measured from its own start rather than an old one.
        try:
            _defer_marker(home).unlink(missing_ok=True)
        except OSError:
            pass
        return due_jobs

    import time as _time

    marker = _defer_marker(home)
    stamp = _time.time() if now is None else now
    try:
        since = float(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        since = None
    if since is None:
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(str(stamp), encoding="utf-8")
        except OSError:
            pass  # an unwritable home must not turn a deferral into a dispatch
        since = stamp
    waited = stamp - since
    if waited >= grace_seconds:
        logger.warning(
            "Dependency environment uncommitted for %.0fs (>= %.0fs grace); dispatching %d "
            "due job(s) so the failure is reported",
            waited, grace_seconds, len(due_jobs))
        return due_jobs

    ids = [str(job["id"]) for job in due_jobs]
    detail = ("Skipped: the install's dependency environment is being rebuilt "
              "(PM commits a new generation shortly)")
    from cron.scheduler_gateway_gate import _note_fire_refused

    _note_fire_refused(ids, detail)
    logger.warning(
        "Dependency environment uncommitted for %.0fs; held %d due job(s) (%s) for the next tick",
        waited, len(ids), ", ".join(ids))
    return []
