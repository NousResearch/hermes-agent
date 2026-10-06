"""Bounded post-run cleanup for cron: the wall-clock lane every fallible finalize step rides.

``_run_cron_cleanup_with_timeout`` runs a cleanup callable under a timeout on a daemon thread
(so a wedged SQLite call can never hold the job's dispatch guard); ``_BoundedCronSessionDB``
proxies the finalizer's SessionDB calls through that lane. Split out of ``cron.scheduler``;
``cron.scheduler`` re-exports the names so existing ``patch("cron.scheduler.*")`` seams hold.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from agent.memory_provider import ctx_bound

logger = logging.getLogger(__name__)


def _cron_cleanup_timeout_seconds() -> float:
    """Return the wall-clock bound for cron post-run cleanup."""
    default = 10.0
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("cleanup_timeout_seconds")
        if configured is not None:
            timeout = float(configured)
            if timeout >= 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron cleanup timeout from config: %s", exc)
    return default


def _run_cron_cleanup_with_timeout(
    cleanup, *, job_id: str, label: str, timeout_seconds: Optional[float] = None,
) -> bool:
    """Run fallible post-run cleanup without permanently wedging a cron ID."""
    timeout = (_cron_cleanup_timeout_seconds() if timeout_seconds is None else float(timeout_seconds))
    if timeout <= 0:
        try:
            cleanup()
            return True
        except (Exception, KeyboardInterrupt) as exc:
            logger.debug("Job '%s': %s failed: %s", job_id, label, exc)
            return False

    done = threading.Event()
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            cleanup()
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    # Daemon thread is deliberate: unlike ThreadPoolExecutor workers it is not joined at interpreter
    # exit if cleanup never returns, so the gateway can still shut down.
    worker = threading.Thread(
        target=ctx_bound(_runner), name=f"cron-cleanup-{job_id}", daemon=True)
    worker.start()
    if not done.wait(timeout):
        logger.error(
            "Job '%s': %s exceeded %.1fs; abandoning cleanup so future runs remain dispatchable",
            job_id,
            label,
            timeout)
        return False
    if error:
        logger.debug("Job '%s': %s failed: %s", job_id, label, error[0])
        return False
    return True


class _BoundedCronSessionDB:
    """Proxy SessionDB cleanup calls through the cron cleanup timeout; after the first failure or
    timeout all later calls fail immediately (a damaged connection leaks at most one worker)."""

    def __init__(self, session_db, job_id: str):
        self._session_db = session_db
        self._job_id = job_id
        self._disabled = False

    def __getattr__(self, name):
        target = getattr(self._session_db, name)
        if not callable(target):
            return target

        def _bounded(*args, **kwargs):
            if self._disabled:
                raise RuntimeError("session finalization disabled after prior cleanup failure")

            result = {}

            def _call():
                try:
                    result["value"] = target(*args, **kwargs)
                except BaseException as exc:
                    result["error"] = exc
                    raise

            ok = _run_cron_cleanup_with_timeout(
                _call, job_id=self._job_id, label=f"session finalization ({name})")
            if not ok:
                error = result.get("error")
                if error is not None:
                    raise error
                # No error yet not complete == timeout: disable so later steps fail fast.
                self._disabled = True
                raise TimeoutError(f"session finalization method {name} timed out")
            return result.get("value")

        return _bounded




def cron_run_history_retention() -> int:
    """``cron.run_history_retention``: SessionDB run rows kept per recurring job (>= 0)."""
    from cron.jobs import _cron_config_number

    return max(0, _cron_config_number("run_history_retention", 50, int))
