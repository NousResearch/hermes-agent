"""Chronos — NAS-mediated managed cron provider (scale-to-zero).

Instead of a 60s ticker, asks NAS to arm one external one-shot per job at its next-fire time;
NAS calls back ``/api/cron/fire`` and the job re-arms after running. start() never blocks or
spawns a periodic wake; reconcile runs only on a warm process (start / on_jobs_changed / fire).
Holds no scheduler credentials — speaks only to NAS ``agent-cron`` endpoints with the Nous token.
Inert unless ``cron.provider: chronos``. Wire contract: ``docs/chronos-managed-cron-contract.md``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from cron.scheduler_provider import CronScheduler

logger = logging.getLogger("cron.chronos")


def _cfg(*keys: str, default: Any = "") -> Any:
    """Read a cron.chronos.* config value (no network)."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), *keys, default=default)
    except Exception:
        return default


class ChronosCronScheduler(CronScheduler):
    """NAS-mediated external cron provider."""

    def __init__(self) -> None:
        # Best-effort job_id -> fire_at cache; a cold process simply re-arms (idempotent).
        self._armed: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._client = None  # lazily constructed (no network in is_available)

    @property
    def name(self) -> str:
        return "chronos"

    def is_available(self) -> bool:
        """Config presence only — NO network: portal URL, publicly reachable callback URL and a
        Nous login; otherwise the resolver falls back to the built-in ticker."""
        if not (_cfg("cron", "chronos", "portal_url") and _cfg("cron", "chronos", "callback_url")):
            return False
        # Stored-token presence only (no refresh); refresh-aware token resolved at provision time.
        try:
            from hermes_cli.auth import get_provider_auth_state
            return bool((get_provider_auth_state("nous") or {}).get("access_token"))
        except Exception:
            return False

    def _get_client(self):
        if self._client is None:
            from ._nas_client import NasCronClient
            self._client = NasCronClient(_cfg("cron", "chronos", "portal_url"))
        return self._client

    def _reconcile_logged(self, log, what: str) -> None:
        try:
            self.reconcile()
        except Exception as e:
            log("Chronos %s reconcile failed: %s", what, e)

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        """Arm all enabled jobs via NAS, then RETURN — no loop, no periodic wake (scale-to-zero)."""
        # A new lifecycle can't prove what an interrupted process did: classify unknown, never requeue.
        self.recover_interrupted()
        self._reconcile_logged(logger.warning, "start()")

    def stop(self) -> None:
        pass

    def on_jobs_changed(self) -> None:
        self._reconcile_logged(logger.debug, "on_jobs_changed")

    def register_job(self, job: Dict[str, Any]) -> None:
        """Arm the first one-shot for a new job; may raise so creation can report it."""
        self._arm_one_shot(job)

    def _arm_one_shot(self, job: Dict[str, Any]) -> None:
        """Arm one one-shot at next_run_at (agent computes the time; NAS executes).
        dedup_key=(job_id, fire_at) makes re-arming the same fire a no-op."""
        job_id = job["id"]
        fire_at = job.get("next_run_at")
        if not fire_at:
            return
        self._get_client().provision(
            job_id=job_id, fire_at=fire_at, dedup_key=f"{job_id}:{fire_at}",
            agent_callback_url=str(_cfg("cron", "chronos", "callback_url") or ""))
        with self._lock:
            self._armed[job_id] = fire_at

    def _arm_logged(self, job: Dict[str, Any], what: str) -> None:
        """Best-effort arm: log a warning instead of raising (reconcile/fire must not die)."""
        try:
            self._arm_one_shot(job)
        except Exception as e:
            logger.warning("Chronos failed to %s: %s", what, e)

    def _cancel(self, job_id: str) -> None:
        try:
            self._get_client().cancel(job_id=job_id)
        finally:
            with self._lock:
                self._armed.pop(job_id, None)

    def _list_armed(self) -> Dict[str, str]:
        """Armed one-shots (job_id -> fire_at): in-memory map when warm, else ask NAS ({} on
        failure — reconcile then re-arms idempotently)."""
        with self._lock:
            if self._armed:
                return dict(self._armed)
        try:
            observed = {item["job_id"]: item.get("fire_at", "")
                        for item in self._get_client().list_armed() if item.get("job_id")}
            with self._lock:
                self._armed.update(observed)
            return observed
        except Exception as e:
            logger.debug("Chronos _list_armed failed (will re-arm idempotently): %s", e)
            return {}

    def reconcile(self) -> None:
        """Converge NAS one-shots toward jobs.json: arm missing/changed, cancel orphans."""
        from cron.jobs import get_job, load_jobs
        desired: Dict[str, str] = {
            j["id"]: j["next_run_at"] for j in load_jobs()
            if j.get("enabled") and j.get("next_run_at") and j.get("state") != "paused"}
        observed = self._list_armed()
        for job_id, fire_at in desired.items():
            if observed.get(job_id) != fire_at and (job := get_job(job_id)):
                self._arm_logged(job, f"arm job {job_id}")
        for job_id in observed.keys() - desired.keys():
            try:
                self._cancel(job_id)
            except Exception as e:
                logger.warning("Chronos failed to cancel orphan %s: %s", job_id, e)

    # No ``fire_due`` override on purpose: ``provider_supports_split_fire`` treats ANY override as
    # the legacy single-phase signal, opting Chronos out of claim admission and cancel-aware drain.

    def fire_claimed(
        self, claimed_job: dict, *, adapters: Any = None, loop: Any = None, cancel_event: Any = None
    ) -> bool:
        ran = super().fire_claimed(claimed_job, adapters=adapters, loop=loop, cancel_event=cancel_event)
        if ran:
            from cron.jobs import get_job
            job_id = claimed_job["id"]
            job = get_job(job_id)
            if job and job.get("enabled") and job.get("next_run_at"):
                self._arm_logged(job, f"re-arm job {job_id} after fire")
        return ran


def register(ctx) -> None:
    """Plugin entrypoint — plugins/cron_providers discovery collects the provider here."""
    ctx.register_cron_scheduler(ChronosCronScheduler())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Optional  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
