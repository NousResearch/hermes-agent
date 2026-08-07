"""Chronos — NAS-mediated managed cron provider (scale-to-zero).

Chronos (the Greek god of time, alongside Hermes) is the first non-default
``CronScheduler``. It lets a hosted gateway scale to zero while idle and still
fire cron jobs: instead of a 60s in-process ticker, it asks NAS to arm exactly
one external one-shot per job at that job's real next-fire time. NAS calls the
agent back at fire time over an authenticated webhook (``/api/cron/fire``); the
agent runs the job via the shared ``run_one_job`` body and re-arms the next
one-shot.

The external scheduler NAS uses is an internal NAS implementation detail —
Chronos names no vendor, holds no scheduler credentials, and speaks only to
NAS's ``agent-cron`` endpoints with the agent's existing Nous token.

Design constraints (see the plan's DQ-1):
  - start() arms all enabled jobs and RETURNS; it never blocks and never spawns
    a periodic wake. Between fires the machine is truly at zero.
  - reconcile runs only on a warm process (start / on_jobs_changed / piggybacked
    on a fire), never as a periodic wake of a sleeping machine.

Inert unless ``cron.provider: chronos``. ``resolve_cron_scheduler`` falls back
to the built-in if Chronos is unavailable, so cron never loses its trigger.

Wire contract: ``docs/chronos-managed-cron-contract.md``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

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
        # In-memory map of job_id → fire_at we've asked NAS to arm. Best-effort
        # cache; reconcile rebuilds desired state from jobs.json, so a cold
        # process simply re-arms (idempotent via dedup_key).
        self._armed: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._client = None  # lazily constructed (no network in is_available)

    # -- identity / availability -----------------------------------------

    @property
    def name(self) -> str:
        return "chronos"

    def is_available(self) -> bool:
        """Config presence only — NO network.

        Chronos needs a portal base URL, the agent's own publicly-reachable
        callback URL (for NAS→agent fires), and a usable Nous token (the agent
        is logged into the portal). If any is missing, resolve_cron_scheduler
        falls back to the built-in ticker.
        """
        if not (_cfg("cron", "chronos", "portal_url") and _cfg("cron", "chronos", "callback_url")):
            return False
        return self._have_nous_token()

    def _have_nous_token(self) -> bool:
        """True if the agent has a Nous Portal login (no network call).

        Checks the stored auth state for a Nous access token — does NOT refresh
        or hit the network (is_available must stay offline). The actual
        refresh-aware token is resolved lazily at provision time.
        """
        try:
            from hermes_cli.auth import get_provider_auth_state
            state = get_provider_auth_state("nous") or {}
            return bool(state.get("access_token"))
        except Exception:
            return False

    # -- client -----------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from ._nas_client import NasCronClient
            self._client = NasCronClient(_cfg("cron", "chronos", "portal_url"))
        return self._client

    def _callback_url(self) -> str:
        return str(_cfg("cron", "chronos", "callback_url") or "")

    # -- lifecycle --------------------------------------------------------

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        """Arm all enabled jobs via NAS, then RETURN immediately.

        Does NOT block and does NOT spawn a 60s wake (DQ-1) — that is the whole
        point of scale-to-zero. The machine wakes only on a NAS→agent fire.
        """
        # A new provider lifecycle cannot prove what an interrupted prior
        # process did. Classify those attempts unknown for audit only; do not
        # requeue them here.
        self.recover_interrupted()
        try:
            self.reconcile()
        except Exception as e:
            logger.warning("Chronos start() reconcile failed: %s", e)
        # Intentionally return — no loop, no periodic wake.

    def stop(self) -> None:
        return None

    def on_jobs_changed(self) -> None:
        """A job was created/updated/removed/paused/resumed — reconcile the NAS
        registry so the affected one-shot is (re-)armed or cancelled."""
        try:
            self.reconcile()
        except Exception as e:
            logger.debug("Chronos on_jobs_changed reconcile failed: %s", e)

    # -- arming -----------------------------------------------------------

    def _arm_one_shot(self, job: Dict[str, Any]) -> None:
        """Ask NAS to arm exactly one one-shot at the job's next_run_at.

        The agent computes the time; NAS+its scheduler are the dumb executor.
        Idempotent per (job_id, fire_at) via dedup_key, so re-arming the same
        fire is a no-op NAS-side.
        """
        job_id = job["id"]
        fire_at = job.get("next_run_at")
        if not fire_at:
            return
        dedup_key = f"{job_id}:{fire_at}"
        self._get_client().provision(
            job_id=job_id,
            fire_at=fire_at,
            agent_callback_url=self._callback_url(),
            dedup_key=dedup_key,
        )
        with self._lock:
            self._armed[job_id] = fire_at

    def _cancel(self, job_id: str) -> None:
        try:
            self._get_client().cancel(job_id=job_id)
        finally:
            with self._lock:
                self._armed.pop(job_id, None)

    def _list_armed(self) -> Dict[str, str]:
        """Observed armed one-shots: job_id → fire_at.

        Prefer the in-memory map (warm process); on a cold/empty map, ask NAS
        (best-effort). If NAS list fails, return what we have — reconcile then
        re-arms desired jobs idempotently.
        """
        with self._lock:
            if self._armed:
                return dict(self._armed)
        try:
            observed = {
                item["job_id"]: item.get("fire_at", "")
                for item in self._get_client().list_armed()
                if item.get("job_id")
            }
            with self._lock:
                self._armed.update(observed)
            return observed
        except Exception as e:
            logger.debug("Chronos _list_armed failed (will re-arm idempotently): %s", e)
            return {}

    # -- reconcile --------------------------------------------------------

    def reconcile(self) -> None:
        """Converge the NAS-armed one-shots toward jobs.json (desired state):
        arm missing / re-arm changed-time, cancel orphaned."""
        from cron.jobs import load_jobs

        desired: Dict[str, str] = {
            j["id"]: j["next_run_at"]
            for j in load_jobs()
            if j.get("enabled") and j.get("next_run_at") and j.get("state") != "paused"
        }
        observed = self._list_armed()

        # Arm missing or changed-time.
        for job_id, fire_at in desired.items():
            if observed.get(job_id) != fire_at:
                # Re-fetch the full job dict to arm (need the whole record).
                from cron.jobs import get_job
                job = get_job(job_id)
                if job:
                    try:
                        self._arm_one_shot(job)
                    except Exception as e:
                        logger.warning("Chronos failed to arm job %s: %s", job_id, e)

        # Cancel orphans (armed but no longer desired).
        for job_id in list(observed.keys()):
            if job_id not in desired:
                try:
                    self._cancel(job_id)
                except Exception as e:
                    logger.warning("Chronos failed to cancel orphan %s: %s", job_id, e)

    # -- fire -------------------------------------------------------------

    def fire_due(self, job_id: str, *, adapters: Any = None, loop: Any = None) -> bool:
        """Run the due job (claim + run_one_job via the ABC default), then
        re-arm the NEXT one-shot through NAS.

        Re-arm happens AFTER the run so next_run_at reflects the completed fire.
        If the job is gone (one-shot completed / repeat-N exhausted), get_job
        returns None → nothing to re-arm (the schedule naturally stops).
        """
        ran = super().fire_due(job_id, adapters=adapters, loop=loop)
        if ran:
            from cron.jobs import get_job
            job = get_job(job_id)
            if job and job.get("enabled") and job.get("next_run_at"):
                try:
                    self._arm_one_shot(job)
                except Exception as e:
                    logger.warning("Chronos failed to re-arm job %s after fire: %s", job_id, e)
        return ran


def register(ctx) -> None:
    """Plugin entrypoint — register the Chronos provider with the loader.

    Mirrors the memory-plugin shape; plugins/cron_providers discovery calls this and
    collects the provider via register_cron_scheduler.
    """
    ctx.register_cron_scheduler(ChronosCronScheduler())
