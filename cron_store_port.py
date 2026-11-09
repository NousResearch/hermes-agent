"""Cron storage port: the scheduled-job store, its fire claims and its execution audit.

One port (:class:`CronStore`) at the level today's call sites use, never SQL and never a file:

- **jobs**: create, read, update, the lifecycle verbs (pause, resume, trigger, re-arm, remove,
  cancel a live run), the due scan a tick reads, the dispatch claim for finite one-shots, the
  pre-dispatch advance of ``next_run_at`` and the run outcome a finished fire writes back
  (``cron/jobs.py``). A job is an agent-level record that also names the principal who created
  it: :meth:`CronStore.create_job` stamps ``created_by`` from the bound scope's actor, never from
  an argument, and no operation may change it afterwards.
- **the at-most-once fire claim**: :meth:`CronStore.claim_job_for_fire` is the compare-and-set
  every fire goes through, so N replicas or triggers competing for one occurrence run it exactly
  once; a claim may be heartbeated while the run is alive and is released by the run's outcome.
- **the execution audit**: one row per attempt a fire makes (claimed, handed off, running,
  terminal), the missed occurrences a scan skipped, and the recovery that closes attempts whose
  owner died (``cron/executions.py``).

A :class:`CronStorageProvider` supplies the store. This is the interface and its local backend
only: no production caller goes through it yet, and nothing in config.yaml selects a backend.
Today's callers (the scheduler, the CLI, the ``cronjob`` tool) still use ``cron.jobs`` and
``cron.executions`` directly — a scheduler provider that fires jobs keeps working unchanged —
and moving those call sites behind the port, one by one, is a follow-up.
:func:`resolve_cron_storage` takes the backend's name explicitly: ``"local"`` (or empty) is
``cron_store_port_local`` (today's per-profile ``jobs.json``, ``output/`` and ``executions.db``
with today's file locking); any other name is a provider from ``plugins/cron_storage/<name>/``
or ``$HERMES_HOME/plugins/<name>/``. A named provider that is missing or unavailable is an
error, never a fallback to the local files.

Providers load in-process only. Under ``plugins.isolation: host`` a user provider is refused
with the reason: a host process would hand back opaque objects for the nested port, and its
calls would not carry the caller's scope. Typed proxies with a scope envelope bound around every
call are a follow-up.

Every operation reads its pin (tenant, agent, owner, actor) from ``hermes_scope.current_scope()``
when it runs, never from a path, ``HERMES_HOME`` or an argument; the port exposes no way to pin
anything else. A backend refuses a pin it cannot serve with :class:`CronPinRefused`, which says
nothing about whether the target exists. On top of the pin, every job mutation requires the
bound scope's administration right: one without it is refused with :class:`CronAdminRefused`
before anything is looked up, while reads and the scheduler's own operations are not gated
(a caller with no scope bound is today's local caller, who administers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Optional

# The ownership levels a store can hold (``Scope`` docs): participant-on-agent, agent, owner-wide,
# tenant-wide. A backend declares which it stores and refuses writes at any other.
LEVELS = frozenset({"participant", "agent", "owner", "tenant"})


class CronPinRefused(PermissionError):
    """The bound scope is not one this store serves: another tenant, another agent, or (for a
    backend that requires one) no scope at all. Says nothing about whether the target exists."""


class CronAdminRefused(PermissionError):
    """The bound scope may not administer this agent, so the job mutation it asked for is
    refused before anything is looked up or written. Says nothing about whether the target
    exists."""


class CronStore(ABC):
    """Cron jobs, their fire claims and their execution audit, for the pinned agent.

    Every operation runs for the agent the bound scope names. A job belongs to one agent: another
    agent's job reads like a missing one and every operation on it under a foreign pin is refused
    whole, so one agent can never list, trigger or re-point another's jobs, and a tick pinned to
    one agent never fires another's due job. Job records are the dicts today's callers exchange:
    the backend may add columns, and a reader must tolerate fields it does not know.

    Every job mutation is the agent's administration: creating, updating, pausing, resuming,
    triggering, re-arming, removing a job and cancelling its live run all require the bound
    scope's administration right (``Scope.can_administer``), and one without it is refused
    whole (:class:`CronAdminRefused`) before anything is looked up, changing nothing. Not gated:
    reads, and the scheduler's own operations — the due scan, dispatch claims, fire claims, the
    claim heartbeats, the run outcome and the execution audit — which run as system activities
    even when a scope is bound. A caller with no scope bound is today's local caller, who
    administers, and is served as today.
    """

    # --- jobs ------------------------------------------------------------------------------

    @abstractmethod
    def create_job(self, prompt: Optional[str], schedule: str, **fields: Any) -> dict[str, Any]:
        """Create a job and return its stored record. *fields* are the creation options today's
        callers pass (name, repeat, deliver, skills, script, ...); an unknown one is refused.
        The record also carries ``created_by``, the id of the principal who created it, stamped
        by the backend from the bound scope's actor — never from *fields* — by the same write
        that creates the job. Requires the administration right."""

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """One job by id, or ``None`` (missing and foreign look the same)."""

    @abstractmethod
    def resolve_job_ref(self, ref: str) -> Optional[dict[str, Any]]:
        """Resolve an id or a name to a job record; ``None`` when nothing matches and
        :class:`LookupError` when the name is ambiguous."""

    @abstractmethod
    def list_jobs(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        """The pinned agent's jobs (all of them with ``include_disabled``)."""

    @abstractmethod
    def update_job(self, job_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Apply *updates* to a job; ``None`` when there was nothing to update. The job's id and
        the principal who created it never change: an update naming either is refused whole.
        Requires the administration right."""

    @abstractmethod
    def pause_job(self, job_id: str, reason: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Pause a job (id or name) without deleting it. Requires the administration right."""

    @abstractmethod
    def resume_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Resume a paused job (id or name); an occurrence that elapsed while paused stays due.
        Requires the administration right."""

    @abstractmethod
    def trigger_job(self, job_id: str, extra_prompt: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Point a job at the next tick (id or name); *extra_prompt* applies to that fire only.
        Requires the administration right."""

    @abstractmethod
    def rearm_oneshot(self, job_id: str, run_at: Any) -> Optional[dict[str, Any]]:
        """Re-arm a completed one-shot (id or name) as an explicit new occurrence. Requires the
        administration right."""

    @abstractmethod
    def remove_job(self, job_id: str) -> bool:
        """Remove a job (id or name); ``False`` when there was nothing to remove. Requires the
        administration right."""

    @abstractmethod
    def cancel_job_run(self, job_id: str, reason: Optional[str] = None) -> dict[str, Any]:
        """Stop the job's live run, if any; its schedule is untouched. Requires the
        administration right."""

    @abstractmethod
    def get_due_jobs(self) -> list[dict[str, Any]]:
        """The pinned agent's jobs due now — the scan a tick walks before dispatching fires. May
        repair records it owns, but never claims a fire: that is :meth:`claim_job_for_fire`."""

    @abstractmethod
    def claim_dispatch(self, job_id: str) -> bool:
        """Claim a finite one-shot's dispatch before execution (at-most-times); ``False`` when
        the limit is already reached."""

    @abstractmethod
    def advance_next_run(self, job_id: str) -> bool:
        """Advance a recurring job's next occurrence before dispatch, so a mid-run crash cannot
        re-fire it; ``True`` when the instant moved."""

    @abstractmethod
    def mark_job_run(self, job_id: str, success: bool, error: Optional[str] = None,
                     delivery_error: Optional[str] = None, status: Optional[str] = None, *,
                     expected_fire_owner: Optional[str] = None,
                     execution_id: Optional[str] = None, model_unreachable: bool = False,
                     ladder_rung: bool = False, quota_hold_seconds: Optional[float] = None,
                     recover_consumed_fire: bool = False) -> bool:
        """Record a finished fire onto the job (outcome stamps, occurrence accounting, next
        occurrence) and release its claims. ``expected_fire_owner`` discards a completion whose
        fire claim another owner now holds. ``False`` when the job is missing, the fence cannot
        be taken, or the fire claim owner changed; nothing is then written."""

    # --- the at-most-once fire claim ----------------------------------------------------------

    @abstractmethod
    def claim_job_for_fire(self, job_id: str, *, claim_ttl_seconds: Optional[int] = None,
                           force: bool = False, manual: bool = False,
                           return_job: bool = False) -> bool | dict[str, Any]:
        """Atomically claim one fire of the job; ``True`` (or the claimed job record with
        ``_scheduled_instant``, when ``return_job``) iff THIS caller won. A missing, terminal or
        paused job is refused unless ``force`` (an explicit manual fire, which also resumes the
        job); a claim younger than *claim_ttl_seconds* (the backend's default lease when
        ``None``) wins and every later claimant loses — so N replicas or triggers competing for
        one occurrence run it exactly once. The claim advances the job past the occurrence it
        took, and the run's outcome (:meth:`mark_job_run`) releases it."""

    @abstractmethod
    def heartbeat_fire_claim(self, job_id: str, *, expected_owner: str) -> bool:
        """Refresh the job's active fire claim without extending another owner's lease."""

    @abstractmethod
    def heartbeat_run_claim(self, job_id: str, *, expected_owner: str) -> bool:
        """Refresh the job's active run claim without extending another owner's lease."""

    @abstractmethod
    def clear_run_claim(self, job_id: str) -> bool:
        """Release the job's run claim; ``False`` when there was none."""

    @abstractmethod
    def fire_claim_fence(self, job_id: str, *, expected_owner: str) -> AbstractContextManager[bool]:
        """Hold the job's fire fence while *expected_owner* performs an external side effect;
        yields whether the owner still holds the claim, ``False`` (fail closed) when the fence
        cannot be taken. A missing record is accepted only for the run that removed exactly
        this job."""

    # --- the execution audit ------------------------------------------------------------------

    @abstractmethod
    def create_execution(self, job_id: str, *, source: str,
                         scheduled_instant: Optional[str] = None) -> dict[str, Any]:
        """Persist one claimed attempt before dispatch; the row names the claiming owner (the
        process or replica that will finish it)."""

    @abstractmethod
    def set_execution_occurrence(self, execution_id: str, instant: Optional[str]) -> None:
        """Bind the claimed occurrence the attempt will run, before a handoff."""

    @abstractmethod
    def mark_execution_handoff_pending(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Fence recovery while a worker adopts the claim; ``None`` when it is not this owner's
        claim to hand off."""

    @abstractmethod
    def adopt_claimed_execution(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Transfer a pending handoff to its worker and start it; the single claimed→running
        gate: ``None`` when this caller did not win the adoption."""

    @abstractmethod
    def mark_execution_running(self, execution_id: str) -> Optional[dict[str, Any]]:
        """Transition this owner's claimed attempt to running exactly once."""

    @abstractmethod
    def touch_execution_progress(self, execution_id: str) -> bool:
        """Stamp progress on a running attempt this owner holds; ``False`` (informational) when
        the row is no longer this owner's or no longer running."""

    @abstractmethod
    def finish_execution(self, execution_id: str, *, success: bool, error: Optional[str] = None,
                         delivery_outcome: Optional[str] = None,
                         cancelled: bool = False) -> Optional[dict[str, Any]]:
        """Write the attempt's terminal result once; ``None`` when it was already terminal, or
        not this owner's to finish. A terminal attempt is never rewritten."""

    @abstractmethod
    def record_missed_occurrences(self, job_id: str, *, count: int, first_instant: Optional[str],
                                  detail: str) -> Optional[dict[str, Any]]:
        """Book occurrences a scan skipped without running as one terminal ``missed`` row."""

    @abstractmethod
    def recover_receipted_execution(self, execution_id: str, job_id: str, *, success: bool,
                                    error: Optional[str] = None,
                                    delivery_outcome: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Settle a verified receipt after the attempt's owner died; ``None`` while that owner
        may still finish it, or for a terminal verdict."""

    @abstractmethod
    def execution_owner_live(self, execution_id: str, job_id: str) -> bool:
        """Whether the attempt's own firer can still finish it, so recovery must not act."""

    @abstractmethod
    def settle_delivery_outcome(self, execution_id: str, outcome: str) -> bool:
        """Replace a terminal attempt's ``queued`` delivery outcome with the drain's real one;
        the only post-terminal rewrite, and only once."""

    @abstractmethod
    def settle_unstarted_execution(self, execution_id: str, job_id: str, error: str) -> None:
        """Close the receipt of a run that never started; best-effort, never raises."""

    @abstractmethod
    def recover_interrupted_executions(self) -> int:
        """Mark this owner's abandoned attempts (its owner provably gone, or silent past the
        stale bound) ``unknown`` without scheduling retries; returns how many."""

    @abstractmethod
    def terminalize_dead_owner(self, execution_id: str, *, reason: str) -> bool:
        """Record one attempt ``unknown`` with a cause this caller observed. ``False`` — leaving
        the caller to the generic sweep — when the row is absent, terminal, owned by a live
        owner, or inside the handoff adoption grace."""

    @abstractmethod
    def give_up_execution(self, execution_id: str, *, reason: str) -> bool:
        """The firer gives up its in-flight attempt: record it ``unknown`` now, owner-fenced,
        instead of leaving it running until a restart's sweep."""

    @abstractmethod
    def get_execution(self, execution_id: str) -> Optional[dict[str, Any]]:
        """One attempt by id, or ``None`` (missing and foreign look the same)."""

    @abstractmethod
    def list_executions(self, *, job_id: Optional[str] = None, limit: int = 50,
                        before_claimed_at: Optional[str] = None) -> list[dict[str, Any]]:
        """Indexed, newest-first execution history with cursor pagination."""

    @abstractmethod
    def latest_execution(self, job_id: str) -> Optional[dict[str, Any]]:
        """The job's newest attempt, or ``None``."""

    @abstractmethod
    def latest_executions(self, job_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Each job's newest attempt, in one read."""

    @abstractmethod
    def live_executions(self, job_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Each job's newest claimed or running attempt under a live owner, in one read."""

    @abstractmethod
    def live_inflight_execution(self, job_id: str) -> Optional[dict[str, Any]]:
        """The job's latest attempt while it is still claimed or running under a live owner."""

    @abstractmethod
    def cancellable_execution(self, job_id: str) -> Optional[dict[str, Any]]:
        """The job's newest claimable attempt under a live owner, even when a newer terminal row
        would hide it from :meth:`latest_execution`."""


class CronStorageProvider(ABC):
    """The plugin kind: supplies the cron store. ``name``, ``open`` and ``assert_isolating`` are
    required; every other method has a safe default, and growth stays additive."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"local"``."""

    #: The ownership levels this backend stores (see :data:`LEVELS`).
    levels: frozenset[str] = frozenset({"agent"})

    def is_available(self) -> bool:
        """Whether this backend can run here. No network calls."""
        return True

    @abstractmethod
    def open(self, home: Any = None) -> CronStore:
        """The store for the bound scope. *home* is the local backend's per-profile home (the
        profile the caller already serves); a backend with no local files ignores it."""

    @abstractmethod
    def assert_isolating(self) -> None:
        """Raise unless this backend isolates tenants and agents from each other in storage
        itself (a Postgres backend whose role is a superuser or has ``BYPASSRLS`` fails).
        Required, with no default: a backend that has not checked must not pass. Nothing calls it
        yet: the first caller that serves a scope from a resolved provider must call it, and
        refuse the provider when it raises, before :meth:`open`."""


def resolve_cron_storage(name: str) -> CronStorageProvider:
    """The provider named *name* (``"local"`` or empty: the local backend). Raises ``LookupError``
    with the reason for a named provider that is missing, refused or unavailable: cron storage
    never falls back to another store."""
    name = name.strip()
    if name in ("", "local"):
        from cron_store_port_local import LocalCronStorageProvider
        return LocalCronStorageProvider()
    from plugins.cron_storage import CronStorageRefused, load_cron_storage
    try:
        provider = load_cron_storage(name)
    except CronStorageRefused as refused:
        raise LookupError(f"cron storage {name!r}: {refused}") from refused
    if provider is None:
        raise LookupError(f"cron storage {name!r}: no such cron storage provider")
    if not provider.is_available():
        raise LookupError(f"cron storage {name!r} is not available here")
    return provider
