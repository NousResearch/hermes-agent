"""The local cron storage backend: today's per-profile ``jobs.json`` and ``executions.db``.

:class:`LocalCronStore` is ``cron.jobs`` (the job store, the fire claim and its file fences) plus
``cron.executions`` (the audit ledger), one profile's files behind the port. Each operation is the
same call the code makes today, after one check of the pin: the bound scope
(``hermes_scope.current_scope()``) must be a local one whose agent is the store's own profile
home. A local store holds one agent and only agent-level records, so any other pin (another
tenant, another profile) is refused with :class:`CronPinRefused` before anything is read or
written. With no scope bound, the pin is the ambient profile (``get_hermes_home()``), as every
local caller has it today. On top of the pin, every job mutation requires the bound scope's
administration right (:class:`CronAdminRefused`): a local profile's person administers, so today's
unbound caller is served as always, while a scope that explicitly withholds the right is refused.

The store's identity is the profile home it was opened for, taken as one canonical absolute
path at open (a relative path is refused: it cannot be established now which profile it names).
A scope's agent is compared as the canonical absolute directory it names when the operation
runs, so a directory alias of the profile (a symlinked home) is the same profile. The files the
operations touch are the ones today's code resolves for that home under the bound scope —
``<home>/cron/jobs.json`` and ``<home>/cron/executions.db`` with today's locking.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Optional

from cron_store_port import CronAdminRefused, CronPinRefused, CronStorageProvider, CronStore

# Job fields no update may change: the id is a path component under the output dir, and the
# creating principal is a fact about the record, not a preference of whoever edits the job.
_IMMUTABLE_JOB_FIELDS = frozenset({"id", "created_by"})


def _jobs():
    """``cron.jobs`` imported at call time: the port module stays free of the local backend."""
    from cron import jobs
    return jobs


def _executions():
    from cron import executions
    return executions


class LocalCronStore(CronStore):
    """One profile's job store, fire claims and execution audit, pinned to its home."""

    def __init__(self, home: Any) -> None:
        if home is None or not os.path.isabs(os.fspath(home)):
            raise CronPinRefused(f"cron store home {home!r} is not an absolute profile path")
        from hermes_constants import hermes_home_key
        self.home = Path(os.fspath(home)).expanduser().resolve()
        self._home_key = hermes_home_key(self.home)

    def _pin(self) -> None:
        from hermes_constants import get_hermes_home, hermes_home_key
        from hermes_scope import LOCAL_TENANT, current_scope
        scope = current_scope()
        if scope is not None and scope.tenant != LOCAL_TENANT:
            raise CronPinRefused("a local cron store serves tenant 'local' only")
        agent = scope.agent if scope is not None else get_hermes_home()
        # Both sides canonical and absolute: a relative agent is the directory it names now.
        if hermes_home_key(agent) != self._home_key:
            raise CronPinRefused("the bound agent is not this cron store's profile")

    def _creator(self) -> str:
        """The principal who created a job: the bound scope's actor, else the local owner."""
        from hermes_scope import LOCAL_OWNER, Principal, System, current_scope
        scope = current_scope()
        actor = scope.actor if scope is not None else LOCAL_OWNER
        if isinstance(actor, Principal):
            return actor.id
        if isinstance(actor, System):
            return f"system:{actor.reason}"
        return str(actor)

    def _require_administration(self) -> None:
        """Every job mutation is the agent's administration: the bound scope must carry the
        administration right. Not gated: reads, the scheduler's own operations (the due scan,
        dispatch claims, fire claims, claim heartbeats, the run outcome, the execution audit),
        and a caller with no scope bound — today's local caller, who administers."""
        from hermes_scope import current_scope
        scope = current_scope()
        if scope is not None and not scope.can_administer:
            raise CronAdminRefused(
                "the bound principal may not administer this profile's cron jobs")

    # --- jobs ------------------------------------------------------------------------------

    def create_job(self, prompt, schedule, **fields):
        self._pin()
        self._require_administration()
        # The creator rides the same durable write as the job itself: an interruption can
        # leave no job, never a runnable one without its creating principal.
        return _jobs().create_job(prompt, schedule, created_by=self._creator(), **fields)

    def get_job(self, job_id):
        self._pin()
        return _jobs().get_job(job_id)

    def resolve_job_ref(self, ref):
        self._pin()
        return _jobs().resolve_job_ref(ref)

    def list_jobs(self, include_disabled=False):
        self._pin()
        return _jobs().list_jobs(include_disabled=include_disabled)

    def update_job(self, job_id, updates):
        self._pin()
        self._require_administration()
        bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
        if bad_fields:
            raise ValueError(
                f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}")
        return _jobs().update_job(job_id, updates)

    def pause_job(self, job_id, reason=None):
        self._pin()
        self._require_administration()
        return _jobs().pause_job(job_id, reason=reason)

    def resume_job(self, job_id):
        self._pin()
        self._require_administration()
        return _jobs().resume_job(job_id)

    def trigger_job(self, job_id, extra_prompt=None):
        self._pin()
        self._require_administration()
        return _jobs().trigger_job(job_id, extra_prompt=extra_prompt)

    def rearm_oneshot(self, job_id, run_at):
        self._pin()
        self._require_administration()
        return _jobs().rearm_oneshot(job_id, run_at)

    def remove_job(self, job_id):
        self._pin()
        self._require_administration()
        return _jobs().remove_job(job_id)

    def cancel_job_run(self, job_id, reason=None):
        self._pin()
        self._require_administration()
        return _jobs().cancel_job_run(job_id, reason=reason)

    def get_due_jobs(self):
        self._pin()
        return _jobs().get_due_jobs()

    def claim_dispatch(self, job_id):
        self._pin()
        return _jobs().claim_dispatch(job_id)

    def advance_next_run(self, job_id):
        self._pin()
        return _jobs().advance_next_run(job_id)

    def mark_job_run(self, job_id, success, error=None, delivery_error=None, status=None, *,
                     expected_fire_owner=None, execution_id=None, model_unreachable=False,
                     ladder_rung=False, quota_hold_seconds=None, recover_consumed_fire=False):
        self._pin()
        return _jobs().mark_job_run(
            job_id, success, error=error, delivery_error=delivery_error, status=status,
            expected_fire_owner=expected_fire_owner, execution_id=execution_id,
            model_unreachable=model_unreachable, ladder_rung=ladder_rung,
            quota_hold_seconds=quota_hold_seconds, recover_consumed_fire=recover_consumed_fire)

    # --- the at-most-once fire claim ----------------------------------------------------------

    def claim_job_for_fire(self, job_id, *, claim_ttl_seconds=None, force=False, manual=False,
                           return_job=False):
        self._pin()
        claim = {"claim_ttl_seconds": claim_ttl_seconds} if claim_ttl_seconds is not None else {}
        return _jobs().claim_job_for_fire(job_id, force=force, manual=manual,
                                          return_job=return_job, **claim)

    def heartbeat_fire_claim(self, job_id, *, expected_owner):
        self._pin()
        return _jobs().heartbeat_fire_claim(job_id, expected_owner=expected_owner)

    def heartbeat_run_claim(self, job_id, *, expected_owner):
        self._pin()
        return _jobs().heartbeat_run_claim(job_id, expected_owner=expected_owner)

    def clear_run_claim(self, job_id):
        self._pin()
        return _jobs().clear_run_claim(job_id)

    def fire_claim_fence(self, job_id, *, expected_owner):
        self._pin()
        jobs = _jobs()

        @contextlib.contextmanager
        def fence():
            # The fence is pinned when it is created AND when it is entered: the scope
            # current at entry decides whose files the held fence runs on, never the one
            # that happened to be bound when the fence was made.
            self._pin()
            with jobs.fire_claim_fence(job_id, expected_owner=expected_owner) as holds:
                yield holds

        return fence()

    # --- the execution audit ------------------------------------------------------------------

    def create_execution(self, job_id, *, source, scheduled_instant=None):
        self._pin()
        return _executions().create_execution(job_id, source=source,
                                              scheduled_instant=scheduled_instant)

    def set_execution_occurrence(self, execution_id, instant):
        self._pin()
        return _executions().set_execution_occurrence(execution_id, instant)

    def mark_execution_handoff_pending(self, execution_id):
        self._pin()
        return _executions().mark_execution_handoff_pending(execution_id)

    def adopt_claimed_execution(self, execution_id):
        self._pin()
        return _executions().adopt_claimed_execution(execution_id)

    def mark_execution_running(self, execution_id):
        self._pin()
        return _executions().mark_execution_running(execution_id)

    def touch_execution_progress(self, execution_id):
        self._pin()
        return _executions().touch_execution_progress(execution_id)

    def finish_execution(self, execution_id, *, success, error=None, delivery_outcome=None,
                         cancelled=False):
        self._pin()
        return _executions().finish_execution(
            execution_id, success=success, error=error, delivery_outcome=delivery_outcome,
            cancelled=cancelled)

    def record_missed_occurrences(self, job_id, *, count, first_instant, detail):
        self._pin()
        return _executions().record_missed_occurrences(
            job_id, count=count, first_instant=first_instant, detail=detail)

    def recover_receipted_execution(self, execution_id, job_id, *, success, error=None,
                                    delivery_outcome=None):
        self._pin()
        return _executions().recover_receipted_execution(
            execution_id, job_id, success=success, error=error, delivery_outcome=delivery_outcome)

    def execution_owner_live(self, execution_id, job_id):
        self._pin()
        return _executions().execution_owner_live(execution_id, job_id)

    def settle_delivery_outcome(self, execution_id, outcome):
        self._pin()
        return _executions().settle_delivery_outcome(execution_id, outcome)

    def settle_unstarted_execution(self, execution_id, job_id, error):
        self._pin()
        return _executions().settle_unstarted_execution(execution_id, job_id, error)

    def recover_interrupted_executions(self):
        self._pin()
        return _executions().recover_interrupted_executions()

    def terminalize_dead_owner(self, execution_id, *, reason):
        self._pin()
        return _executions().terminalize_dead_owner(execution_id, reason=reason)

    def give_up_execution(self, execution_id, *, reason):
        self._pin()
        return _executions().give_up_execution(execution_id, reason=reason)

    def get_execution(self, execution_id):
        self._pin()
        return _executions().get_execution(execution_id)

    def list_executions(self, *, job_id=None, limit=50, before_claimed_at=None):
        self._pin()
        return _executions().list_executions(
            job_id=job_id, limit=limit, before_claimed_at=before_claimed_at)

    def latest_execution(self, job_id):
        self._pin()
        return _executions().latest_execution(job_id)

    def latest_executions(self, job_ids):
        self._pin()
        return _executions().latest_executions(job_ids)

    def live_executions(self, job_ids):
        self._pin()
        return _executions().live_executions(job_ids)

    def live_inflight_execution(self, job_id):
        self._pin()
        return _executions().live_inflight_execution(job_id)

    def cancellable_execution(self, job_id):
        self._pin()
        return _executions().cancellable_execution(job_id)


class LocalCronStorageProvider(CronStorageProvider):
    """Today's storage: one ``jobs.json`` and one ``executions.db`` per profile home, holding
    that agent's records only."""

    levels = frozenset({"agent"})

    @property
    def name(self) -> str:
        return "local"

    def open(self, home: Any = None) -> CronStore:
        """The store for *home*, or for the bound profile's home. The store's home is frozen
        here as one canonical absolute path; the pin is checked by every operation."""
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return LocalCronStore(home)

    def assert_isolating(self) -> None:
        """A local store is one file set per profile and refuses every pin but its own
        (``LocalCronStore._pin``); there is no shared role that could bypass that."""
        return
