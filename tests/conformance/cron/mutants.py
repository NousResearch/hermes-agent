"""Broken local cron backends the conformance suite must reject (``test_mutants.py``).

Each passed an earlier version of the suite. ``double_fire`` drops the live-claim check, so two
claimants both win the same occurrence; ``cross_agent_tick`` drops the pin from the tick's path
(scan and fire claim), so a tick pinned to one agent scans and fires another agent's store;
``claim_ignores_pin`` drops it from the fire claim alone; ``unpinned_reads`` drops it from every
read; ``foreign_tenant_blind`` checks the bound agent but not the tenant; ``creator_not_recorded``
leaves a job without its creating principal and ``creator_repointable`` lets an update change
it; ``terminal_execution_rewritable`` rewrites a terminal audit row. Each mutant is applied by
the ``backend`` fixture when ``HERMES_CRON_MUTANT`` names it.
"""


def _double_fire(monkeypatch):
    """A claim older than the lease never blocks the next claimant: both fire the occurrence."""
    from cron import jobs
    monkeypatch.setattr(jobs, "_claim_is_live", lambda claim, now, ttl_seconds: False)


def _cross_agent_tick(monkeypatch):
    """The tick's path (due scan and fire claim) skips the pin: a tick pinned to one agent scans
    and fires another agent's store."""
    from cron_store_port_local import LocalCronStore

    def get_due_jobs(self):
        from cron import jobs
        return jobs.get_due_jobs()

    def claim_job_for_fire(self, job_id, *, claim_ttl_seconds=None, force=False, manual=False,
                           return_job=False):
        from cron import jobs
        claim = {"claim_ttl_seconds": claim_ttl_seconds} if claim_ttl_seconds is not None else {}
        return jobs.claim_job_for_fire(job_id, force=force, manual=manual,
                                        return_job=return_job, **claim)

    monkeypatch.setattr(LocalCronStore, "get_due_jobs", get_due_jobs)
    monkeypatch.setattr(LocalCronStore, "claim_job_for_fire", claim_job_for_fire)


def _claim_ignores_pin(monkeypatch):
    """Only the fire claim skips the pin: one agent can fire another agent's job by id."""
    from cron_store_port_local import LocalCronStore

    def claim_job_for_fire(self, job_id, *, claim_ttl_seconds=None, force=False, manual=False,
                           return_job=False):
        from cron import jobs
        claim = {"claim_ttl_seconds": claim_ttl_seconds} if claim_ttl_seconds is not None else {}
        return jobs.claim_job_for_fire(job_id, force=force, manual=manual,
                                        return_job=return_job, **claim)

    monkeypatch.setattr(LocalCronStore, "claim_job_for_fire", claim_job_for_fire)


def _unpinned_reads(monkeypatch):
    """Every read skips the pin check entirely."""
    from cron import executions as audit
    from cron import jobs
    from cron_store_port_local import LocalCronStore

    reads = {
        "get_job": lambda self, job_id: jobs.get_job(job_id),
        "resolve_job_ref": lambda self, ref: jobs.resolve_job_ref(ref),
        "list_jobs": lambda self, include_disabled=False: jobs.list_jobs(include_disabled=include_disabled),
        "get_execution": lambda self, execution_id: audit.get_execution(execution_id),
        "list_executions": lambda self, **kw: audit.list_executions(**kw),
        "latest_execution": lambda self, job_id: audit.latest_execution(job_id),
        "latest_executions": lambda self, job_ids: audit.latest_executions(job_ids),
        "live_executions": lambda self, job_ids: audit.live_executions(job_ids),
        "live_inflight_execution": lambda self, job_id: audit.live_inflight_execution(job_id),
        "cancellable_execution": lambda self, job_id: audit.cancellable_execution(job_id),
        "execution_owner_live": lambda self, execution_id, job_id: audit.execution_owner_live(
            execution_id, job_id),
    }
    for name, method in reads.items():
        monkeypatch.setattr(LocalCronStore, name, method)


def _agent_pin(self):
    """The pin's agent half alone: another tenant naming the store's home passes."""
    from cron_store_port import CronPinRefused
    from hermes_constants import get_hermes_home, hermes_home_key
    from hermes_scope import current_scope
    scope = current_scope()
    agent = scope.agent if scope is not None else get_hermes_home()
    if hermes_home_key(agent) != self._home_key:
        raise CronPinRefused("the bound agent is not this cron store's profile")


def _foreign_tenant_blind(monkeypatch):
    """Every operation checks the bound agent but not the tenant: another tenant naming the
    home reads and writes the store."""
    from cron_store_port_local import LocalCronStore
    monkeypatch.setattr(LocalCronStore, "_pin", _agent_pin)


def _creator_not_recorded(monkeypatch):
    """A created job carries no creating principal."""
    from cron_store_port_local import LocalCronStore

    def create_job(self, prompt, schedule, **fields):
        self._pin()
        from cron import jobs
        return jobs.create_job(prompt, schedule, **fields)
    monkeypatch.setattr(LocalCronStore, "create_job", create_job)


def _creator_repointable(monkeypatch):
    """An update may change the creating principal."""
    from cron_store_port_local import LocalCronStore

    def update_job(self, job_id, updates):
        self._pin()
        bad_fields = {"id"}.intersection(updates or {})
        if bad_fields:
            raise ValueError(
                f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}")
        from cron import jobs
        return jobs.update_job(job_id, updates)
    monkeypatch.setattr(LocalCronStore, "update_job", update_job)


def _terminal_execution_rewritable(monkeypatch):
    """A finished attempt can be finished again: the terminal-status guard is missing."""
    from cron import executions as audit

    original = audit.finish_execution

    def finish_execution(execution_id, *, success, error=None, delivery_outcome=None,
                         cancelled=False):
        record = original(execution_id, success=success, error=error,
                          delivery_outcome=delivery_outcome, cancelled=cancelled)
        if record is None:  # the guard would have refused: rewrite the terminal row in place
            now = audit._hermes_now().isoformat()
            status = "cancelled" if cancelled else ("completed" if success else "failed")
            with audit._transaction() as conn:
                conn.execute(
                    "UPDATE executions SET status=?, finished_at=?, error=?, handoff_pending=0 "
                    "WHERE id=?",
                    (status, now, None if success else (str(error) or "unknown failure"),
                     str(execution_id)))
                record = audit._fetch(conn, str(execution_id))
        return record

    monkeypatch.setattr(audit, "finish_execution", finish_execution)


MUTANTS = {
    "double_fire": _double_fire,
    "cross_agent_tick": _cross_agent_tick,
    "claim_ignores_pin": _claim_ignores_pin,
    "unpinned_reads": _unpinned_reads,
    "foreign_tenant_blind": _foreign_tenant_blind,
    "creator_not_recorded": _creator_not_recorded,
    "creator_repointable": _creator_repointable,
    "terminal_execution_rewritable": _terminal_execution_rewritable,
}


def apply_mutant(monkeypatch):
    import os
    name = os.environ.get("HERMES_CRON_MUTANT")
    if name:
        MUTANTS[name](monkeypatch)
