"""Job-level visibility for a recovered, acknowledged worker death."""


def record_unknown_worker_outcome(
    job: dict, *, error=None, adapters=None, loop=None
) -> bool:
    """Project only this fire's uncertain outcome, without retrying its side effects.

    A terminal ledger row alone does not retire the job claim or notify its owner.
    Fence the notice and bookkeeping together so recovery cannot overwrite a newer
    fire or duplicate a failure the worker already recorded.
    """
    from cron import scheduler
    from cron.unreachable_retry import is_retry_run

    execution = scheduler.get_execution(str(job["execution_id"]))
    if not execution or execution.get("status") != "unknown":
        return False
    claim = job.get("fire_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not owner:
        return False
    with scheduler.fire_claim_fence(job["id"], expected_owner=owner) as owns_claim:
        if not owns_claim:
            return True
        error = error or execution["error"]
        scheduler.logger.error("Job '%s': %s", job["id"], error)
        scope_tokens = scheduler._install_fire_secret_scope()
        delivery_error = None
        try:
            delivery_error, _ = scheduler._deliver_crash_failure(
                job, error, adapters=adapters, loop=loop
            )
        finally:
            scheduler._reset_fire_secret_scope(scope_tokens)
            scheduler.mark_job_run(
                job["id"],
                False,
                error,
                delivery_error=delivery_error,
                expected_fire_owner=owner,
                # A ladder re-run's occurrence already counted toward repeat.
                **({"ladder_rung": True} if is_retry_run(job) else {}),
            )
    return True


def external_worker_exited_reason(returncode: int | None, *, adopted: bool | None) -> str:
    """Cause for an attempt whose restart-safe worker this process watched exit.

    The generic dead-owner sweep can only say the owner vanished, so it names a
    scheduler restart -- but this waiter never lost its scheduler: it handed the
    attempt to the external worker and then saw that worker exit with this status.
    The run is still ``unknown`` rather than ``failed`` (whether side effects ran is
    still unknown); only the cause becomes truthful (#128509).

    ``adopted`` is REQUIRED and has no default: it is a factual claim about a
    phase the caller must have observed, and a default would let a caller
    silently assert a claim it cannot substantiate. Required keyword-only mirrors
    the seam #128509 set with ``terminalize_dead_owner(..., reason=...)`` and
    ``recover_interrupted_executions(reason=...)``.

    ``adopted`` is the phase read from the ledger row this waiter re-checked after
    the worker exited — never a guess. ``running`` means the worker won the
    claimed→running adoption gate, so "after adopting" is true of it. ``claimed``
    means it died BEFORE that gate (the observed field case: an import-time death,
    e.g. ModuleNotFoundError in the interpreter it was handed) and cannot be said
    to have adopted. Anything else (already terminalized by a concurrent sweep,
    row gone) has no observable phase; the wording then asserts no phase at all
    rather than defaulting to the post-adoption claim the evidence does not support.
    """
    if adopted:
        return (
            f"Restart-safe cron worker exited with status {returncode} after adopting this "
            "execution, without writing a durable terminal state; whether side effects ran "
            "is unknown."
        )
    if adopted is None:
        return (
            f"Restart-safe cron worker exited with status {returncode} without writing a "
            "durable terminal state and its adoption phase was not observable from this "
            "waiter; whether side effects ran is unknown."
        )
    return (
        f"Restart-safe cron worker exited with status {returncode} before adopting this "
        "execution: it never won the claimed→running adoption gate, which is the contract "
        "seam before any payload side effect may run."
    )
