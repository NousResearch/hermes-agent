"""Ownership reconciliation and job-level visibility for external worker death."""

import os


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


def external_worker_exited_reason(returncode: int | None) -> str:
    """Cause for an attempt whose restart-safe worker this process watched exit.

    The generic dead-owner sweep can only say the owner vanished, so it names a
    scheduler restart — but this waiter never lost its scheduler: it handed the
    attempt to the external worker and then saw that worker exit with this status.
    The run is still ``unknown`` rather than ``failed`` (whether side effects ran is
    still unknown); only the cause becomes truthful (#128509).
    """
    return (
        f"Restart-safe cron worker exited with status {returncode} after adopting this "
        "execution, without writing a durable terminal state; whether side effects ran "
        "is unknown."
    )


def reconcile_external_worker_exit(
    process, *, execution_id: str, job_id: str, handoff_files: tuple,
    returncode: int, dispatch_mode: str, worker_env: dict,
) -> bool:
    """Reconcile the exact attempt after a pre-ack exit observation.

    Adoption and completion can land between the acknowledgement check and the
    exit poll. Durable ownership takes precedence over the missing acknowledgement.
    """
    from cron import scheduler
    from cron.scheduler_diagnostics import external_worker_stderr_tail
    from tools.process_registry import scoped_spawn_lost_user_bus

    current = scheduler.get_execution(execution_id)
    if current and current.get("status") in ("running", "completed", "failed", "unknown"):
        return scheduler._wait_for_external_cron_worker(
            process, execution_id=execution_id, job_id=job_id,
            handoff_files=handoff_files, stderr_path=handoff_files[-1],
        )

    with scheduler._running_lock:
        scheduler._restart_safe_waiter_job_ids.discard(scheduler._inflight_key(job_id))
    stderr_tail = external_worker_stderr_tail(handoff_files[-1])
    for path in handoff_files:
        path.unlink(missing_ok=True)
    if dispatch_mode == "scoped" and scoped_spawn_lost_user_bus(worker_env):
        # systemd-run failed before any worker ran; name the missing bus.
        raise RuntimeError(
            "restart-safe systemd scope could not be created: the user D-Bus session at "
            f"/run/user/{os.getuid()}/bus disappeared after the gateway started. On a "  # windows-footgun: ok — scoped dispatch exists only on Linux
            "system-level service install, run `sudo loginctl enable-linger <gateway-user>`; "
            "the next fire dispatches without scope isolation."
        )
    raise RuntimeError(
        f"cron external worker exited before ownership acknowledgement "
        f"(exit {returncode}){stderr_tail}"
    )
