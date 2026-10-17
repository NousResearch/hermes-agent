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


def external_worker_exited_reason(returncode: int | None) -> str:
    """Cause for an attempt whose restart-safe worker this process watched exit.

    The generic dead-owner sweep can only say the owner vanished, so it names a
    scheduler restart — but this waiter never lost its scheduler: it handed the
    attempt to the external worker and then saw that worker exit with this status.
    The run is still ``unknown`` rather than ``failed`` (whether side effects ran is
    still unknown); only the cause becomes truthful (#128509).

    Loki Q157 (Oct 16 2026): disambiguate negative returncodes. Per Python
    multiprocessing conventions, ``returncode < 0`` means the process was killed
    by signal ``abs(returncode)`` (POSIX). The misleading "Restart-safe cron
    worker" string was previously hiding the fact that the SIGTERM came from
    outside the gateway — empirically a WSL VM reboot cascading through
    systemd-logind, not a Hermes bug. Appending the signal kind + likely
    external-cause label so operator triage doesn't waste cycles looking inside
    the scheduler for a failure that was actually a host event.
    """
    base = (
        f"Restart-safe cron worker exited with status {returncode} after adopting this "
        "execution, without writing a durable terminal state; whether side effects ran "
        "is unknown."
    )
    if returncode is not None and returncode < 0:
        sig = -returncode
        sig_label = {
            15: "SIGTERM (likely host-initiated shutdown — WSL VM reboot / systemd-logind cascade)",
            9: "SIGKILL (likely OOM-killer or manual kill)",
            1: "SIGHUP",
            2: "SIGINT",
            13: "SIGPIPE (subprocess writing to a closed pipe — usually a parent gone)",
        }.get(sig, f"signal {sig}")
        return f"{base} Worker was terminated by {sig_label} from outside the gateway."
    return base
