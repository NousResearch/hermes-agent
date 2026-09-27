"""HERMES_HOME cron-store checks for hermes doctor: silently undelivered results.

A successful run whose delivery failed is recorded (`last_status:
"delivery_failed"` + `last_delivery_error`) but only surfaces through the
pull commands (`cron list`, `cron doctor`). This check pushes it into the one
generic health command an operator already runs.
"""

from __future__ import annotations

from hermes_cli.doctor_report import Finding, check_info, check_warn, doctor_check


@doctor_check()
def _check_cron_delivery(should_fix: bool, f: Finding) -> None:
    """Warn once per job whose last successful run never reached the operator."""
    from cron.jobs import load_jobs

    jobs = load_jobs()
    if not jobs:
        check_info("no cron jobs configured")
        return
    failed = [
        (
            job.get("name") or job.get("id") or "(unnamed)",
            job.get("last_delivery_error"),
        )
        for job in jobs
        if str(job.get("last_status") or "").strip().lower() == "delivery_failed"
    ]
    if not failed:
        check_info(f"last results delivered for all {len(jobs)} job(s)")
        return
    for name, error in failed:
        reason = str(error or "").strip()
        check_warn(
            f"cron job '{name}' ran successfully but its result was not delivered",
            f"({reason[:100] + '…' if len(reason) > 100 else reason or 'no details'})",
        )
    f.manual_issues.append(
        f"{len(failed)} cron job(s) finished but their results were never delivered — "
        f"run 'hermes cron doctor' for details and fix hints"
    )
