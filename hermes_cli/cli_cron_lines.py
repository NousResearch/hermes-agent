"""The classic CLI's `/cron` overview and list as plain lines.

Shared by the classic CLI (`CLICommandsMixin._cron_overview` / `_cron_list`) and the shared gateway, which serves the
same lines to a TUI session (gateway/session_tui_reads.py).
"""
from __future__ import annotations

from hermes_cli.cli_commands_session_tools import _lines, _t


def _cron(**kwargs) -> dict:
    # Looked up on the mixin module at call time so tests that patch `cli_commands_mixin._cron_api` still apply.
    from hermes_cli import cli_commands_mixin
    return cli_commands_mixin._cron_api(**kwargs)


def cron_overview_lines() -> list[str]:
    """Bare ``/cron``: the command card, then the current jobs. Plain text (the shared gateway serves
    the same lines to a TUI session, gateway/session_tui_reads.py)."""
    out = ["", "+" + "-" * 68 + "+", "|" + _t("cron.overview_title").center(68) + "|",
           "+" + "-" * 68 + "+", "", f"  {_t('cron.overview_commands')}",
           *_lines(_t("cron.overview_usage"), pad="    "), ""]
    result = _cron(action="list")
    jobs = result.get("jobs", []) if result.get("success") else []
    if not jobs:
        return [*out, f"  {_t('cron.no_jobs_hint')}", ""]
    from hermes_cli.cron import _next_run_row
    out += [f"  {_t('cron.current_jobs')}", "  " + "-" * 63]
    for job in jobs:
        out.append(f"    {job['job_id'][:12]:<12} | {job['schedule']:<15} | {job.get('repeat', '?'):<8}")
        if job.get("skills"):
            out.append(f"      {_t('cron.skills', skills=', '.join(job['skills']))}")
        out.append(f"      {job.get('prompt_preview', '')}")
        if job.get("next_run_at"):
            # A stamp parked past the scheduler grace must not read as upcoming (#114309).
            label, value = _next_run_row(job)
            out.append(f"      {_t('cron.next_short') if label == 'Next run' else label}: {value}")
        out.append("")
    return [*out, ""]


def cron_list_lines(*, include_disabled: bool = False) -> list[str]:
    """``/cron list [--all]``."""
    result = _cron(action="list", include_disabled=include_disabled)
    jobs = result.get("jobs", []) if result.get("success") else []
    if not jobs:
        return [_t("cron.no_jobs")]
    from hermes_cli.cron import _next_run_row
    out = ["", _t("cron.list_header"), "-" * 80]
    for job in jobs:
        out += [f"  {_t('cron.id', job_id=job['job_id'])}", f"  {_t('cron.name', name=job['name'])}",
                f"  {_t('cron.state', state=job.get('state', '?'))}",
                f"  {_t('cron.schedule_repeat', schedule=job['schedule'], repeat=job.get('repeat', '?'))}",
                "  {}: {}".format(*_next_run_row(job)) if job.get("next_run_at") else f"  {_t('cron.next_run_na')}"]
        if job.get("skills"):
            out.append(f"  {_t('cron.skills', skills=', '.join(job['skills']))}")
        out.append(f"  {_t('cron.prompt', prompt=job.get('prompt_preview', ''))}")
        if job.get("last_run_at"):
            status = job.get("last_status") or "?"
            # delivery_failed: the run succeeded but delivery didn't — the reason lives
            # in last_delivery_error (last_error is None).
            if status == "delivery_failed" and job.get("last_delivery_error"):
                status = _t("cron.status_delivery_failed", error=job["last_delivery_error"])
            elif status == "error" and job.get("last_error"):
                status = _t("cron.status_error", error=job["last_error"])
            out.append(f"  {_t('cron.last_run', when=job['last_run_at'], status=status)}")
        out.append("")
    return out
