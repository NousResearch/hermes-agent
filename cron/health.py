"""Structured cron diagnostics without executing jobs or delivering messages."""

from __future__ import annotations

from typing import Any

from agent.redact import redact_sensitive_text


def _diagnostic_text(message: str) -> str:
    # Diagnostic URLs do not need credentials or query strings. The general
    # tool redactor deliberately preserves login/share URLs, so tighten this boundary.
    return redact_sensitive_text(message, force=True, redact_url_credentials=True)


def inspect_job(
    job: dict[str, Any], *, config: dict[str, Any], check_provider: bool = False,
) -> dict[str, Any]:
    """Inspect a job in the active profile; credential refresh is opt-in."""
    from cron.scheduler_delivery import _normalize_deliver_value, _resolve_delivery_targets
    from hermes_cli.config import resolve_cron_model_drift_defaults
    from hermes_cli.cron import _cron_doctor_issues_for_job

    issues = [
        {"code": "job_health", "message": _diagnostic_text(message)}
        for message in _cron_doctor_issues_for_job(job)
    ]

    def issue(code: str, message: str) -> None:
        issues.append({"code": code, "message": _diagnostic_text(message)})

    global_provider, global_model = resolve_cron_model_drift_defaults(config)
    fleet = config.get("cron") or {}
    if not isinstance(fleet, dict):
        fleet = {}
    provider = job.get("provider") or fleet.get("model_provider") or global_provider or "auto"
    model = job.get("model") or fleet.get("model") or global_model
    provider_check = "not_needed" if job.get("no_agent") else "not_checked"
    if job.get("no_agent"):
        model = None
    elif not model:
        issue("model_unresolved", "No model configured; set a per-job model or run hermes model.")

    if check_provider and not job.get("no_agent"):
        from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error

        try:
            kwargs = {"requested": job.get("provider") or fleet.get("model_provider") or None,
                      "target_model": model}
            if job.get("base_url"):
                kwargs["explicit_base_url"] = job["base_url"]
            runtime = resolve_runtime_provider(**kwargs)
            provider = runtime["provider"]
            provider_check = "credentials_resolved"
        except Exception as exc:
            provider_check = "failed"
            issue("provider_unavailable", format_runtime_provider_error(exc))

    targets = []
    deliveries = _normalize_deliver_value(job.get("deliver", "local"))
    for delivery in (part.strip() for part in deliveries.split(",") if part.strip()):
        if delivery == "local":
            continue
        try:
            resolved = _resolve_delivery_targets({**job, "deliver": delivery})
            if not resolved:
                issue("missing_delivery_target", f"No destination resolves for {delivery!r}.")
        except Exception as exc:
            issue("invalid_delivery_target", str(exc))
    try:
        targets = _resolve_delivery_targets(job)
    except Exception as exc:
        issue("invalid_delivery_target", str(exc))

    return {
        "id": job.get("id"), "name": job.get("name"),
        "provider": provider, "model": model, "provider_check": provider_check,
        "delivery": targets, "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"), "last_status": job.get("last_status"),
        "last_success_at": job.get("last_success_at") or (
            job.get("last_run_at") if job.get("last_status") in {"ok", "delivery_failed", "delivery_queued"} else None
        ),
        "issues": issues,
    }


def inspect_jobs(*, check_provider: bool = False) -> list[dict[str, Any]]:
    """Inspect active-profile jobs using its raw config, as the scheduler does."""
    from cron.jobs import list_jobs
    from hermes_cli.config import _expand_env_vars, read_user_config_raw

    jobs = list_jobs(include_disabled=False)
    if not jobs:
        return []
    try:
        config = _expand_env_vars(read_user_config_raw())
    except Exception as exc:
        return [{"id": job["id"], "name": job.get("name"), "issues": [
            {"code": "configuration_error", "message": _diagnostic_text(str(exc))},
        ]} for job in jobs]
    return [inspect_job(job, config=config, check_provider=check_provider) for job in jobs]
