"""Job-definition schema shared by importers of a foreign cron store.

Cron owns which persisted fields are *authored* (create_job) versus *advanced by the
scheduler* (next_run_at, state, run counters...). Callers merging an authored store
into a live one (profile distributions) import this rather than duplicating the list.
"""
from typing import Any, Dict

from cron.jobs import _apply_schedule_update, _jobs_lock, is_job_runnable, load_jobs, parse_schedule, save_jobs
from cron.quota_hold import clear_state as _clear_quota_hold
from hermes_time import now as _hermes_now

JOB_DEFINITION_FIELDS = frozenset({
    "name", "prompt", "skills", "skill", "model", "provider", "base_url",
    "script", "no_agent", "monitor_script", "monitor_url", "context_from",
    "schedule", "schedule_display", "deliver", "origin", "enabled_toolsets",
    "workdir", "attach_to_session", "reasoning_effort", "failure_deliver",
})


def merge_job_definition(local: Dict[str, Any], authored: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh authored fields while preserving this store's scheduler-owned state.

    Raises ValueError when the authored schedule cannot be scheduled (unparseable string,
    past one-shot for a live job)."""
    merged = {key: value for key, value in local.items() if key not in JOB_DEFINITION_FIELDS}
    merged.update((key, authored[key]) for key in JOB_DEFINITION_FIELDS if key in authored)
    merged["repeat"] = {
        "completed": (local.get("repeat") or {}).get("completed", 0),
        "times": (authored.get("repeat") or {}).get("times"),
    }
    if isinstance(merged.get("schedule"), str):
        merged["schedule"] = parse_schedule(merged["schedule"])

    if local.get("schedule") != merged.get("schedule"):
        merged.pop("pending_slot", None)
        _clear_quota_hold(merged)
        if is_job_runnable(merged):
            updates = {"schedule": merged["schedule"]}
            if "schedule_display" in authored:
                updates["schedule_display"] = authored["schedule_display"]
            _apply_schedule_update(merged, updates, str(merged.get("id") or "imported job"))
        else:
            merged["next_run_at"] = None
    return merged


def import_job_definitions(shipped: Dict[str, Dict[str, Any]], *, paused_reason: str) -> None:
    """Merge *shipped* (job id -> authored record) into the active store under its lock.

    Unknown ids arrive with the marker set ``create_job(paused=True)`` writes; known ids keep
    their scheduler state. Nothing is written when a record cannot be merged: the ValueError
    is re-raised naming the job."""
    now = _hermes_now().isoformat()
    seed = {
        "enabled": False,
        "state": "paused",
        "paused_at": now,
        "paused_reason": paused_reason,
        "created_at": now,
        "next_run_at": None,
    }
    pending = dict(shipped)
    with _jobs_lock():
        merged = []
        for local in load_jobs():
            incoming = pending.pop(local.get("id"), None)
            merged.append(local if incoming is None else _merge_or_name(local, incoming))
        merged.extend(_merge_or_name({"id": job_id, **seed}, incoming) for job_id, incoming in pending.items())
        save_jobs(merged)


def _merge_or_name(local: Dict[str, Any], authored: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return merge_job_definition(local, authored)
    except ValueError as exc:
        raise ValueError(f"cron job {authored.get('name') or local.get('id')!r}: {exc}") from exc
