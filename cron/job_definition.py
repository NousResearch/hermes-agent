"""Job-definition schema shared by importers of a foreign cron store.

Cron owns which persisted fields are *authored* (create_job) versus *advanced by the
scheduler* (next_run_at, state, run counters...). Callers merging an authored store
into a live one (profile distributions) import this rather than duplicating the list.
"""
from typing import Any, Dict

from cron.jobs import _apply_schedule_update
from cron.quota_hold import clear_state as _clear_quota_hold

# Persisted fields authored by create_job rather than advanced by the scheduler.
# Cron owns this schema so import/update callers never need to duplicate it.
JOB_DEFINITION_FIELDS = frozenset({
    "name", "prompt", "skills", "skill", "model", "provider", "base_url",
    "script", "no_agent", "monitor_script", "monitor_url", "context_from",
    "schedule", "schedule_display", "deliver", "origin", "enabled_toolsets",
    "workdir", "attach_to_session", "reasoning_effort", "failure_deliver",
})


def merge_job_definition(local: Dict[str, Any], authored: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh authored fields while preserving this store's scheduler-owned state."""
    merged = {
        key: value for key, value in local.items()
        if key not in JOB_DEFINITION_FIELDS and key != "repeat"
    }
    merged.update((key, authored[key]) for key in JOB_DEFINITION_FIELDS if key in authored)
    merged["repeat"] = {
        "completed": (local.get("repeat") or {}).get("completed", 0),
        "times": (authored.get("repeat") or {}).get("times"),
    }

    if local.get("schedule") != merged.get("schedule"):
        merged.pop("pending_slot", None)
        _clear_quota_hold(merged)
        if merged.get("enabled", True) and merged.get("state") != "paused":
            _apply_schedule_update(
                merged,
                {"schedule": merged["schedule"], "schedule_display": merged.get("schedule_display")},
                str(merged.get("id") or "imported job"),
            )
        else:
            merged["next_run_at"] = None
    return merged
