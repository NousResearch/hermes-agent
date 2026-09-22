"""Explicit, idempotent activation of reviewed Bot Marketplace routine blueprints."""

from __future__ import annotations

import threading
from typing import Any

from hermes_cli.bot_catalog import BotCatalogEntry, BotRoutine

_ROUTINE_LOCK = threading.RLock()


def _read_metadata() -> dict[str, Any]:
    from hermes_cli.bot_metadata import read_bot_metadata

    return read_bot_metadata()


def _routine(entry: BotCatalogEntry, routine_id: str) -> BotRoutine:
    routine = next((item for item in entry.routines if item.id == routine_id), None)
    if routine is None:
        raise ValueError(f"unknown routine id: {routine_id}")
    return routine


def _job_name(entry: BotCatalogEntry, routine: BotRoutine) -> str:
    return f"Bot routine · {entry.name} · {routine.id}"


def _routine_toolsets(entry: BotCatalogEntry) -> list[str]:
    """Merge reviewed extras into the canonical cron surface instead of replacing its workflow tools."""
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.tools_config import _get_platform_tools
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    cfg = load_user_config_effective(config_path) if config_path.is_file() else {}
    baseline = sorted(_get_platform_tools(cfg, "cron"))
    return list(dict.fromkeys([*baseline, *entry.capabilities.toolsets]))


def _matching_job(entry: BotCatalogEntry, routine: BotRoutine) -> dict[str, Any] | None:
    from cron.jobs import list_jobs

    name = _job_name(entry, routine)
    matches = [job for job in list_jobs(include_disabled=True) if job.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate cron jobs already exist for routine {routine.id!r}")
    return matches[0] if matches else None


def list_bot_routines(entry: BotCatalogEntry) -> list[dict[str, Any]]:
    metadata = _read_metadata()
    saved = {
        str(item.get("id")): item
        for item in metadata.get("routines", [])
        if isinstance(item, dict) and item.get("id")
    }
    result = []
    for routine in entry.routines:
        item = saved.get(routine.id, {})
        job = _matching_job(entry, routine)
        state = "active" if job and job.get("enabled", True) else "paused"
        result.append({
            "id": routine.id,
            "name": routine.name,
            "prompt": routine.prompt,
            "schedule": item.get("schedule") or routine.schedule,
            "state": state,
            "job_id": job.get("id") if job else item.get("job_id"),
            "timezone": item.get("timezone"),
            "destination": item.get("destination"),
        })
    return result


def activate_bot_routine(
    entry: BotCatalogEntry,
    routine_id: str,
    *,
    schedule: str,
    timezone_name: str,
    destination: str,
    setup_ready: bool,
) -> dict[str, Any]:
    """Create or resume one cron job only after an explicit, fully specified activation."""
    if not setup_ready:
        raise PermissionError("finish bot setup and its first task before activating routines")
    routine = _routine(entry, routine_id)
    schedule = schedule.strip()
    timezone_name = timezone_name.strip()
    destination = destination.strip()
    if not schedule or not timezone_name or not destination:
        raise ValueError("schedule, timezone, and destination are required")

    with _ROUTINE_LOCK:
        from cron.jobs import _jobs_lock, create_job, resume_job

        # Name lookup and create must share cron's cross-process store lock. create_job's nested
        # lock is re-entrant; without this outer section two backend processes can both observe
        # absence and commit duplicate routine jobs.
        with _jobs_lock():
            job = _matching_job(entry, routine)
            if job is not None:
                if str(job.get("prompt") or "") != routine.prompt:
                    raise RuntimeError("existing routine job does not match the reviewed blueprint")
                from cron.jobs import parse_schedule

                if (
                    job.get("schedule") != parse_schedule(schedule, timezone_name=timezone_name)
                    or str(job.get("deliver") or "") != destination
                ):
                    raise ValueError("routine is already activated with a different schedule, timezone, or destination")
                if not job.get("enabled", True):
                    job = resume_job(str(job["id"]))
                if job is None:
                    raise RuntimeError("existing routine job disappeared during activation")
            else:
                job = create_job(
                    prompt=routine.prompt,
                    schedule=schedule,
                    name=_job_name(entry, routine),
                    deliver=destination,
                    enabled_toolsets=_routine_toolsets(entry),
                    paused=False,
                    timezone_name=timezone_name,
                )

            from hermes_cli.bot_metadata import mutate_bot_metadata

            def record_activation(metadata: dict[str, Any]) -> None:
                records = [item for item in metadata.get("routines", []) if isinstance(item, dict)]
                record = next((item for item in records if item.get("id") == routine.id), None)
                if record is None:
                    record = {"id": routine.id, "name": routine.name, "prompt": routine.prompt}
                    records.append(record)
                record.update({
                    "schedule": schedule,
                    "state": "active",
                    "job_id": job["id"],
                    "timezone": timezone_name,
                    "destination": destination,
                })
                metadata["routines"] = records

            mutate_bot_metadata(record_activation)
        return {
            "id": routine.id,
            "state": "active",
            "job_id": job["id"],
            "schedule": schedule,
            "timezone": timezone_name,
            "destination": destination,
        }


def pause_bot_routine(entry: BotCatalogEntry, routine_id: str) -> dict[str, Any]:
    routine = _routine(entry, routine_id)
    with _ROUTINE_LOCK:
        from cron.jobs import _jobs_lock, pause_job

        # Keep the same lock order as activation: routine → cron store → metadata. The outer
        # cron lock is cross-process and its nested pause_job lock is re-entrant.
        with _jobs_lock():
            job = _matching_job(entry, routine)
            if job is None:
                return {"id": routine.id, "state": "paused", "job_id": None}
            paused = pause_job(str(job["id"]), reason="Paused from Bot Marketplace setup.")
            if paused is None:
                raise RuntimeError("routine job disappeared during pause")

            from hermes_cli.bot_metadata import mutate_bot_metadata

            def record_pause(metadata: dict[str, Any]) -> None:
                records = [item for item in metadata.get("routines", []) if isinstance(item, dict)]
                record = next((item for item in records if item.get("id") == routine.id), None)
                if record is None:
                    record = {
                        "id": routine.id,
                        "name": routine.name,
                        "prompt": routine.prompt,
                        "schedule": routine.schedule,
                        "timezone": None,
                        "destination": None,
                    }
                    records.append(record)
                record.update(state="paused", job_id=job["id"])
                metadata["routines"] = records

            mutate_bot_metadata(record_pause)
            return {"id": routine.id, "state": "paused", "job_id": job["id"]}
