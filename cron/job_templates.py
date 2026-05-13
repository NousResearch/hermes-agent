"""Sync cron job templates from the active Hermes profile."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

from cron.jobs import (
    compute_next_run,
    create_job,
    load_jobs,
    parse_duration,
    parse_schedule,
    save_jobs,
    _normalize_skill_list,
    _normalize_workdir,
)


TEMPLATES_DIR = "cron/templates"

def _templates_path() -> Path:
    return get_hermes_home() / TEMPLATES_DIR


def _template_version(template: Dict[str, Any]) -> Optional[str]:
    version = template.get("template_version", template.get("version"))
    if version is None:
        return None
    text = str(version).strip()
    return text or None


def _template_key(template: Dict[str, Any], path: Path) -> str:
    key = str(template.get("template_key") or path.stem).strip()
    if not key:
        raise ValueError(f"Cron template has no template_key: {path}")
    return key


def _read_template(path: Path) -> Optional[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, dict):
        raise ValueError(f"Cron template must be a JSON object: {path}")
    if template.get("active", True) is False:
        return None
    return template


def _validate_template(template: Dict[str, Any], path: Path) -> None:
    if "schedule" not in template:
        raise ValueError(f"Cron template is missing schedule: {path}")
    if bool(template.get("no_agent", False)) and not str(template.get("script") or "").strip():
        raise ValueError(f"no_agent=True requires a script in cron template: {path}")


def _load_templates() -> List[Dict[str, Any]]:
    templates_dir = _templates_path()
    if not templates_dir.exists():
        return []

    seen_keys: Dict[str, Path] = {}
    templates: List[Dict[str, Any]] = []
    for path in sorted(templates_dir.glob("*.json")):
        template = _read_template(path)
        if template is None:
            continue
        template = dict(template)
        template["template_key"] = _template_key(template, path)
        if template["template_key"] in seen_keys:
            raise ValueError(
                "Duplicate cron template_key "
                f"{template['template_key']!r}: {seen_keys[template['template_key']]} and {path}"
            )
        seen_keys[template["template_key"]] = path
        _validate_template(template, path)
        template["template_version"] = _template_version(template)
        templates.append(template)
    return templates


def _create_kwargs(template: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": template.get("prompt", ""),
        "schedule": template["schedule"],
        "name": template.get("name"),
        "deliver": template.get("deliver"),
        "skill": template.get("skill"),
        "skills": template.get("skills"),
        "model": template.get("model"),
        "provider": template.get("provider"),
        "base_url": template.get("base_url"),
        "script": template.get("script"),
        "context_from": template.get("context_from"),
        "enabled_toolsets": template.get("enabled_toolsets"),
        "workdir": _expand_template_workdir(template.get("workdir")),
        "no_agent": bool(template.get("no_agent", False)),
        "delivery_mode": template.get("delivery_mode"),
        "thread_title_template": template.get("thread_title_template"),
        "template_key": template["template_key"],
        "template_version": template.get("template_version"),
    }


def _optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _expand_template_workdir(value: Any) -> Any:
    """Expand profile-local placeholders in distribution-owned cron templates."""
    if value is None:
        return None
    text = str(value)
    hermes_home = str(get_hermes_home())
    return (
        text.replace("${HERMES_HOME}", hermes_home)
        .replace("$HERMES_HOME", hermes_home)
        .replace("{HERMES_HOME}", hermes_home)
        .replace("{hermes_home}", hermes_home)
    )


def _normalize_context_from(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()] or None
    return None


def _is_relative_oneshot_schedule(schedule: str) -> bool:
    try:
        parse_duration(schedule)
    except ValueError:
        return False
    return True


def _legacy_relative_oneshot_schedule(existing: Dict[str, Any]) -> Optional[str]:
    display = str(existing.get("schedule_display") or "").strip()
    prefix = "once in "
    if display.lower().startswith(prefix):
        return display[len(prefix):].strip()
    return None


def _schedule_changed(existing: Dict[str, Any], schedule: str) -> bool:
    template_schedule = existing.get("template_schedule")
    if template_schedule is not None:
        return template_schedule != schedule
    if existing.get("schedule_display") == schedule:
        return False
    if existing.get("schedule", {}).get("kind") == "once" and _is_relative_oneshot_schedule(schedule):
        legacy_schedule = _legacy_relative_oneshot_schedule(existing)
        if legacy_schedule is not None:
            return legacy_schedule != schedule
        return False
    return parse_schedule(schedule) != existing.get("schedule")


def _apply_template_update(job: Dict[str, Any], template: Dict[str, Any]) -> None:
    raw_schedule = str(template["schedule"])
    schedule_changed = _schedule_changed(job, raw_schedule)

    normalized_skills = _normalize_skill_list(template.get("skill"), template.get("skills"))
    job.update(
        {
            "prompt": str(template.get("prompt") or ""),
            "name": template.get("name") or (str(template.get("prompt") or "")[:50].strip() or "cron job"),
            "skills": normalized_skills,
            "skill": normalized_skills[0] if normalized_skills else None,
            "model": _optional_text(template.get("model")),
            "provider": _optional_text(template.get("provider")),
            "base_url": _optional_text(template.get("base_url"), strip_trailing_slash=True),
            "script": _optional_text(template.get("script")),
            "no_agent": bool(template.get("no_agent", False)),
            "context_from": _normalize_context_from(template.get("context_from")),
            "deliver": template.get("deliver", "local"),
            "enabled_toolsets": [
                str(item).strip()
                for item in template.get("enabled_toolsets", []) or []
                if str(item).strip()
            ]
            or None,
            "workdir": _normalize_workdir(_expand_template_workdir(template.get("workdir"))),
            "delivery_mode": _optional_text(template.get("delivery_mode")),
            "thread_title_template": _optional_text(template.get("thread_title_template")),
            "template_key": template["template_key"],
            "template_version": template.get("template_version"),
            "template_schedule": raw_schedule,
        }
    )

    if schedule_changed:
        parsed_schedule = parse_schedule(raw_schedule)
        job["schedule"] = parsed_schedule
        job["schedule_display"] = parsed_schedule.get("display", raw_schedule)
        if job.get("state") != "paused":
            job["next_run_at"] = compute_next_run(parsed_schedule)


def _remember_template_schedule(job_id: str, raw_schedule: str) -> None:
    jobs = load_jobs()
    for job in jobs:
        if job.get("id") == job_id:
            job["template_schedule"] = raw_schedule
            break
    save_jobs(jobs)


def sync_cron_templates() -> Dict[str, Any]:
    """Upsert active cron templates into cron/jobs.json."""
    templates = _load_templates()
    created: List[str] = []
    updated: List[str] = []

    jobs_by_template_key = {
        job.get("template_key"): job
        for job in load_jobs()
        if job.get("template_key")
    }

    for template in templates:
        key = template["template_key"]
        existing = jobs_by_template_key.get(key)
        if existing is None:
            created_job = create_job(**_create_kwargs(template))
            _remember_template_schedule(created_job["id"], str(template["schedule"]))
            created.append(key)
            continue

        jobs = load_jobs()
        for job in jobs:
            if job.get("id") == existing["id"]:
                _apply_template_update(job, template)
                break
        save_jobs(jobs)
        updated.append(key)

    return {"created": created, "updated": updated}
