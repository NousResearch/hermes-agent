from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from hermes_cli.bot_catalog import BotRoutine, resolve_bot_catalog_entry
from hermes_cli.bot_routines import activate_bot_routine, list_bot_routines, pause_bot_routine


def _entry():
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)
    return entry.model_copy(update={
        "routines": [BotRoutine(
            id="daily-brief",
            name="Daily brief",
            prompt="Prepare the reviewed daily brief.",
            schedule="1h",
        )]
    })


def _home(tmp_path: Path, monkeypatch, *, timezone: str = "UTC") -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(f"timezone: {timezone}\n", encoding="utf-8")
    (home / "profile.yaml").write_text(yaml.safe_dump({"routines": []}), encoding="utf-8")
    return home


def test_activation_is_gated_and_requires_explicit_runtime_fields(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    entry = _entry()

    with pytest.raises(PermissionError, match="finish bot setup"):
        activate_bot_routine(
            entry, "daily-brief", schedule="1h", timezone_name="UTC",
            destination="local", setup_ready=False,
        )
    with pytest.raises(ValueError, match="schedule, timezone, and destination"):
        activate_bot_routine(
            entry, "daily-brief", schedule="", timezone_name="UTC",
            destination="local", setup_ready=True,
        )
    assert list_bot_routines(entry)[0]["job_id"] is None


def test_activation_holds_cron_store_lock_across_lookup_and_create(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    entry = _entry()
    from cron import jobs

    real_create = jobs.create_job
    observed_depth = 0

    def create_while_observing_lock(**kwargs):
        nonlocal observed_depth
        observed_depth = getattr(jobs._jobs_lock_state, "depth", 0)
        return real_create(**kwargs)

    monkeypatch.setattr(jobs, "create_job", create_while_observing_lock)
    activate_bot_routine(
        entry, "daily-brief", schedule="1h", timezone_name="UTC",
        destination="local", setup_ready=True,
    )

    assert observed_depth > 0
    assert len(jobs.list_jobs(include_disabled=True)) == 1


def test_pause_holds_cron_store_lock_through_metadata_receipt(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    entry = _entry()
    from cron import jobs
    from hermes_cli import bot_metadata

    activated = activate_bot_routine(
        entry, "daily-brief", schedule="1h", timezone_name="UTC",
        destination="local", setup_ready=True,
    )
    real_pause = jobs.pause_job
    real_mutate = bot_metadata.mutate_bot_metadata
    observed = []

    def pause_while_observing_lock(*args, **kwargs):
        observed.append(("pause", getattr(jobs._jobs_lock_state, "depth", 0)))
        return real_pause(*args, **kwargs)

    def mutate_while_observing_lock(mutator):
        observed.append(("metadata", getattr(jobs._jobs_lock_state, "depth", 0)))
        return real_mutate(mutator)

    monkeypatch.setattr(jobs, "pause_job", pause_while_observing_lock)
    monkeypatch.setattr(bot_metadata, "mutate_bot_metadata", mutate_while_observing_lock)

    paused = pause_bot_routine(entry, "daily-brief")

    assert paused["job_id"] == activated["job_id"]
    assert observed == [("pause", 1), ("metadata", 1)]


def test_explicit_timezone_schedules_only_the_target_routine_and_resume_is_idempotent(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch, timezone="America/New_York")
    entry = _entry()
    from cron import jobs

    result = activate_bot_routine(
        entry, "daily-brief", schedule="0 9 * * *", timezone_name="Etc/UTC",
        destination="local", setup_ready=True,
    )
    [job] = jobs.list_jobs(include_disabled=True)
    next_run = datetime.fromisoformat(job["next_run_at"])
    assert job["schedule"]["timezone"] == "Etc/UTC"
    assert (next_run.hour, next_run.utcoffset()) == (9, timedelta(0))

    pause_bot_routine(entry, "daily-brief")
    resumed = activate_bot_routine(
        entry, "daily-brief", schedule="0 9 * * *", timezone_name="Etc/UTC",
        destination="local", setup_ready=True,
    )
    assert resumed["job_id"] == result["job_id"]
    assert len(jobs.list_jobs(include_disabled=True)) == 1
    assert jobs.list_jobs(include_disabled=True)[0]["enabled"] is True


def test_activated_routine_keeps_cron_baseline_and_exposes_real_workflow_tools(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    (home / "config.yaml").write_text(
        "timezone: UTC\nplatform_toolsets:\n  cron: [hermes-cron]\n",
        encoding="utf-8",
    )
    entry = _entry()
    from cron import jobs
    from cron.scheduler import _resolve_cron_enabled_toolsets
    from hermes_cli.config_effective import load_user_config_effective
    from model_tools import get_tool_definitions

    activate_bot_routine(
        entry, "daily-brief", schedule="1h", timezone_name="UTC",
        destination="local", setup_ready=True,
    )
    [job] = jobs.list_jobs(include_disabled=True)
    cfg = load_user_config_effective(home / "config.yaml")
    resolved = _resolve_cron_enabled_toolsets(job, cfg)
    definitions = get_tool_definitions(
        enabled_toolsets=resolved,
        disabled_toolsets=["cronjob", "messaging", "clarify"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    names = {
        str(item.get("function", {}).get("name") or "")
        for item in definitions
        if isinstance(item, dict)
    }
    assert {"web_search", "skill_view", "read_file"} <= names


def test_activation_retry_after_schedule_edit_preserves_pin_without_duplicate(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    entry = _entry()
    from cron import jobs

    original = activate_bot_routine(
        entry, "daily-brief", schedule="0 9 * * *", timezone_name="America/Chicago",
        destination="local", setup_ready=True,
    )
    updated = jobs.update_job(original["job_id"], {"schedule": "0 10 * * *"})
    assert updated is not None
    assert updated["schedule"]["timezone"] == "America/Chicago"

    with pytest.raises(ValueError, match="different schedule"):
        activate_bot_routine(
            entry, "daily-brief", schedule="0 9 * * *", timezone_name="America/Chicago",
            destination="local", setup_ready=True,
        )
    retried = activate_bot_routine(
        entry, "daily-brief", schedule="0 10 * * *", timezone_name="America/Chicago",
        destination="local", setup_ready=True,
    )
    assert retried["job_id"] == original["job_id"]
    assert len(jobs.list_jobs(include_disabled=True)) == 1


def test_activation_retry_repairs_receipt_without_duplicate_job(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    entry = _entry()
    from hermes_cli import bot_metadata

    real_mutate = bot_metadata.mutate_bot_metadata
    attempts = 0

    def fail_once(mutator):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("cancelled after cron commit")
        return real_mutate(mutator)

    monkeypatch.setattr(bot_metadata, "mutate_bot_metadata", fail_once)
    with pytest.raises(RuntimeError, match="cancelled"):
        activate_bot_routine(
            entry, "daily-brief", schedule="1h", timezone_name="UTC",
            destination="local", setup_ready=True,
        )

    result = activate_bot_routine(
        entry, "daily-brief", schedule="1h", timezone_name="UTC",
        destination="local", setup_ready=True,
    )
    routines = list_bot_routines(entry)
    assert result["state"] == "active"
    assert [item["job_id"] for item in routines] == [result["job_id"]]

    paused = pause_bot_routine(entry, "daily-brief")
    assert paused == {"id": "daily-brief", "state": "paused", "job_id": result["job_id"]}
    assert list_bot_routines(entry)[0]["state"] == "paused"
