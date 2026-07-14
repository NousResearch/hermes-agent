"""Regression tests for opt-in cron job audit logging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _enable_audit(hermes_home: Path, **cron_overrides) -> None:
    """Enable cron audit logging through config.yaml (the only supported interface)."""
    cron_cfg = {"audit_log": True}
    cron_cfg.update(cron_overrides)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({"cron": cron_cfg}))

    from cron.audit import reload_audit_config

    reload_audit_config()


@pytest.fixture
def cron_audit_env(tmp_path, monkeypatch):
    """Isolated cron environment with audit config cache reset."""
    hermes_home = tmp_path / ".hermes"
    cron_dir = hermes_home / "cron"
    output_dir = cron_dir / "output"
    output_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", output_dir)

    try:
        import cron.audit as audit_mod
    except ModuleNotFoundError:
        audit_mod = None
    if audit_mod is not None:
        audit_mod.reload_audit_config()

    return hermes_home


def _read_audit_entries(hermes_home: Path) -> list[dict]:
    audit_path = hermes_home / "cron" / "audit.log"
    assert audit_path.exists(), f"missing audit log: {audit_path}"
    return [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]


def test_cron_audit_disabled_by_default(cron_audit_env):
    from cron.jobs import create_job

    create_job(prompt="hello", schedule="every 1h", name="quiet job")

    assert not (cron_audit_env / "cron" / "audit.log").exists()


def test_cron_audit_records_lifecycle_once_when_enabled(cron_audit_env):
    _enable_audit(cron_audit_env)

    from cron.jobs import create_job, pause_job, remove_job, resume_job, trigger_job

    job = create_job(prompt="hello", schedule="every 1h", name="audit job")
    pause_job(job["id"], reason="maintenance")
    resume_job(job["id"])
    trigger_job(job["id"])
    remove_job(job["id"])

    entries = _read_audit_entries(cron_audit_env)
    assert [entry["action"] for entry in entries] == [
        "created",
        "paused",
        "resumed",
        "triggered",
        "removed",
    ]
    assert all(entry["job_id"] == job["id"] for entry in entries)
    assert all(entry["job_name"] == "audit job" for entry in entries)
    assert entries[1]["details"]["reason"] == "maintenance"


def test_cron_audit_redacts_prompt_updates(cron_audit_env):
    _enable_audit(cron_audit_env)

    from cron.jobs import create_job, update_job

    job = create_job(prompt="initial", schedule="every 1h", name="redaction job")
    update_job(job["id"], {"prompt": "secret body", "name": "renamed"})

    entries = _read_audit_entries(cron_audit_env)
    assert [entry["action"] for entry in entries] == ["created", "updated"]
    changes = entries[1]["details"]["changes"]
    assert changes["prompt"] == "<11 chars>"
    assert changes["name"] == "renamed"
    assert "secret body" not in (cron_audit_env / "cron" / "audit.log").read_text()


def test_cron_audit_records_run_completion_and_repeat_removal(cron_audit_env):
    _enable_audit(cron_audit_env)

    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="one shot", schedule="30m", name="repeat job", repeat=1)
    mark_job_run(job["id"], success=True)

    assert get_job(job["id"]) is None
    entries = _read_audit_entries(cron_audit_env)
    assert [entry["action"] for entry in entries] == ["created", "completed", "removed"]
    assert entries[-1]["details"]["reason"] == "repeat_limit"


def test_cron_audit_config_is_scoped_per_profile(tmp_path, monkeypatch):
    """A profile with audit enabled must not leak its config into another profile.

    Regression for the process-global config cache: dashboard cron requests
    scope HERMES_HOME per profile, so caching without a per-home key let the
    first profile's enabled flag / log path bleed into every later profile.
    """
    from cron.audit import audit_log_path, is_audit_enabled, reload_audit_config

    enabled_home = tmp_path / "enabled" / ".hermes"
    disabled_home = tmp_path / "disabled" / ".hermes"
    (enabled_home / "cron").mkdir(parents=True)
    (disabled_home / "cron").mkdir(parents=True)

    custom_log = enabled_home / "cron" / "custom-audit.log"
    (enabled_home / "config.yaml").write_text(
        yaml.safe_dump({"cron": {"audit_log": True, "audit_log_path": str(custom_log)}})
    )
    # disabled_home has no config.yaml -> audit stays off.

    reload_audit_config()

    # Populate the cache from the enabled profile first.
    monkeypatch.setenv("HERMES_HOME", str(enabled_home))
    assert is_audit_enabled() is True
    assert audit_log_path() == custom_log

    # Switch profiles WITHOUT reloading: a per-home cache must recompute for
    # the disabled profile rather than serving the enabled profile's entry.
    monkeypatch.setenv("HERMES_HOME", str(disabled_home))
    assert is_audit_enabled() is False
    assert audit_log_path() == disabled_home / "cron" / "audit.log"
