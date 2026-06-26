"""Tests for the cronjob ``get`` action — full-prompt retrieval (issue #18374).

``list`` returns only ``prompt_preview`` (first 100 chars); ``get`` must return
the complete prompt so sandboxed agents can audit/edit a job's prompt without
direct jobs.json access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

LONG_PROMPT = "Monitor solar health and alert on anomalies. " + "detail " * 40  # >100 chars


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def test_get_returns_full_prompt(cron_env):
    from cron.jobs import create_job
    from tools.cronjob_tools import cronjob

    job = create_job(prompt=LONG_PROMPT, schedule="every 1h")
    result = json.loads(cronjob(action="get", job_id=job["id"]))

    assert result["success"] is True
    assert result["job"]["prompt"] == LONG_PROMPT
    assert "prompt_preview" not in result["job"]
    assert len(LONG_PROMPT) > 100  # guard: prompt is long enough to be truncated by list


def test_list_only_returns_truncated_preview(cron_env):
    """Regression guard: list stays a 100-char preview (the asymmetry get fixes)."""
    from cron.jobs import create_job
    from tools.cronjob_tools import cronjob

    create_job(prompt=LONG_PROMPT, schedule="every 1h")
    job = json.loads(cronjob(action="list"))["jobs"][0]

    assert job["prompt_preview"].endswith("...")
    assert len(job["prompt_preview"]) < len(LONG_PROMPT)
    assert "prompt" not in job


def test_get_unknown_job_returns_error(cron_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(cronjob(action="get", job_id="does-not-exist"))
    assert result["success"] is False


def test_schema_action_description_lists_get():
    from tools.cronjob_tools import CRONJOB_SCHEMA

    desc = CRONJOB_SCHEMA["parameters"]["properties"]["action"]["description"]
    assert "get" in desc
