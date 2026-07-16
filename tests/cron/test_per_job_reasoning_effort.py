"""Tests for the per-job ``reasoning_effort`` override (cron jobs).

Covers the change that lets a single cron job pin its own reasoning effort
without touching the global ``config.yaml`` ``agent.reasoning_effort`` that every
session and other cron inherits:

  - cron/jobs.py     create_job/update_job persist (and clear) the field
  - tools/cronjob_tools.py  the cronjob tool validates + threads the param
  - cron/scheduler.py  per-job value wins; invalid/absent falls back to config

The scheduler's resolution lives inside a large run-job function, so the
resolution case asserts the exact decision expression against the real
``parse_reasoning_effort`` (the same call the scheduler makes).
"""
import json
import os

import pytest

import cron.jobs as jobs_mod
from cron.jobs import create_job, update_job, get_job
from hermes_constants import parse_reasoning_effort


# ---------------------------------------------------------------------------
# Self-isolation (2026-07-15 fixture-leak incident).
#
# This file creates jobs with prompt="brief" — the exact fixture that
# cron-config-lint (and eval 1c4a141bd9a9adc6) treats as junk in the LIVE
# ~/.hermes/cron/jobs.json. Normal pytest runs are hermetic via the autouse
# conftest fixture, but this module has leaked via harnesses that run it
# OUTSIDE that fixture (real-agent blackbox session / kanban worker importing
# it from a worktree sharing the live HERMES_HOME). It therefore isolates
# ITSELF and never relies on conftest alone — see the twin block in
# test_ticker_stall_60703.py for the full rationale.
# ---------------------------------------------------------------------------
if not os.environ.get("PYTEST_VERSION"):  # pragma: no cover - non-pytest harness
    import tempfile as _tempfile

    # TemporaryDirectory (not mkdtemp): its finalizer removes the dir at
    # interpreter exit, so repeated harness runs don't accumulate orphans.
    _standalone_tmp = _tempfile.TemporaryDirectory(
        prefix="cron-test-home-reasoning-effort-"
    )
    _standalone_store = jobs_mod.use_cron_store(_standalone_tmp.name)
    _standalone_store.__enter__()  # held for the process lifetime, by design


@pytest.fixture(autouse=True)
def _self_isolated_cron_store(tmp_path):
    """Belt-and-suspenders: never write 'brief' fixture jobs to a real store."""
    with jobs_mod.use_cron_store(tmp_path):
        yield


# ---------------------------------------------------------------------------
# Persistence: create_job / update_job
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_create_persists_reasoning_effort(self):
        job = create_job(prompt="brief", schedule="every 1h", reasoning_effort="medium")
        assert job["reasoning_effort"] == "medium"
        assert get_job(job["id"])["reasoning_effort"] == "medium"

    def test_create_omits_field_when_unset_byte_identical(self):
        # An unset override must NOT add the key, so existing jobs stay
        # byte-identical and the scheduler falls back to config.yaml.
        job = create_job(prompt="brief", schedule="every 1h")
        assert "reasoning_effort" not in job

    def test_update_sets_reasoning_effort(self):
        job = create_job(prompt="brief", schedule="every 1h")
        update_job(job["id"], {"reasoning_effort": "low"})
        assert get_job(job["id"])["reasoning_effort"] == "low"

    def test_update_clears_with_empty_string(self):
        job = create_job(prompt="brief", schedule="every 1h", reasoning_effort="xhigh")
        # Empty string => clear the override (restore config fallback).
        update_job(job["id"], {"reasoning_effort": None})
        assert get_job(job["id"]).get("reasoning_effort") in (None, "")


# ---------------------------------------------------------------------------
# The cronjob tool: validation + threading
# ---------------------------------------------------------------------------

class TestTool:
    def _call(self, **kwargs):
        from tools.cronjob_tools import cronjob
        return json.loads(cronjob(**kwargs))

    def test_tool_create_threads_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRONJOB_TOOL", "1")
        res = self._call(
            action="create", prompt="brief", schedule="every 1h",
            reasoning_effort="medium",
        )
        assert res.get("success") is True
        assert get_job(res["job_id"])["reasoning_effort"] == "medium"

    def test_tool_create_accepts_max_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRONJOB_TOOL", "1")
        res = self._call(
            action="create", prompt="brief", schedule="every 1h",
            reasoning_effort="max",
        )
        assert res.get("success") is True
        assert get_job(res["job_id"])["reasoning_effort"] == "max"

    def test_tool_update_threads_reasoning_effort(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRONJOB_TOOL", "1")
        created = self._call(action="create", prompt="brief", schedule="every 1h")
        res = self._call(
            action="update", job_id=created["job_id"], reasoning_effort="low",
        )
        assert res.get("success") is True
        assert get_job(created["job_id"])["reasoning_effort"] == "low"

    def test_tool_rejects_invalid_value(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRONJOB_TOOL", "1")
        created = self._call(action="create", prompt="brief", schedule="every 1h")
        res = self._call(
            action="update", job_id=created["job_id"], reasoning_effort="garbage",
        )
        assert res.get("success") is False
        assert "reasoning_effort" in res.get("error", "").lower() or "invalid" in res.get("error", "").lower()
        # And it must NOT have written the bad value.
        assert get_job(created["job_id"]).get("reasoning_effort") in (None, "")


# ---------------------------------------------------------------------------
# Scheduler resolution precedence (the exact decision the scheduler makes)
# ---------------------------------------------------------------------------

def _resolve(job_effort: str, config_effort: str):
    """Replicate cron/scheduler.py's per-job > config resolution verbatim."""
    reasoning_config = None
    _job_effort = str(job_effort or "").strip()
    if _job_effort:
        reasoning_config = parse_reasoning_effort(_job_effort)
    if reasoning_config is None:
        reasoning_config = parse_reasoning_effort(str(config_effort or "").strip())
    return reasoning_config


class TestResolution:
    def test_per_job_value_wins_over_config(self):
        assert _resolve("medium", "xhigh") == {"enabled": True, "effort": "medium"}

    def test_no_job_value_falls_back_to_config(self):
        # The unchanged path: no per-job field => config.yaml value.
        assert _resolve("", "xhigh") == {"enabled": True, "effort": "xhigh"}

    def test_invalid_job_value_falls_back_to_config(self):
        # Fail-safe: a typo'd effort must not break the job — fall to config.
        assert _resolve("garbage", "xhigh") == {"enabled": True, "effort": "xhigh"}

    def test_lowest_level_minimal_resolves(self):
        assert _resolve("minimal", "xhigh") == {"enabled": True, "effort": "minimal"}

    def test_none_disables_reasoning(self):
        assert _resolve("none", "xhigh") == {"enabled": False}

    def test_max_resolves(self):
        assert _resolve("max", "medium") == {"enabled": True, "effort": "max"}
