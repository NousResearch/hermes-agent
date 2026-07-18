"""Tests for per-job memory_enabled override in cron jobs.

Covers:
  - jobs.create_job: param plumbing for memory_enabled, default-None preservation
  - scheduler.run_job: skip_memory wiring (memory_enabled=true -> skip_memory=False,
    unset/false -> skip_memory=True, unchanged from pre-PR behavior)
  - tools.cronjob_tools: create + update JSON round-trip for memory_enabled,
    schema includes memory_enabled, _format_job exposes it when set
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate cron job storage into a temp dir so tests don't stomp on real jobs."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


# ---------------------------------------------------------------------------
# jobs.create_job: memory_enabled param plumbing
# ---------------------------------------------------------------------------

class TestCreateJobMemoryEnabled:
    def test_memory_enabled_true_stored(self, tmp_cron_dir):
        """When memory_enabled=True is passed, it is persisted in the job."""
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="hello",
            schedule="every 1h",
            memory_enabled=True,
        )
        stored = get_job(job["id"])
        assert stored["memory_enabled"] is True

    def test_memory_enabled_false_stored(self, tmp_cron_dir):
        """When memory_enabled=False is passed explicitly, it is persisted."""
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="hello",
            schedule="every 1h",
            memory_enabled=False,
        )
        stored = get_job(job["id"])
        assert stored["memory_enabled"] is False

    def test_memory_enabled_none_not_stored(self, tmp_cron_dir):
        """When memory_enabled is left at its default (None), no key is written
        to jobs.json -- the absent key means 'use the cron default of
        skip_memory=True' so existing jobs stay byte-identical."""
        from cron.jobs import create_job, get_job

        job = create_job(prompt="hello", schedule="every 1h")
        stored = get_job(job["id"])
        # The key should not be present at all (not even None), because
        # create_job only writes it when isinstance(memory_enabled, bool).
        assert "memory_enabled" not in stored


# ---------------------------------------------------------------------------
# scheduler.run_job: skip_memory wiring
# ---------------------------------------------------------------------------

class TestRunJobSkipMemory:
    """run_job must wire skip_memory based on the per-job memory_enabled field:
    - memory_enabled=True -> skip_memory=False (memory IS active)
    - memory_enabled=False/unset -> skip_memory=True (memory stays off)

    We stub AIAgent so no real API call happens, capturing the
    skip_memory kwarg passed to its __init__.
    """

    @staticmethod
    def _install_stubs(monkeypatch, observed: dict):
        """Patch enough of run_job's deps that it executes without real creds."""
        import sys
        import cron.scheduler as sched

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["skip_memory"] = kwargs.get("skip_memory")
                observed["skip_context_files"] = kwargs.get("skip_context_files")

            def run_conversation(self, *_a, **_kw):
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        # Bypass the real provider resolver.
        from hermes_cli import runtime_provider as _rtp
        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )

        monkeypatch.setattr(sched, "_build_job_prompt", lambda job, prerun_script=None: "hi")
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None)
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    def test_memory_enabled_true_disables_skip_memory(self, tmp_path, monkeypatch):
        """memory_enabled=True -> skip_memory=False (memory provider is loaded)."""
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "mem-yes",
            "name": "mem-enabled-job",
            "memory_enabled": True,
            "schedule_display": "manual",
        }

        success, _output, response, error = sched.run_job(job)
        assert success is True, f"run_job failed: error={error!r}"
        assert observed["skip_memory"] is False

    def test_memory_enabled_false_keeps_skip_memory_true(self, tmp_path, monkeypatch):
        """memory_enabled=False explicitly -> skip_memory=True (unchanged default)."""
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "mem-no",
            "name": "mem-disabled-job",
            "memory_enabled": False,
            "schedule_display": "manual",
        }

        success, *_ = sched.run_job(job)
        assert success is True
        assert observed["skip_memory"] is True

    def test_memory_enabled_unset_keeps_skip_memory_true(self, tmp_path, monkeypatch):
        """When memory_enabled is absent from the job dict (default-None case),
        skip_memory must stay True -- backward-compatible pre-PR behavior."""
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        job = {
            "id": "mem-unset",
            "name": "mem-unset-job",
            # memory_enabled key intentionally absent
            "schedule_display": "manual",
        }

        success, *_ = sched.run_job(job)
        assert success is True
        assert observed["skip_memory"] is True


# ---------------------------------------------------------------------------
# tools.cronjob_tools: end-to-end JSON round-trip
# ---------------------------------------------------------------------------

class TestCronjobToolMemoryEnabled:
    def test_create_with_memory_enabled_true_json_roundtrip(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                memory_enabled=True,
            )
        )
        assert result["success"] is True
        assert result["job"]["memory_enabled"] is True

    def test_create_without_memory_enabled_hides_field_in_format(self, tmp_cron_dir):
        """When memory_enabled is not set, _format_job should omit it."""
        from tools.cronjob_tools import cronjob

        result = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
            )
        )
        assert result["success"] is True
        assert "memory_enabled" not in result["job"]

    def test_update_sets_memory_enabled(self, tmp_cron_dir):
        from tools.cronjob_tools import cronjob

        created = json.loads(
            cronjob(action="create", prompt="hi", schedule="every 1h")
        )
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, memory_enabled=True)
        )
        assert updated["success"] is True
        assert updated["job"]["memory_enabled"] is True

    def test_update_sets_memory_enabled_false(self, tmp_cron_dir):
        """Update with memory_enabled=False persists to storage, but
        _format_job omits falsy fields (same pattern as workdir, no_agent, etc.).
        We verify the storage directly via load_jobs."""
        from tools.cronjob_tools import cronjob
        from cron.jobs import load_jobs

        created = json.loads(
            cronjob(
                action="create",
                prompt="hi",
                schedule="every 1h",
                memory_enabled=True,
            )
        )
        job_id = created["job_id"]

        updated = json.loads(
            cronjob(action="update", job_id=job_id, memory_enabled=False)
        )
        assert updated["success"] is True
        # _format_job omits falsy fields, so we check the raw stored job.
        stored = [j for j in load_jobs() if j["id"] == job_id][0]
        assert stored.get("memory_enabled") is False

    def test_schema_advertises_memory_enabled(self):
        from tools.cronjob_tools import CRONJOB_SCHEMA

        props = CRONJOB_SCHEMA["parameters"]["properties"]
        assert "memory_enabled" in props, "memory_enabled must be in CRONJOB_SCHEMA"
        desc = props["memory_enabled"]["description"]
        assert "memory" in desc.lower()

    def test_registry_handler_plumbs_memory_enabled(self):
        """The registry handler lambda must pass memory_enabled through to
        cronjob() so that the tool works when called via the tool schema."""
        import inspect
        from tools.cronjob_tools import registry

        tool = registry.get_entry("cronjob")
        source = inspect.getsource(tool.handler)
        assert "memory_enabled" in source, (
            "registry handler must forward memory_enabled to cronjob()"
        )
