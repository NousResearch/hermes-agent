"""Cron job field + scheduler kwargs for memory_provider_mode (PR #18565 salvage).

Cron always keeps skip_memory=True (protect MEMORY.md/USER.md). Per-job
memory_provider may opt external providers into tools|full (default off).
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from cron.jobs import create_job, get_job, update_job
from cron.scheduler import run_job


@pytest.fixture
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestCreateJobMemoryProvider:
    def test_omitted_does_not_persist_field(self, tmp_cron_dir):
        job = create_job(prompt="check", schedule="every 1h")
        assert "memory_provider" not in job
        fetched = get_job(job["id"])
        assert not fetched.get("memory_provider")

    def test_off_does_not_persist_field(self, tmp_cron_dir):
        job = create_job(prompt="check", schedule="every 1h", memory_provider="off")
        assert "memory_provider" not in job

    def test_tools_persisted(self, tmp_cron_dir):
        job = create_job(prompt="check", schedule="every 1h", memory_provider="tools")
        assert job["memory_provider"] == "tools"
        assert get_job(job["id"])["memory_provider"] == "tools"

    def test_full_persisted_case_insensitive(self, tmp_cron_dir):
        job = create_job(prompt="check", schedule="every 1h", memory_provider="FULL")
        assert job["memory_provider"] == "full"

    def test_invalid_rejected(self, tmp_cron_dir):
        with pytest.raises(ValueError, match="Invalid memory_provider"):
            create_job(prompt="check", schedule="every 1h", memory_provider="maybe")

    def test_bool_rejected(self, tmp_cron_dir):
        with pytest.raises(ValueError, match="memory_provider must be"):
            create_job(prompt="check", schedule="every 1h", memory_provider=True)

    def test_update_set_and_clear(self, tmp_cron_dir):
        job = create_job(prompt="check", schedule="every 1h")
        updated = update_job(job["id"], {"memory_provider": "tools"})
        assert updated["memory_provider"] == "tools"
        cleared = update_job(job["id"], {"memory_provider": ""})
        # Cleared value becomes None (scheduler treats falsy as off)
        assert not cleared.get("memory_provider")


class TestRunJobMemoryProviderKwargs:
    @contextlib.contextmanager
    def _run_job_patches(self, tmp_path):
        fake_db = MagicMock()
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        base = [
            patch("cron.scheduler._hermes_home", tmp_path),
            patch("cron.scheduler._resolve_origin", return_value=None),
            patch("hermes_cli.env_loader.load_hermes_dotenv"),
            patch("hermes_cli.env_loader.reset_secret_source_cache"),
            patch("hermes_state.SessionDB", return_value=fake_db),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "test-key",
                    "base_url": "https://example.invalid/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                },
            ),
            patch("run_agent.AIAgent", return_value=mock_agent),
        ]
        with contextlib.ExitStack() as stack:
            entered = [stack.enter_context(cm) for cm in base]
            yield entered[-1]  # AIAgent class mock

    def test_default_job_passes_mode_off_and_skip_memory(self, tmp_path):
        job = {"id": "mem-default", "name": "test", "prompt": "hello"}
        with self._run_job_patches(tmp_path) as mock_agent_cls:
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["skip_memory"] is True
        assert kwargs["memory_provider_mode"] == "off"

    def test_tools_job_passes_mode_tools(self, tmp_path):
        job = {
            "id": "mem-tools",
            "name": "test",
            "prompt": "hello",
            "memory_provider": "tools",
        }
        with self._run_job_patches(tmp_path) as mock_agent_cls:
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["skip_memory"] is True
        assert kwargs["memory_provider_mode"] == "tools"

    def test_full_job_passes_mode_full(self, tmp_path):
        job = {
            "id": "mem-full",
            "name": "test",
            "prompt": "hello",
            "memory_provider": "full",
        }
        with self._run_job_patches(tmp_path) as mock_agent_cls:
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["skip_memory"] is True
        assert kwargs["memory_provider_mode"] == "full"

    def test_empty_memory_provider_field_defaults_to_off(self, tmp_path):
        job = {
            "id": "mem-empty",
            "name": "test",
            "prompt": "hello",
            "memory_provider": None,
        }
        with self._run_job_patches(tmp_path) as mock_agent_cls:
            run_job(job)

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["memory_provider_mode"] == "off"
