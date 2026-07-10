"""Cron inference routing: inherited, pinned, and no-agent jobs.

An agent-backed job with no explicit provider/model must follow the current
main inference configuration on every run. Explicit per-job values remain
pinned. ``no_agent`` jobs bypass inference entirely.

Legacy ``provider_snapshot`` / ``model_snapshot`` fields may still exist in
jobs.json after upgrades; they must not change inherited routing semantics.
"""

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "routing-test",
        "name": "routing test",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run_agent_job(job, current_provider, current_model, tmp_path):
    """Run one agent-backed job against controlled global inference config."""
    (tmp_path / "config.yaml").write_text(
        f"model:\n  default: {current_model}\n",
        encoding="utf-8",
    )
    fake_db = MagicMock()

    def _resolve_runtime_provider(*, requested=None, **_kwargs):
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": requested or current_provider,
            "api_mode": "chat_completions",
        }

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             side_effect=_resolve_runtime_provider,
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        success, output, final_response, error = run_job(job)
        agent_kwargs = (
            mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None
        )

    return success, output, final_response, error, agent_kwargs


class TestCronInferenceRouting:
    def test_unpinned_job_follows_current_global_provider_and_model(self, tmp_path):
        job = _base_job(
            # Stale fields from the removed drift guard must be ignored.
            provider_snapshot="old-provider",
            model_snapshot="old-model",
        )

        success, _output, final_response, error, agent_kwargs = _run_agent_job(
            job,
            current_provider="current-provider",
            current_model="current-model",
            tmp_path=tmp_path,
        )

        assert success is True
        assert error is None
        assert final_response == "ok"
        assert agent_kwargs is not None
        assert agent_kwargs["provider"] == "current-provider"
        assert agent_kwargs["model"] == "current-model"

    def test_explicit_provider_and_model_remain_pinned(self, tmp_path):
        job = _base_job(
            provider="pinned-provider",
            model="pinned-model",
            provider_snapshot="old-provider",
            model_snapshot="old-model",
        )

        success, _output, _final_response, error, agent_kwargs = _run_agent_job(
            job,
            current_provider="current-provider",
            current_model="current-model",
            tmp_path=tmp_path,
        )

        assert success is True
        assert error is None
        assert agent_kwargs is not None
        assert agent_kwargs["provider"] == "pinned-provider"
        assert agent_kwargs["model"] == "pinned-model"


class TestCronJobStorage:
    @staticmethod
    def _isolate_storage(monkeypatch):
        import cron.jobs as jobs

        @contextlib.contextmanager
        def _noop_lock():
            yield

        monkeypatch.setattr(jobs, "_jobs_lock", _noop_lock, raising=True)
        monkeypatch.setattr(jobs, "load_jobs", lambda: [], raising=True)
        monkeypatch.setattr(jobs, "save_jobs", lambda _jobs: None, raising=True)
        return jobs

    def test_unpinned_job_stores_no_inference_snapshot(self, monkeypatch):
        jobs = self._isolate_storage(monkeypatch)

        job = jobs.create_job(prompt="do a thing", schedule="every 1 hour")

        assert job["provider"] is None
        assert job["model"] is None
        assert "provider_snapshot" not in job
        assert "model_snapshot" not in job

    def test_explicit_provider_and_model_are_stored(self, monkeypatch):
        jobs = self._isolate_storage(monkeypatch)

        job = jobs.create_job(
            prompt="do a thing",
            schedule="every 1 hour",
            provider="pinned-provider",
            model="pinned-model",
        )

        assert job["provider"] == "pinned-provider"
        assert job["model"] == "pinned-model"

    def test_no_agent_job_has_no_inference_routing(self, monkeypatch):
        jobs = self._isolate_storage(monkeypatch)

        job = jobs.create_job(
            prompt="ignored",
            schedule="every 1 hour",
            script="watchers/example.py",
            no_agent=True,
        )

        assert job["no_agent"] is True
        assert job["provider"] is None
        assert job["model"] is None
        assert "provider_snapshot" not in job
        assert "model_snapshot" not in job
