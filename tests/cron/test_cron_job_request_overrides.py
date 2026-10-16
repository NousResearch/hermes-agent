"""Invariant tests: per-job request_overrides reach the cron agent runtime.

A cron job dict may carry top-level ``request_overrides`` (top-level keys = API kwargs,
e.g. ``service_tier``), mirroring delegation.request_overrides semantics. The scheduler
merges them into the resolved runtime in ``_resolve_job_runtime`` (both the primary and the
fallback-chain paths), and ``_construct_cron_agent`` already forwards
``runtime.get("request_overrides")`` into AIAgent — so the merged value must land on the
constructed agent.

Contract under test:
  (a) a job with ``request_overrides`` produces an AIAgent carrying the merged overrides,
      even when the provider resolver returns a plain dict WITHOUT its own
      ``request_overrides`` key (the current empirical behaviour);
  (b) a job WITHOUT the key behaves exactly as before — no ``request_overrides`` key is
      added to the runtime and the agent sees none.
These are invariants, not change-detectors: they pin the observable behaviour, not internal
implementation details.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "req-overrides-job",
        "name": "req overrides job",
        "prompt": "hello",
        "model": None,
        "provider": None,
        "provider_snapshot": None,
        "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, tmp_path):
    """Drive run_job with resolve_runtime_provider returning a plain dict (no
    ``request_overrides`` key — matching current resolver behaviour). Returns
    ``(success, error, agent_kwargs)``; agent_kwargs is None when AIAgent was never built."""
    (tmp_path / "config.yaml").write_text("")

    def _resolve(**kwargs):
        return {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": kwargs.get("requested") or "openrouter",
            "api_mode": "chat_completions",
        }

    fake_db = MagicMock()
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, _output, _final, error = run_job(job)
        agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None
    return success, error, agent_kwargs


class TestJobLevelRequestOverrides:
    def test_job_request_overrides_merge_into_runtime_reaching_agent(self, tmp_path):
        job = _base_job(request_overrides={"service_tier": "flex"})
        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None
        assert agent_kwargs.get("request_overrides") == {"service_tier": "flex"}

    def test_job_without_request_overrides_adds_none(self, tmp_path):
        """Absent job key is a None-safe no-op: no request_overrides value is introduced.

        ``_construct_cron_agent`` always forwards ``runtime.get("request_overrides")``
        (None when the key is absent), so the invariant is that the value stays None —
        matching today's behaviour exactly.
        """
        job = _base_job()
        success, error, agent_kwargs = _run(job, tmp_path)

        assert success is True, error
        assert agent_kwargs is not None
        assert agent_kwargs.get("request_overrides") is None

    def test_job_explicit_overrides_win_over_provider_resolved(self, tmp_path):
        """Explicit job values override any request_overrides the resolver may have returned."""
        job = _base_job(request_overrides={"service_tier": "flex"})
        (tmp_path / "config.yaml").write_text("")

        def _resolve(**kwargs):
            return {
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "request_overrides": {"service_tier": "standard"},
            }

        fake_db = MagicMock()
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._get_hermes_home", return_value=tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_resolve), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent
            success, _output, _final, error = run_job(job)
            agent_kwargs = mock_agent_cls.call_args.kwargs if mock_agent_cls.called else None

        assert success is True, error
        assert agent_kwargs is not None
        assert agent_kwargs.get("request_overrides") == {"service_tier": "flex"}
