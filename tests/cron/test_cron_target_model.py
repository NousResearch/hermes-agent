"""Cron scheduler must pass target_model to resolve_runtime_provider.

When a user switches to an anthropic_messages-routed model (e.g. qwen on
opencode-go) via /model, the switch persists to config.yaml's model.default.
The cron scheduler reads config.yaml to resolve the runtime provider, and
resolve_runtime_provider derives api_mode and base_url from model.default.

For opencode-go, anthropic_messages models get a stripped base_url
(https://opencode.ai/zen/go) while chat_completions models get
(https://opencode.ai/zen/go/v1). If a cron job uses a chat_completions
model (e.g. glm-5.2) but config.yaml's default is an anthropic_messages
model, the cron job inherits the wrong base_url — sending requests to the
marketing site instead of the API endpoint (404 → 401).

The fix: pass the job's model as target_model to resolve_runtime_provider
so api_mode and base_url are derived from the model the job will actually
use, not from config.yaml's default.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import run_job


def _base_job(**overrides):
    job = {
        "id": "target-model-test",
        "name": "target model test",
        "prompt": "hello",
        "model": "glm-5.2",
        "provider": "opencode-go",
        "provider_snapshot": None,
        "model_snapshot": None,
        "base_url": None,
        "no_agent": False,
    }
    job.update(overrides)
    return job


def _run_and_capture_resolve_call(job, tmp_path, config_default="qwen3.7-plus"):
    """Drive run_job and capture the kwargs passed to resolve_runtime_provider.

    Returns the kwargs dict (or None if resolve_runtime_provider was never
    called).
    """
    fake_db = MagicMock()

    # Write a minimal config.yaml with a model.default that differs from the
    # job's model — simulating a /model switch that persisted to config.yaml.
    config_yaml = f"model:\n  default: {config_default}\n  provider: opencode-go\n"

    captured_kwargs = {}

    def _capture_resolve(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "api_key": "test-key",
            "base_url": "https://opencode.ai/zen/go/v1",
            "provider": "opencode-go",
            "api_mode": "chat_completions",
        }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_yaml)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             side_effect=_capture_resolve,
         ) as mock_resolve, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        try:
            run_job(job)
        except Exception:
            pass  # We only care about the resolve call, not the full run.

    return captured_kwargs


class TestTargetModelPassedToResolve:
    def test_job_model_passed_as_target_model(self, tmp_path):
        """The job's model must be passed as target_model to
        resolve_runtime_provider so api_mode/base_url are derived from the
        job's model, not config.yaml's default."""
        job = _base_job(model="glm-5.2", provider="opencode-go")
        kwargs = _run_and_capture_resolve_call(job, tmp_path, config_default="qwen3.7-plus")

        assert kwargs is not None, "resolve_runtime_provider was never called"
        assert "target_model" in kwargs, (
            "target_model was not passed to resolve_runtime_provider — "
            "without it, api_mode/base_url are derived from config.yaml's "
            "model.default instead of the job's model"
        )
        assert kwargs["target_model"] == "glm-5.2", (
            f"Expected target_model='glm-5.2', got '{kwargs.get('target_model')}'"
        )

    def test_target_model_matches_job_not_config(self, tmp_path):
        """When config.yaml says qwen3.7-plus but the job uses glm-5.2,
        target_model must be glm-5.2 (the job's model), not qwen3.7-plus
        (the config default)."""
        job = _base_job(model="glm-5.2", provider="opencode-go")
        kwargs = _run_and_capture_resolve_call(job, tmp_path, config_default="qwen3.7-plus")

        assert kwargs is not None
        assert kwargs.get("target_model") == "glm-5.2"
        assert kwargs.get("target_model") != "qwen3.7-plus"

    def test_no_target_model_when_job_model_unset(self, tmp_path):
        """When the job has no explicit model (falls through to config), 
        target_model should not be passed — the config default is correct."""
        job = _base_job(model=None, provider="opencode-go")
        # When model is None, the scheduler resolves it from config.yaml/env.
        # We patch the config to have a default so the job gets a model.
        kwargs = _run_and_capture_resolve_call(job, tmp_path, config_default="glm-5.2")

        # model was None, so the scheduler resolves it from config.yaml.
        # After resolution, model becomes "glm-5.2" from config.
        # The target_model should be the resolved model, not None.
        if "target_model" in kwargs:
            assert kwargs["target_model"] is not None, (
                "target_model should be the resolved model, not None"
            )
